"""
This file is derived from the following repository:
https://github.com/CyberAgentAILab/layout-dm/tree/9cf122d3be74b1ed8a8d51e3caabcde8164d238e

Original file: src/trainer/trainer/main.py
Author: naoto0804
License: Apache-2.0 License

Modifications have been made to the original file to fit the requirements of this project.
"""

import json
import logging
import os
import shutil
import sys
import time

import hydra
import torch
from fsspec.core import url_to_fs
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from timm.utils import ModelEmaV2
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader  # noqa
from trainer.data.util import compose_transform, sparse_to_dense, split_num_samples
from trainer.fid.model import load_fidnet_v3
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.layout_diffusion_tokenizer import LayoutDiffusionTokenizer
from trainer.helpers.metric import compute_generative_model_scores
from trainer.helpers.sampling import register_sampling_config
from trainer.helpers.scheduler import ReduceLROnPlateauWithWarmup
from trainer.helpers.util import set_seed, dict2str
from trainer.helpers.visualization import save_image
from trainer.hydra_configs import DataConfig, TrainConfig
from trainer.models.common.util import save_model
from typing import Optional

from .crossplatform_util import filter_args_for_ai_platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "1"  # to see full tracelog for hydra

torch.autograd.set_detect_anomaly(True)
total_iter_count = 0

WEIGHTS_DIR = './download/pretrained_weights/'


def _to(inputs, device):
    """
    recursively send tensor to the specified device
    """
    outputs = {}
    for k, v in inputs.items():
        if isinstance(v, dict):
            outputs[k] = _to(v, device)
        elif isinstance(v, Tensor):
            outputs[k] = v.to(device)
    return outputs


# if config is not used by hydra.utils.instantiate, define schema to validate args
cs = ConfigStore.instance()
cs.store(group="data", name="base_data_default", node=DataConfig)
cs.store(group="training", name="base_training_default", node=TrainConfig)
register_sampling_config(cs)

def build_layoutdm_model(cfg, tokenizer, **kwargs):
    try:
        model = instantiate(cfg.model)(backbone_cfg=cfg.backbone, 
                                    tokenizer=tokenizer,
                                    use_padding_as_vocab=cfg.model.use_padding_as_vocab,
                                    **kwargs)
    except:
        model = instantiate(cfg.model)(backbone_cfg=cfg.backbone, 
                                    tokenizer=tokenizer,
                                    **kwargs)
    return model


def build_tokenizer(cfg):
    if "LayoutDiffusion" in cfg.model._target_:
        tokenizer = LayoutDiffusionTokenizer(data_cfg=cfg.data, dataset_cfg=cfg.dataset)
    else:
        tokenizer = LayoutSequenceTokenizer(data_cfg=cfg.data, dataset_cfg=cfg.dataset)
    return tokenizer


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    global total_iter_count
    if cfg.debug:
        _job_dir = cfg.job_dir
        dbg_job_dir = os.path.split(_job_dir)[0] + '/DEBUG_' + os.path.split(_job_dir)[1]
        cfg.job_dir = dbg_job_dir
    job_dir = os.path.join(cfg.job_dir, str(cfg.seed))

    fs, _ = url_to_fs(job_dir)
    if not fs.exists(job_dir):
        fs.mkdir(job_dir)

    handler = logging.FileHandler(os.path.join(job_dir, "train.log"))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s  %(asctime)s  [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    writer = SummaryWriter(os.path.join(job_dir, "logs"))

    _indent = ' ' * 4
    logger.info(f'ARGS:\n{_indent}' + f'\n{_indent}'.join(sys.argv[1:]))
    logger.info('train_config:\n' + dict2str(cfg, level=0))

    if cfg.debug:
        cfg.data.num_workers = 1
        cfg.training.epochs = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = compose_transform(cfg.data.transforms)
    train_dataset = instantiate(cfg.dataset)(split="train", transform=transform)
    val_dataset = instantiate(cfg.dataset)(split="val", transform=transform)

    if cfg.debug and cfg.training.debug_num_samples > 0:
        train_dataset = train_dataset[:cfg.training.debug_num_samples]

    logger.info(f'len(train_dastet) = {len(train_dataset)}')
    logger.info(f'len(val_dataset) = {len(val_dataset)}')

    kwargs = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": True,
    }
    train_dataloader = DataLoader(train_dataset, shuffle=True, **kwargs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **kwargs)
    tokenizer = build_tokenizer(cfg)

    # Load pretrained model (For fine-tuning)
    if cfg.training.pretrain_job_dir:
        pretrain_job_dir = os.path.join(cfg.training.pretrain_job_dir, str(cfg.seed))
        pretrain_cfg_path = os.path.join(pretrain_job_dir, "config.yaml")

        with open(pretrain_cfg_path, "rb") as f:
            pretrain_cfg = OmegaConf.load(f)

        # Set same pos_emb_length as pretraining to avoid size-mismatch
        max_seq_length = pretrain_cfg.dataset.max_seq_length
        N_var_per_element= len(pretrain_cfg.data.get("var_order", "c-x-y-w-h").split('-'))
        max_token_length = max_seq_length * N_var_per_element
        model = build_layoutdm_model(cfg, tokenizer, pos_emb_length=max_token_length)
        cfg.model.pos_emb_length = max_token_length

        # Load pretrained model (best or final)
        best_or_final = cfg.training.pretrain_model_type
        assert best_or_final in ["best", "final"]
        model_path = os.path.join(pretrain_job_dir, f"{best_or_final}_model.pt")
        logger.info(f"Load pretrained_model: {model_path}")
        fs, _ = url_to_fs(model_path)
        with fs.open(model_path, "rb") as file_obj:
            model.load_state_dict(torch.load(file_obj))
    else:
        model = build_layoutdm_model(cfg, tokenizer)

    model = model.to(device)

    use_ema = cfg.training.ema_decay > 0.0
    if use_ema:
        ema_model = ModelEmaV2(
            model,
            decay=cfg.training.ema_decay,
            device=device,
        )
    else:
        ema_model = None

    optim_groups = model.optim_groups(cfg.training.weight_decay)
    optimizer = instantiate(cfg.optimizer)(optim_groups)
    scheduler = instantiate(cfg.scheduler)(optimizer=optimizer)

    with fs.open(os.path.join(job_dir, "config.yaml"), "wb") as file_obj:
        file_obj.write(OmegaConf.to_yaml(cfg).encode("utf-8"))

    if not cfg.pretrain:
        fid_model = load_fidnet_v3(train_dataset, cfg.fid_weight_dir, device)

    best_val_loss = float("Inf")
    for epoch in range(cfg.training.epochs):
        model.update_per_epoch(epoch, cfg.training.epochs)

        start_time = time.time()
        train_loss = train(model, train_dataloader, optimizer, cfg, device, writer, ema_model)
        if use_ema:
            val_loss = evaluate(ema_model.module, val_dataloader, cfg, device)
        else:
            val_loss = evaluate(model, val_dataloader, cfg, device)
        logger.info(
            "Epoch %d: elapsed = %.1fs, train_loss = %.4f, val_loss = %.4f, lr = %.7f"
            % (epoch + 1, time.time() - start_time, train_loss, val_loss, optimizer.param_groups[0]["lr"])
        )
        if any(
            isinstance(scheduler, s)
            for s in [ReduceLROnPlateau, ReduceLROnPlateauWithWarmup]
        ):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
        writer.add_scalar("train_loss_epoch_avg", train_loss, epoch + 1)
        writer.add_scalar("val_loss_epoch_avg", val_loss, epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, job_dir, best_or_final="best")
            if use_ema:
                save_model(ema_model.module, job_dir, best_or_final="ema_best")

        if cfg.training.save_step > 0 and (epoch + 1) % cfg.training.save_step == 0:
            save_model(model, job_dir, best_or_final=f"epoch{epoch + 1}")
            if use_ema:
                save_model(ema_model.module, job_dir, best_or_final=f"ema_epoch{epoch + 1}")

        if cfg.pretrain:
            continue

        if (epoch + 1) % cfg.training.sample_plot_epoch_interval == 0:
            test_model = ema_model.module if use_ema else model

            with torch.set_grad_enabled(False):
                layouts = test_model.sample(
                    batch_size=cfg.data.batch_size,
                    sampling_cfg=cfg.sampling,
                    device=device,
                )
            images = save_image(
                layouts["bbox"],
                layouts["label"],
                layouts["mask"],
                val_dataset.colors,
            )
            tag = f"{cfg.sampling.name} sampling results"
            writer.add_images(tag, images, epoch + 1)

            if cfg.debug:
                save_image(
                    layouts["bbox"],
                    layouts["label"],
                    layouts["mask"],
                    val_dataset.colors,
                    f"tmp/debug_{total_iter_count}.png",
                )

        fid_epoch_interval = 1 if cfg.debug else max(cfg.training.epochs // 10, 1)
        # fid_epoch_interval = 3 if cfg.debug else cfg.training.epochs // 10

        if (epoch + 1) % fid_epoch_interval == 0:
            N = cfg.training.fid_plot_num_samples
            val_dataloader_fid = DataLoader(val_dataset[:N], shuffle=False, **kwargs)
            feats_1, feats_2 = [], []

            for batch in val_dataloader_fid:
                remove_canvas = (
                    cfg.model._target_ == "trainer.models.unilayout.UniLayout"
                )
                bbox, label, padding_mask, _ = sparse_to_dense(
                    batch, device, remove_canvas=remove_canvas
                )
                with torch.set_grad_enabled(False):
                    feat = fid_model.extract_features(bbox, label, padding_mask)
                feats_1.append(feat.cpu())

            dataloader = split_num_samples(
                cfg.training.fid_plot_num_samples, cfg.training.fid_plot_batch_size
            )
            t_total = 0.
            test_model = ema_model.module if use_ema else model
            for batch_size in dataloader:
                with torch.set_grad_enabled(False):
                    t_start = time.time()
                    layouts = test_model.sample(
                        batch_size=batch_size,
                        sampling_cfg=cfg.sampling,
                        device=device,
                    )
                    t_end = time.time()
                    t_total += t_end - t_start
                    feat = fid_model.extract_features(
                        layouts["bbox"].to(device),
                        layouts["label"].to(device),
                        ~layouts["mask"].to(device),
                    )

                feats_2.append(feat.cpu())

            logger.info(f"{N} samples: ms per sample: {1e3 * t_total / N}")
            fid_results = compute_generative_model_scores(feats_1, feats_2)
            logger.info(str(fid_results))
            for k, v in fid_results.items():
                writer.add_scalar(f"val_{k}", v, epoch + 1)

    test_dataset = instantiate(cfg.dataset)(split="test", transform=transform)
    test_dataloader = DataLoader(test_dataset, shuffle=False, **kwargs)
    if use_ema:
        test_loss = evaluate(ema_model.module, test_dataloader, cfg, device)
    else:
        test_loss = evaluate(model, test_dataloader, cfg, device)
    logger.info("test_loss = %.4f" % (test_loss))
    result = {"test_loss": test_loss}

    # Save results and model weights.
    with fs.open(os.path.join(job_dir, "result.json"), "wb") as file_obj:
        file_obj.write(json.dumps(result).encode("utf-8"))
    save_model(model, job_dir, best_or_final="final")
    if use_ema:
        save_model(ema_model.module, job_dir, best_or_final="ema_final")

    if not cfg.debug:
        dataset_name = train_dataset.name
        dst_dir = os.path.join(WEIGHTS_DIR, dataset_name, os.path.basename(cfg.job_dir), str(cfg.seed))
        logger.info(f'SAVE: {dst_dir}')
        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
        shutil.copytree(src=job_dir, dst=dst_dir)

    total_iter_count = 0  # reset iter count for multirun
    logger.removeHandler(handler) # reset logger for multirun


def train(
    model: torch.nn.Module,
    train_data: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    writer: SummaryWriter,
    ema_model: Optional[ModelEmaV2] = None,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    global total_iter_count

    for batch in train_data:
        batch = model.preprocess(batch)
        batch = _to(batch, device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs, losses = model(batch)
            loss = sum(losses.values())
        loss.backward()  # type: ignore

        if cfg.training.grad_norm_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.grad_norm_clip
            )

        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
        total_iter_count += 1

        if ema_model is not None:
            ema_model.update(model)

        if total_iter_count % cfg.training.loss_plot_iter_interval == 0:
            for (k, v) in losses.items():
                writer.add_scalar(k, v.cpu().item(), total_iter_count + 1)

        # below are for development

        # if cfg.debug:
        #     break

        # if cfg.debug and total_iter_count % 10 == 0:
        #     text = ""
        #     for (k, v) in losses.items():
        #         text += f"{k}: {v} "
        #     print(total_iter_count, text)

        # if cfg.debug and total_iter_count % (cfg.training.loss_plot_iter_interval * 10) == 0:
        #     # sanity check
        #     if cfg.debug:
        #         layouts = model.tokenizer.decode(outputs["outputs"].cpu())
        #         save_image(
        #             layouts["bbox"],
        #             layouts["label"],
        #             layouts["mask"],
        #             train_data.dataset.colors,
        #             f"tmp/debug_{total_iter_count}.png",
        #         )

    return total_loss / steps


def evaluate(
    model: torch.nn.Module,
    test_data: DataLoader,
    cfg: DictConfig,
    device: torch.device,
) -> float:
    total_loss = 0.0
    steps = 0

    model.eval()
    with torch.set_grad_enabled(False):
        for batch in test_data:
            batch = model.preprocess(batch)
            # batch = {k: v.to(device) for (k, v) in batch.items()}
            batch = _to(batch, device)
            _, losses = model(batch)
            loss = sum(losses.values())
            total_loss += float(loss.item())
            steps += 1

            if cfg.debug:
                break

    return total_loss / steps


if __name__ == "__main__":
    filter_args_for_ai_platform()
    main()
