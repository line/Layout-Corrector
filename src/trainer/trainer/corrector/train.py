"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from copy import deepcopy

import hydra
import torch
from fsspec.core import url_to_fs
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader  # noqa
from trainer.data.util import compose_transform, sparse_to_dense, split_num_samples
from trainer.global_configs import DATASET_DIR
from trainer.helpers.layout_diffusion_tokenizer import LayoutDiffusionTokenizer
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.sampling import register_sampling_config
from trainer.helpers.scheduler import ReduceLROnPlateauWithWarmup
from trainer.helpers.util import set_seed, dict2str
from trainer.hydra_configs import DataConfig, CorrectorTrainConfig
from trainer.models.common.util import save_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "1"  # to see full tracelog for hydra

torch.autograd.set_detect_anomaly(True)
total_iter_count = 0

WEIGHTS_DIR = './download/pretrained_weights/'

cs = ConfigStore.instance()
cs.store(group="data", name="base_data_default", node=DataConfig)
cs.store(group="training", name="base_training_default", node=CorrectorTrainConfig)
register_sampling_config(cs)


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


def build_tokenizer(cfg):
    if "LayoutDiffusion" in cfg.model._target_:
        tokenizer = LayoutDiffusionTokenizer(data_cfg=cfg.data, dataset_cfg=cfg.dataset)
    else:
        tokenizer = LayoutSequenceTokenizer(data_cfg=cfg.data, dataset_cfg=cfg.dataset)
    return tokenizer


def build_pretrained_diffusion_model(train_cfg, job_dir, device, best_or_final="best"):
    if 'pad_until_max' not in train_cfg.data:
        train_cfg.data.pad_until_max = train_cfg.model.get("use_padding_as_vocab", False)
    tokenizer = build_tokenizer(train_cfg)
    try:
        model = instantiate(train_cfg.model)(backbone_cfg=train_cfg.backbone, 
                                            tokenizer=tokenizer,
                                            use_padding_as_vocab=train_cfg.model.use_padding_as_vocab)
    except:
        model = instantiate(train_cfg.model)(backbone_cfg=train_cfg.backbone, 
                                            tokenizer=tokenizer)
    model_path = os.path.join(job_dir, f"{best_or_final}_model.pt")
    logger.info(f'load Diffusion Model: "{model_path}"')
    fs, _ = url_to_fs(model_path)
    with fs.open(model_path, "rb") as file_obj:
        model.load_state_dict(torch.load(file_obj))
    model.eval()
    model = model.to(device)
    return model


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Build job_dir and set up
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
        cfg.data.batch_size = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Read diffusion_model's config
    ## NOTE: use the same seed for diffusion and corrector model traiining
    dm_job_dir = os.path.join(cfg.dm_job_dir, str(cfg.seed))
    config_path = os.path.join(dm_job_dir, "config.yaml")
    fs, _ = url_to_fs(config_path)
    if fs.exists(config_path):
        with fs.open(config_path, "rb") as file_obj:
            diffusion_cfg = OmegaConf.load(file_obj)
    else:
        raise FileNotFoundError(f'Cannot find dm_config: {config_path}')

    diffusion_model = build_pretrained_diffusion_model(
        diffusion_cfg, dm_job_dir, device, best_or_final=cfg.training.diffusion_model_ckpt)

    ## build dataset & dataloader. Use same dataset as diffusion model
    if diffusion_cfg.dataset.dir is None:
        diffusion_cfg.dataset.dir = DATASET_DIR
    transform = compose_transform(diffusion_cfg.data.transforms)
    train_dataset = instantiate(diffusion_cfg.dataset)(split="train", transform=transform)
    val_dataset = instantiate(diffusion_cfg.dataset)(split="val", transform=transform)

    kwargs = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": True,
    }
    train_dataloader = DataLoader(train_dataset, shuffle=True, **kwargs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **kwargs)

    tokenizer = deepcopy(diffusion_model.tokenizer) ## same tokenizer as difusion model
    model = instantiate(cfg.model)(backbone_cfg=cfg.backbone, tokenizer=tokenizer)
    model = model.to(device)

    sampling_cfg = cfg.sampling
    logger.info(sampling_cfg)

    optim_groups = model.optim_groups(cfg.training.weight_decay)
    optimizer = instantiate(cfg.optimizer)(optim_groups)
    scheduler = instantiate(cfg.scheduler)(optimizer=optimizer)

    best_val_loss = float("Inf")

    with fs.open(os.path.join(job_dir, "config.yaml"), "wb") as file_obj:
        file_obj.write(OmegaConf.to_yaml(cfg).encode("utf-8"))

    for epoch in range(cfg.training.epochs):
        model.update_per_epoch(epoch, cfg.training.epochs)

        start_time = time.time()
        train_loss = train(model, diffusion_model, train_dataloader, optimizer, cfg, sampling_cfg, device, writer)
        val_loss = evaluate(model, diffusion_model, val_dataloader, cfg, sampling_cfg, device)
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

    test_dataset = instantiate(cfg.dataset)(split="test", transform=transform)
    test_dataloader = DataLoader(test_dataset, shuffle=False, **kwargs)
    test_loss = evaluate(model, diffusion_model, test_dataloader, cfg, sampling_cfg, device)
    logger.info("test_loss = %.4f" % (test_loss))
    result = {"test_loss": test_loss}

    # Save results and model weights.
    with fs.open(os.path.join(job_dir, "result.json"), "wb") as file_obj:
        file_obj.write(json.dumps(result).encode("utf-8"))
    save_model(model, job_dir, best_or_final="final")

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
    diffusion_model: torch.nn.Module,
    train_data: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    sampling_cfg: DictConfig,
    device: torch.device,
    writer: SummaryWriter,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    global total_iter_count

    for raw_batch in train_data:
        batch = diffusion_model.preprocess(raw_batch)
        batch = _to(batch, device)
        batch['raw_batch'] = raw_batch
        batch = model.preprocess(batch, diffusion_model, sampling_cfg)

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

        if total_iter_count % cfg.training.loss_plot_iter_interval == 0:
            for (k, v) in losses.items():
                writer.add_scalar(k, v.cpu().item(), total_iter_count + 1)

    return total_loss / steps

def evaluate(
    model: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    test_data: DataLoader,
    cfg: DictConfig,
    sampling_cfg: DictConfig,
    device: torch.device,
) -> float:
    total_loss = 0.0
    steps = 0

    model.eval()
    with torch.set_grad_enabled(False):
        for raw_batch in test_data:
            batch = diffusion_model.preprocess(raw_batch)
            batch = _to(batch, device)
            batch['raw_batch'] = raw_batch
            batch = model.preprocess(batch, diffusion_model, sampling_cfg)

            _, losses = model(batch)
            loss = sum(losses.values())
            total_loss += float(loss.item())
            steps += 1

            if cfg.debug:
                break

    return total_loss / steps

if __name__ == "__main__":
    # filter_args_for_ai_platform()
    main()
