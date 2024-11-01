"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Tuple

import hydra
import torch
import torch.nn as nn
from fsspec.core import url_to_fs
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from trainer.crossplatform_util import filter_args_for_ai_platform
from trainer.data.util import split_num_samples
from trainer.helpers.layout_diffusion_tokenizer import LayoutDiffusionTokenizer
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.sampling import SAMPLING_CONFIG_DICT
from trainer.helpers.task import get_cond
from trainer.helpers.util import set_seed
from trainer.models.common.util import load_model

from .global_configs import DATASET_DIR
from .hydra_configs import TestConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def _filter_invalid(layouts: Dict[str, Tensor]):
    outputs = []
    for b in range(layouts["bbox"].size(0)):
        bbox = layouts["bbox"][b].numpy()
        label = layouts["label"][b].numpy()
        mask = layouts["mask"][b].numpy()
        outputs.append((bbox[mask], label[mask]))
    return outputs


def build_tokenizer(data_cfg, dataset_cfg):
    var_order = data_cfg.get("var_order")
    if "l-t-r-b" in var_order:
        tokenizer = LayoutDiffusionTokenizer(data_cfg=data_cfg, dataset_cfg=dataset_cfg)
    else:
        tokenizer = LayoutSequenceTokenizer(data_cfg=data_cfg, dataset_cfg=dataset_cfg)
    return tokenizer


# instantiate a hydra config for test
cs = ConfigStore.instance()
cs.store(name="test_config", node=TestConfig)


def search_ckpt_dirs(job_dir: str) -> Tuple[List, List]:
    fs, _ = url_to_fs(job_dir)
    if not fs.exists(job_dir):
        raise FileNotFoundError

    config_path = os.path.join(job_dir, "config.yaml")
    if fs.exists(config_path):
        with fs.open(config_path, "rb") as file_obj:
            train_cfg = OmegaConf.load(file_obj)
        ckpt_dirs = [job_dir]
        seeds = [train_cfg.seed]
    else:
        # multi-seed experiment
        # assume seed is 0, 1, 2, ...
        ckpt_dirs = []
        seeds = []
        seed = 0
        while True:
            tmp_job_dir = os.path.join(job_dir, str(seed))
            config_path = os.path.join(tmp_job_dir, "config.yaml")

            if fs.exists(config_path):
                if seed == 0:
                    with fs.open(config_path, "rb") as file_obj:
                        train_cfg = OmegaConf.load(file_obj)
                ckpt_dirs.append(tmp_job_dir)
                seeds.append(seed)
            else:
                assert seed > 0, "At least seed=0 should exist"
                break

            seed += 1

    return ckpt_dirs, seeds


def build_models(
    train_cfg: DictConfig, data_cfg: DictConfig, dataset_cfg: DictConfig, device: str
) -> Tuple[nn.Module, nn.Module]:
    ## Define Models
    tokenizer = build_tokenizer(data_cfg, dataset_cfg)
    corrector = instantiate(train_cfg.model)(
        backbone_cfg=train_cfg.backbone, tokenizer=tokenizer
    )
    corrector = corrector.to(device)

    ## Load Diffusion Model
    dm_job_dir = os.path.join(train_cfg.dm_job_dir, str(train_cfg.seed))
    dm_config_path = os.path.join(dm_job_dir, "config.yaml")
    fs, _ = url_to_fs(dm_config_path)
    if fs.exists(dm_config_path):
        with fs.open(dm_config_path, "rb") as file_obj:
            dm_cfg = OmegaConf.load(file_obj)
    else:
        raise FileNotFoundError(f'"{dm_config_path}" not found')
    dm_cfg.dataset.dir = DATASET_DIR
    dm_cfg.data.pad_until_max = True
    # dm_cfg.data.pad_until_max = dm_cfg.model.get("use_padding_as_vocab", False)
    dm_tokenizer = build_tokenizer(dm_cfg.data, dm_cfg.dataset)
    diffusion = instantiate(dm_cfg.model)(
        backbone_cfg=dm_cfg.backbone, tokenizer=dm_tokenizer
    ).to(device)
    return diffusion, corrector


def run(
    diffusion: torch.nn.Module,
    corrector: torch.nn.Module,
    dataloader: DataLoader,
    test_cfg: DictConfig,
    sampling_cfg: Dict,
    multi_per_input: bool = False,
) -> Tuple[List, List, float, int]:
    t_total = 0.0
    N_total = 0
    inputs, results = [], []

    for j, batch in enumerate(tqdm(dataloader)):
        if test_cfg.cond == "unconditional":
            cond = None
            batch_size = batch
        else:
            cond = get_cond(
                batch=batch,
                tokenizer=diffusion.tokenizer,
                cond_type=test_cfg.cond,
                model_type=type(diffusion).__name__,
                refinement_noise_std=test_cfg.refine_noise_std,
            )
            batch_size = cond["seq"].size(0)
            if multi_per_input:
                batch_size = test_cfg.num_run

        t_start = time.time()
        with torch.no_grad():
            layouts = diffusion.sample(
                batch_size=batch_size,
                cond=cond,
                sampling_cfg=sampling_cfg,
                cond_type=test_cfg.cond,
                corrector=corrector,
            )
        t_end = time.time()
        t_total += t_end - t_start
        N_total += batch_size

        if (
            cond
            and "type" in cond
            and cond["type"] in ["partial", "refinement", "partial_shift"]
        ):
            if "bos" in diffusion.tokenizer.special_tokens:
                input_layouts = diffusion.tokenizer.decode(cond["seq"][:, 1:].cpu())
            else:
                is_diffusion = type(diffusion).__name__ == "LayoutDM"
                type_key = (
                    "seq_orig"
                    if cond["type"] == "refinement" and is_diffusion
                    else "seq"
                )
                input_layouts = diffusion.tokenizer.decode(cond[type_key].cpu())
            inputs.extend(_filter_invalid(input_layouts))
        results.extend(_filter_invalid(layouts))

        # relation violation detection if necessary
        if test_cfg.cond == "relation":
            raise NotImplementedError()

    return inputs, results, t_total, N_total


@hydra.main(version_base="1.2", config_name="test_config")
def main(test_cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    multi_per_input = test_cfg.num_run > 1

    ckpt_dirs, seeds = search_ckpt_dirs(test_cfg.job_dir)

    if test_cfg.debug:
        ckpt_dirs = [ckpt_dirs[0]]
        seeds = [seeds[0]]

    fs, _ = url_to_fs(test_cfg.job_dir)
    config_path = os.path.join(ckpt_dirs[0], "config.yaml")
    with fs.open(config_path, "rb") as file_obj:
        train_cfg = OmegaConf.load(file_obj)

    dataset_cfg = train_cfg.dataset
    if test_cfg.get("dataset_dir", None):
        dataset_cfg.dir = test_cfg.dataset_dir
    else:
        dataset_cfg.dir = DATASET_DIR

    data_cfg = train_cfg.data
    data_cfg.pad_until_max = True

    ## Define Models
    diffusion, corrector = build_models(train_cfg, data_cfg, dataset_cfg, device)

    sampling_cfg = OmegaConf.structured(SAMPLING_CONFIG_DICT[test_cfg.sampling])
    OmegaConf.set_struct(sampling_cfg, False)
    if "temperature" in test_cfg:
        sampling_cfg.temperature = test_cfg.temperature
    if "top_p" in test_cfg and sampling_cfg.name == "top_p":
        sampling_cfg.top_p = test_cfg.top_p

    sampling_cfg = corrector.aggregate_sampling_settings(sampling_cfg, test_cfg)

    dir_name = test_cfg.cond

    dir_name += f"_{sampling_cfg.corrector_mask_mode}"
    if sampling_cfg.corrector_mask_mode == "thresh":
        dir_name += f"_{sampling_cfg.corrector_mask_threshold}"

    if test_cfg.is_validation:
        dir_name += "_validation"
    if test_cfg.debug:
        dir_name += "_debug"
    if test_cfg.debug_num_samples > 0:
        dir_name += f"_only_{test_cfg.debug_num_samples}_samples"

    if test_cfg.cond == "unconditional":
        dir_name += f"_{test_cfg.num_uncond_samples}_samples"

    if multi_per_input:
        assert test_cfg.cond
        test_cfg.max_batch_size = 1  # load single sample and generate multiple results
        dir_name += f"_{test_cfg.num_run}samples_per_input"

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    result_dir = os.path.join(test_cfg.result_dir, f"{dir_name}_{timestamp}")

    if not fs.exists(result_dir):
        fs.mkdir(result_dir)

    handler = logging.FileHandler(os.path.join(result_dir, f"test_{timestamp}.log"))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s  %(asctime)s  [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    with fs.open(os.path.join(result_dir, "test_config.yaml"), "wb") as f:
        f.write(OmegaConf.to_yaml(test_cfg).encode("utf-8"))
    with fs.open(os.path.join(result_dir, "sampling_config.yaml"), "wb") as f:
        f.write(OmegaConf.to_yaml(sampling_cfg).encode("utf-8"))

    logger.info(f"Diffusion #params: {diffusion.calc_num_params() / 1e6:.3f} M")
    logger.info(f"Corrector #params: {corrector.calc_num_params() / 1e6:.3f} M")
    logger.info(f"Results saved to {result_dir}")
    logger.info(f"Test config: {test_cfg}")
    logger.info(f"Sampling config: {sampling_cfg}")

    for seed_no, ckpt_dir in zip(seeds, ckpt_dirs):
        logger.info(f"Run seed = {seed_no}")
        logger.info(f"ckpt_dir: {ckpt_dir}")
        set_seed(seed_no)

        corrector = load_model(
            model=corrector,
            ckpt_dir=ckpt_dir,
            device=device,
            best_or_final="best",
        )
        corrector.eval()
        diffusion = load_model(
            model=diffusion,
            ckpt_dir=os.path.join(train_cfg.dm_job_dir, str(seed_no)),
            device=device,
            best_or_final=test_cfg.best_or_final,
        )
        diffusion.eval()

        if test_cfg.cond == "relation":
            raise NotImplementedError()
        test_transform = None

        split = "val" if test_cfg.is_validation else "test"
        split = (
            "train" if test_cfg.use_train_data else split
        )
        dataset = instantiate(dataset_cfg)(split=split, transform=test_transform)
        if test_cfg.debug_num_samples > 0:
            dataset = dataset[: test_cfg.debug_num_samples]

        if test_cfg.cond == "unconditional":
            dataloader = split_num_samples(
                test_cfg.num_uncond_samples, test_cfg.max_batch_size
            )
        else:
            dataloader = DataLoader(
                dataset, batch_size=test_cfg.max_batch_size, shuffle=False
            )

        inputs, results, t_total, N_total = run(
            diffusion,
            corrector,
            dataloader,
            test_cfg,
            sampling_cfg,
            multi_per_input,
        )

        # dummy_cfg = copy.deepcopy(train_cfg)
        config_path = os.path.join(ckpt_dir, "config.yaml")
        with fs.open(config_path, "rb") as file_obj:
            dummy_cfg = OmegaConf.load(file_obj)
        dummy_cfg.sampling = sampling_cfg

        data = {
            "results": results,
            "train_cfg": dummy_cfg,
            "test_cfg": test_cfg,
            "t_total": t_total,
            "N_total": N_total,
        }
        if len(inputs) > 0:
            data["inputs"] = inputs

        pkl_file = os.path.join(result_dir, f"seed_{seed_no}.pkl")
        with fs.open(pkl_file, "wb") as file_obj:
            pickle.dump(data, file_obj)

        logger.info(f"N_total = {N_total}")
        logger.info(f"ms per sample: {1e3 * t_total / N_total}")


if __name__ == "__main__":
    filter_args_for_ai_platform()
    main()
