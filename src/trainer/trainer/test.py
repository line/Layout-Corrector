"""
This file is derived from the following repository:
https://github.com/CyberAgentAILab/layout-dm/tree/9cf122d3be74b1ed8a8d51e3caabcde8164d238e

Original file: src/trainer/trainer/test.py
Author: naoto0804
License: Apache-2.0 License

Modifications have been made to the original file to fit the requirements of this project.
"""

import logging
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict

import hydra
import numpy as np
import torch
import torchvision.transforms as T
from einops import repeat
from fsspec.core import url_to_fs
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from trainer.crossplatform_util import filter_args_for_ai_platform
from trainer.data.util import (
    AddCanvasElement,
    AddRelationConstraints,
    split_num_samples,
)
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.layout_diffusion_tokenizer import LayoutDiffusionTokenizer
from trainer.helpers.metric import compute_violation
from trainer.helpers.sampling import SAMPLING_CONFIG_DICT
from trainer.helpers.task import get_cond
from trainer.helpers.util import set_seed
from trainer.helpers.visualization import save_image
from trainer.models.common.util import load_model

from .global_configs import DATASET_DIR
from .hydra_configs import TestConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = True

def _filter_invalid(layouts: Dict[str, Tensor]):
    outputs = []
    for b in range(layouts["bbox"].size(0)):
        bbox = layouts["bbox"][b].numpy()
        label = layouts["label"][b].numpy()
        mask = layouts["mask"][b].numpy()
        outputs.append((bbox[mask], label[mask]))
    return outputs


# instantiate a hydra config for test
cs = ConfigStore.instance()
cs.store(name="test_config", node=TestConfig)


@hydra.main(version_base="1.2", config_name="test_config")
def main(test_cfg: DictConfig) -> None:
    # print(test_cfg)
    # breakpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    multi_per_input = test_cfg.num_run > 1

    fs, _ = url_to_fs(test_cfg.job_dir)
    if not fs.exists(test_cfg.job_dir):
        raise FileNotFoundError
    
    config_path = os.path.join(test_cfg.job_dir, "config.yaml")
    if fs.exists(config_path):
        with fs.open(config_path, "rb") as file_obj:
            train_cfg = OmegaConf.load(file_obj)
        ckpt_dirs = [test_cfg.job_dir]
        seeds = [train_cfg.seed]
    else:
        # multi-seed experiment
        # assume seed is 0, 1, 2, ...
        ckpt_dirs = []
        seeds = []
        seed = 0
        while True:
            tmp_job_dir = os.path.join(test_cfg.job_dir, str(seed))
            config_path = os.path.join(tmp_job_dir, "config.yaml")

            if fs.exists(config_path):
                if seed == 0:
                    with fs.open(config_path, "rb") as file_obj:
                        train_cfg = OmegaConf.load(file_obj)
                ckpt_dirs.append(tmp_job_dir)
                seeds.append(seed)
            else:
                break

            seed += 1

    if test_cfg.debug:
        ckpt_dirs = [ckpt_dirs[0]]

    dataset_cfg = train_cfg.dataset
    if test_cfg.get("dataset_dir", None):
        dataset_cfg.dir = test_cfg.dataset_dir
    else:
        dataset_cfg.dir = DATASET_DIR

    data_cfg = train_cfg.data
    data_cfg.pad_until_max = True

    # if test_cfg.cond not in ["refinement", "unconditional"]:
    #     assert train_cfg.data.transforms == ["RandomOrder"]

    sampling_cfg = OmegaConf.structured(SAMPLING_CONFIG_DICT[test_cfg.sampling])
    OmegaConf.set_struct(sampling_cfg, False)
    if "temperature" in test_cfg:
        sampling_cfg.temperature = test_cfg.temperature
    if "top_p" in test_cfg and sampling_cfg.name == "top_p":
        sampling_cfg.top_p = test_cfg.top_p

    if "LayoutDiffusion" in train_cfg.model._target_:
        tokenizer = LayoutDiffusionTokenizer(data_cfg=data_cfg, dataset_cfg=dataset_cfg)
    else:
        tokenizer = LayoutSequenceTokenizer(data_cfg=data_cfg, dataset_cfg=dataset_cfg)
    # tokenizer = LayoutSequenceTokenizer(data_cfg=data_cfg, dataset_cfg=dataset_cfg)
    model = instantiate(train_cfg.model)(
        backbone_cfg=train_cfg.backbone, tokenizer=tokenizer
    )
    model = model.to(device)

    sampling_cfg = model.aggregate_sampling_settings(sampling_cfg, test_cfg)
    logger.warning(f"Test config: {test_cfg}")
    logger.warning(f"Sampling config: {sampling_cfg}")

    key = "_".join([f"{k}_{v}" for (k, v) in sampling_cfg.items()])
    if test_cfg.is_validation:
        key += "_validation"
    if test_cfg.debug:
        key += "_debug"
    if test_cfg.use_train_data:
        key += "_train_data"
    if test_cfg.debug_num_samples > 0:
        key += f"_only_{test_cfg.debug_num_samples}_samples"

    if test_cfg.cond == "unconditional":
        key += f"_{test_cfg.num_uncond_samples}_samples"

    if test_cfg.best_or_final != "best":
        key += f"_{test_cfg.best_or_final}"

    if multi_per_input:
        assert test_cfg.cond
        test_cfg.max_batch_size = 1  # load single sample and generate multiple results
        key += f"_{test_cfg.num_run}samples_per_input"

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    result_dir = os.path.join(test_cfg.result_dir, f"{test_cfg.cond}_{key}_{timestamp}")

    if not fs.exists(result_dir):
        fs.mkdir(result_dir)

    handler = logging.FileHandler(os.path.join(result_dir, f"test_{timestamp}.log"))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s  %(asctime)s  [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    with fs.open(os.path.join(result_dir, "test_config.yaml"), "wb") as f:
        f.write(OmegaConf.to_yaml(test_cfg).encode("utf-8"))
    with fs.open(os.path.join(result_dir, "sampling_config.yaml"), "wb") as f:
        f.write(OmegaConf.to_yaml(sampling_cfg).encode("utf-8"))

    scores = defaultdict(list)
    for seed_no, ckpt_dir in zip(seeds, ckpt_dirs):
        pkl_file = os.path.join(result_dir, f"seed_{seed_no}.pkl")
        if os.path.exists(pkl_file):
            logger.warning(f"Results for seed {seed_no} exists in {result_dir}")
            continue

        set_seed(seed_no)
        batch_metrics = defaultdict(float)
        model = load_model(
            model=model,
            ckpt_dir=ckpt_dir,
            device=device,
            best_or_final=test_cfg.best_or_final,
        )
        logger.info(f'Loaded "{test_cfg.best_or_final}" model from {ckpt_dir}')
        model.eval()

        if test_cfg.cond == "relation":
            test_transform = T.Compose(
                [
                    AddCanvasElement(),
                    AddRelationConstraints(seed=seed_no, edge_ratio=0.1),
                ]
            )
        else:
            test_transform = None

        split = "val" if test_cfg.is_validation else "test"
        split = "train" if test_cfg.use_train_data else split
        dataset = instantiate(dataset_cfg)(split=split, transform=test_transform)
        if test_cfg.debug_num_samples > 0:
            dataset = dataset[: test_cfg.debug_num_samples]

        t_total = 0.0
        N_total = 0
        inputs, relations, relation_scores, results = [], [], [], []
        if test_cfg.cond == "unconditional":
            dataloader = split_num_samples(
                test_cfg.num_uncond_samples, test_cfg.max_batch_size
            )
        else:
            dataloader = DataLoader(
                dataset, batch_size=test_cfg.max_batch_size, shuffle=False
            )

        for j, batch in enumerate(tqdm(dataloader)):
            if test_cfg.cond == "unconditional":
                cond = None
                batch_size = batch
            else:
                cond = get_cond(
                    batch=batch,
                    tokenizer=model.tokenizer,
                    cond_type=test_cfg.cond,
                    model_type=type(model).__name__,
                    refinement_noise_std=test_cfg.refine_noise_std,
                )
                batch_size = cond["seq"].size(0)
                if multi_per_input:
                    batch_size = test_cfg.num_run

            t_start = time.time()
            with torch.no_grad():
                layouts = model.sample(
                    batch_size=batch_size,
                    cond=cond,
                    sampling_cfg=sampling_cfg,
                    cond_type=test_cfg.cond,
                )
            t_end = time.time()
            t_total += t_end - t_start
            N_total += batch_size

            # visualize the results for sanity check, since the generation takes minutes to hours
            if j == 0:
                if not ckpt_dir.startswith("gs://"):
                    save_image(
                        layouts["bbox"],
                        layouts["label"],
                        layouts["mask"],
                        dataset.colors,
                        f"tmp/test_generated.png",
                    )

            if cond and "type" in cond and cond["type"] in ["partial", "refinement"]:
                if "bos" in model.tokenizer.special_tokens:
                    input_layouts = model.tokenizer.decode(cond["seq"][:, 1:].cpu())
                else:
                    is_diffusion = type(model).__name__ == "LayoutDM"
                    type_key = (
                        "seq_orig"
                        if cond["type"] == "refinement" and is_diffusion
                        else "seq"
                    )
                    input_layouts = model.tokenizer.decode(cond[type_key].cpu())
                inputs.extend(_filter_invalid(input_layouts))
            results.extend(_filter_invalid(layouts))

            # relation violation detection if necessary
            if test_cfg.cond == "relation":
                B = layouts["bbox"].size(0)
                canvas = torch.FloatTensor([0.5, 0.5, 1.0, 1.0])
                canvas_mask = torch.full((1,), fill_value=True)
                bbox_c = torch.cat(
                    [
                        repeat(canvas, "c -> b 1 c", b=B),
                        layouts["bbox"],
                    ],
                    dim=1,
                )
                mask_c = torch.cat(
                    [
                        repeat(canvas_mask, "1 -> b 1", b=B),
                        layouts["mask"],
                    ],
                    dim=1,
                )
                bbox_flatten = bbox_c[mask_c]
                if len(batch.edge_index) > 0:
                    v = compute_violation(bbox_flatten.to(device), batch)
                    v = v[~v.isnan()].sum().item()
                    batch_metrics["violation_score"] += v

                relation_scores.append(v)

        # dummy_cfg = copy.deepcopy(train_cfg)
        config_path = os.path.join(ckpt_dir, "config.yaml")
        with fs.open(config_path, "rb") as file_obj:
            dummy_cfg = OmegaConf.load(file_obj)
        dummy_cfg.sampling = sampling_cfg
        data = {"results": results, "train_cfg": dummy_cfg, "test_cfg": test_cfg, "t_total": t_total, "N_total": N_total}
        if len(inputs) > 0:
            data["inputs"] = inputs
        if len(relations) > 0:
            data["relations"] = relations
            data["relation_scores"] = relation_scores

        pkl_file = os.path.join(result_dir, f"seed_{seed_no}.pkl")
        with fs.open(pkl_file, "wb") as file_obj:
            pickle.dump(data, file_obj)

        logger.info(f'N_total = {N_total}')
        logger.info(f"ms per sample: {1e3 * t_total / N_total}")

        for k, v in batch_metrics.items():
            scores[k].append(v / len(results))

    keys, values = [], []
    for k, v in scores.items():
        v = np.array(v)
        mean, std = np.mean(v), np.std(v)
        keys += [f"{k}-mean", f"{k}-std"]
        values += [str(mean), str(std)]
    print(",".join(keys))
    print(",".join(values))


if __name__ == "__main__":
    filter_args_for_ai_platform()
    main()
