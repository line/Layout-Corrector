"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php

-------------------------------------------------------------------------------

Test Corrupted Token Detection Capability of Layout-Corrector.
We synthesize erroneous token sequences and check the detection accurcy of these tokens.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import random
from collections import defaultdict
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
from hydra.utils import instantiate
from trainer.analysis_helpers.corruption_util import get_corrupted_seq
from trainer.data.util import sparse_to_dense
from trainer.global_configs import DATASET_DIR
from trainer.helpers.sampling import SAMPLING_CONFIG_DICT
from trainer.hydra_configs import TestConfig
from trainer.models.layoutdm import LayoutDM
from trainer.models.common.util import build_model, load_model
from trainer.helpers.util import load_config, set_seed


def run_corrector_test(
    dataset, 
    model,
    corrector,
    device, 
    num_replace=1, 
    corrector_timesteps=[10]
):
    N_test_per_sample = 1
    timestep2matched = defaultdict(list)
    timestep2num_replaced = defaultdict(list)

    for t in corrector_timesteps:
        set_seed(0)
        # for data in dataset:
        for data in tqdm(dataset):
            bbox, label, _, mask = sparse_to_dense(data)
            gt_cond = model.tokenizer.encode({"label": label, "mask": mask, "bbox": bbox})
        
            for _ in range(N_test_per_sample):
                noise_cond, replaced_index, num_valid = get_corrupted_seq(
                    gt_cond["seq"],
                    model.model,
                    replace_with_mask=False,
                    num_corrupted=num_replace,
                )
                if noise_cond is None and replaced_index is None and num_valid is None:
                    continue

                model_t = torch.full((1, ), t, device=device, dtype=torch.long)
                noise_cond_conf = corrector.calc_confidence_score(
                    noise_cond.to(device), model_t
                )
                _, topk_lowest_indices = torch.topk(-noise_cond_conf, num_replace, dim=-1)
                num_matched = sum([index.cpu().item() in replaced_index for index in topk_lowest_indices[0]])
                timestep2matched[t].append(num_matched)
                timestep2num_replaced[t].append(len(replaced_index))
    
    timestep2acc = dict(
        token_wise_acc=dict(),
        full_acc=dict()
    )
    for t, results in timestep2matched.items():
        results = np.array(results)
        token_wise_acc = np.sum(results) / np.sum(timestep2num_replaced[t]) * 100
        timestep2acc["token_wise_acc"][t] = token_wise_acc

        full_acc = np.sum(results == num_replace) / len(results) * 100
        timestep2acc["full_acc"][t] = full_acc

    return timestep2acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", 
                        type=str, 
                        help="Layout Generation Model's job_dir including ckpt.")
    parser.add_argument("--corr_job_dir", 
                        type=str, 
                        help="LayoutCorrector's job_dir includibg ckpt.")
    parser.add_argument("--corr_timesteps",
                        nargs="*",
                        type=int, 
                        help="The timesteps at which the corrector is applied to detecto erroneous tokens.",
                        default=[10, 20, 30])
    parser.add_argument("-n",
                        "--num_replace",
                        help="The number of tokens to be replaced with random token.", 
                        type=int,
                        default=3)
    parser.add_argument("--save_dir",
                        type=str, 
                        help="A directory path where the results are saved.", 
                        default="error_token_detection_results")
    args = parser.parse_args()


    test_cfg = OmegaConf.structured(TestConfig)
    test_cfg.cond = "unconditional"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NOTE: you may change sampling algorithm (default: random)
    sampling_cfg = OmegaConf.structured(SAMPLING_CONFIG_DICT[test_cfg.sampling])
    OmegaConf.set_struct(sampling_cfg, False)

    config_path = os.path.join(args.job_dir, "config.yaml")
    train_cfg = load_config(config_path)
    train_cfg.dataset.dir = DATASET_DIR

    if args.corr_job_dir is not None:
        corr_config_path = os.path.join(args.corr_job_dir, "config.yaml")
        corr_train_cfg = load_config(corr_config_path)
    else:
        corr_train_cfg = None

    # initialize model
    model = build_model(train_cfg, device=device)
    assert isinstance(model, LayoutDM), f"Only supports LayoutDM, received {model.__name__}"
    model = load_model(
        model=model,
        ckpt_dir=args.job_dir,
        device=device,
        best_or_final="best"
    )
    model.eval()

    corrector = build_model(corr_train_cfg, device=device)
    corrector = load_model(
        model=corrector,
        ckpt_dir=args.corr_job_dir,
        device=device,
        best_or_final="best"
    )
    corrector.eval()

    num_timesteps = train_cfg.backbone.encoder_layer.diffusion_step
    correction_timesteps = [t for t in args.corr_timesteps if t < num_timesteps]

    # NOTE: split is test
    print(f"job_dir = {args.job_dir}\n"
          f"corr_job_dir = {args.corr_job_dir}")
    dataset = instantiate(train_cfg.dataset)(split="test", transform=None)
    
    basename = args.job_dir.replace("/", "_")
    corr_name = args.corr_job_dir.replace("/", "_")
    filename = f"{basename}_{corr_name}n{args.num_replace}.json"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    time2mask_acc = run_corrector_test(
        dataset, 
        model, 
        corrector, 
        device, 
        num_replace=args.num_replace,
        corrector_timesteps=correction_timesteps,
    )

    maskacc_filename = f"corrector_mask_acc_{filename}"
    with open(os.path.join(args.save_dir, maskacc_filename), "w") as f:
        json.dump(time2mask_acc, f, indent=2)


if __name__ == "__main__":
    main()
    