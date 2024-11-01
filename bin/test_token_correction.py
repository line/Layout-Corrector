"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php

-------------------------------------------------------------------------------

Test Token-Correction Capability on LayoutDM and its conjunction with Layout-Corrector.
We synthesize erroneous token sequences and check the ablity of restoring these tokens to the ground truth.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
from collections import defaultdict
from omegaconf import OmegaConf

import torch
from hydra.utils import instantiate
from trainer.analysis_helpers.corruption_util import get_corrupted_seq
from trainer.data.util import sparse_to_dense
from trainer.global_configs import DATASET_DIR
from trainer.helpers.sampling import SAMPLING_CONFIG_DICT
from trainer.hydra_configs import TestConfig
from trainer.models.layoutdm import LayoutDM
from trainer.models.categorical_diffusion.util import index_to_log_onehot, log_onehot_to_index
from trainer.models.common.util import build_model, load_model
from trainer.helpers.util import load_config, set_seed


def run_test(
    dataset, 
    model, 
    sampling_cfg, 
    device, 
    replace_with_mask=False, 
    num_replace=1, 
    refinement_timesteps=[10]
):
    # replaced: evaluate matching of replaced tokens
    # full: require full matching
    N_test_per_sample = 1
    full_timestep2results = defaultdict(list)
    replaced_timestep2results = defaultdict(list)
    timestep2num_replaced = defaultdict(list)

    for t in refinement_timesteps:
        set_seed(0)
        for data in dataset[:100]:
            bbox, label, _, mask = sparse_to_dense(data)
            gt_cond = model.tokenizer.encode({"label": label, "mask": mask, "bbox": bbox})
        
            for _ in range(N_test_per_sample):
                noise_cond, replaced_index, num_valid = get_corrupted_seq(
                    gt_cond["seq"],
                    model.model,
                    replace_with_mask=replace_with_mask,
                    num_corrupted=num_replace
                )
                if noise_cond is None and replaced_index is None and num_valid is None:
                    continue

                noise_log_z = index_to_log_onehot(noise_cond, model.tokenizer.N_total)
                for t_local in range(t, -1, -1):
                    model_t = torch.full((1, ), t_local, device=device, dtype=torch.long)
                    noise_log_z = model.model._sample_single_step(
                        noise_log_z.to(device), model_t, 0, sampling_cfg=sampling_cfg
                    )
  
                index_output_noise = log_onehot_to_index(noise_log_z).cpu()
                # matching in replaced tokens
                matched_tokens = gt_cond["seq"][0, replaced_index] == index_output_noise[0, replaced_index]
                num_matched = matched_tokens.sum().cpu().item()
                replaced_timestep2results[t].append(num_matched)
                # full matching
                full_matched = (gt_cond["seq"] == index_output_noise.cpu()).all().item()
                full_timestep2results[t].append(full_matched)
                timestep2num_replaced[t].append(len(replaced_index))
                
    timestep2success_rate = dict(
        token_wise=dict(),
        full=dict()
    )
    for t, results in replaced_timestep2results.items():
        results = np.array(results)
        avg = np.sum(results) / np.sum(timestep2num_replaced[t])
        timestep2success_rate["token_wise"][t] = avg * 100

    for t, results in full_timestep2results.items():
        results = np.array(results)
        avg = np.mean(results)
        timestep2success_rate["full"][t] = avg * 100

    return timestep2success_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", 
                        type=str, 
                        help="Layout Generation Model's job_dir including ckpt.")
    parser.add_argument("--start_timesteps",
                        nargs="*",
                        type=int, 
                        help="The generation start timestep which should be [0, T], where 0 means the last timestep in generation.", 
                        default=[10])
    parser.add_argument("-m",
                        "--mask",
                        help="Whether to replace random a selected token with MASK.", 
                        action="store_true")
    parser.add_argument("-n",
                        "--num_replace",
                        help="The number of tokens to be replaced with random token.", 
                        type=int,
                        default=3)
    parser.add_argument("--save_dir",
                        type=str, 
                        help="A directory path where the results are saved.", 
                        default="token_correction_results")
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
    sampling_cfg = model.aggregate_sampling_settings(sampling_cfg, test_cfg)

    # NOTE: split is test
    print(f"job_dir = {args.job_dir}")
    dataset = instantiate(train_cfg.dataset)(split="test", transform=None)
    # sr: sticking rate
    time2sr = run_test(
        dataset, 
        model, 
        sampling_cfg, 
        device, 
        replace_with_mask=args.mask, 
        num_replace=args.num_replace,
        refinement_timesteps=args.start_timesteps
    )
    
    basename = args.job_dir.replace("/", "_") + f"n{args.num_replace}"
    if args.mask:
        basename += "_mask"
    
    filename = basename + ".json"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, filename), "w") as f:
        json.dump(time2sr, f, indent=2)



if __name__ == "__main__":
    main()
