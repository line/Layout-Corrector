"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import Dict


def run_test_command(
    device: int,
    cond: str,
    job_dir: str,
    result_dir: str,
    other_arg_dict: Dict,
    is_val=True,
    batch_size: int = 512,
    num_uncond_samples: int = 1000,
    best_or_final: str = "best",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    cmd = [
        "python3",
        "-m",
        "src.trainer.trainer.test",
        "dataset_dir=./download/datasets",
        f"cond={cond}",
        f"job_dir={job_dir}",
        f"result_dir={result_dir}",
        f"max_batch_size={batch_size}",
        f"num_uncond_samples={num_uncond_samples}",
        f"best_or_final={best_or_final}",
    ]
    if is_val:
        cmd.append("is_validation=True")
    for k, v in other_arg_dict.items():
        cmd.append(f"{k}={v}")
    subprocess.run(cmd, env=os.environ, check=True)


def main():
    COND_LIST = ["unconditional", "c", "cwh"]
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("-t", "--timesteps", type=int, help="number of diffusion steps")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch_size", type=int, default=512)
    parser.add_argument("-n", "--num_uncond_samples", type=int, default=1000)
    parser.add_argument("-c", "--cond_list", nargs='*', default=COND_LIST, type=str)
    parser.add_argument("--sampling", default='random', type=str)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--best_or_final", type=str, default="best")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--test_only", action="store_true", help="Run test_data only"
    )
    parser.add_argument(
        "--val_only", action="store_true", help="Run validation_data only"
    )
    args = parser.parse_args()

    job_dir = os.path.normpath(args.job_dir)
    if (
        os.path.split(job_dir)[1]
    ).isnumeric():  # if job_dir == "**/**/<JOB_NAME>/<SEED_NUM>"
        job_dir_wo_seed = os.path.split(job_dir)[0]
        job_name = os.path.basename(job_dir_wo_seed)
    else:  # if job_dir == "**/**/<JOB_NAME>"
        job_name = os.path.basename(job_dir)

    result_root = './results'
    result_dir = os.path.join(result_root, args.dataset, job_name)

    conditions = args.cond_list
    for c in args.cond_list:
        assert c in COND_LIST, f"Invalid cond: {c}"

    cfg = dict(
        num_timesteps=args.timesteps,
        sampling=args.sampling,
        temperature=args.temperature,
    )
    if args.sampling == 'top_p':
        cfg['top_p'] = args.top_p

    split_list = ['test', 'validation']
    assert not (
        args.test_only and args.val_only
    ), "Cannot set both --test_only and --val_only"
    if args.test_only:
        split_list = ['test']
    elif args.val_only:
        split_list = ['validation']

    # Run test (generate layouts)
    for split in split_list:
        for cond in conditions:
            run_test_command(
                args.device,
                cond,
                job_dir,
                result_dir,
                cfg,
                is_val=(split == 'validation'),
                batch_size=args.batch_size,
                num_uncond_samples=args.num_uncond_samples,
                best_or_final=args.best_or_final,
            )

    # calculate metrics
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    cmd = ["python3", "./bin/calc_metrics.py", result_dir]
    if args.force:
        cmd.append("--force")
    subprocess.run(cmd, env=os.environ, check=True)


if __name__ == "__main__":
    main()
