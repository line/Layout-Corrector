"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

import argparse
import os
import subprocess


def run_test_command(
    cond,
    job_dir,
    result_dir,
    cfg_dict,
    device,
    is_val=True,
    batch_size: int = 512,
    num_uncond_samples: int = 1000,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    cmd = [
        "python3",
        "-m",
        "src.trainer.trainer.corrector_test",
        "dataset_dir=./download/datasets",
        f"cond={cond}",
        f"job_dir={job_dir}",
        f"result_dir={result_dir}",
        f"max_batch_size={batch_size}",
        f"num_uncond_samples={num_uncond_samples}",
    ]
    if is_val:
        cmd.append("is_validation=True")
    for k, v in cfg_dict.items():
        if isinstance(v, list):
            v = ",".join([str(x) for x in v])
            v = f"[{v}]"
        cmd.append(f"{k}={v}")
    subprocess.run(cmd, env=os.environ, check=True)


def main():
    COND_LIST = ["unconditional", "c", "cwh"]
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("-t", "--timesteps", type=int, help="LayoutDM diffusion step number")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch_size", type=int, default=512)
    parser.add_argument("-n", "--num_uncond_samples", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_gumbel_noise", action="store_true")
    parser.add_argument(
        "--test_only", action="store_true", help="Run test_data only"
    )
    parser.add_argument(
        "--val_only", action="store_true", help="Run validation_data only"
    )
    parser.add_argument("-c", "--cond_list", nargs='*', default=COND_LIST, type=str)
    args = parser.parse_args()

    for c in args.cond_list:
        assert c in COND_LIST, f"Invalid cond: {c}"

    split_list = ['test', 'validation']
    assert not (
        args.test_only and args.val_only
    ), "Cannot set both --test_only and --val_only"
    if args.test_only:
        split_list = ['test']
    elif args.val_only:
        split_list = ['validation']

    job_dir = os.path.normpath(args.job_dir)
    if (os.path.split(job_dir)[1]).isnumeric():
        # job_dir = "**/**/<JOB_NAME>/<SEED_NUM>"
        job_dir_wo_seed = os.path.split(job_dir)[0]
        job_name = os.path.basename(job_dir_wo_seed)
    else:
        # job_dir = "**/**/<JOB_NAME>"
        job_name = os.path.basename(job_dir)

    result_root = './results'
    result_dir = os.path.join(result_root, args.dataset, job_name)

    cfg = dict(
        num_timesteps=args.timesteps,
        corrector_start=-1,
        corrector_end=-1,
        corrector_steps=1,
        use_gumbel_noise=not args.no_gumbel_noise,
        corrector_t_list=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        corrector_mask_mode='thresh',
        corrector_mask_threshold=0.3,
        corrector_temperature=4.0,
    )

    # Run test commands
    for split in split_list:
        for cond in args.cond_list:
            run_test_command(
                cond=cond,
                job_dir=job_dir,
                result_dir=result_dir,
                cfg_dict=cfg,
                device=args.device,
                is_val=(split == 'validation'),
                batch_size=args.batch_size,
                num_uncond_samples=args.num_uncond_samples,
            )

    # calculate metrics
    cmd = ["python3", "./bin/calc_metrics.py", result_dir]
    if args.force:
        cmd.append("--force")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
