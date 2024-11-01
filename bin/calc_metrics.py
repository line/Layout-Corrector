"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

import argparse
import json
import os
import subprocess
from glob import glob
from typing import Dict, List

AVG_COMMON_KEYS = ["fid", "precision", "recall", "density", "coverage", "overlap-LayoutGAN++", "alignment-LayoutGAN++"]
AVG_CONDITIONAL_KEYS = ["maximum_iou", "DocSim"]

def run_eval_command(dir_name):
    cmd = [
        'python3',
        'eval.py',
        dir_name,
        # '--compute_real'
    ]
    subprocess.run(cmd, check=False)

def sort_dict_by_keys(data: Dict[str, float], target_keys: List[str]) -> Dict[str, float]:
    sorted_dict = {}
    for key in target_keys:
        if key in data:
            sorted_dict[key] = data[key]
    return sorted_dict

def display_metrics(result_dir):
    avg_json_path = os.path.join(result_dir, "scores_fake_avg.json")

    with open(avg_json_path, "r") as f:
        avg_result = json.load(f)
    
    avg_result = sort_dict_by_keys(
        avg_result, AVG_COMMON_KEYS + AVG_CONDITIONAL_KEYS
    )

    avg_result_str, avg_result_for_spread_sheet = "", ""
    for key, score in avg_result.items():
        if "alignment" in key:
            score = 100 * score
        avg_result_str = avg_result_str + f"{key}: {score:.4f}, "
        avg_result_for_spread_sheet = avg_result_for_spread_sheet + f"{score:.4f}\n"

    seed_json_paths = glob(f"{result_dir}/scores_fake_seed_*.json")
    seed_fid_str, seed_fid_for_spread_sheet = "", ""
    for seed_json_path in seed_json_paths:
        with open(seed_json_path) as f:
            seed_result = json.load(f)
        seed_no, seed_fid = seed_result["seed"], seed_result["fid"]
        seed_fid_str += f"seed{seed_no}: {seed_fid:.4f}, "
        seed_fid_for_spread_sheet += f"{seed_fid:.4f}\n"
    
    _avg_result = avg_result_for_spread_sheet.split('\n')
    _seed_fid = seed_fid_for_spread_sheet.split('\n')

    base_dir_name = os.path.basename(result_dir)
    condition = base_dir_name.split("_")[0]
    _spread_sheet_horizontal = [condition] + _avg_result + _seed_fid

    print(f"{condition}: {base_dir_name}")
    print(",".join(_spread_sheet_horizontal))
    print('')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str)
    parser.add_argument("--force", action='store_true')
    args = parser.parse_args()

    result_dir_list = glob(os.path.join(args.base_dir, '*'))

    for result_dir in result_dir_list:
        pkl_path_list = glob(os.path.join(result_dir, 'seed_*.pkl'))
        if not pkl_path_list:
            continue

        if args.force:
            run_eval = True
        else:
            run_eval = False
            for pkl_path in pkl_path_list:
                seed = os.path.basename(pkl_path).split('.')[0]
                seed = int(seed.replace('seed_', ''))
                json_path = os.path.join(result_dir, f'scores_fake_seed_{seed}.json')
                if not os.path.exists(json_path):
                    run_eval = True
                    break
            if not run_eval:
                json_path = os.path.join(result_dir, f'scores_fake_avg.json')
                run_eval = not(os.path.join(json_path))

        if run_eval:
            print('RUN:', result_dir)

            run_eval_command(result_dir)
            try:
                display_metrics(result_dir)
            except:
                print('Error in display_metrics')
        else:
            continue

if __name__ == "__main__":
    main()