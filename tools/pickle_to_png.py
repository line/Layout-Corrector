import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from trainer.data.util import sparse_to_dense
from trainer.global_configs import DATASET_DIR
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.helpers.metric import Layout
from trainer.helpers.visualization import save_image

# necessary to load pickle file
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

SIZE = (360, 240)


def preprocess(layouts: List[Layout], max_len: int, device: torch.device):
    layout = defaultdict(list)
    for b, l in layouts:
        pad_len = max_len - l.shape[0]
        bbox = torch.tensor(
            np.concatenate([b, np.full((pad_len, 4), 0.0)], axis=0),
            dtype=torch.float,
        )
        layout["bbox"].append(bbox)
        label = torch.tensor(
            np.concatenate([l, np.full((pad_len,), 0)], axis=0),
            dtype=torch.long,
        )
        layout["label"].append(label)
        mask = torch.tensor(
            [True for _ in range(l.shape[0])] + [False for _ in range(pad_len)]
        )
        layout["mask"].append(mask)
    bbox = torch.stack(layout["bbox"], dim=0).to(device)
    label = torch.stack(layout["label"], dim=0).to(device)
    mask = torch.stack(layout["mask"], dim=0).to(device)
    padding_mask = ~mask
    return bbox, label, padding_mask, mask


def load_pkl(pkl_path):
    assert os.path.exists(pkl_path), pkl_path
    with open(pkl_path, "rb") as file_obj:
        x = pickle.load(file_obj)
    return x


def gt_preprocess(batch: List, tokenizer):
    key_list = ["bbox", "label", "mask"]
    gt_batch = defaultdict(list)

    for b in batch:
        bbox, label, _, mask = sparse_to_dense(b)
        gt_cond = tokenizer.encode({"label": label, "mask": mask, "bbox": bbox})
        if "bos" in tokenizer.special_tokens:
            gt = tokenizer.decode(gt_cond["seq"][:, 1:])
        else:
            gt = tokenizer.decode(gt_cond["seq"])

        for k in key_list:
            gt_batch[k].append(gt[k])
    gt_dic = {k: torch.stack(v) for k, v in gt_batch.items()}
    padding_mask = ~gt_dic["mask"]
    return gt_dic["bbox"], gt_dic["label"], padding_mask, gt_dic["mask"]


def main():
    parser = argparse.ArgumentParser(
        description="Save PNG files from a .pkl file containing raw data."
    )

    parser.add_argument("pkl_path", type=str, help="Path to the `.pkl` file.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name. The save directory will be `{SAVE_DIR}/{DATASET}_{COND}`.",
    )
    parser.add_argument(
        "--cond",
        type=str,
        choices=["unconditional", "c", "cwh"],
        help="Condition name.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name to be used in the file name `{IDX}_{LABEL}.png`. Something like layout_corrector, maskgit, etc.",
    )
    parser.add_argument(
        "-s", "--start_ind", type=int, default=0, help="Start index of samples to save."
    )
    parser.add_argument(
        "-e", "--end_ind", type=int, default=100, help="End index of samples to save."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="sample_pngs",
        help="Directory path where the results will be saved.",
    )
    parser.add_argument(
        "--save_real",
        action="store_true",
        help="Save real layout as well. Filename will be `{IDX}_real.png`.",
    )
    args = parser.parse_args()

    pkl_path = args.pkl_path
    obj = load_pkl(pkl_path)

    cond = args.cond
    seed = (
        os.path.basename(pkl_path).split(".")[0].split("_")[-1]
    )  # assume the file name is `seed_{SEED}.pkl`
    save_dir = os.path.join(args.save_dir, f"{args.dataset}_{cond}")
    os.makedirs(save_dir, exist_ok=True)

    train_cfg = obj["train_cfg"]

    train_cfg.dataset.dir = DATASET_DIR

    dataset = instantiate(train_cfg.dataset)(split="test", transform=None)
    tokenizer = LayoutSequenceTokenizer(
        data_cfg=train_cfg.data, dataset_cfg=train_cfg.dataset
    )

    save_kwargs = {
        "colors": dataset.colors,
        "names": dataset.labels,
        "line_width": 2,
        "canvas_size": SIZE,
        "use_grid": True,
    }

    for ind in range(args.start_ind, args.end_ind):
        batch = [obj["results"][ind]]
        max_len = max(len(g[-1]) for g in batch)
        bbox, label, padding_mask, mask = preprocess(batch, max_len, "cpu")

        out_path = os.path.join(save_dir, f"seed{seed}_{ind:05}_{args.name}.png")
        save_image(bbox, label, mask, nrow=1, out_path=out_path, **save_kwargs)

        if args.save_real:
            batch = [dataset[ind]]
            bbox, label, padding_mask, mask = gt_preprocess(batch, tokenizer)
            out_path = os.path.join(save_dir, f"seed{seed}_{ind:05}_real.png")
            save_image(bbox, label, mask, nrow=1, out_path=out_path, **save_kwargs)


if __name__ == "__main__":
    main()
