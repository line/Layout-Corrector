"""
Analyze token sticking rate from a dumped generation process by visualize_generation_process.py
"""
from __future__ import annotations

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import torch
from einops import rearrange, repeat


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_tensor(ids, decoded_ids):
    tensor_ids = torch.Tensor(ids).long()
    tensor_decoded_ids = [
        dict(
            bbox=torch.Tensor(_dec_ids["bbox"]).long(),
            label=torch.Tensor(_dec_ids["label"]).long(),
            mask=torch.Tensor(_dec_ids["mask"]).bool(),
        )
        for _dec_ids in decoded_ids 
    ]
    return tensor_ids, tensor_decoded_ids


def calc_token_sticking_rate(
    ids: list[torch.Tensor],
    pad_id: int,
    mask_id: int,
) -> list[float]:
    """Calculate token sticking rate between x_0 and x_t
    
    The results are average of batch samples.
    MASK tokens are ignored and optionally PAD tokens are too.
    """
    num_timesteps = len(ids)
    token_sticking_rates = list()
    x_0 = ids[-1]
    for t in range(num_timesteps):
        x_t = ids[t]  # (B, num_attributes * num_max_elements)
        mask_idx = x_t == mask_id
        pad_idx = x_t == pad_id
        invalid_idx = torch.logical_or(mask_idx, pad_idx)
        num_valid_idx = (~invalid_idx).sum()

        match_idx = x_t == x_0
        match_idx[invalid_idx] = False
        token_sticking_rate = (match_idx.sum() / num_valid_idx).cpu().item() if num_valid_idx > 0 else 0.0
        token_sticking_rates.append(token_sticking_rate)

    token_sticking_rates = 100 * np.array(token_sticking_rates)
    return token_sticking_rates


def calc_elem_sticking_rate(
    ids: list[torch.Tensor],
    pad_id: int,
    mask_id: int,
    num_attributes: int = 5,
) -> list[float]:
    """Calculate element sticking rate between x_0 and x_t
    
    The results are average of batch samples.
    If an element includes a MASK token, the corresponding element is ignored.
    We focus on completly generated elements only.
    """
    num_timesteps = len(ids)
    elem_sticking_rates = list()
    
    # e = num_tokens_in_elem, s = num_elements
    assert ids[-1].shape[-1] % num_attributes == 0 
    x_0 = rearrange(ids[-1], "b (e s) -> b e s", e=num_attributes)
    for t in range(num_timesteps):
        x_t = rearrange(ids[t], "b (e s) -> b e s", e=num_attributes)
        # When an element includes MASK, we don't count the element as effective one.
        elem_mask_seq = (x_t == mask_id).any(dim=1)  # (b, s)
        elem_pad_seq = (x_t == pad_id).any(dim=1)
        invalid_elem_seq = torch.logical_or(elem_mask_seq, elem_pad_seq)
        num_valid_seq = (~invalid_elem_seq).sum()

        token_match_idx = x_t == x_0
        invalid_token_idx = repeat(invalid_elem_seq, "b s -> b e s", e=num_attributes)
        token_match_idx[invalid_token_idx] = False
        elem_match_idx = token_match_idx.all(dim=1)
        
        elem_sticking_rate = (elem_match_idx.sum() / num_valid_seq).cpu().item() if num_valid_seq > 0 else 0.0
        elem_sticking_rates.append(elem_sticking_rate)

    elem_sticking_rates = 100 * np.array(elem_sticking_rates)
    return elem_sticking_rates


def calc_mask_rate(
    ids: list[torch.Tensor],
    mask_id: int,
) -> list[float]:
    """Calculate mask rate on each timestep
    
    The results are average of batch samples.
    """
    num_timesteps = len(ids)
    mask_rates = list()
    for t in range(num_timesteps):
        x_t = ids[t]  # (B, num_attributes * num_max_elements)
        mask_idx = x_t == mask_id
        num_mask_tokens = mask_idx.sum()
        num_all_tokens = x_t.shape[0] * x_t.shape[1]
        mask_rates.append(num_mask_tokens / num_all_tokens)

    mask_rates = 100 * np.array(mask_rates)
    return mask_rates


def plot_sticking_rate(ax, sticking_rates, label="label", title="title"):
    timestep_axis = np.arange(len(sticking_rates))[::-1]  # flip to align the generation process
    ax.plot(timestep_axis, sticking_rates, label=label)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Sticking Rate [%]")
    ax.set_xticks(np.arange(0, 102, 10))
    ax.set_yticks(np.arange(0, 120, 20))
    ax.set_ylim(-5, 110)
    ax.set_aspect(0.5)
    ax.grid()
    ax.legend(loc="lower left")
    ax.set_title(title)
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path", 
                        type=str, 
                        help="A pickle file path dumped by dump_generation_process.py")
    args = parser.parse_args()

    assert args.pkl_path.split("/")[-1].endswith(".pickle")
    with open(args.pkl_path, "rb") as f:
        outputs = pickle.load(f)

    # Extract constants
    SAVE_DIR = os.path.dirname(args.pkl_path)
    GEN_JOB_DIR = outputs["gen_job_dir"]
    CORR_JOB_DIR = outputs.get("corr_job_dir")
    PAD_ID = outputs["pad_id"]
    MASK_ID = outputs["mask_id"]
    NUM_ATTRIBUTES = outputs["num_attributes"]
    NUM_CATEGORY = outputs["num_category"]
    NUM_BBOX_BINS = outputs["num_bbox_bins"]
    CORRECTOR_TIMESTEPS = outputs.get("corrector_timesteps")

    logger.info(f"GEN_JOB_DIR: {GEN_JOB_DIR}\n"
                f"CORR_JOB_DIR: {CORR_JOB_DIR}\n"
                f"SAVE_DIR: {SAVE_DIR}")
    
    # Set custom configurations
    SAVE_TIMESTEPS = list(range(0, 10, 2))
    NUM_ELEMENT_LIMIT = 10
    
    # Extract generation results
    orig_ids = outputs["original_ids"]
    orig_decoded_ids = outputs["original_decoded_ids"]
    corr_ids = outputs.get("corrector_ids")
    corr_decoded_ids = outputs.get("corrector_decoded_ids")

    # Convert list to Tensor
    orig_ids, orig_decoded_ids = to_tensor(orig_ids, orig_decoded_ids)
    if corr_ids is not None:
        corr_ids, corr_decoded_ids = to_tensor(corr_ids, corr_decoded_ids)

    # Analyze layout generation process for the original model
    orig_token_sticking_rates = calc_token_sticking_rate(
        orig_ids, PAD_ID, MASK_ID
    )
    orig_elem_sticking_rates = calc_elem_sticking_rate(
        orig_ids, PAD_ID, MASK_ID, num_attributes=NUM_ATTRIBUTES
    )
    orig_mask_rates = calc_mask_rate(orig_ids, MASK_ID)

    if CORR_JOB_DIR is not None:
        # Analyze layout generation process for the corrector model
        corr_token_sticking_rates = calc_token_sticking_rate(
            corr_ids, PAD_ID, MASK_ID
        )
        corr_elem_sticking_rates = calc_elem_sticking_rate(
            corr_ids, PAD_ID, MASK_ID, num_attributes=NUM_ATTRIBUTES
        )
        corr_mask_rates = calc_mask_rate(corr_ids, MASK_ID)

    # plot sticking rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # (nrow, ncol)
    ax1 = plot_sticking_rate(ax1, orig_token_sticking_rates, label="Original", title="Token Sticking Rate")
    ax2 = plot_sticking_rate(ax2, orig_elem_sticking_rates, label="Original", title="Element Sticking Rate")
    if CORR_JOB_DIR is not None:
        ax1 = plot_sticking_rate(ax1, corr_token_sticking_rates, label="Corrector", title="Token Sticking Rate")
        ax2 = plot_sticking_rate(ax2, corr_elem_sticking_rates, label="Corrector", title="Element Sticking Rate")
    ax1.grid(visible=True, alpha=0.2)
    ax2.grid(visible=True, alpha=0.2)
    fig.savefig(os.path.join(SAVE_DIR, "original_sticking_rate.pdf"))
    