"""
This file is derived from the following repository:
https://github.com/CyberAgentAILab/layout-dm/tree/9cf122d3be74b1ed8a8d51e3caabcde8164d238e

Original file: src/trainer/trainer/helpers/util.py
Author: naoto0804
License: Apache-2.0 License

Modifications have been made to the original file to fit the requirements of this project.
"""

import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from fsspec.core import url_to_fs
from omegaconf import DictConfig, OmegaConf
from torch import BoolTensor, FloatTensor, LongTensor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def convert_xywh_to_ltrb(bbox: Union[np.ndarray, FloatTensor]):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def batch_topk_mask(
    scores: FloatTensor,
    topk: LongTensor,
    mask: Optional[BoolTensor] = None,
) -> Tuple[BoolTensor, FloatTensor]:
    assert scores.ndim == 2 and topk.ndim == 1 and scores.size(0) == topk.size(0)
    if mask is not None:
        assert mask.size() == scores.size()
        assert (scores.size(1) >= topk).all()

    # ignore scores where mask = False by setting extreme values
    if mask is not None:
        const = -1.0 * float("Inf")
        const = torch.full_like(scores, fill_value=const)
        scores = torch.where(mask, scores, const)

    sorted_values, _ = torch.sort(scores, dim=-1, descending=True)
    topk = rearrange(topk, "b -> b 1")

    # sorted_values: [B, S]
    # To avoid index error when topk contains S
    _dummy = torch.empty((scores.size(0), 1), device=scores.device).fill_(-1.0 * float("Inf"))
    sorted_values = torch.cat((sorted_values, _dummy), dim=1) # [B, S] -> [B, S+1]
    # Now sorted_values has [B, S+1] and still keeps descending order

    k_th_scores = torch.gather(sorted_values, dim=1, index=topk)

    topk_mask = scores > k_th_scores
    return topk_mask, k_th_scores


def batch_shuffle_index(
    batch_size: int,
    feature_length: int,
    mask: Optional[BoolTensor] = None,
) -> LongTensor:
    """
    Note: masked part may be shuffled because of unpredictable behaviour of sorting [inf, ..., inf]
    """
    if mask:
        assert mask.size() == [batch_size, feature_length]
    scores = torch.rand((batch_size, feature_length))
    if mask:
        scores[~mask] = float("Inf")
    _, indices = torch.sort(scores, dim=1)
    return indices


def dict2str(dic, level=0, indent_width=2, prefix=None) -> str:
    """dict to string for printing options.
    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.
    Return:
        (str): Option string for printing.
    """
    if prefix is None:
        prefix = " " * indent_width

    def key_prefix(is_last):
        s = "┗" if is_last else "┣"
        return prefix + s + " " * (indent_width - 1)

    def val_prefix(is_last):
        s = " " if is_last else "┃"
        return prefix + s + " " * (indent_width - 1)

    msg = "" if level == 0 else "\n"
    dict_len = len(dic)
    for i, (k, v) in enumerate(dic.items()):
        is_last_key = i == dict_len - 1
        key_str = key_prefix(is_last_key) + f"{k}: "
        if isinstance(v, (dict, DictConfig)):
            msg += key_str
            new_prefix = val_prefix(is_last_key)
            msg += dict2str(
                v, level + 1, indent_width=indent_width, prefix=new_prefix
            )
        elif isinstance(v, list):
            msg += key_str + "\n"
            new_prefix = val_prefix(is_last_key)
            for s in v:
                msg += new_prefix + f" - {s}\n"
        else:
            msg += key_str + f"{v}\n"
    return msg


def load_config(config_path: str) -> DictConfig:
    """Load config from file path.
    
    Args:
        config_path (str): Path to config file.

    Returns:
        DictConfig: Config object.
    """
    fs, _ = url_to_fs(config_path)
    if fs.exists(config_path):
        with fs.open(config_path, "rb") as file_obj:
            cfg = OmegaConf.load(file_obj)
        return cfg
    else:
        raise FileNotFoundError


if __name__ == "__main__":
    scores = torch.arange(6).view(2, 3).float()
    # topk = torch.arange(2) + 1
    topk = torch.full((2,), 3)
    mask = torch.full((2, 3), False)
    # mask[1, 2] = False
    print(batch_topk_mask(scores, topk, mask=mask))
