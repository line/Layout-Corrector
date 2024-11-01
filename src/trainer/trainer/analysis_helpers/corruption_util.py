"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

import copy
import random
import torch

from trainer.models.layoutdm import LayoutDM


def get_corrupted_seq(
      gt_seq: torch.Tensor,
      model: LayoutDM,
      replace_with_mask: bool = False,
      num_corrupted: int = 1
) -> tuple[torch.Tensor, list[int], int]:
    """Create corrupted token sequence from the ground truth sequence.

    Args:
        gt_seq (torch.Tensor): Ground truth sequence.
        model (LayoutDM): LayoutDM model.
        replace_with_mask (bool): Replace the selected tokens with mask token. Defaults to False.
        num_corrupted (int): The number of tokens to be replaced with random token. Defaults to 1.

    Returns:
        corrupted_seq (torch.Tensor | None): Corrupted sequence.
        replace_target_index (list[int] | None): List of replaced token indices.
        num_valid (int | None): The number of valid tokens.

    Note:
        Need to add `var2index_range` property to work on models except LayoutDM.
    
    """
    assert hasattr(model, "tokenizer"), "Model should have tokenizer."
    assert gt_seq.shape[0] == 1, f"Allow batch_size = 1 only, recieved {gt_seq.shape[0]}."
    
    # The pad_id is the maximum token value in the sequence, in which mask tokens are not in.
    # To focus on regular tokens, we extract the first PAD index.
    if model.tokenizer.name_to_id("pad") in gt_seq:
        first_pad_index = torch.argmax(gt_seq)
    else:
        first_pad_index = model.tokenizer.max_token_length

    # we require at least 4 tokens to estimate corrupted tokens
    valid_indices = list(range(first_pad_index))
    num_valid = len(valid_indices)
    if num_valid < num_corrupted + model.tokenizer.N_var_per_element - 1:
        return None, None, None

    replace_target_index = random.sample(valid_indices, num_corrupted)
    corrupted_seq = copy.deepcopy(gt_seq)
    var2index_range = model.var2index_range
    for tgt_index in replace_target_index:
        if replace_with_mask:
            corrupted_seq[0, tgt_index] = model.tokenizer.name_to_id("mask")
            continue

        tgt_var = model.get_var_from_index(tgt_index)  # tgt_var: ("c", "x", "y", "w", "h")
        tgt_var_range = var2index_range[tgt_var]

        original_token = gt_seq[0, tgt_index].item()
        replace_token = original_token
        while original_token == replace_token:
            replace_token = random.randint(tgt_var_range[0], tgt_var_range[1] - 1)

        corrupted_seq[0, tgt_index] = replace_token
    return corrupted_seq, replace_target_index, num_valid


def get_mild_corrupted_seq(
      gt_seq: torch.Tensor,
      model: LayoutDM,
      num_corrupted: int = 1,
      max_token_transition_step: int = 3,
) -> tuple[torch.Tensor, list[int], int]:
    """Create moderately corrupted token sequence from the ground truth sequence.

    Args:
        gt_seq (torch.Tensor): Ground truth sequence.
        model (LayoutDM): LayoutDM model.
        num_corrupted (int): The number of tokens to be replaced with random token. Defaults to 1.
        max_token_transition_step (int): The maximum step of token transition.

    Returns:
        corrupted_seq (torch.Tensor | None): Corrupted sequence.
        replace_target_index (list[int] | None): List of replaced token indices.
        num_valid (int | None): The number of valid tokens.

    Note:
        Need to add `var2index_range` property to work on other models.
    
    """
    assert hasattr(model, "tokenizer"), "Model should have tokenizer."
    assert gt_seq.shape[0] == 1, f"Allow batch_size = 1 only, recieved {gt_seq.shape[0]}."
    
    # The pad_id is the maximum token value in the sequence, in which mask tokens are not in.
    # To focus on regular tokens, we extract the first PAD index.
    if model.tokenizer.name_to_id("pad") in gt_seq:
        first_pad_index = torch.argmax(gt_seq)
    else:
        first_pad_index = model.tokenizer.max_token_length

    # we require at least 4 tokens to estimate corrupted tokens
    valid_indices = list(range(first_pad_index))
    num_valid = len(valid_indices)
    if num_valid < num_corrupted + model.tokenizer.N_var_per_element - 1:
        return None, None, None

    replace_target_index = random.sample(valid_indices, num_corrupted)
    corrupted_seq = copy.deepcopy(gt_seq)
    var2index_range = model.var2index_range
    for tgt in replace_target_index:
        tgt_var = model.get_var_from_index(tgt)
        tgt_var_range = var2index_range[tgt_var]

        original_token = gt_seq[0, tgt].item()
        replace_token = original_token
        while original_token == replace_token:
            if tgt_var == "c":
                # For category, we allow any token
                replace_token = random.randint(tgt_var_range[0], tgt_var_range[1] - 1)
            else:
                # geometry var
                replace_token = random.randint(original_token - max_token_transition_step, 
                                               original_token + max_token_transition_step)
        replace_token = max(tgt_var_range[0], min(tgt_var_range[1] - 1, replace_token))

        corrupted_seq[0, tgt] = replace_token
    return corrupted_seq, replace_target_index, num_valid
