"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php

-------------------------------------------------------------------------------

This file is derived from the following repository:
https://github.com/CyberAgentAILab/layout-dm/tree/9cf122d3be74b1ed8a8d51e3caabcde8164d238e

Original file: src/trainer/trainer/models/layoutdm.py
Author: naoto0804
License: Apache-2.0 License

Modifications have been made to the original file to fit the requirements of this project.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
from copy import deepcopy
from einops import repeat
from omegaconf import DictConfig
from torch import Tensor
from trainer.data.util import sparse_to_dense
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.models.base_model import BaseModel
from trainer.models.categorical_diffusion.constrained import (
    ConstrainedMaskAndReplaceDiffusion,
)
from trainer.models.categorical_diffusion.vanilla import VanillaMaskAndReplaceDiffusion
from trainer.models.common.nn_lib import CustomDataParallel
from trainer.models.common.util import shrink
from trainer.models.common.nn_lib import (
    CustomDataParallel,
    SeqLengthDistribution,
)
from trainer.helpers.task import duplicate_cond

logger = logging.getLogger(__name__)

Q_TYPES = {
    "vanilla": VanillaMaskAndReplaceDiffusion,
    "constrained": ConstrainedMaskAndReplaceDiffusion,
}


class LayoutDM(BaseModel):
    def __init__(
        self,
        backbone_cfg: DictConfig,
        tokenizer: LayoutSequenceTokenizer,
        transformer_type: str = "flattened",
        pos_emb: str = "elem_attr",
        num_timesteps: int = 100,
        auxiliary_loss_weight: float = 1e-1,
        q_type: str = "single",
        seq_type: str = "poset",
        use_padding_as_vocab: bool = True,
        backbone_shrink_ratio: float = 29 / 32,
        **kwargs,
    ) -> None:
        super().__init__()
        assert q_type in Q_TYPES
        assert seq_type in ["set", "poset"]

        self.pos_emb = pos_emb
        self.seq_type = seq_type
        # make sure MASK is the last vocabulary
        assert tokenizer.id_to_name(tokenizer.N_total - 1) == "mask"

        pos_emb_length = kwargs.pop('pos_emb_length', None)
        if pos_emb_length is None:
            pos_emb_length = tokenizer.max_token_length

        # Note: make sure learnable parameters are inside self.model
        self.tokenizer = tokenizer
        model = Q_TYPES[q_type]

        _backbone_cfg = deepcopy(backbone_cfg)
        if backbone_shrink_ratio != 1.0: # for fair comparison
            _backbone_cfg = shrink(backbone_cfg, backbone_shrink_ratio)

        self.model = CustomDataParallel(
            model(
                backbone_cfg=_backbone_cfg,
                num_classes=tokenizer.N_total,
                max_token_length=tokenizer.max_token_length,
                num_timesteps=num_timesteps,
                pos_emb=pos_emb,
                pos_emb_length=pos_emb_length,
                transformer_type=transformer_type,
                auxiliary_loss_weight=auxiliary_loss_weight,
                tokenizer=tokenizer,
                use_padding_as_vocab=use_padding_as_vocab,
                **kwargs,
            )
        )

        self.apply(self._init_weights)
        self.compute_stats()

        self.use_padding_as_vocab = use_padding_as_vocab
        if not self.use_padding_as_vocab:
            self.seq_dist = SeqLengthDistribution(tokenizer.max_seq_length)

    def forward(self, inputs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor]]:
        outputs, losses = self.model(inputs["seq"])

        # aggregate losses for multi-GPU mode (no change in single GPU mode)
        new_losses = {k: v.mean() for (k, v) in losses.items()}

        return outputs, new_losses

    def sample(
        self,
        batch_size: Optional[int] = 1,
        cond: Optional[Dict] = None,
        sampling_cfg: Optional[DictConfig] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        if not self.use_padding_as_vocab:
            if cond:
                if cond['type'] in ['c', 'cwh']:
                    pad_id = self.tokenizer.name_to_id("pad")
                    cond = duplicate_cond(cond, batch_size)
                    padding_mask = (cond["seq"] == pad_id)
                elif cond['type'] in ['partial', 'partial_shift']:
                    raise NotImplementedError(f'cond=partial, partial_shift are not supported for use_padding_as_vocab=False')
                else:
                    raise NotImplementedError()
            else:
                n_elements = self.seq_dist.sample(batch_size) * self.tokenizer.N_var_per_element
                padding_mask = torch.full((batch_size, self.tokenizer.max_token_length), fill_value=False)
                for i, n in enumerate(n_elements):
                    padding_mask[i, n:] = True
            
            if self.seq_type == "set":
                # randomly shuffle [PAD]'s location
                B, S = padding_mask.size()
                C = self.tokenizer.N_var_per_element
                for i in range(B):
                    indices = torch.randperm(S // C)
                    indices = repeat(indices * C, "b -> (b c)", c=C)
                    indices += torch.arange(S) % C
                    padding_mask[i, :] = padding_mask[i, indices]
        else:
            padding_mask = None
        if cond:
            if cond['type'] == 'partial_shift':
                assert cond['mask'][:, 0:5].all() # check if condition_tokens are at the beginning of seq

        # if cond:
        #     if cond['type'] == 'partial':
        #         # shift valid condition at the beginning of seq
        #         mask_id = self.tokenizer.name_to_id("mask")
        #         new_seq = torch.full_like(cond["seq"], mask_id)
        #         new_mask = torch.full_like(cond["mask"], False)
        #         n_seen_elements = cond["mask"].sum(dim=-1) / self.tokenizer.N_var_per_element
        #         n_seen_elements = n_seen_elements.long()
        #         for i, n in enumerate(n_seen_elements):
        #             ind_end = n * self.tokenizer.N_var_per_element
        #             s = cond["seq"][i]
        #             new_seq[i][:ind_end] = s[cond["mask"][i]]
        #             new_mask[i][:ind_end] = True
        #         cond["seq"] = new_seq
        #         cond["mask"] = new_mask

        ids = self.model.sample(
            batch_size=batch_size, cond=cond, sampling_cfg=sampling_cfg, padding_mask=padding_mask, **kwargs
        ).cpu()
        layouts = self.tokenizer.decode(ids)
        return layouts

    def aggregate_sampling_settings(
        self, sampling_cfg: DictConfig, args: DictConfig
    ) -> DictConfig:
        sampling_cfg = super().aggregate_sampling_settings(sampling_cfg, args)
        if args.time_difference > 0:
            sampling_cfg.time_difference = args.time_difference

        return sampling_cfg

    def preprocess(self, batch):
        bbox, label, _, mask = sparse_to_dense(batch)
        inputs = {"label": label, "mask": mask, "bbox": bbox}
        if not self.use_padding_as_vocab and self.training:
            self.seq_dist(mask) # Update SeqLengthDistribution

        ids = self.tokenizer.encode(inputs)
        if self.seq_type == "set":
            # randomly shuffle [PAD]'s location
            B, S = ids["mask"].size()
            C = self.tokenizer.N_var_per_element
            for i in range(B):
                indices = torch.randperm(S // C)
                indices = repeat(indices * C, "b -> (b c)", c=C)
                indices += torch.arange(S) % C
                for k in ids:
                    ids[k][i, :] = ids[k][i, indices]
        return ids

    def optim_groups(
        self, weight_decay: float = 0.0
    ) -> Union[Iterable[Tensor], Dict[str, Tensor]]:
        base = "model.module.transformer.pos_emb"
        additional_no_decay = [
            f"{base}.{name}"
            for name in self.model.module.transformer.pos_emb.no_decay_param_names
        ]
        return super().optim_groups(
            weight_decay=weight_decay, additional_no_decay=additional_no_decay
        )
