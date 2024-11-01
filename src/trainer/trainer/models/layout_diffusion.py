"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php

-------------------------------------------------------------------------------

This file is derived from the following repository:
https://github.com/microsoft/LayoutGeneration/blob/main/LayoutDiffusion/improved-diffusion/improved_diffusion/gaussian_diffusion.py

Original file: gaussian_diffusion.py
Author: Junyi42
License: MIT License (https://github.com/microsoft/LayoutGeneration/blob/main/LICENSE)
"""

## LayoutDiffusion ICCV2023

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor
from trainer.data.util import sparse_to_dense
from trainer.helpers.layout_diffusion_tokenizer import LayoutDiffusionTokenizer
from trainer.models.base_model import BaseModel
from trainer.models.categorical_diffusion.mild_corruption import MildCorruptionDiffusion
from trainer.models.common.util import shrink
from trainer.models.common.nn_lib import (
    CustomDataParallel,
    SeqLengthDistribution,
)

logger = logging.getLogger(__name__)


class LayoutDiffusion(BaseModel):
    def __init__(
        self,
        backbone_cfg: DictConfig,
        tokenizer: LayoutDiffusionTokenizer,
        transformer_type: str = "bert",
        pos_emb: str = "elem_attr",
        num_timesteps: int = 200,
        auxiliary_loss_weight: float = 1e-1,
        use_padding_as_vocab: bool = False,
        adaptive_auxiliary_loss: bool = True,
        backbone_shrink_ratio: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.transformer_type = transformer_type
        self.pos_emb = pos_emb
        # make sure MASK is the last vocabulary
        assert tokenizer.id_to_name(tokenizer.N_total - 1) == "mask"

        pos_emb_length = kwargs.pop('pos_emb_length', None)
        if pos_emb_length is None:
            pos_emb_length = tokenizer.max_token_length

        # Note: make sure learnable parameters are inside self.model
        self.tokenizer = tokenizer

        if backbone_shrink_ratio != 1.0:
            backbone_cfg = shrink(backbone_cfg, backbone_shrink_ratio)

        self.model = CustomDataParallel(
            MildCorruptionDiffusion(
                backbone_cfg=backbone_cfg,  # for fair comparison
                num_classes=tokenizer.N_total,
                max_token_length=tokenizer.max_token_length,
                num_timesteps=num_timesteps,
                pos_emb=pos_emb,
                pos_emb_length=pos_emb_length,
                transformer_type=transformer_type,
                auxiliary_loss_weight=auxiliary_loss_weight,
                tokenizer=tokenizer,
                use_padding_as_vocab=use_padding_as_vocab,
                adaptive_auxiliary_loss=adaptive_auxiliary_loss,
                **kwargs,
            )
        )
        self.apply(self._init_weights)
        self.compute_stats()

        self.use_padding_as_vocab = use_padding_as_vocab
        self.seq_dist = SeqLengthDistribution(tokenizer.max_seq_length)

    def forward(self, inputs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor]]:
        outputs, losses = self.model(inputs["seq"])
        loss = losses["loss"].mean()
        return outputs, {'loss': loss}

    def sample(
        self,
        batch_size: Optional[int] = 1,
        cond: Optional[Dict] = None,
        sampling_cfg: Optional[DictConfig] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        if cond: # TODO implement conditional generation
            raise NotImplementedError()
        else:
            init_seq = self.build_uncond_init_seq(batch_size)

        ids = self.model.sample(
            init_seq=init_seq, cond=cond, sampling_cfg=sampling_cfg, **kwargs
        ).cpu()
        layouts = self.tokenizer.decode(ids)
        return layouts
    
    def build_uncond_init_seq(self, batch_size: int) -> Tensor:
        """Build initial sequence for unconditional generation
        NOTE: maybe this function should be moved to tokenizer
        """
        n_elements = self.seq_dist.sample(batch_size) # [batch_size]

        N_max_element = self.tokenizer.max_seq_length
        special_tokens = self.tokenizer.special_tokens
        pad_id = self.tokenizer.name_to_id('pad')
        mask_id = self.tokenizer.name_to_id('mask')

        mask = torch.full((batch_size, N_max_element), fill_value=False)
        for i, n in enumerate(n_elements):
            mask[i, :n] = True
        
        _labels = torch.full((batch_size, N_max_element), pad_id).long()
        _bbox = torch.full((batch_size, N_max_element), pad_id).long()
        _labels[mask] = mask_id
        _bbox[mask] = mask_id

        _labels = _labels.unsqueeze(-1)
        _bbox = _bbox.unsqueeze(-1).repeat(1, 1, 4)

        if 'sep' in special_tokens:
            sep_id = self.tokenizer.name_to_id('sep')
            eos_id = self.tokenizer.name_to_id('eos')
            _sep = torch.full((batch_size, N_max_element), pad_id).long()
            _sep[mask] = sep_id

            seq_len = rearrange(n_elements, "b -> b 1")
            indices = rearrange(torch.arange(0, N_max_element), "s -> 1 s")
            eos_mask = (seq_len - 1) == indices
            _sep[eos_mask] = eos_id
            _sep = _sep.unsqueeze(-1)
            init_seq = torch.cat([_labels, _bbox, _sep], axis=-1)
        else:
            init_seq = torch.cat([_labels, _bbox], axis=-1)

        init_seq = rearrange(init_seq, "b s x -> b (s x)")

        return init_seq

    def aggregate_sampling_settings(
        self, sampling_cfg: DictConfig, args: DictConfig
    ) -> DictConfig:
        sampling_cfg = super().aggregate_sampling_settings(sampling_cfg, args)
        if args.time_difference > 0:
            sampling_cfg.time_difference = args.time_difference

        if hasattr(args, "num_timesteps"):
            sampling_cfg.num_timesteps = args.num_timesteps

        return sampling_cfg

    def preprocess(self, batch):
        bbox, label, _, mask = sparse_to_dense(batch)
        inputs = {"label": label, "mask": mask, "bbox": bbox}
        if self.training:
            self.seq_dist(mask) # Update SeqLengthDistribution

        ids = self.tokenizer.encode(inputs)
        return ids

    def optim_groups(
        self, weight_decay: float = 0.0
    ) -> Union[Iterable[Tensor], Dict[str, Tensor]]:
        if self.transformer_type == "layout_diffusion":
            ## no weight_decay in LayoutDiffusion(ICCV2023) implementation
            additional_no_decay = []
        else:
            base = "model.module.transformer.pos_emb"
            additional_no_decay = [
                f"{base}.{name}"
                for name in self.model.module.transformer.pos_emb.no_decay_param_names
            ]
        return super().optim_groups(
            weight_decay=weight_decay, additional_no_decay=additional_no_decay
        )
