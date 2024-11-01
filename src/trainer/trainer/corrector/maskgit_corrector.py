"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from einops import rearrange, repeat
from omegaconf import DictConfig
from torch import Tensor
from trainer.helpers.sampling import sample
from trainer.models.maskgit import MaskGIT
from .layout_corrector import LayoutCorrector


class MaskGitLayoutCorrector(LayoutCorrector):
    """Corrector trained with MaskGIT
    """

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor]]:
        mask_ratio = batch["mask_ratio"]  # [batch_size]
        x0_recon = batch[
            "x0_recon"
        ]  # LongTensor [batch_size, num_token] (token-index)
        gt = batch[self.target]  # [batch_size, num_token]
        padding_mask = batch["padding_mask"]
        num_token = x0_recon.shape[-1]
        num_element = num_token // self.n_attr_per_elem

        out = self.model(
            x0_recon,
            timestep=mask_ratio,
            self_cond=None,
            src_key_padding_mask=None
            if self.use_padding_as_vocab
            else padding_mask,
            attention_bias=None,
        )  # [batch_size, num_token, 1]
        if not self.use_padding_as_vocab:
            out["logits"][padding_mask.unsqueeze(-1)] = 1000.0

        loss_weight = repeat(
            self.attr_loss_weights, "1 s -> 1 (n s) 1", n=num_element
        ).to(x0_recon.device)

        bce_loss = self.bce_loss(out["logits"], gt.float().unsqueeze(-1))
        bce_loss = bce_loss * loss_weight
        if not self.use_padding_as_vocab:
            bce_loss[padding_mask.unsqueeze(-1)] = 0.0

        losses = {"bce_loss": bce_loss.mean()}
        return out, losses

    def calc_confidence_score(
        self,
        x0_recon: Tensor,
        timestep: Tensor,
        padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        confidence = self.model(
            x0_recon,
            timestep=timestep,
            self_cond=None,
            attention_bias=None,
            src_key_padding_mask=None
            if self.use_padding_as_vocab
            else padding_mask,
        )["logits"]
        confidence = rearrange(
            confidence, "b s 1 -> b s"
        )  # high confidence indicates clean token
        return confidence

    def run_maskgit(self, maskgit, batch, sampling_cfg):
        outputs, _ = maskgit(batch)
        logits = outputs['logits']
        seq_pred = sample(rearrange(logits, "b s c -> b c s"), sampling_cfg)
        seq_pred = rearrange(seq_pred, "b 1 s -> b s")
        out = batch["input"].clone()
        loss_mask = batch['loss_mask']
        out[loss_mask] = seq_pred[loss_mask]
        return out

    @torch.no_grad()
    def preprocess(
        self,
        batch: Dict,
        maskgit: MaskGIT,
        sampling_cfg: DictConfig,
        t: Optional[Tensor] = None,
    ) -> Dict[str, Union[Tensor, None]]:
        if not isinstance(maskgit, MaskGIT):
            raise NotImplementedError(
                "For now, only MaskGIT is supported as diffusion_model"
            )
        x0 = batch["target"]
        padding_mask = batch["padding_mask"] # [b, S]
        mask_ratio = batch["mask_ratio"] # [b]
        xt = batch["input"] # [b, s]
        mask_t = (~batch["loss_mask"]).long() # 1: clean token, 0: mask token

        x0_recon = self.run_maskgit(maskgit, batch, sampling_cfg)

        # token-wise reconstruction accuracy
        recon_acc = (x0 == x0_recon).long()

        return {
            "mask_ratio": mask_ratio,
            "x0": x0,
            "xt": xt,
            "x0_recon": x0_recon,
            "mask": mask_t,
            "recon_acc": recon_acc,
            "padding_mask": padding_mask,
        }
    