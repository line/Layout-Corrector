"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Union, List

import torch
import torch.nn as nn
from einops import rearrange, repeat
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
from trainer.models.base_model import BaseModel
from trainer.models.common.util import shrink, get_dim_model
from trainer.models.common.nn_lib import (
    CustomDataParallel,
    CategoricalAggregatedTransformer,
)
from trainer.helpers.sampling import sample
from trainer.models.layoutdm import LayoutDM


class LayoutCorrector(BaseModel):
    def __init__(
        self,
        backbone_cfg: DictConfig,
        tokenizer: LayoutSequenceTokenizer,
        recon_type: str = "x_0",
        target: str = "mask",
        shrink_ratio: float = 29 / 32,  # same as LayoutDM
        time_step_range: Optional[Tuple[int, int]] = None,
        pos_emb: str = "default",
        attr_loss_weights: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0],
        use_padding_as_vocab: bool = True,
        transformer_type: str = "aggregated",
        **kwargs,
    ) -> None:
        super().__init__()

        backbone_cfg = shrink(backbone_cfg, shrink_ratio)
        backbone = instantiate(backbone_cfg)
        self.tokenizer = tokenizer
        assert recon_type in ["x_0", "x_t-1"]
        self.recon_type = recon_type

        assert target in [
            "mask",
            "recon_acc",
        ]
        self.target = target
        self.pos_emb = pos_emb
        self.use_padding_as_vocab = use_padding_as_vocab
        self.n_attr_per_elem = len(tokenizer.var_names)

        assert len(attr_loss_weights) == len(tokenizer.var_names)

        self.attr_loss_weights = torch.as_tensor(attr_loss_weights).reshape(1, -1)

        assert transformer_type in ["aggregated"]
        self.transformer_type = transformer_type
        self.model = CustomDataParallel(
            CategoricalAggregatedTransformer(
                backbone=backbone,
                dim_model=get_dim_model(backbone_cfg),
                num_classes=tokenizer.N_total,
                tokenizer=tokenizer,
                max_token_length=tokenizer.max_token_length,
                dim_head=1,  ## Corrector is a binary classifier
                use_additional_input=None,
                additional_input_dim=None,
                pos_emb=pos_emb,
                n_attr_per_elem=len(tokenizer.var_names),
                **kwargs,
            )
        )

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        if time_step_range is not None:
            assert len(time_step_range) == 2
        self.time_step_range = time_step_range

        self.model.apply(self._init_weights)
        self.compute_stats()

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor]]:
        t = batch["t"]  # [batch_size]
        x0_recon = batch["x0_recon"]  # LongTensor [batch_size, num_token] (token-index)
        gt = batch[self.target]  # [batch_size, num_token]
        padding_mask = batch["padding_mask"]
        num_token = x0_recon.shape[-1]
        num_element = num_token // self.n_attr_per_elem

        out = self.model(
            x0_recon,
            timestep=t,
            self_cond=None,
            src_key_padding_mask=None if self.use_padding_as_vocab else padding_mask,
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
            src_key_padding_mask=None if self.use_padding_as_vocab else padding_mask,
        )["logits"]
        confidence = rearrange(
            confidence, "b s 1 -> b s"
        )  # high confidence indicates clean token
        if not self.use_padding_as_vocab:
            confidence[padding_mask] = 1000.0
        return confidence

    def sample_time(self, b: int, device: torch.device):
        t0, t1 = self.time_step_range
        t = torch.randint(t0, t1, (b,), device=device).long()
        return t

    def run_diffusion_model(self, diffusion_model, t, padding_mask, x0, sampling_cfg):
        xt, log_xt = diffusion_model.model.corrupt_x(x0, t)
        log_x0_recon = diffusion_model.model.predict_start(
            log_xt,
            t=t,
            padding_mask=None if diffusion_model.use_padding_as_vocab else padding_mask,
        )
        if self.recon_type == "x_t-1":
            log_x_t_1 = diffusion_model.model.q_posterior(
                log_x_start=log_x0_recon, log_x_t=log_xt, t=t
            )
            mask_id = self.tokenizer.name_to_id("mask")
            log_x_t_1[:, mask_id, :] = -70.0  # disable mask token
            log_x0_recon = log_x_t_1

        # Copy PAD token from x0 to the corrupted token
        if not diffusion_model.use_padding_as_vocab:
            pad_id = self.tokenizer.name_to_id("pad")
            pad_log_onehot = (
                torch.zeros_like(log_x0_recon).log().clamp(-70.0, 0.0)
            )  # [bs, num_class, num_token]
            pad_log_onehot[:, pad_id, :] = 0.0
            log_x0_recon = torch.where(
                padding_mask.unsqueeze(1), pad_log_onehot, log_x0_recon
            )
        x0_recon = sample(log_x0_recon, sampling_cfg).squeeze(1)
        return xt, x0_recon

    @torch.no_grad()
    def preprocess(
        self,
        batch: Dict,
        diffusion_model: LayoutDM,
        sampling_cfg: DictConfig,
        t: Optional[Tensor] = None,
    ) -> Dict[str, Union[Tensor, None]]:
        if not isinstance(diffusion_model, (LayoutDM)):
            raise NotImplementedError(
                "For now, only LayoutDM is supported as diffusion_model"
            )
        x0 = batch["seq"]
        padding_mask = ~batch["mask"]
        b = x0.size(0)

        if t is not None:
            t = t.to(x0.device)
            if t.shape == (1,):
                t = t.repeat(b)
            else:
                assert (
                    t.size(0) == b
                ), f"t.shape should be (1,) or (batch_size,) but get t.shape = {t.shape} and batch_size = {b}"
        elif self.time_step_range is None:
            t, _ = diffusion_model.model.sample_time(b, x0.device, "importance")
        else:
            t = self.sample_time(b, x0.device)

        xt, x0_recon = self.run_diffusion_model(
            diffusion_model, t, padding_mask, x0, sampling_cfg
        )
        mask_t = (xt == x0).long()  # 1: clean token, 0: corrupted token

        # 1: correct recon, 0: wrong recon
        # token-wise reconstruction accuracy
        recon_acc = (x0 == x0_recon).long()

        return {
            "t": t,
            "x0": x0,
            "xt": xt,
            "x0_recon": x0_recon,
            "mask": mask_t,
            "recon_acc": recon_acc,
            "padding_mask": padding_mask,
        }

    def aggregate_sampling_settings(
        self, sampling_cfg: DictConfig, args: DictConfig
    ) -> DictConfig:
        """
        Set user-specified args for sampling cfg
        """
        # Aggregate refinement-related parameters
        is_ruite = type(self).__name__ == "RUITE"
        if args.cond == "refinement" and args.refine_lambda > 0.0 and not is_ruite:
            sampling_cfg.refine_mode = args.refine_mode
            sampling_cfg.refine_offset_ratio = args.refine_offset_ratio
            sampling_cfg.refine_lambda = args.refine_lambda
            sampling_cfg.refine_noise_std = args.refine_noise_std

        if args.cond == "relation" and args.relation_lambda > 0.0:
            sampling_cfg.relation_mode = args.relation_mode
            sampling_cfg.relation_lambda = args.relation_lambda
            sampling_cfg.relation_tau = args.relation_tau
            sampling_cfg.relation_num_update = args.relation_num_update

        if "num_timesteps" not in sampling_cfg:
            # for dec or enc-dec
            if "eos" in self.tokenizer.special_tokens:
                sampling_cfg.num_timesteps = self.tokenizer.max_token_length
            else:
                sampling_cfg.num_timesteps = args.num_timesteps

        ## Corrector Configs
        sampling_cfg.corrector_steps = getattr(args, "corrector_steps", 1)
        sampling_cfg.use_gumbel_noise = getattr(args, "use_gumbel_noise", False)
        sampling_cfg.corrector_start = getattr(args, "corrector_start", 0)
        sampling_cfg.corrector_end = getattr(
            args, "corrector_end", sampling_cfg.num_timesteps + 1
        )

        sampling_cfg.time_adaptive_temperature = getattr(
            args, "time_adaptive_temperature", False
        )
        if getattr(args, "corrector_t_list", None):
            sampling_cfg.corrector_t_list = getattr(args, "corrector_t_list")
        else:
            sampling_cfg.corrector_t_list = [-1]

        sampling_cfg.corrector_temperature = getattr(args, "corrector_temperature", 1.0)
        sampling_cfg.gumbel_temperature = getattr(args, "gumbel_temperature", 1.0)

        sampling_cfg.corrector_mask_mode = getattr(
            args, "corrector_mask_mode", "thresh"
        )
        sampling_cfg.corrector_mask_threshold = getattr(
            args, "corrector_mask_threshold", 0.7
        )

        return sampling_cfg

    def optim_groups(
        self, weight_decay: float = 0.0
    ) -> Union[Iterable[Tensor], Dict[str, Tensor]]:
        if self.pos_emb == "default" or self.pos_emb == "shuffle":
            additional_no_decay = [
                "model.module.pos_emb.pos_emb",
            ]
        elif self.pos_emb == "elem_attr":
            base = "model.module.pos_emb"
            additional_no_decay = [
                f"{base}.{name}"
                for name in self.model.module.pos_emb.no_decay_param_names
            ]
        elif self.pos_emb == "none":
            additional_no_decay = []
        elif self.pos_emb in ["pos_enc", "shuffle_pos_enc"]:
            additional_no_decay = []
        else:
            raise NotImplementedError()

        return super().optim_groups(
            weight_decay=weight_decay, additional_no_decay=additional_no_decay
        )
    