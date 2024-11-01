"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php

-------------------------------------------------------------------------------

We refer to the following repository:
https://github.com/microsoft/LayoutGeneration
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from trainer.helpers.layout_diffusion_tokenizer import LayoutDiffusionTokenizer
from trainer.models.common.nn_lib import (CategoricalTransformer,
                                          LayoutDiffusionTransformer)
from trainer.models.common.util import get_dim_model

from .base import BaseMaskAndReplaceDiffusion
from .mild_corruption_utils import (alpha_schedule, 
                                    extract, 
                                    gaussian_matrix2,
                                    sum_except_batch)
from .util import (index_to_log_onehot, 
                   log_1_min_a, 
                   log_add_exp,
                   log_categorical, 
                   log_onehot_to_index)

logger = logging.getLogger(__name__)


class MildCorruptionDiffusion(BaseMaskAndReplaceDiffusion):
    """Mild Forward Corruption Discrete Diffusion from LayoutDiffusion (ICCV2023)
    original implementation: 
    https://github.com/microsoft/LayoutGeneration/blob/main/LayoutDiffusion/improved-diffusion/improved_diffusion/discrete_diffusion.py
    """
    def __init__(
        self,
        *,
        backbone_cfg: DictConfig,
        tokenizer: LayoutDiffusionTokenizer,
        num_classes: int = 159,
        num_timesteps: int = 200,
        T_tilde: Optional[int] = None,
        max_token_length: int = 125,
        ## Discrete Diffusion Training
        adaptive_auxiliary_loss = False,
        auxiliary_loss_weight: float = 0.001,
        mask_weight = [1, 1],
        rescale_weight: bool = False,
        alignment_loss: bool = False,
        alignment_weight: float = 100.,
        pow_num: float = 2.5,
        mul_num: float = 12.4,
        ## Transformer args
        transformer_type: str = "layout_diffusion",
        dropout: float = 0.1,
        hidden_size: int = 768,
        time_encode_dim: int = 128,
        pos_emb: str = "elem_attr",
        pos_emb_length: Optional[int] = None,
        **kwargs,
    ):
        super(BaseMaskAndReplaceDiffusion, self).__init__()

        self.tokenizer = tokenizer

        if T_tilde is None:
            T_tilde = int(num_timesteps * 0.8)
        assert isinstance(T_tilde, int) and T_tilde > 0
        self.T_tilde = T_tilde

        self.num_classes = num_classes

        if pos_emb_length is None:
            pos_emb_length = max_token_length

        if pos_emb == "elem_attr":
            kwargs["n_attr_per_elem"] = len(tokenizer.var_names)

        backbone = instantiate(backbone_cfg)
        if transformer_type == "flattened":
            self.transformer = CategoricalTransformer(
                backbone=backbone,
                num_classes=self.num_classes,
                max_token_length=pos_emb_length,
                dim_model=get_dim_model(backbone_cfg),
                pos_emb=pos_emb,
                **kwargs,
            )
        elif transformer_type == "layout_diffusion":
            self.transformer = LayoutDiffusionTransformer(
                backbone=backbone,
                num_classes=tokenizer.N_total,
                time_encode_dim=time_encode_dim,
                hidden_size=hidden_size,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Invalide transformer_type: {transformer_type}")

        # hard coding to make it simple
        # NOTE: some of these parameters are not used in the current simplified implementation
        self.use_padding_as_vocab = False # number of elements have to be given
        alpha_init_type = "gaussian_refine_pow2.5"
        self.wo_bbox_absorb = False
        self.gaussian_matrix = True

        self.alignment_loss = alignment_loss
        self.alignment_weight = alignment_weight
        self.ori_schedule_type = alpha_init_type

        self.schedule_type = alpha_init_type
        self.max_token_length = max_token_length

        self.N_special = tokenizer.N_sp_token - 1 # -1 to exclude "MASK"
        self.N_type = tokenizer.N_category
        self.N_pos = tokenizer.N_bbox

        self.loss_type = "vb_stochastic"
        self.shape = max_token_length
        self.num_timesteps = num_timesteps
        self.parametrization = "x0"
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.rescale_weight = rescale_weight
        if adaptive_auxiliary_loss:
            self.auxiliary_loss_weight = auxiliary_loss_weight
        else:
            self.auxiliary_loss_weight = 0

        self.mask_weight = mask_weight

        self._register_transition_matrices(pow_num, mul_num)

        self.register_buffer("Lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(self.num_timesteps))

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps

        self.zero_vector = None

        self.prior_rule = 0  # inference rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
        self.prior_ps = 1024  # max number to sample per step
        self.prior_weight = 0  # probability adjust parameter, 'r' in Equation.11 of Improved VQ-Diffusion

        self.learnable_cf = False # not used

    def _register_transition_matrices(self, pow_num: float, mul_num: float) -> None:
        if not self.gaussian_matrix:  # gaussian refine
            raise NotImplementedError()

        # at, bt, ct, cumprod_at, cumprod_bt, cumprod_ct for type_token and pos_token
        alpha_schedule_dict = alpha_schedule(
            self.num_timesteps, T_tilde=self.T_tilde, type_classes=self.N_type, pos_classes=self.N_pos,
        )

        def _np_to_log_tensor(x: np.ndarray) -> Tensor:
            y = torch.tensor(x.astype("float64"))
            y = torch.log(y).clamp(-70, 0)
            return y

        # to_tensor -> log -> clamp
        log_alpha_shcedule_list = {
            f"log_{k}": _np_to_log_tensor(v) for k, v in alpha_schedule_dict.items()
        }
        # register schedules
        for k, v in log_alpha_shcedule_list.items():
            self.register_buffer(k, v.float())

        # register log_1_min_ct, log_1_min_cumprod_ct schedules
        for suffix in ["type", "pos"]:
            log_ct = log_alpha_shcedule_list[f"log_ct_{suffix}"]
            log_cumprod_ct = log_alpha_shcedule_list[f"log_cumprod_ct_{suffix}"]

            # calc log (1 - ct) -> check
            log_1_min_ct = log_1_min_a(log_ct)
            log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)
            assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1e-5
            assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1e-5

            self.register_buffer(f"log_1_min_ct_{suffix}", log_1_min_ct.float())
            self.register_buffer(
                f"log_1_min_cumprod_ct_{suffix}", log_1_min_cumprod_ct.float()
            )

        # register Gaussian matrix schedules
        bt_pos = torch.tensor(alpha_schedule_dict["bt_pos"].astype("float64"))
        bt_pos = torch.where(bt_pos == 0.0, bt_pos.max(), bt_pos)

        _q_onestep_mats = []
        for t in range(0, self.num_timesteps):
            _q_onestep_mats.append(
                gaussian_matrix2(t, bt=bt_pos.pow(2).pow(pow_num / 2) * mul_num)
            )
        _q_onestep_mats.append(np.ones((self.N_pos, self.N_pos)) / (self.N_pos**2))

        q_onestep_mats = np.stack(_q_onestep_mats, axis=0)
        q_onestep_mats = torch.from_numpy(q_onestep_mats).float()
        self.register_buffer("q_onestep_mats", q_onestep_mats)

        q_mat_t = self.q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):  # calc cumulative product
            # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
            q_mat_t = np.tensordot(q_mat_t, self.q_onestep_mats[t], axes=[[1], [0]])
            q_mats.append(q_mat_t)
        q_mats.append(np.ones((self.N_pos, self.N_pos)) / (self.N_pos**2))
        q_mats = np.stack(q_mats, axis=0)
        q_mats = torch.from_numpy(q_mats).float()
        self.register_buffer("q_mats", q_mats)
        assert self.q_mats.shape == (
            self.num_timesteps + 1, self.N_pos, self.N_pos,
        ), f"Invalid q_mats.shape: {self.q_mats.shape}"


    def q_pred_one_timestep(self, log_x_t, t):  # q(xt|xt_1)
        if not self.gaussian_matrix:
            raise NotImplementedError()

        device = log_x_t.device
        bz = log_x_t.shape[0]

        log_at_pos = extract(self.log_at_pos, t, log_x_t.shape)  # at
        log_bt_pos = extract(self.log_bt_pos, t, log_x_t.shape)  # bt
        log_ct_pos = extract(self.log_ct_pos, t, log_x_t.shape)  # ct
        log_1_min_ct_pos = extract(self.log_1_min_ct_pos, t, log_x_t.shape)  # 1-ct

        log_at_type = extract(self.log_at_type, t, log_x_t.shape)  # at~
        log_bt_type = extract(self.log_bt_type, t, log_x_t.shape)  # bt
        log_ct_type = extract(self.log_ct_type, t, log_x_t.shape)  # ct~
        log_1_min_ct_type = extract(self.log_1_min_ct_type, t, log_x_t.shape)  # 1-ct~

        mask = (t < self.T_tilde).unsqueeze(-1).unsqueeze(-1)

        num_special = self.N_special  # 0~4 are special tokens: ["START", "END", "UNK", "PAD", "|"]
        num_type = self.N_type  # type tokens
        num_pos = self.N_pos  # position tokens
        TINY = 1e-30

        # create tensor with batch size of `bz`
        def _zero_tensor(*shape):
            return torch.zeros(bz, *shape, device=device)

        def _eye_tensor(n):
            return torch.eye(n, device=device).expand(bz, -1, -1)  # [bz, n, n]

        ### Transition probability
        # something -> special_tokens [bz, num_special, num_classes]
        special2special = _eye_tensor(num_special)
        type2special    = _zero_tensor(num_special, num_type)
        pos2special     = _zero_tensor(num_special, num_pos)
        mask2special    = _zero_tensor(num_special, 1)
        trans2special   = torch.cat([special2special, type2special, pos2special, mask2special], dim=-1)

        # something -> type [bz, num_type, num_classes]
        special2type = _zero_tensor(num_type, num_special)
        type2type    = log_add_exp(_eye_tensor(num_type).clamp_min(TINY).log() + log_at_type, log_bt_type).exp()
        pos2type     = _zero_tensor(num_type, num_pos)
        mask2type    = _zero_tensor(num_type, 1)
        trans2type   = torch.cat([special2type, type2type, pos2type, mask2type], dim=-1)

        # something -> position [bz, num_pos, num_classes]
        special2pos = _zero_tensor(num_pos, num_special)
        type2pos    = _zero_tensor(num_pos, num_type)
        pos2pos     = log_add_exp(_eye_tensor(num_pos).clamp_min(TINY).log() + log_at_pos, log_bt_pos).exp()
        mask2pos    = _zero_tensor(num_pos, 1)
        trans2pos   = torch.cat([special2pos, type2pos, pos2pos, mask2pos], dim=-1)

        # something -> MASK [bz, 1, num_classes]
        special2mask = _zero_tensor(1, num_special)
        type2mask    = log_add_exp(_zero_tensor(1, num_type).clamp_min(TINY).log() + log_1_min_ct_type, log_ct_type).exp()
        pos2mask     = log_add_exp(_zero_tensor(1, num_pos).clamp_min(TINY).log() + log_1_min_ct_pos, log_ct_pos).exp()
        mask2mask    = torch.ones(bz, 1, 1, device=device)
        trans2mask   = torch.cat([special2mask, type2mask, pos2mask, mask2mask], dim=-1)

        ### Build matrix Q
        # when t > 0.8T
        matrix_1 = torch.cat(
            [
                trans2special,
                trans2type,
                trans2pos,
                trans2mask,
            ],
            dim=-2,
        )

        # when t <= 0.8T
        pos2pos_gaussian = self.q_onestep_mats[t].to(device)
        trans2pos_gaussian = torch.cat(
            [special2pos, type2pos, pos2pos_gaussian, mask2pos], dim=-1
        )
        matrix_2 = torch.cat(
            [
                trans2special,
                trans2type,
                trans2pos_gaussian,
                trans2mask,
            ],
            dim=-2,
        )
        matrix = torch.where(mask, matrix_2, matrix_1)

        log_probs = matrix.matmul(log_x_t.exp()).clamp(min=1e-30).log()
        return log_probs

    def q_pred(self, log_x_start, t):  # q(xt|x0)
        if not self.gaussian_matrix:
            raise NotImplementedError()

        device = log_x_start.device
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        bz = log_x_start.shape[0]

        log_cumprod_at_pos = extract(self.log_cumprod_at_pos, t, log_x_start.shape)
        log_cumprod_bt_pos = extract(self.log_cumprod_bt_pos, t, log_x_start.shape)
        log_cumprod_ct_pos = extract(self.log_cumprod_ct_pos, t, log_x_start.shape)
        log_1_min_cumprod_ct_pos = extract(self.log_1_min_cumprod_ct_pos, t, log_x_start.shape)

        log_cumprod_at_type = extract(self.log_cumprod_at_type, t, (1, 1, 1)) # at~ (batch_size, 1, 1)
        log_cumprod_bt_type = extract(self.log_cumprod_bt_type, t, log_x_start.shape)
        log_cumprod_ct_type = extract(self.log_cumprod_ct_type, t, (1, 1, 1)) # ct~ (batch_size, 1, 1)
        log_1_min_cumprod_ct_type = extract(self.log_1_min_cumprod_ct_type, t, (1, 1, 1)) # 1-ct~

        mask = (t < self.T_tilde).unsqueeze(-1).unsqueeze(-1)

        num_special = self.N_special  # 0~4 are special tokens: ["START", "END", "UNK", "PAD", "|"]
        num_type = self.N_type  # type tokens
        num_pos = self.N_pos  # position tokens
        TINY = 1e-30

        # create tensor with batch size of `bz`
        def _zero_tensor(*shape):
            return torch.zeros(bz, *shape, device=device)

        def _eye_tensor(n):
            return torch.eye(n, device=device).expand(bz, -1, -1)  # [bz, n, n]

        ### Transition probability
        # something -> special_tokens [bz, num_special, num_classes]
        special2special = _eye_tensor(num_special)
        others2special  = _zero_tensor(num_special, self.num_classes-num_special)
        trans2special   = torch.cat([special2special, others2special], dim=-1)

        # something -> type [bz, num_type, num_classes]
        special2type = _zero_tensor(num_type, num_special)
        type2type    = log_add_exp(_eye_tensor(num_type).clamp_min(TINY).log() + log_cumprod_at_type, log_cumprod_bt_type).exp()
        pos2type     = _zero_tensor(num_type, num_pos)
        mask2type    = _zero_tensor(num_type, 1)
        trans2type   = torch.cat([special2type, type2type, pos2type, mask2type], dim=-1)

        # something -> position [bz, num_pos, num_classes]
        special2pos = _zero_tensor(num_pos, num_special)
        type2pos    = _zero_tensor(num_pos, num_type)
        pos2pos     = log_add_exp(_eye_tensor(num_pos).clamp_min(TINY).log() + log_cumprod_at_pos, log_cumprod_bt_pos).exp()
        mask2pos    = _zero_tensor(num_pos, 1)
        trans2pos   = torch.cat([special2pos, type2pos, pos2pos, mask2pos], dim=-1)

        # something -> MASK [bz, 1, num_classes]
        special2mask = _zero_tensor(1, num_special)
        type2mask    = log_add_exp(_zero_tensor(1, num_type).clamp_min(TINY).log() + log_1_min_cumprod_ct_type, log_cumprod_ct_type).exp()
        pos2mask     = log_add_exp(_zero_tensor(1, num_pos).clamp_min(TINY).log() + log_1_min_cumprod_ct_pos, log_cumprod_ct_pos).exp()
        mask2mask    = torch.ones(bz, 1, 1, device=device)
        trans2mask   = torch.cat([special2mask, type2mask, pos2mask, mask2mask], dim=-1)

        ### Build matrix Q
        # when t > 0.8T
        matrix_1 = torch.cat(
            [
                trans2special,
                trans2type,
                trans2pos,
                trans2mask,
            ],
            dim=-2,
        )

        # when t <= 0.8T
        pos2pos_gaussian = self.q_mats[t].to(device)
        trans2pos_gaussian = torch.cat(
            [special2pos, type2pos, pos2pos_gaussian, mask2pos], dim=-1
        )
        matrix_2 = torch.cat(
            [
                trans2special,
                trans2type,
                trans2pos_gaussian,
                trans2mask,
            ],
            dim=-2,
        )

        matrix = torch.where(mask, matrix_2, matrix_1)
        log_probs = matrix.matmul(log_x_start.exp()).clamp(min=1e-30).log()

        return log_probs

    def predict_start(self, log_x_t, t, **kwargs):
        x_t = log_onehot_to_index(log_x_t)
        out = self.transformer(x_t, timestep=t)['logits']
        out = out[:, :, :-1]  # ignore MASK
        out = rearrange(out, "b s c -> b c s")

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes - 1
        assert out.size()[2:] == x_t.size()[1:]

        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
            self.zero_vector = (
                torch.zeros(batch_size, 1, self.max_token_length).type_as(log_x_t) - 70
            )
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes - 1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(
            -1, -1, self.max_token_length
        )
        log_zero_vector_aux = torch.log(log_one_vector + 1.0e-30).expand(-1, -1, -1)

        log_qt = self.q_pred(log_x_t, t)  # q(xt|x0)
        log_qt = log_qt[:, :-1, :]
        log_cumprod_ct_pos = extract(
            self.log_cumprod_ct_pos, t, log_x_start.shape
        )  # ct~

        if self.schedule_type == "relu":
            log_cumprod_ct_type = extract(
                self.log_cumprod_ct_type, t, log_x_start.shape
            )  # ct~
            ct_cumprod_vector = torch.cat(
                [
                    log_zero_vector_aux.expand(-1, self.N_special, -1),
                    log_cumprod_ct_type.expand(-1, self.N_type, -1),
                    log_cumprod_ct_pos.expand(-1, self.num_classes - 1 - self.N_type - self.N_special, -1),
                ],
                dim=1,
            )
        else:
            ct_cumprod_vector = torch.cat(
                [
                    log_zero_vector_aux.expand(-1, self.N_special, -1),
                    log_cumprod_ct_pos.expand(-1, self.num_classes - 1 - self.N_special, -1),
                ],
                dim=1,
            )

        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat(
            (log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1
        )
        log_ct_pos = extract(self.log_ct_pos, t, log_x_start.shape)  # ct

        if self.schedule_type == "relu":
            log_ct_type = extract(self.log_ct_type, t, log_x_start.shape)  # ct~
            ct_vector = torch.cat(
                [
                    log_zero_vector_aux.expand(-1, self.N_special, -1),
                    log_ct_type.expand(-1, self.N_type, -1),
                    log_ct_pos.expand(-1, self.num_classes - 1 - self.N_type - self.N_special, -1),
                ],
                dim=1,
            )
        else:
            ct_vector = torch.cat(
                [
                    log_zero_vector_aux.expand(-1, self.N_special, -1),
                    log_ct_pos.expand(-1, self.num_classes - 1 - self.N_special, -1),
                ],
                dim=1,
            )

        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector
        q = log_x_start[:, :-1, :] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)

        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t - 1) + log_qt_one_timestep

        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def q_sample(self, log_x_start, t):  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def q_sample_onestep(
        self, log_x_start, t
    ):  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred_one_timestep(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    @property
    def device(self) -> torch.device:
        if hasattr(self, "transformer"):
            return next(self.transformer.parameters()).device
        raise NotImplementedError

    def corrupt_x(
        self, x: Tensor, t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        log_x_start = index_to_log_onehot(x, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)
        return xt, log_xt

    def forward(
        self, x: Tensor, is_train: bool = True
    ):
        b, device = x.size(0), x.device

        assert self.loss_type == "vb_stochastic"
        x_start = x
        t, pt = self.sample_time(b, device, "importance")

        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)  # gt x_t #use matrix
        xt = log_onehot_to_index(log_xt)
        # breakpoint()

        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(
            log_xt, t=t
        )  # P_theta(x0|xt)
        log_model_prob = self.q_posterior(
            log_x_start=log_x0_recon, log_x_t=log_xt, t=t
        )  # go through q(xt_1|xt,x0) #pred x_t

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            # the ratio of same codebook
            same_rate = (
                x0_recon[index] == x0_real[index]
            ).sum().cpu() / x0_real.size()[1]
            self.diffusion_acc_list[this_t] = (
                same_rate.item() * 0.1 + self.diffusion_acc_list[this_t] * 0.9
            )
            same_rate = (
                xt_1_recon[index] == xt_recon[index]
            ).sum().cpu() / xt_recon.size()[1]
            self.diffusion_keep_list[this_t] = (
                same_rate.item() * 0.1 + self.diffusion_keep_list[this_t] * 0.9
            )

        # compute log_true_prob now
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)

        # compute loss
        kl = self.multinomial_kl(log_true_prob, log_model_prob)

        mask_region = (xt == self.num_classes - 1).float()
        mask_weight = (
            mask_region * self.mask_weight[0]
            + (1.0 - mask_region) * self.mask_weight[1]
        )

        kl = kl * mask_weight
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()

        kl_loss = mask * decoder_nll + (1.0 - mask) * kl

        Lt2 = kl_loss.pow(2)
        _device = self.device
        Lt2_prev = self.Lt_history.to(_device).gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.to(_device).scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.to(_device).scatter_add_(
            dim=0, index=t, src=torch.ones_like(Lt2)
        )

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt
        vb_loss = {}
        vb_loss["loss1"] = loss1

        if self.auxiliary_loss_weight != 0 and is_train == True:
            kl_aux = self.multinomial_kl(
                log_x_start[:, :-1, :], log_x0_recon[:, :-1, :]
            )
            kl_aux = kl_aux * mask_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1.0 - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1 - t / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss["loss2"] = loss2
            vb_loss["loss"] = vb_loss["loss1"] + vb_loss["loss2"]
        else:
            vb_loss["loss"] = vb_loss["loss1"]

        if self.rescale_weight:
            _time_scale = (self.num_timesteps - t) / self.num_timesteps
            vb_loss["loss"] = (
                0.2 * vb_loss["loss"] + 0.8 * 2 * vb_loss["loss"] * _time_scale
            )

        outputs = {"probs": log_model_prob.exp()}
        return outputs, vb_loss

    @torch.no_grad()
    def _sample_single_step(self, log_z, t, skip_step, **kwargs):
        log_x_recon = self.predict_start(log_z, t)
        if t[0].item() > skip_step:
            model_log_prob = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_z, t=t - skip_step
            )
        else:
            model_log_prob = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_z, t=t
            )

        log_z = self.log_sample_categorical(model_log_prob)
        return log_z
    
    @torch.no_grad()
    def _sample_single_step_with_corrector(self,*args, **kwargs):
        raise NotImplementedError('MildCorruptionDiffusion + Corrector is not supported yet')

    def sample(
        self,
        init_seq: Tensor,
        sampling_cfg: DictConfig,
        cond: Optional[Dict] = None,
        get_intermediate_results=False,
        **kwargs,
    ):
        if cond is not None:
            raise NotImplementedError('Conditional sampling is not supported yet')

        batch_size = init_seq.size(0)

        init_seq = init_seq.to(self.device)
        mask_logits = index_to_log_onehot(init_seq, self.num_classes).exp()
        log_z = torch.log(mask_logits)

        if get_intermediate_results:  # save intermediate state
            intermediate_results = []

        num_timesteps_eval = sampling_cfg.get("num_timesteps", self.num_timesteps)
        assert num_timesteps_eval == self.num_timesteps, 'num_timesteps_eval should be equal to self.num_timesteps'
        diffusion_list = []
        for i in range(num_timesteps_eval - 1, -1, -1):
            diffusion_list.append(int(i * self.num_timesteps / num_timesteps_eval))
        if diffusion_list[-1] != 0:
            diffusion_list.append(0)

        #### Denoising ####
        prev_diffusion_index = self.num_timesteps
        for diffusion_index in diffusion_list:

            delta_t = prev_diffusion_index - diffusion_index
            t = torch.full(
                (batch_size,), diffusion_index, device=self.device, dtype=torch.long
            )
            log_z = self._sample_single_step(log_z, t, skip_step=delta_t-1)

            if get_intermediate_results:
                intermediate_results.append(log_onehot_to_index(log_z))
            prev_diffusion_index = diffusion_index

        content_token = log_onehot_to_index(log_z)

        if get_intermediate_results:
            return intermediate_results
        return content_token


    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == "none":
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear,)
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                    if pn.endswith("bias"):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(
                        m, whitelist_weight_modules
                    ):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(
                        m, blacklist_weight_modules
                    ):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ["content_emb"]
            pos_emb_name = [
                "pos_emb",
                "width_emb",
                "height_emb",
                "pad_emb",
                "token_type_emb",
            ]
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(
                                getattr(getattr(self, mn), pn), torch.nn.Parameter
                            ):
                                no_decay.add("{}.{}".format(mn, pn))

            # validate that we considered every parameter
            param_dict = {
                pn: p for pn, p in self.transformer.named_parameters()
            }  # if p.requires_grad} ##todo
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert (
                len(inter_params) == 0
            ), "parameters %s made it into both decay/no_decay sets!" % (
                str(inter_params),
            )
            assert (
                len(param_dict.keys() - union_params) == 0
            ), "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params),
            )

            # create the pytorch optimizer object
            optim_groups = [
                {
                    "params": [param_dict[pn] for pn in sorted(list(decay))],
                    "weight_decay": 0.01,
                },
                {
                    "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                    "weight_decay": 0.0,
                },
            ]
            return optim_groups

    # this function is not used, apparently
    def noise_process_new(self, x_start, sep):
        xt_all = []
        bz = x_start.shape[0]
        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        for t in range(self.num_timesteps):
            log_xt = self.q_sample_onestep(
                log_x_start=log_x_start, t=torch.tensor([t]).cuda().expand(bz)
            )
            xt = log_onehot_to_index(log_xt)

            if t in np.linspace(0, self.num_timesteps - 1, num=sep).astype(int):
                xt_all.append(xt.cpu().detach().numpy())
            log_x_start = index_to_log_onehot(xt, self.num_classes)
        return xt_all
    