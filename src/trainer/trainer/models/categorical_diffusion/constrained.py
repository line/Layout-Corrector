"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php

-------------------------------------------------------------------------------

This file is derived from the following repository:
https://github.com/CyberAgentAILab/layout-dm/tree/9cf122d3be74b1ed8a8d51e3caabcde8164d238e

Original file: src/trainer/trainer/models/categorical_diffusion/constrained.py
Author: naoto0804
License: Apache-2.0 License

Modifications have been made to the original file to fit the requirements of this project.
"""

import logging
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment
from torch import Tensor  # for type hints
from trainer.helpers.layout_tokenizer import Converter, LayoutSequenceTokenizer
from trainer.helpers.sampling import RandomSamplingConfig, sample


from .base import BaseMaskAndReplaceDiffusion
from .util import (
    extract,
    index_to_log_onehot,
    log_1_min_a,
    log_add_exp,
    log_categorical,
    log_onehot_to_index,
    mean_except_batch,
    alpha_schedule,
)

logger = logging.getLogger(__name__)


class ConstrainedMaskAndReplaceDiffusion(BaseMaskAndReplaceDiffusion):
    def __init__(
        self,
        backbone_cfg: DictConfig,
        num_classes: int,
        max_token_length: int,
        num_timesteps: int = 100,
        tokenizer: LayoutSequenceTokenizer = None,
        hungarian_match: bool = False,
        use_padding_as_vocab: bool = True,
        training_condition: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            backbone_cfg=backbone_cfg,
            num_classes=num_classes,
            max_token_length=max_token_length,
            num_timesteps=num_timesteps,
            tokenizer=tokenizer,
            use_padding_as_vocab=use_padding_as_vocab,
            **kwargs,
        )

        self.use_padding_as_vocab = use_padding_as_vocab
        # if not use_padding_as_vocab:
            # raise NotImplementedError('use_padding_as_vocab = False is not supported yet!')

        # tokenizer is required to separate corruption matrix
        self.tokenizer = tokenizer
        self.converter = Converter(self.tokenizer)
        self.hungarian_match = hungarian_match

        if training_condition is not None:
            assert training_condition in ['c', 'cwh']
            self.training_condition = list(training_condition)
        else:
            self.training_condition = []
        # special scheduling for conditional training
        self.conditional_alpha_schedule_partial_func = partial(
            alpha_schedule,
            num_timesteps=num_timesteps,
            att_1=0.99999,
            att_T=0.99999, # alpha is always 1
            ctt_1=0.000009,
            ctt_T=0.000009,
        )

        # set vocabulari size for each corruption matrix (w/ pad)
        self.mat_size = {"c": self.tokenizer.N_category + 2}
        num_bin = self.tokenizer.N_bbox_per_var
        for key in ["x", "y", "w", "h"]:
            self.mat_size.update({key: num_bin + 2})

        for key in self.tokenizer.var_names:
            if self.alpha_init_type == "alpha1":
                N = self.mat_size[key] - 1
                if not self.use_padding_as_vocab:
                    N -= 1 # exclude [PAD]
                if key in self.training_condition:
                    at, bt, ct, att, btt, ctt = self.conditional_alpha_schedule_partial_func(N=N)
                else:
                    at, bt, ct, att, btt, ctt = self.alpha_schedule_partial_func(N=N)
            else:
                print("alpha_init_type is Wrong !! ")
                raise NotImplementedError

            log_at, log_bt, log_ct = torch.log(at), torch.log(bt), torch.log(ct)
            log_cumprod_at, log_cumprod_bt, log_cumprod_ct = (
                torch.log(att),
                torch.log(btt),
                torch.log(ctt),
            )
            log_at = log_at.clamp(-70., 0.) # avoid -inf
            log_bt = log_bt.clamp(-70., 0.)
            log_ct = log_ct.clamp(-70., 0.)
            log_cumprod_at = log_cumprod_at.clamp(-70., 0.)
            log_cumprod_bt = log_cumprod_bt.clamp(-70., 0.)
            log_cumprod_ct = log_cumprod_ct.clamp(-70., 0.)

            log_1_min_ct = log_1_min_a(log_ct)
            log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

            assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.0e-5
            assert (
                log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item()
                < 1.0e-5
            )

            # Convert to float32 and register buffers.
            self.register_buffer(f"{key}_log_at", log_at.float())
            self.register_buffer(f"{key}_log_bt", log_bt.float())
            self.register_buffer(f"{key}_log_ct", log_ct.float())
            self.register_buffer(f"{key}_log_cumprod_at", log_cumprod_at.float())
            self.register_buffer(f"{key}_log_cumprod_bt", log_cumprod_bt.float())
            self.register_buffer(f"{key}_log_cumprod_ct", log_cumprod_ct.float())
            self.register_buffer(f"{key}_log_1_min_ct", log_1_min_ct.float())
            self.register_buffer(
                f"{key}_log_1_min_cumprod_ct", log_1_min_cumprod_ct.float()
            )

    def hungarian_multinomial_kl(self, log_prob1: Tensor, log_prob2: Tensor) -> Tensor:
        n_attr_per_elem = len(self.tokenizer.var_names)
        num_element = self.tokenizer.max_seq_length
        bs = log_prob1.size(0)
        log_prob1 = log_prob1.reshape(bs, -1, num_element, n_attr_per_elem)
        log_prob2 = log_prob2.reshape(bs, -1, num_element, n_attr_per_elem)

        # calc cost matrix
        with torch.no_grad():
            C = self.multinomial_kl(log_prob1.unsqueeze(2), log_prob2.unsqueeze(3)) # [bs, num_element, num_element, n_attr_per_elem]
            C = C.sum(dim=-1).cpu() # [bs, num_element, num_element] C[i, j] = kl(log_prob1[j], log_prob2[i])

        sorted_log_prob1_list = []

        for b in range(bs):
            row_ind, col_ind = linear_sum_assignment(C[b])
            # sort log_prob1 so that kl(_log_prob1, log_prob2) is minimized
            _log_prob1 = log_prob1[b] # [num_class, num_ele, n_attr_per_elem]
            _log_prob1 = _log_prob1.permute(1, 2, 0)[col_ind] # [num_ele, n_attr_per_elem, num_class]
            _log_prob1 = _log_prob1.permute(2, 0, 1)# [num_class, num_ele, n_attr_per_elem]
            sorted_log_prob1_list.append(_log_prob1)

            # For checking
            cost1 = C[b][row_ind, col_ind]
            cost2 = (_log_prob1.exp() * (_log_prob1 - log_prob2[b])).sum(dim=(0, 2))
            assert torch.allclose(cost1.cpu(), cost2.cpu())

        sorted_log_prob1 = torch.stack(sorted_log_prob1_list, dim=0)
        loss = self.multinomial_kl(sorted_log_prob1, log_prob2)
        loss = loss.reshape(bs, -1)
        return loss

    def q_pred_one_timestep(
        self, log_x_t: Tensor, t: Tensor, key: str
    ) -> Tensor:  # q(xt|xt_1)
        log_at = extract(getattr(self, f"{key}_log_at"), t, log_x_t.shape)  # at
        log_bt = extract(getattr(self, f"{key}_log_bt"), t, log_x_t.shape)  # bt
        log_ct = extract(getattr(self, f"{key}_log_ct"), t, log_x_t.shape)  # ct
        log_1_min_ct = extract(
            getattr(self, f"{key}_log_1_min_ct"), t, log_x_t.shape
        )  # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct),
            ],
            dim=1,
        )

        return log_probs

    def q_pred(self, log_x_start: Tensor, t: Tensor, key: str) -> Tensor:  # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        s = log_x_start.shape
        log_cumprod_at = extract(getattr(self, f"{key}_log_cumprod_at"), t, s)  # at~
        log_cumprod_bt = extract(getattr(self, f"{key}_log_cumprod_bt"), t, s)  # bt~
        log_cumprod_ct = extract(getattr(self, f"{key}_log_cumprod_ct"), t, s)  # ct~
        log_1_min_cumprod_ct = extract(
            getattr(self, f"{key}_log_1_min_cumprod_ct"), t, s
        )  # 1-ct~

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(
                    log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct
                ),
            ],
            dim=1,
        )

        return log_probs

    def q_posterior(
        self, log_x_start: Tensor, log_x_t: Tensor, t: Tensor
    ) -> Tensor:  # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        if self.converter.get_device() == torch.device("cpu"):
            self.converter.to(self.device)

        log_x_start_full, log_x_t_full = log_x_start, log_x_t  # for API compatibiliry

        batch_size = log_x_start_full.size()[0]
        step = self.tokenizer.N_var_per_element

        index_x_t_full = log_onehot_to_index(log_x_t_full)
        log_one_vector_full = torch.zeros(batch_size, 1, 1).type_as(log_x_t_full)
        seq_len = self.max_token_length // step
        log_zero_vector_full = torch.log(log_one_vector_full + 1.0e-30).expand(
            -1, -1, seq_len
        )
        mask_reshaped = rearrange(
            index_x_t_full == self.tokenizer.name_to_id("mask"),
            "b (s x) -> b s x",
            s=seq_len,
            x=step,
        )

        log_EV_xtmin_given_xt_given_xstart_full = []
        for i, key in enumerate(self.tokenizer.var_names):
            mask = mask_reshaped[..., i].unsqueeze(1)
            log_x_start = self.converter.f_to_p_log(log_x_start_full[..., i::step], key)
            log_x_t = self.converter.f_to_p_log(log_x_t_full[..., i::step], key)
            log_qt = self.q_pred(log_x_t, t, key)  # q(xt|x0)

            log_qt = log_qt[:, :-1, :]
            log_cumprod_ct = extract(
                getattr(self, f"{key}_log_cumprod_ct"), t, log_x_t.shape
            )  # ct~
            ct_cumprod_vector = log_cumprod_ct.expand(-1, self.mat_size[key] - 1, -1)
            log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

            log_qt_one_timestep = self.q_pred_one_timestep(
                log_x_t, t, key
            )  # q(xt|xt_1)

            log_qt_one_timestep = torch.cat(
                (log_qt_one_timestep[:, :-1, :], log_zero_vector_full), dim=1
            )
            log_ct = extract(getattr(self, f"{key}_log_ct"), t, log_x_t.shape)  # ct
            ct_vector = log_ct.expand(-1, self.mat_size[key] - 1, -1)
            ct_vector = torch.cat((ct_vector, log_one_vector_full), dim=1)
            log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

            # below just does log(ab/c)=loga+logb-logc in eq.5 of VQDiffusion
            q = log_x_start[:, :-1, :] - log_qt
            q = torch.cat((q, log_zero_vector_full), dim=1)
            q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
            q = q - q_log_sum_exp
            log_EV_xtmin_given_xt_given_xstart = (
                self.q_pred(q, t - 1, key) + log_qt_one_timestep + q_log_sum_exp
            )
            log_EV_xtmin_given_xt_given_xstart = torch.clamp(
                log_EV_xtmin_given_xt_given_xstart, -70, 0
            )
            log_EV_xtmin_given_xt_given_xstart_full.append(
                self.converter.p_to_f_log(log_EV_xtmin_given_xt_given_xstart, key)
            )

        log_EV_xtmin_given_xt_given_xstart_full = torch.stack(
            log_EV_xtmin_given_xt_given_xstart_full, dim=-1
        ).view(batch_size, self.num_classes, -1)

        return log_EV_xtmin_given_xt_given_xstart_full

    def log_sample_categorical(
        self, logits: Tensor, key: str
    ) -> Tensor:  # use gumbel to sample onehot vector from log probability
        if self.train_sampling == "gumbel":
            uniform = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sampled = (gumbel_noise + logits).argmax(dim=1)
        elif self.train_sampling == "random":
            sampling_cfg = OmegaConf.structured(RandomSamplingConfig)
            sampled = sample(logits, sampling_cfg)
            sampled = rearrange(sampled, "b 1 s -> b s")

        log_sample = index_to_log_onehot(sampled, self.mat_size[key])
        return log_sample

    def q_sample(
        self, log_x_start: Tensor, t: Tensor, key: str
    ) -> Tensor:  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t, key)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0, key)

        return log_sample

    def forward(
        self, x: Tensor, is_train: bool = True
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        b, s = x.size()[:2]
        device = x.device
        step = self.tokenizer.N_var_per_element

        pad_mask = (x == self.tokenizer.N_total - 2) # [bs, n_token]
        pad_mask_for_log = pad_mask.unsqueeze(1).repeat(1, self.tokenizer.N_total, 1) # [bs, num_class, n_token]

        assert self.loss_type == "vb_stochastic"
        x_start_full = x
        t, pt = self.sample_time(b, device, "importance")

        log_x_start_full = index_to_log_onehot(x_start_full, self.num_classes)

        if self.converter.get_device() == torch.device("cpu"):
            self.converter.to(self.device)
        x_start_reshaped = self.converter.f_to_p_id_all(
            rearrange(x_start_full, "b (s x) -> b s x", s=s // step, x=step)
        )
        log_x_t_full = []
        xt_full = []
        for i, key in enumerate(self.tokenizer.var_names):
            log_x_start = index_to_log_onehot(
                x_start_reshaped[..., i], self.mat_size[key]
            )
            log_x_t = self.q_sample(log_x_start=log_x_start, t=t, key=key)
            log_x_t_full.append(self.converter.p_to_f_log(log_x_t, key))
            xt_full.append(log_onehot_to_index(log_x_t))

        xt_full = self.converter.p_to_f_id_all(torch.stack(xt_full, dim=-1)).view(b, -1)
        log_x_t_full = torch.stack(log_x_t_full, dim=-1).view(b, self.num_classes, -1)

        if not self.use_padding_as_vocab:
            log_x_t_full[pad_mask_for_log] = log_x_start_full[pad_mask_for_log]

        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_x_t_full, t=t, padding_mask=None if self.use_padding_as_vocab else pad_mask)  # P_theta(x0|xt)
        log_model_prob = self.q_posterior(
            log_x_start=log_x0_recon, log_x_t=log_x_t_full, t=t
        )  # go through q(xt_1|xt,x0)

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start_full
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_x_t_full)
        for index in range(t.size()[0]):
            this_t = t[index].item()
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
        log_true_prob = self.q_posterior( # [bs, num_cls, num_token]
            log_x_start=log_x_start_full, log_x_t=log_x_t_full, t=t
        )

        if self.hungarian_match:
            kl = self.hungarian_multinomial_kl(log_true_prob, log_model_prob)
        else:
            kl = self.multinomial_kl(log_true_prob, log_model_prob) ## <<======= Loss (GT x_{t-1} vs pred x_{t-1}) out: [bs, 125]
        mask_region = (xt_full == self.num_classes - 1).float()
        mask_weight = (
            mask_region * self.mask_weight[0]
            + (1.0 - mask_region) * self.mask_weight[1]
        )
        kl = kl * mask_weight
        if not self.use_padding_as_vocab:
            kl[pad_mask] = 0.
        kl = mean_except_batch(kl) # out: [bs]

        decoder_nll = -log_categorical(log_x_start_full, log_model_prob) ## <<======= Loss (GT one-hot vs pred x_{t-1}) out: [bs, 125]
        if not self.use_padding_as_vocab:
            decoder_nll[pad_mask] = 0.
        decoder_nll = mean_except_batch(decoder_nll) # out: [bs]

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1.0 - mask) * kl # decoder_nll if t == 0 else kl

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        loss1 = kl_loss / pt # [bs]

        losses = {"kl_loss": loss1.mean()}
        if self.auxiliary_loss_weight != 0 and is_train == True:
            if self.hungarian_match:
                kl_aux = self.hungarian_multinomial_kl(
                    log_x_start_full[:, :-1, :], log_x0_recon[:, :-1, :]
                )
            else:
                kl_aux = self.multinomial_kl( ## <<======= Loss (GT x0 vs pred x0) (except MASK token) out: [bs, 125]
                    log_x_start_full[:, :-1, :], log_x0_recon[:, :-1, :]
                )
            kl_aux = kl_aux * mask_weight
            if not self.use_padding_as_vocab:
                kl_aux[pad_mask] = 0.
            kl_aux = mean_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1.0 - mask) * kl_aux # decoder_nll if t == 0 else kl
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1 - t / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt

            losses["aux_loss"] = loss2.mean()

        outputs = {"probs": log_model_prob.exp()}
        return outputs, losses


    def corrupt_x(
        self, x: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        b, s = x.size()[:2]
        step = self.tokenizer.N_var_per_element
        x_start_full = x
        if self.converter.get_device() == torch.device("cpu"):
            self.converter.to(self.device)
        x_start_reshaped = self.converter.f_to_p_id_all(
            rearrange(x_start_full, "b (s x) -> b s x", s=s // step, x=step)
        )
        log_x_t_full = []
        xt_full = []
        for i, key in enumerate(self.tokenizer.var_names):
            log_x_start = index_to_log_onehot(
                x_start_reshaped[..., i], self.mat_size[key]
            )
            log_x_t = self.q_sample(log_x_start=log_x_start, t=t, key=key)
            log_x_t_full.append(self.converter.p_to_f_log(log_x_t, key))
            xt_full.append(log_onehot_to_index(log_x_t))

        xt_full = self.converter.p_to_f_id_all(torch.stack(xt_full, dim=-1)).view(b, -1)
        log_x_t_full = torch.stack(log_x_t_full, dim=-1).view(b, self.num_classes, -1)
        return xt_full, log_x_t_full
    
    @property
    def var2index_range(self) -> Dict[str, List[int]]:
        """Return index range per var"""
        n_cateogry = self.tokenizer.N_category
        n_bbox_per_var = self.tokenizer.N_bbox_per_var
        var2range = {
            "c": [0, n_cateogry],
            "x": [n_cateogry, n_cateogry + n_bbox_per_var],
            "y": [n_cateogry + n_bbox_per_var, n_cateogry + n_bbox_per_var * 2],
            "w": [n_cateogry + n_bbox_per_var * 2, n_cateogry + n_bbox_per_var * 3],
            "h": [n_cateogry + n_bbox_per_var * 3, n_cateogry + n_bbox_per_var * 4],
        }
        return var2range

    def get_var_from_index(self, index: int) -> str:
        """Return var from index"""
        res2var = {
            0: "c",
            1: "x",
            2: "y",
            3: "w",
            4: "h"
        }
        res = index % self.tokenizer.N_var_per_element
        return res2var[res]
