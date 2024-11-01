"""
This file is derived from the following repository:
https://github.com/CyberAgentAILab/layout-dm/tree/9cf122d3be74b1ed8a8d51e3caabcde8164d238e

Original file: src/trainer/trainer/models/transformer_utils.py
Author: naoto0804
License: Apache-2.0 License

Modifications have been made to the original file to fit the requirements of this project.
"""

# Implement TransformerEncoder that can consider timesteps as optional args for Diffusion.

from __future__ import annotations

import copy
import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder as _BertEncoder


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _gelu2(x):
    return x * F.sigmoid(1.702 * x)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "gelu2":
        return _gelu2
    else:
        raise RuntimeError(
            "activation should be relu/gelu/gelu2, not {}".format(activation)
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps: int, dim: int, rescale_steps: int = 4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x: Tensor):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class _AdaNorm(nn.Module):
    def __init__(
        self,
        n_embd: int,
        max_timestep: int,
        emb_type: str = "adalayernorm_abs",
    ):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(max_timestep, n_embd)
        elif "mlp" in emb_type:
            self.emb = nn.Sequential(
                Rearrange("b -> b 1"),
                nn.Linear(1, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_embd),
            )
        else:
            self.emb = nn.Embedding(max_timestep, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)


class AdaLayerNorm(_AdaNorm):
    def __init__(
        self,
        n_embd: int,
        max_timestep: int,
        emb_type: str = "adalayernorm_abs",
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x: Tensor, timestep: int):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class AdaInsNorm(_AdaNorm):
    def __init__(
        self,
        n_embd: int,
        max_timestep: int,
        emb_type: str = "adalayernorm_abs",
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = (
            self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2)
            * (1 + scale)
            + shift
        )
        return x


class LayoutLMv2SelfAttention(nn.Module):
    # LayoutLMv2's Multi-Head Self-Attention Layer
    # This layer is implemented for "spatial_attention_bias".
    # Source https://github.com/microsoft/unilm/blob/b60c741f746877293bb85eed6806736fc8fa0ffd/layoutlmft/layoutlmft/models/layoutlmv2/modeling_layoutlmv2.py#L87C11-L87C11
    # NOTE: Some features implemented in torch.nn.MultiheadAttention are not supported (e.g., key_padding_mask)
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        spatial_attention_bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        assert batch_first, "For now, `batch_first=False` is not supported!"
        self.batch_first = batch_first
        self.fast_qkv = False

        self.has_spatial_attention_bias = spatial_attention_bias
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
                q = q + self.q_bias.view(*_sz)
                v = v + self.v_bias.view(*_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[Tensor] = False,
        key_padding_mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Union[Tensor, None]]:
        if key_padding_mask is not None:
            raise NotImplementedError(
                "For now, key_padding_mask is not supported"
            )

        if not self.batch_first:
            raise NotImplementedError()  # TODO

        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.head_dim)
        # [BSZ, NAT, L, L]
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )

        if self.has_spatial_attention_bias:
            attention_scores += attention_bias
        if attention_mask is not None:
            attention_scores = attention_scores.float().masked_fill_(
                attention_mask.to(torch.bool), float("-inf")
            )
        attention_probs = F.softmax(
            attention_scores, dim=-1, dtype=torch.float32
        ).type_as(value_layer)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.out_linear(context_layer)

        outputs = (
            (context_layer, attention_probs)
            if output_attentions
            else (context_layer,)
        )
        return outputs


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        d_model=1024,
        nhead=16,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        # extension for diffusion
        diffusion_step: int = 100,
        timestep_type: str = None,
        # spatial attention bias used in LayoutLMv2
        spatial_attention_bias: bool = False,
    ) -> None:
        super().__init__()

        assert norm_first  # minGPT-based implementations are designed for prenorm only
        assert timestep_type in [
            None,
            "adalayernorm",
            "adainnorm",
            "adalayernorm_abs",
            "adainnorm_abs",
            "adalayernorm_mlp",
            "adainnorm_mlp",
        ]
        layer_norm_eps = 1e-5  # fixed

        self.norm_first = norm_first
        self.diffusion_step = diffusion_step
        self.timestep_type = timestep_type

        factory_kwargs = {"device": device, "dtype": dtype}

        self.has_spatial_attention_bias = spatial_attention_bias
        if self.has_spatial_attention_bias:
            self.self_attn = LayoutLMv2SelfAttention(
                d_model,
                nhead,
                dropout=dropout,
                batch_first=batch_first,
                spatial_attention_bias=True,
                **factory_kwargs
            )
        else:
            self.self_attn = torch.nn.MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                batch_first=batch_first,
                **factory_kwargs
            )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        if timestep_type is not None:
            if "adalayernorm" in timestep_type:
                self.norm1 = AdaLayerNorm(
                    d_model, diffusion_step, timestep_type
                )
            elif "adainnorm" in timestep_type:
                self.norm1 = AdaInsNorm(d_model, diffusion_step, timestep_type)
        else:
            self.norm1 = nn.LayerNorm(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Tensor = None,
        attention_bias: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            if self.timestep_type is not None:
                x = self.norm1(x, timestep)
            else:
                x = self.norm1(x)
            x = x + self._sa_block(
                x, src_mask, src_key_padding_mask, attention_bias
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = x + self._sa_block(
                x, src_mask, src_key_padding_mask, attention_bias
            )
            if self.timestep_type is not None:
                x = self.norm1(x, timestep)
            else:
                x = self.norm1(x)
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        attention_bias: Optional[Tensor] = None,
    ) -> Tensor:
        if self.has_spatial_attention_bias:
            x = self.self_attn(
                x,
                attention_mask=attn_mask,
                output_attentions=False,
                key_padding_mask=key_padding_mask,
                attention_bias=attention_bias,
            )[0]
        else:
            x = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    """
    Close to torch.nn.TransformerEncoder, but with timestep support for diffusion
    """

    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Tensor = None,
        attention_bias: Optional[Tensor] = None,
    ) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                timestep=timestep,
                attention_bias=attention_bias,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

class MlpBlock(Block):
    """Transformer block WITHOUT self-attention"""

    def __init__(
        self,
        d_model=1024,
        nhead=16,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        # extension for diffusion
        diffusion_step: int = 100,
        timestep_type: str = None,
        # spatial attention bias used in LayoutLMv2
        spatial_attention_bias: bool = False,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first,
            norm_first,
            device,
            dtype,
            diffusion_step,
            timestep_type,
            spatial_attention_bias,
        )
        self.self_attn = None ## <<=== NOTE: no self-attention
        import warnings
        warnings.warn("This is MlpBlock. No self-attention is used.")


    def forward(
        self,
        src: Tensor,
        timestep: Tensor = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            if self.timestep_type is not None:
                x = self.norm1(x, timestep)
            else:
                x = self.norm1(x)
            x = x + self._ff_block(self.norm2(x))
        else:
            if self.timestep_type is not None:
                x = self.norm1(x, timestep)
            else:
                x = self.norm1(x)
            x = self.norm2(x + self._ff_block(x))

        return x


class BertEncoder(_BertEncoder):
    def __init__(self, config_name: str = "bert-base-uncased", **kwargs):
        """BertEncoder from transformers library.
        config can be modified by passing kwargs. e.g., hidden_dropout_prob=0.1

        Args:
            config_name (str, optional): Defaults to "bert-base-uncased".
        """
        config = AutoConfig.from_pretrained(config_name)
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        super().__init__(config)

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        return out.last_hidden_state