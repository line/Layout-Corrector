"""
This file is derived from the following repository:
https://github.com/CyberAgentAILab/layout-dm/tree/9cf122d3be74b1ed8a8d51e3caabcde8164d238e

Original file: src/trainer/trainer/models/common/nn_lib.py
Author: naoto0804
License: Apache-2.0 License

Modifications have been made to the original file to fit the requirements of this project.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from trainer.models.transformer_utils import TransformerEncoder

from .layout import LayoutDecoder, LayoutEncoder
from .util import generate_causal_mask

logger = logging.getLogger(__name__)


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        # https://github.com/pytorch/pytorch/issues/16885#issuecomment-551779897
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class SeqLengthDistribution(nn.Module):
    def __init__(self, max_seq_length: int, weight=0.999) -> None:
        super().__init__()
        self.max_seq_length = max_seq_length
        self.weight = weight

        # logger.warning("EMA for seq_length is computed during training")
        fill_value = 1 / max_seq_length
        self.register_buffer(
            "n_elements_prob",
            torch.full((max_seq_length,), fill_value=fill_value),
        )

    def __call__(self, mask: BoolTensor):
        N = self.max_seq_length
        batch_prob = mask.sum(dim=1).bincount(minlength=N + 1)[1:] / mask.size(0)
        self.n_elements_prob = self.weight * self.n_elements_prob
        self.n_elements_prob += (1.0 - self.weight) * batch_prob.to(
            self.n_elements_prob
        )

    def sample(self, batch_size: int) -> LongTensor:
        n_elements = torch.multinomial(
            self.n_elements_prob.cpu(), batch_size, replacement=True
        )
        n_elements += 1  # shoule be in range [1, cfg.dataset.max_seq_length]
        return n_elements


class VAEModule(nn.Module):
    def __init__(self, dim_input: int, dim_latent: int) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(dim_input, dim_latent)
        self.fc_var = nn.Linear(dim_input, dim_latent)

    def reparameterize(self, mu: FloatTensor, logvar: FloatTensor) -> FloatTensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: FloatTensor) -> Dict[str, FloatTensor]:
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        z = self.reparameterize(mu, logvar)
        return {"z": z, "mu": mu, "logvar": logvar}


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dim_model: int, max_token_length: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.rand(max_token_length, dim_model))
        self.max_token_length = max_token_length

    def forward(self, h: Tensor):
        B, S = h.shape[:2]
        emb = rearrange(self.pos_emb[:S], "s d -> 1 s d")
        emb = repeat(emb, "1 s d -> b s d", b=B)
        return emb

    @property
    def no_decay_param_names(self) -> List[str]:
        return [
            "pos_emb",
        ]


class ShufflePositionalEmbedding(PositionalEmbedding):

    def forward(self, h: Tensor, do_shuffle: bool=True):
        if not do_shuffle:
            return super().forward(h) # [B, S, D]

        B, S = h.shape[:2]
        emb = self.pos_emb # [S, D]
        shuffle_emb = []
        for b in range(B):
            ind = torch.randperm(self.max_token_length, device=emb.device)
            ind = ind[:S]
            shuffle_emb.append(emb[ind])
        return torch.stack(shuffle_emb, dim=0)

class ElementPositionalEmbedding(torch.nn.Module):
    """
    Positional embedding to indicate j-th attr of i-th element
    """

    def __init__(self, dim_model: int, max_token_length: int, n_attr_per_elem=5):
        super().__init__()
        remainder = max_token_length % n_attr_per_elem
        if remainder == 1:
            self.bos_emb = nn.Parameter(torch.rand(1, dim_model))
        elif remainder == 0:
            pass
        else:
            raise NotImplementedError

        self.max_len = max_token_length
        self.n_elem = max_token_length // n_attr_per_elem
        self.n_attr_per_elem = n_attr_per_elem
        self.elem_emb = nn.Parameter(torch.rand(self.n_elem, dim_model))
        self.attr_emb = nn.Parameter(torch.rand(self.n_attr_per_elem, dim_model))

    def forward(self, h: Tensor):
        if getattr(self, "bos_emb", None) is not None:
            h = h[:, 1:]
        B, S = h.size()[:2]

        # (1, 2, 3) -> (1, ..., 1, 2, ..., 2, 3, ..., 3, ...)
        elem_emb = repeat(self.elem_emb, "s d -> (s x) d", x=self.n_attr_per_elem)
        # (1, 2, 3) -> (1, 2, 3, 1, 2, 3, ...)
        attr_emb = repeat(self.attr_emb, "x d -> (s x) d", s=self.n_elem)
        emb = elem_emb + attr_emb

        emb = emb[:S]
        if getattr(self, "bos_emb", None) is not None:
            emb = torch.cat([self.bos_emb, emb], dim=0)
        emb = repeat(emb, "s d -> b s d", b=B)
        return emb

    @property
    def no_decay_param_names(self) -> List[str]:
        decay_list = ["elem_emb", "attr_emb"]
        if getattr(self, "bos_emb", None) is not None:
            decay_list.append("bos_emb")
        return decay_list


class PositionalEncoding(nn.Module):
    """Unlearnable Positional Encoding
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]``
        """
        B, S = x.size()[:2]
        pe = self.pe[:S]
        pe = repeat(pe, "s d -> b s d", b=B)
        return pe


class ShufflePositionalEncoding(PositionalEncoding):
    """Unlearnable Positional Encoding + Shuffle
    """

    def forward(self, x: Tensor, do_shuffle: bool=True) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        if not do_shuffle:
            return super().forward(x)
        B, S = x.size()[:2]
        pe = torch.stack([self.pe[torch.randperm(self.max_len)] for _ in range(B)], dim=0) #[B, max_len, D]
        pe = pe[:, :S] # [B, S, D]
        return pe


class CategoricalTransformer(torch.nn.Module):
    """
    Model to perform one-shot / auto-regressive generation of 1D sequence
    """

    def __init__(
        self,
        # backbone_cfg: DictConfig,
        backbone: TransformerEncoder,
        num_classes: int,
        max_token_length: int,
        dim_model: int,
        lookahead: bool = True,
        pos_emb: str = "default",
        dim_head: Optional[int] = None,
        use_additional_input: Optional[str] = None,  # for self-conditioned generation
        additional_input_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.lookahead = lookahead
        self.use_additional_input = use_additional_input
        self.backbone = backbone
        self.cat_emb = nn.Embedding(num_classes, dim_model)

        if self.use_additional_input == "seq":
            self.cat_emb_additional = nn.Embedding(num_classes, dim_model)
        elif self.use_additional_input in ["logit", "prob"]:
            self.cat_emb_additional = nn.Linear(num_classes, dim_model)
        elif self.use_additional_input == "bbox":
            assert isinstance(additional_input_dim, int)
            self.cat_emb_additional = nn.Linear(additional_input_dim, dim_model)
        elif self.use_additional_input == 'cond_mask':
            assert isinstance(additional_input_dim, int)
            self.self_cond_emb = nn.Linear(additional_input_dim, dim_model)

        if self.use_additional_input:
            self.self_cond_fuser = nn.Sequential(
                nn.Linear(dim_model * 2, dim_model),
                nn.ReLU(),
            )

        if pos_emb == "default":
            self.pos_emb = PositionalEmbedding(
                dim_model=dim_model, max_token_length=max_token_length
            )
        elif pos_emb == "elem_attr":
            self.pos_emb = ElementPositionalEmbedding(
                dim_model=dim_model,
                max_token_length=max_token_length,
                n_attr_per_elem=kwargs["n_attr_per_elem"],
            )
        elif pos_emb == "shuffle":
            self.pos_emb = ShufflePositionalEmbedding(
                dim_model=dim_model, max_token_length=max_token_length,
            )
        elif pos_emb == 'none':
            self.pos_emb = None
        elif pos_emb == 'pos_enc':
            self.pos_emb = PositionalEncoding(dim_model, max_token_length)
        elif pos_emb == 'shuffle_pos_enc':
            self.pos_emb = ShufflePositionalEncoding(dim_model, max_token_length)
        else:
            raise NotImplementedError

        self.drop = nn.Dropout(0.1)
        d_last = dim_head if dim_head else num_classes
        self.head = nn.Sequential(
            nn.LayerNorm(dim_model), nn.Linear(dim_model, d_last, bias=False)
        )

    def forward(
        self,
        seq: Union[Tensor, Tuple[Tensor, Tensor]],
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Input: 1D sequence of shape (B, S)
        Output: 2D sequence of logits (B, S, C)
        """

        S = seq.shape[1]
        h = self.cat_emb(seq)

        if self.use_additional_input:
            h = self.fuse_self_cond(h, self_cond)

        if self.pos_emb is not None:
            h = h + self.pos_emb(seq)
        h = self.drop(h)

        if self.lookahead:
            # non-autoregressive generation
            if timestep is not None:
                h = self.backbone(
                    h, src_key_padding_mask=src_key_padding_mask,
                    timestep=timestep, attention_bias=attention_bias,
                )
            else:
                h = self.backbone(h, src_key_padding_mask=src_key_padding_mask, attention_bias=attention_bias)
        else:
            # autoregressive generation
            mask = generate_causal_mask(S).to(h)
            h = self.backbone(h, mask=mask, src_key_padding_mask=src_key_padding_mask, attention_bias=attention_bias)
        logits = self.head(h)  # (B, S, C)
        outputs = {"logits": logits, "feat": h}
        return outputs
    
    def fuse_self_cond(self, h, self_cond):
        if self_cond is not None:
            if self.use_additional_input == "seq":
                h_add = self.cat_emb_additional(self_cond)
            elif self.use_additional_input == 'bbox':
                h_add = self.cat_emb_additional(self_cond)
                h_add = torch.repeat_interleave(h_add, repeats=5, dim=1)
            elif self.use_additional_input in ["logit", "prob"]:
                h_add = self.cat_emb_additional(
                    rearrange(self_cond, "b c s -> b s c")
                )
            else:
                raise NotImplementedError
        elif self.use_additional_input == 'bbox':
            raise ValueError(f'self_cond is required, but self_cond={self_cond}')
        else:
            h_add = torch.zeros_like(h)
        h = self.self_cond_fuser(torch.cat([h, h_add], dim=-1))
        return h


class ContinuousTransformer(torch.nn.Module):
    """
    Model to perform one-shot / auto-regressive generation of 1D sequence (B, S, C)
    """

    def __init__(
        self,
        backbone: TransformerEncoder,
        max_token_length: int,
        dim_model: int,
        dim_in: int,
        lookahead: bool = True,
        pos_emb: str = "default",
        # use_additional_input: Optional[str] = None,  # for self-conditioned generation
        **kwargs,
    ) -> None:
        super().__init__()

        self.lookahead = lookahead
        self.backbone = backbone
        self.emb = nn.Linear(dim_in * 2, dim_model)

        if pos_emb == "default":
            self.pos_emb = PositionalEmbedding(
                dim_model=dim_model, max_token_length=max_token_length
            )
        elif pos_emb == "elem_attr":
            self.pos_emb = ElementPositionalEmbedding(
                dim_model=dim_model,
                max_token_length=max_token_length,
                n_attr_per_elem=kwargs["n_attr_per_elem"],
            )
        else:
            raise NotImplementedError

        self.drop = nn.Dropout(0.1)
        self.head = nn.Sequential(
            nn.LayerNorm(dim_model), nn.Linear(dim_model, dim_in, bias=False)
        )

    def forward(
        self,
        x: FloatTensor,
        src_key_padding_mask: Optional[BoolTensor] = None,
        timestep: Optional[Union[LongTensor, FloatTensor]] = None,
        x_self_cond: Optional[FloatTensor] = None,
    ) -> Tensor:
        """
        Input: 2D sequence of shape (B, S, C)
        Output: 2D sequence of logits (B, S, C)
        """
        if x_self_cond is None:
            x_self_cond = torch.zeros_like(x)
        x = torch.cat((x_self_cond, x), dim=-1)

        h = self.emb(x)
        h = h + self.pos_emb(h)
        h = self.drop(h)

        if self.lookahead:
            # non-autoregressive generation
            if timestep is not None:
                h = self.backbone(
                    h, src_key_padding_mask=src_key_padding_mask, timestep=timestep
                )
            else:
                h = self.backbone(h, src_key_padding_mask=src_key_padding_mask)
        else:
            # autoregressive generation
            # mask = generate_causal_mask(S).to(h)
            # h = self.backbone(h, mask=mask, src_key_padding_mask=src_key_padding_mask)
            raise NotImplementedError
        outputs = {"outputs": self.head(h)}  # (B, S, C)
        return outputs


class CategoricalEncDecTransformer(torch.nn.Module):
    """
    For bart-like models
    """

    def __init__(
        self,
        backbone_enc: TransformerEncoder,
        backbone_dec: nn.TransformerDecoder,
        num_classes_dec: int,
        max_token_length_dec: int,
        dim_model: int,
        pos_emb: str = "default",
        dim_head: Optional[int] = None,
        num_classes_enc: Optional[int] = None,
        max_token_length_enc: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = backbone_enc
        self.decoder = backbone_dec

        if num_classes_enc is None:
            num_classes_enc = num_classes_dec
        if max_token_length_enc is None:
            max_token_length_enc = max_token_length_dec

        self.input_cat_emb = nn.Embedding(num_classes_enc, dim_model)
        self.target_cat_emb = nn.Embedding(num_classes_dec, dim_model)

        if pos_emb == "default":
            self.input_pos_emb = PositionalEmbedding(
                dim_model=dim_model, max_token_length=max_token_length_enc
            )
            self.target_pos_emb = PositionalEmbedding(
                dim_model=dim_model, max_token_length=max_token_length_dec
            )
        elif pos_emb == "elem_attr":
            self.input_pos_emb = ElementPositionalEmbedding(
                dim_model=dim_model,
                max_token_length=max_token_length_enc,
                n_attr_per_elem=kwargs["n_attr_per_elem"],
            )
            self.target_pos_emb = ElementPositionalEmbedding(
                dim_model=dim_model,
                max_token_length=max_token_length_dec,
                n_attr_per_elem=kwargs["n_attr_per_elem"],
            )
        else:
            raise NotImplementedError

        self.drop = nn.Dropout(0.1)
        d_last = dim_head if dim_head else num_classes_dec
        self.head = nn.Sequential(
            nn.LayerNorm(dim_model), nn.Linear(dim_model, d_last, bias=False)
        )

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Input: 1D sequence of shape (B, S), first token is always [BOS]
        Output: 2D sequence of logits (B, S, C)
        """

        h_enc = self.input_cat_emb(input)
        h_enc += self.input_pos_emb(input)

        h_enc = self.drop(h_enc)
        memory = self.encoder(h_enc, src_key_padding_mask=src_key_padding_mask)

        tgt = self.target_cat_emb(target)
        tgt += self.target_pos_emb(target)

        S = target.shape[1]
        tgt_mask = generate_causal_mask(S).to(tgt)
        h = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        logits = self.head(h)  # (B, S, C)
        outputs = {"logits": logits}
        return outputs


class CategoricalAggregatedTransformer(CategoricalTransformer):
    """
    Model to perform one-shot / auto-regressive generation of 1D sequence
    """

    def __init__(
        self,
        # backbone_cfg: DictConfig,
        backbone: TransformerEncoder,
        num_classes: int,
        max_token_length: int,
        dim_model: int,
        lookahead: bool = True,
        pos_emb: str = "default",
        dim_head: Optional[int] = None,
        use_additional_input: Optional[str] = None,  # for self-conditioned generation
        additional_input_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        assert use_additional_input in [None, 'cond_mask']
        assert pos_emb in ['default', 'none', 'pos_enc', 'shuffle_pos_enc', 'shuffle']
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            max_token_length=max_token_length//5,
            dim_model=dim_model,
            lookahead=lookahead,
            pos_emb=pos_emb,
            dim_head=dim_head,
            use_additional_input=use_additional_input,
            additional_input_dim=additional_input_dim,
            **kwargs,
        )
        self.additional_input_dim = additional_input_dim
        assert self.lookahead
        self.d_model = dim_model
        self.enc = nn.Sequential(
            nn.Linear(5 * self.d_model, self.d_model),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(self.d_model, 5 * self.d_model),
            nn.ReLU(),
        )

    def forward(
        self,
        seq: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Input: 1D sequence of shape (B, S)
        Output: 2D sequence of logits (B, S, C)
        """
        assert attention_bias is None
        b, S = seq.shape
        # (guess): positional info. should be added just before transformer blocks
        h = self.drop(self.cat_emb(seq))
        h = rearrange(h, "b (s x) d -> b s (x d)", x=5)
        h = self.enc(h)
        if self.pos_emb is not None:
            h = h + self.pos_emb(h)

        if src_key_padding_mask is not None:
            src_key_padding_mask = rearrange(src_key_padding_mask, "b (s x) -> b s x", x=5)
            src_key_padding_mask = src_key_padding_mask.any(dim=-1)

        if self.use_additional_input is not None:
            if self_cond is None:
                raise ValueError('self_cond must be given when use_additional_input is not None')
            self_cond = self_cond.reshape(b, -1, self.additional_input_dim)
            h_add = self.self_cond_emb(self_cond.float())
            h = self.self_cond_fuser(torch.cat([h, h_add], dim=-1))

        if timestep is not None:
            h = self.backbone(
                h, src_key_padding_mask=src_key_padding_mask, timestep=timestep
            )
        else:
            h = self.backbone(h, src_key_padding_mask=src_key_padding_mask)

        h = self.dec(h)
        h = rearrange(h, "b s (x d) -> b (s x) d ", x=5)
        outputs = {"logits": self.head(h)}  # (B, S, C)
        return outputs


class ElementTransformer(torch.nn.Module):
    """
    Model to perform one-shot / auto-regressive generation of elemtns
    """

    def __init__(
        self,
        # backbone_cfg: DictConfig,
        backbone: TransformerEncoder,
        num_labels: int,
        num_bin_bboxes: int,
        max_len: int,
        dim_model: int,
        lookahead: bool = False,
    ) -> None:
        super().__init__()
        # d_model = get_dim_model(backbone_cfg)
        self.backbone = backbone

        self.lookahead = lookahead

        self.layout_enc = LayoutEncoder(dim_model, num_labels, num_bin_bboxes)
        self.layout_dec = LayoutDecoder(dim_model, num_labels, num_bin_bboxes)

        self.drop = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(dim_model)

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Input: 1D sequence of shape (B, S)
        Output: 2D sequence of logits (B, S, C)
        """
        h = self.layout_enc(src)
        h = self.drop(h)

        if self.lookahead:
            # non-autoregressive generation
            if timestep is not None:
                h = self.backbone(
                    h, src_key_padding_mask=src_key_padding_mask, timestep=timestep
                )
            else:
                h = self.backbone(h, src_key_padding_mask=src_key_padding_mask)
        else:
            # autoregressive generation
            mask = generate_causal_mask(h.size(1)).to(h)
            h = self.backbone(h, mask=mask, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        outputs = self.layout_dec(h)
        return outputs


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class LayoutDiffusionTransformer(nn.Module):
    """BERT_encoder-based Transformer used in LayoutDiffusion (ICCV2023)
    original implementation:
    https://github.com/microsoft/LayoutGeneration/blob/main/LayoutDiffusion/improved-diffusion/improved_diffusion/transformer_model.py#L216
    """
    def __init__(
        self,
        backbone,
        hidden_size: int,
        num_classes: int,
        time_encode_dim: int = 128,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.backbone = backbone

        self.in_channels = hidden_size
        self.time_encode_dim = time_encode_dim
        self.out_channels = num_classes
        self.dropout = dropout
        self.num_classes = num_classes

        self.word_embedding = nn.Embedding(num_classes, self.in_channels)
            
        time_embed_dim = time_encode_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(time_encode_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_size),
        )

        max_position_embeddings = 512 # set large enough number

        self.input_up_proj = nn.Sequential(nn.Linear(self.in_channels, hidden_size),
                                                nn.Tanh(), nn.Linear(hidden_size, hidden_size))
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        self.pos_emb = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        self.output_down_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                              nn.Tanh(), nn.Linear(hidden_size, self.out_channels))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def index_to_log_onehot(self,x):
        num_classes=self.out_channels
        assert x.max().item() < num_classes, \
            f'Error: {x.max().item()} >= {num_classes}'
        x_onehot = F.one_hot(x, num_classes)
        permute_order = (0, -1) + tuple(range(1, len(x.size())))
        x_onehot = x_onehot.permute(permute_order)
        log_x = torch.log(x_onehot.float().clamp(min=1e-30))
        return log_x

    def forward(self, seq, timestep, y=None):
        seq = self.word_embedding(seq)
        emb = self.time_embed(timestep_embedding(timestep, self.time_encode_dim))

        emb_x = self.input_up_proj(seq)
        seq_length = seq.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        pe = self.pos_emb(position_ids)
        emb_inputs = pe + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        input_trans_hidden_states = self.backbone(emb_inputs)
        logits = self.output_down_proj(input_trans_hidden_states)  # (B, S, C)

        outputs = {"logits": logits}
        return outputs