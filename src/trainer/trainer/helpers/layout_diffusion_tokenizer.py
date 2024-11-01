"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php

-------------------------------------------------------------------------------

Special Tokenizer for LayoutDiffusion (ICCV2023)
"""

import logging
from copy import deepcopy
from typing import Dict

import torch
from einops import rearrange, reduce, repeat
from omegaconf import DictConfig
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from trainer.datasets import DATASETS
from trainer.helpers.bbox_tokenizer import BboxTokenizer
from trainer.helpers.layout_tokenizer import LayoutTokenizer

logger = logging.getLogger(__name__)

# "sep" is used for separating bbox tokens
SPECIAL_TOKEN_VOCABULARIES = ["bos", "eos", "sep", "pad", "mask", "unk"]


class LayoutDiffusionBboxTokenizer(BboxTokenizer):
    """
    If N is number of bins, 0 <= x, y <= (N - 1) / N
    'bbox' variable is assumed to have "xywh" order
    """
    def __init__(
        self,
        num_bin_bboxes: int,
        var_order: str = "c-l-t-r-b",
        shared_bbox_vocab: str = "ltrb",
        bbox_quantization: str = "linear",
        dataset_name: str = "rico25_max25",
    ):
        assert bbox_quantization == 'linear'
        assert var_order in ["c-l-t-r-b", "c-l-t-r-b-sep"]
        assert shared_bbox_vocab == 'ltrb'
        assert num_bin_bboxes == 128
        self._num_bin_bboxes = num_bin_bboxes
        self._var_order = var_order.lstrip("c-").split("-")
        self._shared_bbox_vocab = shared_bbox_vocab
        self._bbox_quantization = bbox_quantization
        self._dataset_name = dataset_name
        self._var_names = ["l", "r", "t", "b"]

        self._clustering_models = None

    @staticmethod
    def xywh_to_ltrb( bbox: FloatTensor) -> FloatTensor:
        ltrb_bbox = torch.clone(bbox)
        ltrb_bbox[..., 0] = bbox[..., 0] - (bbox[..., 2] / 2)  # Left
        ltrb_bbox[..., 1] = bbox[..., 1] - (bbox[..., 3] / 2)  # Top
        ltrb_bbox[..., 2] = bbox[..., 0] + (bbox[..., 2] / 2)  # Right
        ltrb_bbox[..., 3] = bbox[..., 1] + (bbox[..., 3] / 2)  # Bottom
        return ltrb_bbox

    @staticmethod
    def ltrb_to_xywh(bbox: FloatTensor) -> FloatTensor:
        xywh_bbox = torch.clone(bbox)

        xywh_bbox[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2  # X center
        xywh_bbox[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2  # Y center
        xywh_bbox[..., 2] = bbox[..., 2] - bbox[..., 0]  # Width
        xywh_bbox[..., 3] = bbox[..., 3] - bbox[..., 1]  # Height
        return xywh_bbox

    def encode(self, xywh_bbox: FloatTensor) -> LongTensor:
        bbox = self.xywh_to_ltrb(xywh_bbox)
        bbox_q = torch.clamp(bbox, 0.0, 1.0)
        indices = ((self.num_bin_bboxes - 1) * bbox_q).round().long()
        return indices

    def decode(self, bbox_indices: LongTensor) -> FloatTensor:
        arr = torch.clone(bbox_indices) # avoid overriding
        arr = torch.clamp(arr, 0, self.num_bin_bboxes - 1)

        bbox = arr.float() / (self.num_bin_bboxes - 1)
        bbox_xywh = self.ltrb_to_xywh(bbox)
        # avoid negative width/height
        bbox_xywh[..., 2:] = torch.clamp(bbox_xywh[..., 2:], 0.0, 1.0)
        return bbox_xywh
    
    @property
    def token_mask(self) -> Dict[str, BoolTensor]:
        raise NotImplementedError


class LayoutDiffusionTokenizer(LayoutTokenizer):
    """
    Tokenizer for LayoutDiffusion
    Different from LayoutTokenizer, specials tokens must come first.
    i.e., [special_tokens, category_tokens, bbox_tokens, MASK]
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        dataset_cfg: DictConfig,
    ) -> None:
        self._data_cfg = data_cfg
        self._dataset_cfg = dataset_cfg
        self._sort_by = None

        name = dataset_cfg._target_.split(".")[-1]
        inv_dic = {str(v.__name__): k for (k, v) in DATASETS.items()}

        # validation
        self._var_order = data_cfg.get("var_order", "c-x-y-w-h")
        # assert self.var_order in ["c-x-y-w-h", "c-w-h-x-y", "c-xw-yh", "c-xywh"]
        assert self._var_order[0] == "c"
        assert all(token in SPECIAL_TOKEN_VOCABULARIES for token in self.special_tokens)
        if "mask" in self.special_tokens:
            assert self.special_tokens.index("mask") == self.N_sp_token - 1

        dataset_name = f"{inv_dic[name]}_max{dataset_cfg.max_seq_length}"
        self._bbox_tokenizer = LayoutDiffusionBboxTokenizer(
            num_bin_bboxes=data_cfg.num_bin_bboxes,
            var_order=self._var_order,
            shared_bbox_vocab=data_cfg.shared_bbox_vocab,
            bbox_quantization=data_cfg.bbox_quantization,
            dataset_name=dataset_name,
        )

        self._N_category = len(DATASETS[inv_dic[name]].labels)

        logger.info(
            f"N_total={self.N_total}, (N_label, N_bbox, N_sp_token)=({self.N_category},{self.N_bbox},{self.N_sp_token})"
        )

        ## Set Token Order
        # original_order: [category_tokens, bbox_tokens, special_tokens (including MASK)]
        # new_order: [special_tokens (except MASK), category_tokens, bbox_tokens, MASK]
        self._special_token_name_to_id = {
            token: self.special_tokens.index(token) for token in self.special_tokens
        }
        self._special_token_name_to_id["mask"] = self.N_total - 1
        self._special_token_id_to_name = {
            v: k for (k, v) in self._special_token_name_to_id.items()
        }

    def encode(
        self,
        inputs: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        inputs has the following items
            mask: torch.BoolTensor of shape (B, S)
            label: torch.LongTensor of shape (B, S)
            bbox: torch.FloatTensor of shape (B, S, 4)
        """
        label = deepcopy(rearrange(inputs["label"], "b s -> b s 1"))
        bbox = deepcopy(self._bbox_tokenizer.encode(inputs["bbox"]))
        mask = deepcopy(inputs["mask"])

        label, bbox, mask = self._pad_until(label, bbox, mask)

        ###### add offset
        label += (self.N_sp_token - 1)
        bbox += (self.N_sp_token - 1) + self.N_category
        label, bbox = self._fix_padded_sequences(label, bbox, mask)

        B, S = label.size()[:2]
        C = self.N_var_per_element

        # sanity check
        seq_len = reduce(mask.int(), "b s -> b 1", reduction="sum")
        indices = rearrange(torch.arange(0, S), "s -> 1 s")
        assert torch.all(torch.logical_not(mask) == (seq_len <= indices)).item()

        if "sep" in self.special_tokens:
            separator = torch.clone(mask).long()
            separator[mask] = self.name_to_id("sep")
            separator[~mask] = self.name_to_id("pad")
            eos_mask = (seq_len - 1) == indices
            separator[eos_mask] = self.name_to_id("eos")
            separator = rearrange(separator, "b s -> b s 1")

        if self.sort_by == "category_alphabetical":
            label, index = torch.sort(label, dim=1)
            bbox = torch.gather(bbox, dim=1, index=repeat(index, "b s 1 -> b s c", c=4))
            mask = torch.gather(mask, dim=1, index=rearrange(index, "b s 1 -> b s"))

        # make 1d sequence
        if "sep" in self.special_tokens:
            seq = torch.cat([label, bbox, separator], axis=-1)
        else:
            seq = torch.cat([label, bbox], axis=-1)

        seq = rearrange(seq, "b s x -> b (s x)")
        mask = repeat(mask, "b s -> b (s c)", c=C)

        if "bos" in self.special_tokens:
            bos = torch.full((B, 1), self.name_to_id("bos"))
            seq = torch.cat([bos, seq], axis=-1)
            mask = torch.cat([torch.full((B, 1), fill_value=True), mask], axis=-1)

        return {"seq": seq.long(), "mask": mask}

    def decode(self, ids: LongTensor) -> Dict[str, Tensor]:
        if "bos" in self.special_tokens:
            ids = ids[..., 1:]
        ids = rearrange(ids, "b (s c) -> b s c", c=self.N_var_per_element)
        label, bbox = deepcopy(ids[..., 0]), deepcopy(ids[..., 1:5])
        label -= (self.N_sp_token - 1)
        bbox -= (self.N_sp_token - 1) + self.N_category
        invalid = self._filter_eos(label)
        invalid = invalid | self._filter_invalid_labels_and_bboxes(label, bbox)

        bbox = self.bbox_tokenizer.decode(bbox)
        label[invalid] = 0
        bbox[invalid] = 0.0
        return {"bbox": bbox, "label": label, "mask": torch.logical_not(invalid)}

    @property
    def max_token_length(self) -> int:
        out = self.max_seq_length * self.N_var_per_element
        if "bos" in self.special_tokens:
            out += 1
        return out

    @property
    def token_mask(self) -> BoolTensor:
        ## May not be used in LayoutDiffusion
        """
        Returns a bool tensor in shape (S, C), which is used to filter our invalid predictions
        E.g., predict high probs on x=1, while the location of token is for predicting a category
        """
        masks = self.bbox_tokenizer.token_mask
        top = BoolTensor(
            [
                False if x in ["bos"] else True
                for x in self.special_tokens
                if x != "mask"
            ]
        )
        last = BoolTensor([False])

        masks["c"] = torch.cat(
            [
                top,
                torch.full((self.N_category,), True),
                torch.full((self.N_bbox,), False),
                last,
            ]
        )
        for key in self.var_names:
            if key == "c":
                continue
            masks[key] = torch.cat(
                [top, torch.full((self.N_category,), False), masks[key], last]
            )
        mask = torch.stack([masks[k] for k in self.var_names], dim=0)
        mask = repeat(mask, "x c -> (s x) c", s=self.max_seq_length)
        return mask

    def get_slice(self, name: str) -> slice:
        raise NotImplementedError()
    