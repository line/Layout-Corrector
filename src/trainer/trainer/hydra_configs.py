"""
This file is derived from the following repository:
https://github.com/CyberAgentAILab/layout-dm/tree/9cf122d3be74b1ed8a8d51e3caabcde8164d238e

Original file: src/trainer/trainer/hydra_configs.py
Author: naoto0804
License: Apache-2.0 License

Modifications have been made to the original file to fit the requirements of this project.

--------------------------------------------

A file to declare dataclass instances used for hydra configs at ./config/*
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

from trainer.helpers.layout_tokenizer import CHOICES
from trainer.corrector.util import CorrectorMaskingMode


@dataclass
class TestConfig:
    job_dir: str
    result_dir: str
    dataset_dir: Optional[str] = None  # set if it is different for train/test
    max_batch_size: int = 512
    num_run: int = 1  # number of outputs per input
    cond: str = "unconditional"
    num_timesteps: int = 100
    is_validation: bool = False  # for evaluation in validation set (e.g. HP search)
    debug: bool = False  # disable some features to enable fast runtime
    debug_num_samples: int = -1  # in debug mode, reduce the number of samples when > 0
    best_or_final: str = "best"  # which model to load ("best" or "final")

    # for sampling
    sampling: str = "random"  # see ./helpers/sampling.py for options
    # below are additional parameters for sampling modes
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: float = 5

    # for unconditional models
    num_uncond_samples: int = 1000

    # for diffusion models
    # assymetric time-difference (https://arxiv.org/abs/2208.04202)
    time_difference: float = 0.0

    # for diffusion models, refinement only
    refine_lambda: float = 3.0  # if > 0.0, trigger refinement mode
    refine_mode: str = "uniform"
    refine_offset_ratio: float = 0.1  # 0.2
    refine_noise_std: float = 0.1

    # for diffusion models, relation only
    relation_lambda: float = 3e6  # if > 0.0, trigger relation mode
    relation_mode: str = "average"
    relation_tau: float = 1.0
    relation_num_update: int = 3

    # for continuous diffusion models
    use_ddim: bool = False

    use_train_data: bool = False # Check if the model working well on Training Dataset

    # For test with Corrector
    corrector_steps: int = 1
    dm_job_dir: str = "" # job_dir of Diffusion model for Corrector
    use_gumbel_noise: bool = False
    corrector_start: int = 0
    corrector_end: int = 9999
    time_adaptive_temperature: bool = False
    corrector_t_list: Optional[List[int]] = None
    corrector_temperature: float = 1.0
    gumbel_temperature: float = 1.0

    corrector_mask_mode: str = CorrectorMaskingMode.THRESH.value
    corrector_mask_threshold: float = 0.7


@dataclass
class TrainConfig:
    epochs: int = 50
    save_step: int = -1 # default: -1 (save only best model)
    grad_norm_clip: float = 1.0
    weight_decay: float = 1e-1
    loss_plot_iter_interval: int = 50
    sample_plot_epoch_interval: int = 1
    fid_plot_num_samples: int = 1000
    fid_plot_batch_size: int = 512
    pretrain_job_dir: Optional[str] = None
    pretrain_model_type: str = "best" # best or final
    debug_num_samples: int = -1 # in debug mode, reduce the number of training samples when > 0
    ema_decay: float = -1.0 # default: no EMA


@dataclass
class CorrectorTrainConfig:
    epochs: int = 50
    grad_norm_clip: float = 1.0
    weight_decay: float = 1e-1
    loss_plot_iter_interval: int = 50
    weight_init_w_diffusion: bool = False
    pretrained_job_dir: Optional[str] = None
    diffusion_model_ckpt: str = "best" # best or final


@dataclass
class DataConfig:
    batch_size: int = 64
    bbox_quantization: str = "linear"
    num_bin_bboxes: int = 32
    num_workers: int = os.cpu_count()
    pad_until_max: bool = (
        False  # True for diffusion models, False for others for efficient batching
    )
    shared_bbox_vocab: str = "xywh"
    special_tokens: Tuple[str] = ("pad", "mask")
    # special_tokens: Tuple[str] = ("pad",)
    # transforms: Tuple[str] = ("SortByLabel", "LexicographicOrder")
    transforms: Tuple[str] = ("RandomOrder",)
    var_order: str = "c-x-y-w-h"

    def __post_init__(self) -> None:
        # advanced validation like choices in argparse
        for key in ["shared_bbox_vocab", "bbox_quantization", "var_order"]:
            assert getattr(self, key) in CHOICES[key]
