"""
This file is derived from the following repository:
https://github.com/CyberAgentAILab/layout-dm/tree/9cf122d3be74b1ed8a8d51e3caabcde8164d238e

Original file: src/trainer/trainer/global_configs.py
Author: naoto0804
License: Apache-2.0 License
"""

from pathlib import Path

ROOT = f"{str(Path(__file__).parent)}/../../../download"
KMEANS_WEIGHT_ROOT = f"{ROOT}/clustering_weights"
DATASET_DIR = f"{ROOT}/datasets"
FID_WEIGHT_DIR = f"{ROOT}/fid_weights/FIDNetV3"
JOB_DIR = f"{ROOT}/pretrained_weights"
