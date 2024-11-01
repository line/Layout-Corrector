"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

from PIL import Image

from .base import BaseDataset


CRELLO_LABELS = [
    'coloredBackground', 'imageElement', 'maskElement', 'svgElement', 'textElement'
]


class CrelloBboxDataset(BaseDataset):
    name = "crello-bbox"
    labels = CRELLO_LABELS

    def __init__(self, dir: str, split: str, max_seq_length: int, transform=None):
        super().__init__(dir, split, max_seq_length, transform)

    def process(self):
        raise NotImplementedError('Implemented in RicoDatset, but removed')

    def download(self):
        pass

    def get_original_resource(self, batch) -> Image:
        raise NotImplementedError('Implemented in RicoDatset, but removed')
