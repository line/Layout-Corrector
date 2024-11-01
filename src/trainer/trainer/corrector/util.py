"""
Copyright (c) 2024 LY Corporation and Tohoku University
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

from __future__ import annotations

from enum import Enum


class CorrectorMaskingMode(Enum):
    TOPK = "topk"
    THRESH = "thresh"

    @classmethod
    def get_names(cls) -> list:
        return [e.name for e in cls]

    @classmethod
    def get_values(cls) -> list:
        return [e.value for e in cls]
    