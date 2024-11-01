"""
This file is derived from the following repository:
https://github.com/CyberAgentAILab/layout-dm/tree/9cf122d3be74b1ed8a8d51e3caabcde8164d238e

Original file: src/trainer/trainer/crossplatform_util.py
Author: naoto0804
License: Apache-2.0 License
"""

import logging
import sys

logger = logging.getLogger(__name__)


def filter_args_for_ai_platform():
    """
    This is to filter out "--job-dir <JOB_DIR>" which is passed from AI Platform training command,
    """
    key = "--job_dir"
    if key in sys.argv:
        logger.warning(f"{key} removed")
        arguments = sys.argv
        ind = arguments.index(key)
        sys.argv = [a for (i, a) in enumerate(arguments) if i not in [ind, ind + 1]]

    key = "--job-dir"
    for i, arg in enumerate(sys.argv):
        if len(arg) >= len(key) and arg[: len(key)] == key:
            sys.argv = [a for (j, a) in enumerate(sys.argv) if i != j]
