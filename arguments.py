# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import transformers

from self_speculation.generator_base import (
    GenerationConfig,
)

raw_types = Union[str, float, int, Dict, List, Tuple]

@dataclass
class Arguments:
    model: str
    model_args: Optional[Dict[str, Any]] = None
    dist_url: str = "env://"
    world_size: int = 1
    rank: int = 0
    seed: Optional[int] = 42
    output_dir: str = "./logs"
    distributed: bool = False
    dist_url: str = "env://"
    local_rank: int = -1  


# Source: https://github.com/EleutherAI/lm-evaluation-harness/blob/a9eaaf46f1e246e5ce090e37f2f99fe1cfe5a919/lm_eval/utils.py
def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]
    }
    return args_dict


# Source: https://github.com/EleutherAI/lm-evaluation-harness/blob/a9eaaf46f1e246e5ce090e37f2f99fe1cfe5a919/lm_eval/utils.py
def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg
