from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import transformers

from self_speculation.generator_base import (
    GenerationConfig,
)

raw_types = Union[str, float, int, Dict, List, Tuple]


@dataclass
class BenchmarkArguments:
    data_format: str
    model_path: str
    data_path: Optional[str] = None
    random_shuffle: bool = True
    num_samples: Optional[int] = None
    seed: Optional[int] = 42
    n_shot: Optional[int] = 0
    model_args: Optional[str] = None
    output_dir: str = "./logs"

@dataclass
class Arguments:
    benchmark_arguments: BenchmarkArguments
    generation_config: GenerationConfig


def process_cli_arguments() -> Tuple[BenchmarkArguments, GenerationConfig]:
    parser = transformers.HfArgumentParser((BenchmarkArguments, GenerationConfig))
    (
        benchmark_arguments,
        generation_config,
        _remaining,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if benchmark_arguments.model_args:
        benchmark_arguments.model_args = simple_parse_args_string(benchmark_arguments.model_args)
    else:
        benchmark_arguments.model_args = {}

    args: Arguments = Arguments(benchmark_arguments=benchmark_arguments, generation_config=generation_config)
    return args


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
