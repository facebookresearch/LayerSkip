from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import transformers

from self_speculation.generator_base import (
    GenerationConfig,
)

raw_types = Union[str, float, int, Dict, List, Tuple]


@dataclass
class BenchmarkArguments:
    manifold_output_dir: str
    data_path: str
    data_format: str
    model_path: str
    random_shuffle: bool = True
    num_samples: Optional[int] = None
    seed: int = 1


def process_cli_arguments() -> Tuple[BenchmarkArguments, GenerationConfig]:
    parser = transformers.HfArgumentParser((BenchmarkArguments, GenerationConfig))
    (
        benchmark_arguments,
        generation_config,
        _remaining,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    return benchmark_arguments, generation_config
