import datetime
import os
import random

import aix
import torch

# @manual=fbsource//third-party/pypi/transformers:transformers
import transformers

# @manual=//deeplearning/projects/core_ai:core_ai
from core_ai.utils import utils

from fblearner.flow.projects.assistant.llms.trainer.core.workflow_tc import (
    WorkflowTCRunner,
)

from fblearner.flow.projects.assistant.llms.trainer.projects.layerskip.arguments import (
    process_cli_arguments,
)

from fblearner.flow.projects.assistant.llms.trainer.projects.layerskip.data import (
    get_data,
)

from fblearner.flow.projects.assistant.llms.trainer.projects.layerskip.self_speculation.autoregressive_generator import (
    AutoRegressiveGenerationStrategy,
)

from fblearner.flow.projects.assistant.llms.trainer.projects.layerskip.self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    HuggingfaceLlamaGenerator,
)

from fblearner.flow.projects.assistant.llms.trainer.projects.layerskip.self_speculation.self_speculation_generator import (
    SelfSpeculativeGenerationStrategy,
)
from tqdm import tqdm

log = utils.get_logger(__name__)


def main():
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl", timeout=datetime.timedelta(hours=48)
    )
    rank = int(os.environ["LOCAL_RANK"])
    benchmark_arguments, generation_config = process_cli_arguments()
    random.seed(benchmark_arguments.seed)
    torch.manual_seed(benchmark_arguments.seed)

    if rank != 0:
        # only run on rank 0, we don't support parallel inference yet
        return

    pathmgr = utils.get_path_manager()
    local_model_path: str = pathmgr.get_local_path(benchmark_arguments.model_path)

    # initialize model
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        local_model_path, use_fast=False
    )
    config = transformers.LlamaConfig.from_pretrained(local_model_path)
    model = transformers.LlamaForCausalLM.from_pretrained(
        local_model_path,
        config=config,
        torch_dtype=torch.float16,
    )
    model.cuda()
    model.half()
    model.eval()

    # initialize generator
    spec_generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer,
        model=model,
        generation_strategy=SelfSpeculativeGenerationStrategy(),
    )

    ar_generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer,
        model=model,
        generation_strategy=AutoRegressiveGenerationStrategy(),
    )

    evaluation_set = get_data(
        data_path=benchmark_arguments.data_path,
        random_shuffle=benchmark_arguments.random_shuffle,
        num_samples=benchmark_arguments.num_samples,
        data_format=benchmark_arguments.data_format,
    )

    errors: int = 0
    for i, example in enumerate(tqdm(evaluation_set)):
        aix.set_run_progress(i, total_steps=len(evaluation_set))
        spec_response: GenerationResult = spec_generator.generate(
            prompt=example.input,
            generation_config=generation_config,
        )
        ar_response: GenerationResult = ar_generator.generate(
            prompt=example.input,
            # generation config to use the full model
            generation_config=GenerationConfig(
                max_steps=generation_config.max_steps,
                exit_layer=-1,
                num_speculations=-1,
                generation_strategy="autoregressive",
            ),
        )

        if spec_response.decoded_prediction != ar_response.decoded_prediction:
            errors += 1
            log.info("Error found")
            log.info(f"Spec response: {spec_response}")
            log.info(f"AR response: {ar_response}")

    metric_result = {"errors": errors, "error_pct": errors / len(evaluation_set)}
    print(metric_result)
    WorkflowTCRunner.upload_output_to_manifold(
        folder_path=benchmark_arguments.manifold_output_dir,
        output=metric_result,
    )


if __name__ == "__main__":
    main()
