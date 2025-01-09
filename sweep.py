# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict, List, Optional, Tuple
import pandas as pd
import transformers
from datetime import datetime
import os
import tabulate
import torch
from dataclasses import dataclass
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

import arguments
from arguments import Arguments, simple_parse_args_string
from benchmark import benchmark, load_model_and_tokenizer, process_cli_arguments, setup, BenchmarkArguments
from self_speculation.generator_base import (
    GenerationConfig,
)

@dataclass
class SweepArguments:
    exit_layer_first: Optional[int] = 1
    exit_layer_last: Optional[int] = 15
    exit_layer_step: Optional[int] = 1
    num_speculations_first: Optional[int] = 1
    num_speculations_last: Optional[int] = 6
    num_speculations_step: Optional[int] = 1

def sweep(args: Arguments, benchmark_arguments: BenchmarkArguments, generation_config: GenerationConfig, sweep_arguments: SweepArguments):
    results: List[Dict] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    model, tokenizer = load_model_and_tokenizer(args, device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_fname = f"{args.output_dir}/sweep_{timestamp}.csv"
    pdf_fname = f"{args.output_dir}/sweep_{timestamp}.pdf"

    for exit_layer in range(sweep_arguments.exit_layer_first, sweep_arguments.exit_layer_last+1, sweep_arguments.exit_layer_step):
        for num_speculations in range(sweep_arguments.num_speculations_first, sweep_arguments.num_speculations_last+1, sweep_arguments.num_speculations_step):
            generation_config.exit_layer = exit_layer
            generation_config.num_speculations = num_speculations

            metric_result = benchmark(model, tokenizer, benchmark_arguments, generation_config, args.seed)

            results.append({
                "exit_layer": exit_layer,
                "num_speculations": num_speculations,
                "acceptance_rate": metric_result['acceptance_rate']['mean'],
                "total_time": metric_result['total_time']['mean'],
                "time_per_token": metric_result['time_per_token']['mean'],
                "tokens_per_second": metric_result['tokens_per_second']['mean'],
            })
            df = pd.DataFrame(results) 
            # Update table every iteration
            df.to_csv(csv_fname, index=False)
            print(f"exit_layer: {exit_layer}, num_speculations: {num_speculations}, time_per_token: {metric_result['time_per_token']['mean']}")

    # Print summary table
    print("\n")
    header = results[0].keys()
    rows =  [x.values() for x in results]
    print(tabulate.tabulate(rows, header))

    # Plot contour plot
    plot_contour(df, pdf_fname)

def plot_contour(df, pdf_fname):
    ## Prepare grid coordinates (assuming exit_layer and num_speculations are integer indices)
    grid_x, grid_y = np.mgrid[df['exit_layer'].min():df['exit_layer'].max():100j,
                            df['num_speculations'].min():df['num_speculations'].max():100j]
    ## Interpolate missing data
    grid_z = griddata((df['exit_layer'], df['num_speculations']), df['tokens_per_second'],
                    (grid_x, grid_y), method='linear')
    ## Create the contour plot
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis')
    plt.colorbar(contour)
    ## Overlay the data points
    plt.scatter(df['exit_layer'], df['num_speculations'], color='black', s=25, zorder=5)
    plt.title('Tokens Per Second')
    plt.xlabel('Exit Layer')
    plt.ylabel('Number of Speculations')
    ## Save the plot
    plt.savefig(pdf_fname, format="pdf", dpi=300)
    ## Show the plot
    plt.show()

def process_cli_arguments() -> Tuple[arguments.Arguments, BenchmarkArguments, GenerationConfig, SweepArguments]:
    parser = transformers.HfArgumentParser((arguments.Arguments, BenchmarkArguments, GenerationConfig, SweepArguments))
    (
        general_arguments,
        benchmark_arguments,
        generation_config,
        sweep_arguments,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=False)

    if general_arguments.model_args:
        general_arguments.model_args = simple_parse_args_string(general_arguments.model_args)
    else:
        general_arguments.model_args = {}

    return general_arguments, benchmark_arguments, generation_config, sweep_arguments

if __name__ == "__main__":
    args, benchmark_arguments, generation_config, sweep_arguments = process_cli_arguments()
    sweep(args, benchmark_arguments, generation_config, sweep_arguments)
