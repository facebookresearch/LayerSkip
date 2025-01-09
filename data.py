# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset
import pandas as pd

# for language modeling problems how long to use the prefix as
PREFIX_LENGTH: int = 100


@dataclass
class EvaluationExample:
    input: str
    output: str


class DatasetFormat:
    CHAT_FORMAT: str = "chat_format"
    CNN_DM_SUMMARIZATION: str = "cnn_dm_summarization"
    CNN_DM_LM: str = "cnn_dm_lm"
    XSUM_SUMMARIZATION: str = "xsum_summarization"
    HUMAN_EVAL: str = "human_eval"
    CUSTOM_JSONL: str = "custom_jsonl"
    TOP_V2: str = "top_v2"


def get_valid_dataset_formats():
    # Extract the values of class attributes, excluding internal dunder methods
    return [value for key, value in DatasetFormat.__dict__.items() if not key.startswith('__')]

def apply_template(message:str, template:str) -> str:
    """
    Applies a template to a given message.
    
    Parameters:
        message (str): The message to insert into the template.
        template (str): The template with a placeholder for the message in `{message}`.
        
    Returns:
        str: The formatted message with the template applied.
    """
    if template is None:
        return message
    return template.format(message=message) 


def LowercaseProcessingFunction(input: str) -> str:
    return input.lower()


# TODO: fix or remove TOPv2 benchmarking
def prepare_evaluation_examples_chat_format(data_path: str, template: str = None) -> List[EvaluationExample]:
    SINGLE_TURN_TEMPLATE: str = "\n[{role}]\n{message}\n[/{role}]"
    evaluation_data_points = []

    def stringify_conversation(conversation: List[Dict[str, str]]) -> str:
        return "".join(
            [
                SINGLE_TURN_TEMPLATE.format(role=x["role"], message=x["message"])
                for x in conversation
            ]
        )

    for line in open(data_path):
        json_line = json.loads(line)
        i: int = 0
        while i < len(json_line["data"]):
            if json_line["data"][i]["role"] == "PARSER":
                prompt = apply_template(message=stringify_conversation(json_line["data"][1:i]) + "\n[PARSER]\n", 
                                        template=template) 
                evaluation_data_points.append(
                    EvaluationExample(
                        input=prompt,
                        output=stringify_conversation([json_line["data"][i]]),
                    )
                )
            i += 1
    return evaluation_data_points


def prepare_cnn_dm_lm_format(template: str = None) -> List[EvaluationExample]:
    evaluation_data_points = []
    for data_point in load_dataset("cnn_dailymail", "3.0.0")["test"]:
        words = data_point["article"].split()
        prompt = apply_template(message=" ".join(words[:PREFIX_LENGTH]), template=template)
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=" ".join(words[PREFIX_LENGTH:]),
            )
        )
    return evaluation_data_points


def prepare_cnn_dm_summarization_format(n_shot: int = 0, seed: int = 42, template: str = None) -> List[EvaluationExample]:
    prompt_shots = ""
    if n_shot > 0:
        prompt_keys=["article", "highlights"]
        shots = load_dataset("cnn_dailymail", name="3.0.0", split="train").shuffle(seed=seed).select(range(n_shot))
        for i in range(n_shot):
            prompt = "Article: " + shots[i][prompt_keys[0]] + "\nSummary: " + shots[i][prompt_keys[1]].replace("\n", "") + "\n"
            prompt_shots += prompt
        prompt_shots += "\n"

    evaluation_data_points = []
    for data_point in load_dataset("cnn_dailymail", name="3.0.0", split="test"):
        article = data_point["article"]
        highlights = data_point["highlights"]
        prompt = apply_template(message=prompt_shots + f"Article: {article}\nSummary:", template=template) 
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=f" {highlights}",
            )
        )
    return evaluation_data_points

def prepare_xsum_summarization_format(n_shot: int = 0, seed: int = 42, template: str = None) -> List[EvaluationExample]:
    prompt_shots = ""
    if n_shot > 0:
        prompt_keys=["document", "summary"]
        shots = load_dataset("xsum", split="train").shuffle(seed=seed).select(range(n_shot))
        for i in range(n_shot):
            prompt = "Article: " + shots[i][prompt_keys[0]] + "\nSummary: " + shots[i][prompt_keys[1]].replace("\n", "") + "\n"
            prompt_shots += prompt
        prompt_shots += "\n"

    evaluation_data_points = []
    for data_point in load_dataset('xsum', split='test'):
        article = data_point["document"]
        highlights = data_point["summary"]
        prompt = apply_template(message=prompt_shots + f"Article: {article}\nSummary:", template=template) 
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=f" {highlights}",
            )
        )
    return evaluation_data_points

def prepare_human_eval(template: str = None) -> List[EvaluationExample]:
    evaluation_data_points = []
    for data_point in load_dataset('openai_humaneval', split='test'):
        prompt = apply_template(message=data_point["prompt"], template=template) 
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=data_point["canonical_solution"],
            )
        )
    return evaluation_data_points

def prepare_top_v2(template: str = None) -> List[EvaluationExample]:
    evaluation_data_points = []
    for data_point in load_dataset('WillHeld/top_v2', split='test'):
        # apply template if it exists
        prompt = apply_template(message=data_point["utterance"], template=template)
        evaluation_data_points.append(
            EvaluationExample(
               input= prompt,
                output=data_point["semantic_parse"],
            )
        )
    return evaluation_data_points

def prepare_custom(data_path: str, prompt_field: str = "prompt", response_field: str = "response", template: str = None) -> List[EvaluationExample]:
    evaluation_data_points = []
    for _, data_point in pd.read_json(data_path, lines=True).iterrows():
        prompt = apply_template(message=data_point[prompt_field], template=template)  
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=data_point[response_field],
            )
        )
    return evaluation_data_points

def get_data(
    random_shuffle: bool,
    num_samples: int,
    dataset: str,
    data_path: Optional[str] = None,
    n_shot: int = 0,
    seed: int = 42,
    prompt_field: str = "prompt",
    response_field: str = "response",
    template: str = None
) -> List[EvaluationExample]:
    if dataset == DatasetFormat.CHAT_FORMAT:
        evaluation_data_points = prepare_evaluation_examples_chat_format(data_path, template=template)
    elif dataset == DatasetFormat.CNN_DM_SUMMARIZATION:
        evaluation_data_points = prepare_cnn_dm_summarization_format(n_shot=n_shot, seed=seed, template=template)
    elif dataset == DatasetFormat.XSUM_SUMMARIZATION:
        evaluation_data_points = prepare_xsum_summarization_format(n_shot=n_shot, seed=seed, template=template)
    elif dataset == DatasetFormat.CNN_DM_LM:
        evaluation_data_points = prepare_cnn_dm_lm_format(template)
    elif dataset == DatasetFormat.HUMAN_EVAL:
        evaluation_data_points = prepare_human_eval(template)
    elif dataset == DatasetFormat.CUSTOM_JSONL:
        evaluation_data_points = prepare_custom(data_path, prompt_field=prompt_field, 
                                                response_field=response_field, template=template)
    elif dataset == DatasetFormat.TOP_V2:
        evaluation_data_points = prepare_top_v2(template)
    else:
        raise NotImplementedError(f"Unknown dataset format {dataset}")

    if random_shuffle:
        random.shuffle(evaluation_data_points)

    if num_samples:
        evaluation_data_points = evaluation_data_points[:num_samples]

    return evaluation_data_points
