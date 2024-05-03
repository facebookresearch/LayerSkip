import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

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


def LowercaseProcessingFunction(input: str) -> str:
    return input.lower()


def prepare_evaluation_examples_chat_format(data_path: str) -> List[EvaluationExample]:
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
                evaluation_data_points.append(
                    EvaluationExample(
                        input=stringify_conversation(json_line["data"][1:i])
                        + "\n[PARSER]\n",
                        output=stringify_conversation([json_line["data"][i]]),
                    )
                )
            i += 1
    return evaluation_data_points


def prepare_cnn_dm_lm_format(data_path: str) -> List[EvaluationExample]:
    evaluation_data_points = []
    for line in open(data_path):
        dataset = json.loads(line)
        for _i, data_point in enumerate(dataset):
            words = data_point["article"].split()
            evaluation_data_points.append(
                EvaluationExample(
                    input=" ".join(words[:PREFIX_LENGTH]),
                    output=" ".join(words[PREFIX_LENGTH:]),
                )
            )
        return evaluation_data_points


def prepare_cnn_dm_summarization_format(data_path: str) -> List[EvaluationExample]:
    evaluation_data_points = []
    for line in open(data_path):
        dataset = json.loads(line)
        for _i, data_point in enumerate(dataset):
            article = data_point["article"]
            highlights = data_point["highlights"]
            evaluation_data_points.append(
                EvaluationExample(
                    input=f"Article: {article}\nSummary:",
                    output=f" {highlights}",
                )
            )
        return evaluation_data_points


def get_data(
    data_path: str,
    random_shuffle: bool,
    num_samples: int,
    data_format: str,
) -> List[EvaluationExample]:
    if data_format == DatasetFormat.CHAT_FORMAT:
        evaluation_data_points = prepare_evaluation_examples_chat_format(data_path)
    elif data_format == DatasetFormat.CNN_DM_SUMMARIZATION:
        evaluation_data_points = prepare_cnn_dm_summarization_format(data_path)
    elif data_format == DatasetFormat.CNN_DM_LM:
        evaluation_data_points = prepare_cnn_dm_lm_format(data_path)
    else:
        raise NotImplementedError(f"Unknown dataset format {data_format}")

    if random_shuffle:
        random.shuffle(evaluation_data_points)

    if num_samples:
        evaluation_data_points = evaluation_data_points[:num_samples]

    return evaluation_data_points
