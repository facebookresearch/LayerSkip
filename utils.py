from torchmetrics.metric import Metric
from torchmetrics.wrappers.abstract import WrapperMetric
from torchmetrics.text import ROUGEScore

from typing import Any
from torch import Tensor

class ROUGEScoreWrapper(WrapperMetric):
    def __init__(
        self,
        base_metric: ROUGEScore,
        score: str = "fmeasure",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(base_metric, ROUGEScore):
            raise ValueError(
                f"Expected base metric to be an instance of `torchmetrics.Metric` but received {type(base_metric)}"
            )
        if len(base_metric.rouge_keys) != 1:
            raise NotImplementedError(
                f"ROUGEScoreWrapper is only implemented to wrap a ROUGEScore with 1 rouge key but instead got {len(base_metric.rouge_keys)} keys."
            )
        self._base_metric = base_metric
        self._score = score

    def compute(self) -> Tensor:
        return self._base_metric.compute()[f"{self._base_metric.rouge_keys[0]}_{self._score}"]

    def update(
        self, 
        *args: Any,
        **kwargs: Any,
    ) -> None:
        return self._base_metric.update(*args, **kwargs)
