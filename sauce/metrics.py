import logging
from typing import Any, List, Optional, Dict

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
from torchmetrics import Accuracy, ConfusionMatrix, MeanSquaredError, Metric
from torchmetrics.classification import BinaryFairness
from hydra.utils import instantiate

from sauce.utils.dcor import DistanceCorrelation
from sauce.utils.cdcor import ConditionalDistanceCorrelation


class DistanceCorrelationMetric(Metric):
    def __init__(self, num_classes: int | None, bandwidth: torch.Tensor | float | None, device: str, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        if num_classes is None:
            assert bandwidth is not None
            self.dcor = ConditionalDistanceCorrelation(bandwidth=bandwidth)
            self.add_state("cdcor_total", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum")
            self.add_state("count", default=torch.tensor(0, device=device), dist_reduce_fx="sum")
        elif num_classes == 1:
            self.dcor = DistanceCorrelation()
            for i in range(num_classes + 1):
                self.add_state(f"dcor_{i}", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum")
                self.add_state(f"count_{i}", default=torch.tensor(0, device=device), dist_reduce_fx="sum")
        elif num_classes > 1:
            self.dcor = DistanceCorrelation()
            for i in range(num_classes):
                self.add_state(f"dcor_{i}", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum")
                self.add_state(f"count_{i}", default=torch.tensor(0, device=device), dist_reduce_fx="sum")
        else:
            raise ValueError(f"Unknown task")

    def update(self, preds: torch.Tensor, target: torch.Tensor, distractors: torch.Tensor):
        if self.num_classes is None:
            cdcor = self.dcor(preds, target, distractors)
            self.cdcor_total += cdcor
            self.count += 1
        elif self.num_classes == 1:
            for i in range(self.num_classes + 1):
                mask = target == i
                if mask.sum() > 2:
                    dcor = self.dcor(preds[mask], distractors[mask])
                    dcor_i = getattr(self, f"dcor_{i}")
                    count_i = getattr(self, f"count_{i}")
                    dcor_i += dcor
                    count_i += 1

                    setattr(self, f"dcor_{i}", dcor_i)
                    setattr(self, f"count_{i}", count_i)

        # todo: target needs to be correctly encoded
        elif self.num_classes > 1:
            for i in range(self.num_classes):
                mask = target == i
                if mask.sum() > 2:
                    dcor = self.dcor(preds[mask], distractors[mask])
                    dcor_i = getattr(self, f"dcor_{i}")
                    count_i = getattr(self, f"count_{i}")
                    dcor_i += dcor
                    count_i += 1

                    setattr(self, f"dcor_{i}", dcor_i)
                    setattr(self, f"count_{i}", count_i)

    def compute(self):
        if self.num_classes is None:
            return {"total": self.cdcor_total / self.count}
        elif self.num_classes == 1:
            return {f"{i}": getattr(self, f"dcor_{i}") / getattr(self, f"count_{i}") for i in range(self.num_classes + 1)}
        elif self.num_classes > 1:
            return {f"{i}": getattr(self, f"dcor_{i}") / getattr(self, f"count_{i}") for i in range(self.num_classes)}

    def reset(self):
        if self.num_classes is None:
            device = self.cdcor_total.device
            self.cdcor_total = torch.tensor(0.0, device=device)
            self.count = torch.tensor(0, device=device)
        elif self.num_classes == 1:
            device = getattr(self, f"dcor_0").device
            for i in range(self.num_classes + 1):
                setattr(self, f"dcor_{i}", torch.tensor(0.0, device=device))
                setattr(self, f"count_{i}", torch.tensor(0, device=device))
        elif self.num_classes > 1:
            device = getattr(self, f"dcor_0").device
            for i in range(self.num_classes):
                setattr(self, f"dcor_{i}", torch.tensor(0.0, device=device))
                setattr(self, f"count_{i}", torch.tensor(0, device=device))
