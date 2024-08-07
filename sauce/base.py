import logging
from typing import Any, List, Optional, Dict

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
from torchmetrics import Accuracy, ConfusionMatrix, MeanSquaredError, Metric, R2Score
from torchmetrics.classification import BinaryFairness, BinaryAUROC
from hydra.utils import instantiate
from sauce.metrics import DistanceCorrelationMetric

LOG = logging.getLogger(__name__)


class BalancedAccuracy(Metric):
    def __init__(self, task: str, num_classes: int, *args, **kwargs):
        super().__init__()
        self.cmat = ConfusionMatrix(task=task, num_classes=num_classes)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.cmat.update(preds, target)

    def compute(self):
        cmat = self.cmat.compute()
        per_class = cmat.diag() / cmat.sum(dim=1)
        per_class = per_class[~torch.isnan(per_class)]  # remove classes that are not present in this split
        LOG.debug("Confusion matrix:\n%s", cmat)

        return per_class.mean()

    def reset(self):
        self.cmat.reset()


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        distractors: dict,
        num_classes: Optional[int],
    ) -> None:
        super().__init__()

        self.net = net
        self.task = "regression"
        self.num_classes = num_classes
        self.distractors = distractors
        if num_classes is not None:
            self.task = "binary" if num_classes <= 2 else "multiclass"
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        if self.task == "regression":
            self.train_metrics = nn.ModuleDict({
                "mse": MeanSquaredError(),
            })
            self.val_metrics = nn.ModuleDict({
                "mse": MeanSquaredError(),
                "r2": R2Score(),
            })
            self.test_metrics = nn.ModuleDict({
                "mse": MeanSquaredError(),
                "r2": R2Score(),
            })
            # TODO more metrics
        else:
            self.train_metrics = nn.ModuleDict({
                "acc": Accuracy(task=self.task, num_classes=num_classes),
                "bacc": BalancedAccuracy(task=self.task, num_classes=num_classes),
            })
            self.val_metrics = nn.ModuleDict({
                "acc": Accuracy(task=self.task, num_classes=num_classes),
                "bacc": BalancedAccuracy(task=self.task, num_classes=num_classes),
                "dcor": DistanceCorrelationMetric(num_classes=num_classes, bandwidth=None, device=self.device),
                "auroc": BinaryAUROC(),
            })
            self.test_metrics = nn.ModuleDict({
                "acc": Accuracy(task=self.task, num_classes=num_classes),
                "bacc": BalancedAccuracy(task=self.task, num_classes=num_classes),
                "dcor": DistanceCorrelationMetric(num_classes=num_classes, bandwidth=None, device=self.device),
                "auroc": BinaryAUROC(),
            })
            if len(self.distractors) == 1:
                for name, distractor in self.distractors.items():
                    distractor_type = distractor["distractor_type"]
                    if distractor_type == "binary":
                        self.val_metrics["eq_opp"] =  BinaryFairness(2, task='equal_opportunity', threshold=0.5, validate_args=True)
                        self.test_metrics["eq_opp"] =  BinaryFairness(2, task='equal_opportunity', threshold=0.5, validate_args=True)

    def _log_metrics(self, metrics: Dict, prefix: str) -> None:
        for name, func in metrics.items():
            result: torch.Tensor | dict = func.compute()

            if isinstance(func, DistanceCorrelationMetric):
                for k, v in result.items():
                    self.log(f"{prefix}/{name}_{k}", v)
                    print(f"{prefix}/{name}_{k}: {v}")

            elif isinstance(result, dict):
                assert len(result.keys()) == 1
                result = list(result.values())[0]
                self.log(f"{prefix}/{name}", result)

            else:
                self.log(f"{prefix}/{name}", result)
                print(result)

            func.reset()

    def _update_metrics(self, metrics: Dict[str,Metric], preds: torch.Tensor, targets: torch.Tensor,
                        distractors: torch.Tensor) -> None:
        for func in metrics.values():
            if isinstance(func, DistanceCorrelationMetric):
                func.update(preds, targets, distractors)
            elif isinstance(func, BinaryFairness):
                func.update(preds, targets, distractors.long())
            else:
                func.update(preds, targets)

    def _log_train_metrics(
        self, loss: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor,
    ) -> None:
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, func in self.train_metrics.items():
            result = func(preds, targets)
            self.log(f"train/{name}", result, on_step=False, on_epoch=True, prog_bar=True)

    def _update_validation_metrics(
        self, loss: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, distractors: torch.Tensor,
    ) -> None:
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self._update_metrics(self.val_metrics, preds, targets, distractors)

    def _log_validation_metrics(self) -> None:
        self._log_metrics(self.val_metrics, "val")

    def _update_test_metrics(
        self, loss: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, distractors: torch.Tensor,
    ) -> None:
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self._update_metrics(self.test_metrics, preds, targets, distractors)

    def _log_test_metrics(self) -> None:
        self._log_metrics(self.test_metrics, "test")

    def on_train_start(self):

        if isinstance(self.logger, TensorBoardLogger):
            tb_logger = self.logger.experiment
            tb_logger.flush()

        if self.task == "regression":
            bandwidth: torch.Tensor = self.trainer.datamodule.get_target_bandwidth()
            bandwidth = bandwidth.to(self.device)
            print(f"Bandwidth: {bandwidth}")
            self.val_metrics["dcor"] = DistanceCorrelationMetric(num_classes=self.num_classes, bandwidth=bandwidth, device=self.device)

    def on_train_epoch_end(self):
        for func in self.train_metrics.values():
            func.reset()
        
        # update schedulers if not automatic optimization
        if not self.automatic_optimization:
            schedulers = self.lr_schedulers()
            # check if schedulers is iterable
            if schedulers:
                if hasattr(schedulers, "__iter__"):
                    for s in self.lr_schedulers():
                        self.lr_scheduler_step(s, metric=None)
                else:
                    self.lr_scheduler_step(schedulers, metric=None)


    def on_validation_epoch_end(self):
        for func in self.val_metrics.values():
            func.reset()

    def on_test_epoch_end(self):
        for func in self.test_metrics.values():
            func.reset()

    def on_test_start(self) -> None:
        print("Im in on test start hook")
        for func in self.test_metrics.values():
            func.reset()

    def on_validation_start(self) -> None:
        print("Im in on validation start hook")
        for func in self.val_metrics.values():
            func.reset()

    def forward(self, x, y=None):
        return self.net(x, y)
