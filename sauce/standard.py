import logging
from typing import Any, List, Dict, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.linalg import solve as scp_solve
import torch

from sauce.base import BaseModule
from sauce.utils.circe_utils import circe_estimate, gaussian_kernel, leave_one_out_reg, leave_one_out_reg_kernels, hscic
from sauce.utils.cdcor import ConditionalDistanceCorrelation

LOG = logging.getLogger(__name__)

STANDARDARGS = [
    "net",
    "lr",
    "num_classes",
    "wd",
    "distractors"]

def split_batch_by_target(batch: List) -> List:
        x, targets, z = batch
        outputs = []
        for cls in targets.unique():
            idx = targets == cls
            idx = idx.squeeze()
            outputs.append((x[idx], targets[idx], z[idx]))
        return outputs


def split_target_z_feats_by_target(target: torch.Tensor, z: torch.Tensor, feats: List[torch.Tensor]) -> List:
        outputs = []
        for cls in target.unique():
            idx = target == cls
            idx = idx.squeeze()
            outputs.append((target[idx], z[idx], [feat[idx] for feat in feats]))
        return outputs


def get_indices_per_target(batch: List) -> List:
        _, targets, _ = batch
        indices = []
        for cls in targets.unique():
            idx = targets == cls
            indices.append(idx)
        return indices

def split_batch(batch: List, idx: torch.Tensor) -> List:
        if idx.ndim > 1:
            idx = idx.squeeze()
        x, targets, z = batch
        return x[idx], targets[idx], z[idx]


class StandardModule(BaseModule):
    def __init__(
        self,
        net,
        distractors: dict,
        lr: float = 0.001,
        num_classes: int | None = 3,
        wd: float = 0.0005,
        ce_weights: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(
            net=net,
            distractors=distractors,
            num_classes=num_classes,
        )
        self.save_hyperparameters(logger=False)

        # loss function
        if num_classes is None:
            self.criterion = torch.nn.MSELoss()
        elif num_classes > 2:
            # stability for imbalanced classes
            ce_weights = torch.tensor(ce_weights) if ce_weights is not None else None
            self.criterion = torch.nn.CrossEntropyLoss(weight=ce_weights)
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()

    def step(self, batch: List[torch.Tensor]):
        x, y, z = batch
        features: torch.Tensor = self.forward(x)
        logits = features[-1]
        loss = 0
        if self.task == "regression":
            preds = logits
            loss = self.criterion(preds.squeeze(), y.squeeze())
        elif self.task == "binary":
            preds = logits
            loss = self.criterion(preds.squeeze(), y.squeeze())
        elif self.task == "multiclass":
            preds = logits
            loss = self.criterion(logits, y)
        else:
            raise ValueError(f"Unknown task: {self.task} for network outputs {features[-1].size()}")

        return loss, preds, y, z, features

    def training_step(self, batch: List[torch.Tensor], batch_idx: int):
        loss, preds, targets, z, features = self.step(batch)

        # log train metrics
        self._log_train_metrics(loss, preds, targets)

        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets, "disctractors": z}

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int):
        loss, preds, targets, z, features = self.step(batch)

        self._update_validation_metrics(loss, preds, targets, z)

        return {"loss": loss, "preds": preds, "targets": targets, "distractors": z}

    def on_validation_epoch_end(self):
        self._log_validation_metrics()

    def test_step(self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        loss, preds, targets, z, features = self.step(batch)

        self._update_test_metrics(loss, preds, targets, z)

        return {"loss": loss, "preds": preds, "targets": targets, "distractors": z}

    def on_test_epoch_end(self):
        self._log_test_metrics()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.AdamW(
            params=[p for p in self.net.parameters() if p.requires_grad], lr=self.hparams.lr, weight_decay=self.hparams.wd
        )
        # n_iters = len(self.trainer.datamodule.train_dataloader())
        max_epochs = self.trainer.max_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
             optimizer, T_max=max_epochs
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


class CIRCE(StandardModule):
    def __init__(self,  **kwargs):
        standardargs = {k: kwargs.pop(k) for k in STANDARDARGS}
        super().__init__(**standardargs)
        self.kwargs = kwargs

    def _leave_one_out_regressors(self, reg_list, sigma2_list, Kz):
        LOO_error_sanity_check = np.zeros((len(sigma2_list), len(reg_list)))
        LOO_error = np.zeros((len(sigma2_list), len(reg_list)))
        LOO_tol = np.zeros(len(sigma2_list))
        for idx, sigma2 in enumerate(sigma2_list):
            print(idx, sigma2)
            self.kwargs['kernel_y']['sigma2'] = sigma2
            loo_size = self.targets_heldout.shape[0] // 2
            K_YY = gaussian_kernel(self.targets_heldout, **self.kwargs['kernel_y'])
            LOO_error_sanity_check[idx] = leave_one_out_reg(K_YY[:loo_size, :loo_size].cpu().numpy(),
                                                            labels=Kz[:loo_size, loo_size:].cpu().numpy(),
                                                            reg_list=reg_list)
            LOO_error[idx], under_tol, LOO_tol[idx] = \
                leave_one_out_reg_kernels(K_YY.cpu().numpy(), Kz.cpu().numpy(), reg_list)
            # if not np.any(under_tol)
            LOO_error[idx, under_tol] = 2.0 * LOO_error[idx].max()  # a hack to remove < tol lambdas
        LOO_idx = np.unravel_index(np.argmin(LOO_error, axis=None), LOO_error.shape)
        self.kwargs['kernel_y']['sigma2'] = sigma2_list[LOO_idx[0]]
        self.ridge_lambda = reg_list[LOO_idx[1]]
        print('Best LOO parameters: sigma2 {}, lambda {}'.format(sigma2_list[LOO_idx[0]], reg_list[LOO_idx[1]]))
        if self.ridge_lambda < LOO_tol[LOO_idx[0]]:
            print('POORLY CONDITIONED MATRIX, switching lambda to SVD tolerance: {}'.format(LOO_tol[LOO_idx[0]]))
            self.ridge_lambda = LOO_tol[LOO_idx[0]]

        LOO_idx = np.unravel_index(np.argmin(LOO_error_sanity_check, axis=None), LOO_error_sanity_check.shape)
        print('Best LOO parameters (sanity check): sigma2 {}, lambda {}'.format(
            sigma2_list[LOO_idx[0]], reg_list[LOO_idx[1]]))

        print('LOO results\n{}'.format(LOO_error))
        print('LOO results (sanity check)\n{}'.format(LOO_error_sanity_check))

    def _regress_yz(self):
        '''
        Create a held-out set and learn a linear regressor from Y to Z on it.
        '''
        # del self.trainer.datamodule.train_data.linear_reg
        targets, distractors = self.trainer.datamodule.get_circe_data()
        self.targets_heldout = targets #torch.FloatTensor(targets.numpy().reshape(-1, 1))
        self.distractors_heldout = distractors #torch.FloatTensor(distractors.numpy().reshape(-1, 1))

        n_points = self.targets_heldout.shape[0]
        Kz = gaussian_kernel(self.distractors_heldout, **self.kwargs['kernel_z'])

        if self.kwargs['loo_cond_mean']:
            print('Estimating regressions parameters with LOO')
            reg_list = [1e-2, 1e-1, 1.0, 10.0, 100.0]
            sigma2_list = [1.0, 0.1, 0.01, 0.001]
            self._leave_one_out_regressors(reg_list, sigma2_list, Kz)

        Ky = gaussian_kernel(self.targets_heldout, **self.kwargs['kernel_y'])
        I = torch.eye(n_points, device=Ky.device)
        print('All gram matrices computed')
        W_all = torch.tensor(scp_solve(np.float128((Ky + self.kwargs['ridge_lambda'] * I).cpu().numpy()),
                                            np.float128(torch.cat((I, Kz), 1).cpu().numpy()),
                                            assume_a='pos')).float().to(Ky.device)
        print('W_all computed')
        self.W_1 = W_all[:, :n_points].to(self.device)
        self.W_2 = W_all[:, n_points:].to(self.device)

        self.targets_heldout = self.targets_heldout.to(self.device)
        self.distractors_heldout = self.distractors_heldout.to(self.device)

    def on_train_start(self):
        self._regress_yz()
        super().on_train_start()

    def regularize_feats(self, feats, z, targets):
        loss = 0
        for feat in feats[-self.kwargs['n_last_reg_layers']:]:
            loss += self.kwargs['circe_lambda'] * circe_estimate(feat, z, self.distractors_heldout,
                            targets, self.targets_heldout, self.W_1, self.W_2,
                            "gaussian", self.kwargs['kernel_ft'],
                            "gaussian", self.kwargs['kernel_z'], "gaussian", self.kwargs['kernel_y'],
                            self.kwargs['biased'], cond_cov=not self.kwargs['centered_circe'])
        return loss / self.kwargs['n_last_reg_layers']

    def training_step(self, batch: Any, batch_idx: int):
        targetloss, preds, targets, z, all_features = self.step(batch)

        # circe regularizer
        if self.kwargs['n_last_reg_layers'] == -1 or self.kwargs['n_last_reg_layers'] > len(all_features):
                self.kwargs['n_last_reg_layers'] = len(all_features)
        circe_reg = self.regularize_feats(all_features, z, targets)

        self.log('train/circe', circe_reg.item())
        self.log('train/targetloss', targetloss.item())
        loss = targetloss + circe_reg

        # log train metrics
        self._log_train_metrics(loss, preds, targets)

        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets, "distractors": z}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, z, all_features = self.step(batch)

        loss += self.regularize_feats(all_features, z, targets)

        self._update_validation_metrics(loss, preds, targets, z)

        return {"loss": loss, "preds": preds, "targets": targets, "distractors": z}


class HSCIC(StandardModule):
    def __init__(self,  **kwargs):
        standardargs = {k: kwargs.pop(k) for k in STANDARDARGS}
        super().__init__(**standardargs)
        self.kwargs = kwargs
        self.LOO_done = False

    def _leave_one_out_regressors(self, reg_list, sigma2_list, Kz, y):
        LOO_error = np.zeros((len(sigma2_list), len(reg_list)))
        LOO_tol = np.zeros(len(sigma2_list))
        for idx, sigma2 in enumerate(sigma2_list):
            print(idx, sigma2)
            self.kwargs['kernel_y']['sigma2'] = sigma2
            K_YY = gaussian_kernel(y, **self.kwargs['kernel_y'])
            LOO_error[idx], under_tol, LOO_tol[idx] = leave_one_out_reg_kernels(K_YY.cpu().numpy(), Kz.cpu().numpy(), reg_list)
            LOO_error[idx, under_tol] = 2.0 * LOO_error[idx].max()  # a hack to remove < tol lambdas
        LOO_idx = np.unravel_index(np.argmin(LOO_error, axis=None), LOO_error.shape)
        self.kwargs['kernel_y']['sigma2'] = sigma2_list[LOO_idx[0]]
        self.kwargs['ridge_lambda'] = reg_list[LOO_idx[1]]
        print('Best LOO parameters: sigma2 {}, lambda {}'.format(sigma2_list[LOO_idx[0]], reg_list[LOO_idx[1]]))
        if self.kwargs['ridge_lambda'] < LOO_tol[LOO_idx[0]]:
            print('POORLY CONDITIONED MATRIX, switching lambda to SVD tolerance: {}'.format(LOO_tol[LOO_idx[0]]))
            self.kwargs['ridge_lambda'] = LOO_tol[LOO_idx[0]]

        print('LOO results\n{}'.format(LOO_error))

    def reg_hscic(self, all_features, z, y):
        loss = 0
        for int_ft in all_features[-self.kwargs['n_last_reg_layers']:]:
            loss += self.kwargs['hscic_lambda'] * hscic(int_ft, z, y, self.kwargs['ridge_lambda'], "gaussian", self.kwargs['kernel_ft'],
                                  "gaussian", self.kwargs['kernel_z'], "gaussian", self.kwargs['kernel_y'])
        return loss


    def training_step(self, batch: Any, batch_idx: int):
        '''
        Run a single training step
        '''

        if self.kwargs['loo_cond_mean'] and not self.LOO_done:
            x, y, z = batch
            with torch.no_grad():
                Kz = gaussian_kernel(z, **self.kwargs['kernel_z'])
                print('Estimating regressions parameters with LOO')
                reg_list = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
                sigma2_list = [1.0, 0.1, 0.01, 0.001]
                self._leave_one_out_regressors(reg_list, sigma2_list, Kz, y)
            self.LOO_done = True
        
        targetloss, preds, targets, z, all_features = self.step(batch) 

        # HSCIC regularizer:
        if self.kwargs['n_last_reg_layers'] == -1 or self.kwargs['n_last_reg_layers'] > len(all_features):
            self.kwargs['n_last_reg_layers'] = len(all_features)
        hscic = self.reg_hscic(all_features, z, targets)
        loss = targetloss + hscic

        self.log('train/hscic', hscic.item())
        self.log('train/targetloss', targetloss.item())

        # log train metrics
        self._log_train_metrics(loss, preds, targets)

        return {"loss": loss, "preds": preds, "targets": targets, "distractors": z}
    
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, z, all_features = self.step(batch)

        loss += self.reg_hscic(all_features, z, targets)

        self._update_validation_metrics(loss, preds, targets, z)

        return {"loss": loss, "preds": preds, "targets": targets, "distractors": z}


class ConditionalDcorRegularization(StandardModule):
    def __init__(self, dcor_lambda: float, n_last_reg_layers: float, bandwidth: float | None,
                 rule_of_thumb_bw: bool = False, **kwargs):
        if bandwidth is None and not rule_of_thumb_bw:
            raise ValueError("Must specify bandwidth or set rule_of_thumb_bw=True")
        elif bandwidth and rule_of_thumb_bw:
            raise ValueError("Cannot specify both bandwidth and rule_of_thumb_bw=True")
        
        if rule_of_thumb_bw:
            bandwidth = self.trainer.datamodule.get_target_bandwidth()
        standardargs = {k: kwargs.pop(k) for k in STANDARDARGS}
        super().__init__(**standardargs)
        self.kwargs = kwargs
        self.cdcor = ConditionalDistanceCorrelation(bandwidth=bandwidth)
        self.dcor_lamda = dcor_lambda
        self.n_last_reg_layers = n_last_reg_layers

    def regularize_feats(self, feats: torch.Tensor, z: torch.Tensor, target: torch.Tensor, *args, **kwargs):
        loss = 0.0
        if z.size(0) < 3:
            LOG.debug("Not enough samples for regularization")
            return 0
        for feat in feats[-self.n_last_reg_layers:]:
            loss += self.cdcor(feat, z, target)

        return loss / self.n_last_reg_layers

    def training_step(self, batch: List[torch.Tensor], batch_idx: int):
        loss, preds, targets, z, all_features = self.step(batch)

        if self.n_last_reg_layers == -1 or self.n_last_reg_layers > len(all_features):
                self.n_last_reg_layers = len(all_features)

        cdcor_loss = self.regularize_feats(all_features, z, targets)
        loss += cdcor_loss * self.dcor_lamda

        self.log('train/cdcor', cdcor_loss.item())

        # log train metrics
        self._log_train_metrics(loss, preds, targets)

        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets, "distractors": z}

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int):
        loss, preds, targets, z, all_features = self.step(batch)

        if self.n_last_reg_layers == -1 or self.n_last_reg_layers > len(all_features):
                self.n_last_reg_layers = len(all_features)

        cdcor_loss = self.regularize_feats(all_features, z, targets)
        loss += cdcor_loss * self.dcor_lamda

        self.log('val/cdcor', cdcor_loss.item())

        self._update_validation_metrics(loss, preds, targets, z)

        return {"loss": loss, "preds": preds, "targets": targets, "distractors": z}
