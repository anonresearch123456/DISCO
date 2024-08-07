import logging

import pandas as pd
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch.nn as nn

LOG = logging.getLogger(__name__)


def train(config: DictConfig):

    exp_ckpt_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "results.csv"
    if exp_ckpt_dir.exists() and (config.debug is False):
        LOG.info(f"Found existing experiment in directory {exp_ckpt_dir.parent}")
        return None

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    LOG.info("Instantiating datamodule <%s>", config.data._target_)
    data: pl.LightningDataModule = hydra.utils.instantiate(config.data)
    # data.setup()
    LOG.info("Instantiating model <%s>", config.model._target_)
    module: pl.LightningModule = hydra.utils.instantiate(config.model, _recursive_=True)

    # Init lightning callbacks
    callbacks: List[pl.Callback] = []
    if "callbacks" in config:
        for cb_conf in config.callbacks.values():
            if "_target_" in cb_conf:
                LOG.info("Instantiating callback <%s>", cb_conf._target_)
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[pl.LightningLoggerBase] = []
    if "logger" in config:
        for lg_conf in config.logger.values():
            if "_target_" in lg_conf:
                LOG.info("Instantiating logger <%s>", lg_conf._target_)
                logger.append(hydra.utils.instantiate(lg_conf))

    LOG.info("Instantiating trainer <%s>", config.trainer._target_)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    LOG.info("Starting training!")
    trainer.fit(module, data)

    LOG.info("Starting testing!")
    results = trainer.validate(module, data, ckpt_path='best')[0]  # if multiple dataloaders are used, update this!
    results.update(trainer.test(module, data, ckpt_path='best')[0])
    results = {f"best_{k}": v for k, v in results.items()}
    # and evaluate last saved checkpoint
    results_last = trainer.validate(module, data, ckpt_path='last')[0]
    results_last.update(trainer.test(module, data, ckpt_path='last')[0])
    results_last = {f"last_{k}": v for k, v in results_last.items()}
    results = results | results_last
    results = pd.Series(results)
    print(results)
    results.to_csv(Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "results.csv")
    return results


@hydra.main(config_path="configs", config_name="default.yaml", version_base="1.3")
def main(config: DictConfig):
    return train(config)


if __name__ == "__main__":
    main()
