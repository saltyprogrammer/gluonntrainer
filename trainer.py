"""
General trainer for Gluon models.
"""
import os

from typing import Tuple, Union

from mxnet.metric import CompositeEvalMetrics
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from mxnet.gluon.loss import Loss
from mxnet.gluon.nn import HybridBlock

class GluonTrainer:
    """
    Trainer class for Gluon models.

    Parameters
    ----------
    store : str
        Path where metadata and checkpoints are going to be stored. 
    trainer : Trainer
        Gluon trainer to be used during the training process.
    loss : Loss
        Loss function to use. It must match the Gluon Loss protocol.
    metrics : CompsiteEvalMetrics
        Metrics to be calculated for measuirng network performance.
    data : Union[DataLoader, Tuple[DataLoader, DataLoader]]
        Data to be used for training and validation. If a single
        instance is provided, then no validation metrics are calculated.
    epochs : int
        Number of iterations.
    last_epoch : int
        Epoch from the last stored checkpoint.
    net : HybridBlock
        Network to be trained.
    log_freq : int
        Batch frequency to generate metrics logging.
    store_freq : int
        Epoch frequency to store model checkpoints. It must be smaller
        than the number of epochs.
    """
    def __init__(
            self,
            store: str,
            trainer: Trainer,
            loss: Loss,
            metrics: CompositeEvalMetrics,
            data: Union[DataLoader, Tuple[DataLoader, DataLoader]],
            epochs: int = 10,
            last_epoch: int = 0,
            net: HybridBlock = None,
            log_freq: int = 50,
            store_freq: int = 5,
        ):
        self.store = store
        self.trainer = trainer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.log_freq = log_freq
        self.store_freq = store_freq

        # Check if store dir is not empty.
        if os.path.exists(self.store):
            self.net, self.last_epoch = self._load_checkpoint()
        else:
            self.net = net

        # Load and prepare the datasets
        if isinstance(data, tuple):
            self.train_iter, self.valid_iter = data
        else:
            self.train_iter = data
            self.valid_iter = None

    def train(self):
        pass

    def _load_checkpoint(self) -> Tuple[HybridBlock, int]:
        pass

    def _save_checkpoint(self):
        pass

    def _evaluate_progress(self):
        pass
