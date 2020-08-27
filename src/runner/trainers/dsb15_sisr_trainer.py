import torch
import functools

from src.runner.trainers.base_trainer import BaseTrainer
from src.utils import denormalize


class Dsb15SISRTrainer(BaseTrainer):
    """The DSB15 trainer for the Single-Image Super-Resolution.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._denormalize = functools.partial(denormalize, dataset='dsb15')

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.

        Returns:
            iutput (torch.Tensor): The data input.
            target (torch.Tensor): The data target.
        """
        return batch['lr_img'], batch['hr_img']

    def _compute_losses(self, output, target):
        """Compute the losses.
        Args:
            output (torch.Tensor): The model output.
            target (torch.Tensor): The data target.

        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        losses = [loss_fn(output, target) for loss_fn in self.loss_fns]
        return losses

    def _compute_metrics(self, output, target):
        """Compute the metrics.
        Args:
            output (torch.Tensor): The model output.
            target (torch.Tensor): The data target.

        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        output, target = self._denormalize(output), self._denormalize(target)
        metrics = [metric_fn(output, target) for metric_fn in self.metric_fns]
        return metrics