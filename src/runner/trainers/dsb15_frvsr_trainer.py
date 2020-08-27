import torch
from tqdm import tqdm
import functools

from src.runner.trainers.base_trainer import BaseTrainer
from src.utils import denormalize


class Dsb15FRVSRTrainer(BaseTrainer):
    """The DSB15 trainer for the FRVSR network.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._denormalize = functools.partial(denormalize, dataset='dsb15')

    def _run_epoch(self, mode):
        """Run an epoch for training.
        Args:
            mode (str): The mode of running an epoch ('training' or 'validation').
        Returns:
            log (dict): The log information.
            batch (dict or sequence): The last batch of the data.
            outputs (torch.Tensor or sequence of torch.Tensor): The corresponding model outputs.
        """
        if mode == 'training':
            self.net.train()
        else:
            self.net.eval()
        dataloader = self.train_dataloader if mode == 'training' else self.valid_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc=mode)

        log = self._init_log()
        count = 0
        for batch in trange:
            batch = self._allocate_data(batch)
            inputs, targets = self._get_inputs_targets(batch)
            T = len(inputs)
            if mode == 'training':
                outputs = self.net(inputs)
                losses = self._compute_losses(outputs, inputs, targets)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.net(inputs)
                    losses = self._compute_losses(outputs, inputs, targets)
                    loss = (torch.stack(losses) * self.loss_weights).sum()
            metrics =  self._compute_metrics(outputs, targets)

            batch_size = self.train_dataloader.batch_size if mode == 'training' else self.valid_dataloader.batch_size
            self._update_log(log, batch_size, T, loss, losses, metrics)
            count += batch_size * T
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))
        
        outputs = outputs[0] # SR images
        for key in log:
            log[key] /= count
        return log, batch, outputs

    def _get_inputs_targets(self, batch):
        """Specify the data inputs and targets.
        Args:
            batch (dict): A batch of data.

        Returns:
            inputs (list of torch.Tensor): The data inputs.
            targets (list of torch.Tensor): The data targets.
        """
        return batch['lr_imgs'], batch['hr_imgs']

    def _compute_losses(self, outputs, lr_imgs, hr_imgs):
        """Compute the losses.
        Args:
            outputs (list of torch.Tensor): The model outputs.
            lr_imgs (list of torch.Tensor): The low resolution frames.
            hr_imgs (list of torch.Tensor): The high resolution frames.

        Returns:
            loss (torch.Tensor): The computed loss.
        """
        sr_imgs, lr_imgs_ = outputs
        flow_loss = torch.stack([self.loss_fns[0](lr_img_, lr_img) for lr_img_, lr_img in zip(lr_imgs_, lr_imgs)]).mean()
        sr_loss = torch.stack([self.loss_fns[1](sr_img, hr_img) for sr_img, hr_img in zip(sr_imgs, hr_imgs)]).mean()
        return [flow_loss, sr_loss]

    def _compute_metrics(self, outputs, targets):
        """Compute the metrics.
        Args:
            outputs (list of torch.Tensor): The model outputs.
            targets (list of torch.Tensor): The data targets.

        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        outputs, _ = outputs
        outputs = list(map(self._denormalize, outputs))
        targets = list(map(self._denormalize, targets))

        # Average the metric of every frame in a video.
        metrics = []
        for metric_fn in self.metric_fns:
            metric = torch.stack([metric_fn(output, target) for output, target in zip(outputs, targets)]).mean()
            metrics.append(metric)
        return metrics

    def _update_log(self, log, batch_size, T, loss, losses, metrics):
        """Update the log.
        Args:
            log (dict): The log to be updated.
            batch_size (int): The batch size.
            T (int): The total number of the frames.
            loss (torch.Tensor): The weighted sum of the computed losses.
            losses (sequence of torch.Tensor): The computed losses.
            metrics (sequence of torch.Tensor): The computed metrics.
        """
        log['Loss'] += loss.item() * batch_size * T
        for loss_fn, loss in zip(self.loss_fns, losses):
            log[loss_fn.__class__.__name__] += loss.item() * batch_size * T
        for metric_fn, metric in zip(self.metric_fns, metrics):
            log[metric_fn.__class__.__name__] += metric.item() * batch_size * T