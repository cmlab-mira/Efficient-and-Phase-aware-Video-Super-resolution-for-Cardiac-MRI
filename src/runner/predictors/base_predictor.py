import torch
import logging
from tqdm import tqdm


class BasePredictor:
    """The base class for all predictors.
    Args:
        device (torch.device): The device.
        test_dataloader (Dataloader): The testing dataloader.
        net (BaseNet): The network architecture.
        loss_fns (list of torch.nn.Module): The loss functions.
        loss_weights (list of float): The corresponding weights of loss functions.
        metric_fns (list of torch.nn.Module): The metric functions.
    """
    def __init__(self, device, test_dataloader, net, loss_fns, loss_weights, metric_fns):
        self.device = device
        self.test_dataloader = test_dataloader
        self.net = net.to(device)
        self.loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]
        self.loss_weights = torch.tensor(loss_weights, dtype=torch.float, device=device)
        self.metric_fns = [metric_fn.to(device) for metric_fn in metric_fns]

    def predict(self):
        """The testing process.
        """
        self.net.eval()
        trange = tqdm(self.test_dataloader,
                      total=len(self.test_dataloader),
                      desc='testing')

        log = self._init_log()
        count = 0
        for batch in trange:
            batch = self._allocate_data(batch)
            inputs, targets = self._get_inputs_targets(batch)
            with torch.no_grad():
                outputs = self.net(inputs)
                losses = self._compute_losses(outputs, targets)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                metrics =  self._compute_metrics(outputs, targets)

            batch_size = self.test_dataloader.batch_size
            self._update_log(log, batch_size, loss, losses, metrics)
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        for key in log:
            log[key] /= count
        logging.info(f'Test log: {log}.')

    def _allocate_data(self, batch):
        """Allocate the data to the device.
        Args:
            batch (dict or sequence): A batch of the data.

        Returns:
            batch (dict or sequence): A batch of the allocated data.
        """
        if isinstance(batch, dict):
            return dict((key, self._allocate_data(data)) for key, data in batch.items())
        elif isinstance(batch, list):
            return list(self._allocate_data(data) for data in batch)
        elif isinstance(batch, tuple):
            return tuple(self._allocate_data(data) for data in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)

    def _get_inputs_targets(self, batch):
        """Specify the data inputs and targets.
        Args:
            batch (dict or sequence): A batch of data.

        Returns:
            inputs (torch.Tensor or sequence of torch.Tensor): The data inputs.
            targets (torch.Tensor or sequence of torch.Tensor): The data targets.
        """
        raise NotImplementedError

    def _compute_losses(self, outputs, targets):
        """Compute the losses.
        Args:
            outputs (torch.Tensor or sequence of torch.Tensor): The model outputs.
            targets (torch.Tensor or sequence of torch.Tensor): The data targets.

        Returns:
            losses (sequence of torch.Tensor): The computed losses.
        """
        raise NotImplementedError

    def _compute_metrics(self, outputs, targets):
        """Compute the metrics.
        Args:
            outputs (torch.Tensor or sequence of torch.Tensor): The model outputs.
            targets (torch.Tensor or sequence of torch.Tensor): The data targets.

        Returns:
            metrics (sequence of torch.Tensor): The computed metrics.
        """
        raise NotImplementedError

    def _init_log(self):
        """Initialize the log.
        Returns:
            log (dict): The initialized log.
        """
        log = {}
        log['Loss'] = 0
        for loss_fn in self.loss_fns:
            log[loss_fn.__class__.__name__] = 0
        for metric_fn in self.metric_fns:
            log[metric_fn.__class__.__name__] = 0
        return log

    def _update_log(self, log, batch_size, loss, losses, metrics):
        """Update the log.
        Args:
            log (dict): The log to be updated.
            batch_size (int): The batch size.
            loss (torch.Tensor): The weighted sum of the computed losses.
            losses (sequence of torch.Tensor): The computed losses.
            metrics (sequence of torch.Tensor): The computed metrics.
        """
        log['Loss'] += loss.item() * batch_size
        for loss_fn, loss in zip(self.loss_fns, losses):
            log[loss_fn.__class__.__name__] += loss.item() * batch_size
        for metric_fn, metric in zip(self.metric_fns, metrics):
            log[metric_fn.__class__.__name__] += metric.item() * batch_size

    def load(self, path):
        """Load the model checkpoint.
        Args:
            path (Path): The path to load the model checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
