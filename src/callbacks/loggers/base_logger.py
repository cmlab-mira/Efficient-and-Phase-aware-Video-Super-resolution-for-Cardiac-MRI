import torch
from torch.utils.tensorboard import SummaryWriter


class BaseLogger:
    """The base class for all loggers.
    Args:
        log_dir (str): The saved directory.
        net (BaseNet): The network architecture.
        dummy_input (torch.Tensor): The dummy input for plotting the network architecture.
    """
    def __init__(self, log_dir, net, dummy_input):
        """
        # TODO: Plot the network architecture.
        # There are some errors: ONNX runtime errors.
        with SummaryWriter(log_dir) as w:
            w.add_graph(net, dummy_input)
        """
        self.writer = SummaryWriter(log_dir)

    def write(self, epoch, train_log, train_batch, train_outputs, valid_log, valid_batch, valid_outputs):
        """Plot the network architecture and the visualization results.
        Args:
            epoch (int): The number of trained epochs.
            train_log (dict): The training log information.
            train_batch (dict or sequence): The training batch.
            train_outputs (torch.Tensor or sequence of torch.Tensor): The training outputs.
            valid_log (dict): The validation log information.
            valid_batch (dict or sequence): The validation batch.
            valid_outputs (torch.Tensor or sequence of torch.Tensor): The validation outputs.
        """
        self._add_scalars(epoch, train_log, valid_log)
        self._add_images(epoch, train_batch, train_outputs, valid_batch, valid_outputs)

    def close(self):
        """Close the writer.
        """
        self.writer.close()

    def _add_scalars(self, epoch, train_log, valid_log):
        """Plot the training curves.
        Args:
            epoch (int): The number of trained epochs.
            train_log (dict): The training log information.
            valid_log (dict): The validation log information.
        """
        for key in train_log:
            self.writer.add_scalars(key, {'train': train_log[key], 'valid': valid_log[key]}, epoch)

    def _add_images(self, epoch, train_batch, train_outputs, valid_batch, valid_outputs):
        """Plot the visualization results.
        Args:
            epoch (int): The number of trained epochs.
            train_batch (dict or sequence): The training batch.
            train_outputs (torch.Tensor or sequence of torch.Tensor): The training outputs.
            valid_batch (dict or sequence): The validation batch.
            valid_outputs (torch.Tensor or sequence of torch.Tensor): The validation outputs.
        """
        raise NotImplementedError
