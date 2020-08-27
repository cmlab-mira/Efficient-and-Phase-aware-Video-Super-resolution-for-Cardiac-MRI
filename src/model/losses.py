import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    """The implementation of the HuberLoss.
    Args:
        delta (float): The threshold (ref: http://openaccess.thecvf.com/content_cvpr_2018/papers/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.pdf).
    """
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, output, target):
        abs_error = torch.abs(output - target)
        delta = torch.ones_like(output) * self.delta
        quadratic = torch.min(abs_error, delta)
        linear = (abs_error - quadratic)
        loss = 0.5 * quadratic ** 2 + delta * linear
        return torch.mean(loss)


class CharbonnierLoss(nn.Module):
    """The implementation of the CharbonnierLoss.
    Args:
        epsilon (float): The smooth term.
    """
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, output, target):
        loss = torch.sqrt((output - target) ** 2 + self.epsilon)
        return torch.mean(loss)


class FlowLoss(nn.MSELoss):
    """The implementation of the flow loss in the frvsr network.
    """
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        return super().forward(outputs, targets)

