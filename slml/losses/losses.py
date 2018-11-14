import torch
from torch import nn, Tensor
from torch.nn.functional import logsigmoid


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input_: Tensor, target):
        if not (target.size() == input_.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input_.size()))

        max_val = (-input_).clamp(min=0)
        loss = input_ - input_ * target + max_val + ((-max_val).exp() +
                                                     (-input_ - max_val).exp()).log()

        invprobs = logsigmoid(-input_ * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))