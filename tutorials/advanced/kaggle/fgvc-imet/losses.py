from torch.nn.modules.loss import _Loss
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from lovasz_losses import lovasz_hinge2

rare_class = [   3,    6,    7,   10,   11,   19,   20,   21,   30,   31,   34,
         36,   37,   38,   39,   47,   53,   54,   56,   68,   71,   73,
         80,   81,   82,   84,   87,   88,   92,   94,  100,  101,  103,
        104,  108,  112,  115,  118,  119,  123,  126,  128,  129,  130,
        132,  137,  140,  142,  143,  146,  159,  160,  164,  166,  167,
        168,  174,  176,  177,  181,  183,  186,  187,  190,  193,  197,
        198,  199,  200,  201,  203,  207,  209,  211,  214,  215,  219,
        220,  221,  224,  225,  230,  233,  235,  240,  241,  242,  243,
        250,  254,  260,  262,  264,  268,  271,  277,  278,  281,  284,
        286,  288,  290,  291,  293,  296,  297,  298,  302,  303,  305,
        310,  311,  312,  314,  327,  328,  329,  333,  340,  343,  346,
        355,  363,  364,  365,  366,  367,  370,  372,  376,  381,  388,
        389,  391,  394,  395,  396,  431,  452,  460,  476,  523,  527,
        544,  561,  599,  635,  643,  652,  719,  727,  752,  787,  798,
        805,  812,  843,  845,  854,  855,  873,  883,  892,  904,  917,
        919,  987, 1017, 1060]

def bce_with_pos_weight():
    pos_weights = np.ones((1103,))
    pos_weights[rare_class] = 20
    pos_weights = torch.FloatTensor(pos_weights)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weights)

def binary_focal_loss(gamma=2, **_):
    def func(input, target):
        assert target.size() == input.size()

        max_val = (-input).clamp(min=0)

        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * gamma).exp() * loss
        return loss.mean()

    return func
    
def sigmoid_focal_loss(input: torch.Tensor,
                       target: torch.Tensor,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean'):
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
    References::
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(input.type())

    logpt = -F.binary_cross_entropy_with_logits(input, target, reduction='none')
    pt = torch.exp(logpt)

    # compute the loss
    loss = -((1 - pt).pow(gamma)) * logpt

    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))

    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)

    return loss

def reduced_focal_loss(input: torch.Tensor,
                       target: torch.Tensor,
                       threshold=0.5,
                       gamma=2.0,
                       reduction='mean'):
    """Compute reduced focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
    References::
        https://arxiv.org/abs/1903.01347
    """
    target = target.type(input.type())

    logpt = -F.binary_cross_entropy_with_logits(input, target, reduction='none')
    pt = torch.exp(logpt)

    # compute the loss
    focal_reduction = ((1. - pt) / threshold).pow(gamma)
    focal_reduction[pt < threshold] = 1

    loss = - focal_reduction * logpt

    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)

    return loss

class BinaryFocalLoss(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore=None, reduction='mean', reduced=False, threshold=0.5):
        """
        :param alpha:
        :param gamma:
        :param ignore:
        :param reduced:
        :param threshold:
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore = ignore
        if reduced:
            self.focal_loss = partial(reduced_focal_loss, gamma=gamma, threshold=threshold, reduction=reduction)
        else:
            self.focal_loss = partial(sigmoid_focal_loss, gamma=gamma, alpha=alpha, reduction=reduction)

    def forward(self, label_input, label_target):
        """Compute focal loss for binary classification problem.
        """
        label_target = label_target.view(-1)
        label_input = label_input.view(-1)

        if self.ignore is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = label_target != self.ignore
            label_input = label_input[not_ignored]
            label_target = label_target[not_ignored]

        loss = self.focal_loss(label_input, label_target)
        return loss


class FocalLoss2d(nn.modules.loss._WeightedLoss):
    def __init__(self, gamma=2, weight=None, size_average=None,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        # self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        # weight = Variable(self.weight)

        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, reduction='none') # pos_weight=weight,
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        # balanced_focal_loss = self.balance_param * focal_loss
        if self.reduction =='mean':
            focal_loss = focal_loss.mean()
        return focal_loss
    
class FbetaLoss(nn.Module):
    def __init__(self, beta=1):
        super(FbetaLoss, self).__init__()
        self.small_value = 1e-6
        self.beta = beta

    def forward(self, logits, labels):
        beta = self.beta
        batch_size = logits.size()[0]
        p = F.sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + self.small_value
        num_pos_hat = torch.sum(l, 1) + self.small_value
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + self.small_value)
        loss = fs.sum() / batch_size
        return 1 - loss

class LovaszLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logit, target):
        if not (target.size() == logit.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), logit.size()))        
        
        return lovasz_hinge2(logit, target)
    
class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight

class JointLoss(_Loss):
    def __init__(self, first, second, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)