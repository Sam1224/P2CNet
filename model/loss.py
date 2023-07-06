import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================
# bce_loss
# ========================================
def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


# ========================================
# bce_dice_loss
# ========================================
class BCEDiceLoss(nn.Module):
    def __init__(self, penalty_weight=None, size_average=True):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE Loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2. * (pred * truth).double().sum() + 1) / (pred.double().sum() + truth.double().sum() + 1)

        if self.penalty_weight:
            dice_loss = self.penalty_weight * (1 - dice_coef)
        else:
            dice_loss = (1 - dice_coef)

        loss = bce_loss + dice_loss
        return loss, bce_loss, dice_loss


def bce_dice_loss(input, target, penalty_weight=1, weight=1):
    return weight * BCEDiceLoss(penalty_weight=penalty_weight)(input, target)


# ========================================
# bce_dice_logits_loss
# ========================================
class BCEDiceLogitsLoss(nn.Module):
    def __init__(self, penalty_weight=None, size_average=True):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE Loss
        bce_loss = nn.BCEWithLogitsLoss()(pred, truth).double()

        # Dice Loss
        pred = nn.Sigmoid()(pred)
        dice_coef = (2. * (pred * truth).double().sum() + 1) / (pred.double().sum() + truth.double().sum() + 1)

        if self.penalty_weight:
            dice_loss = self.penalty_weight * (1 - dice_coef)
        else:
            dice_loss = (1 - dice_coef)

        loss = bce_loss + dice_loss
        return loss, bce_loss, dice_loss


def bce_dice_logits_loss(input, target, penalty_weight=1, weight=1):
    return weight * BCEDiceLogitsLoss(penalty_weight=penalty_weight)(input, target)


# ========================================
# mp_loss
# - missing part loss
# ========================================
def mp_loss(maps, partials, completes):
    # missing part
    mp = (completes - partials)
    loss = nn.BCELoss(reduction='none')(maps, mp) * mp
    loss = loss.mean()
    return loss


if __name__ == '__main__':
    maps = torch.randint(0, 2, (4, 1, 4, 4)).float().requires_grad_()
    partials = torch.randint(0, 2, (4, 1, 4, 4))
    completes = torch.ones((4, 1, 4, 4))
    # loss = mp_loss(maps, partials, completes)
    loss = nn.BCELoss(reduction='none')(maps, (completes - partials))
    print(loss)
    loss_mask = loss * (completes - partials)
    print(loss_mask)
    # loss = loss.mean()
    # loss.backward()
