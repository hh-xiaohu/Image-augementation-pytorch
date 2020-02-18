import numpy as np
import torch

def rand_bbox(size, lam):
    """

    :param size: torch.Size. Shape:(batch_size, channel, image_size, image_size)
    :param lam: cutmix rate
    :return: box boundary
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
def cutmix(data, targets1, targets2, targets3, alpha):
    """

    :param data: torch.Size. Shape:(batch_size, channel, image_size, image_size)
    :param targets1: torch.Tensor. multitask, multi-classification, target1. shape(batch_size, 1)
    :param targets2: torch.Tensor. multitask, multi-classification, target2. shape(batch_size, 1)
    :param targets3: torch.Tensor. multitask, multi-classification, target3. shape(batch_size, 1)
    :param alpha: float, positive
    :return: resulting image, multi-target
    """
    indices = torch.randperm(data.size(0))
    # shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
    return data, targets
