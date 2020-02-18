import torch
import numpy as np


def mixup(data, targets1, targets2, targets3, alpha):
    """

        :param data: torch.Size. Shape:(batch_size, channel, image_size, image_size)
        :param targets1: torch.Tensor. multitask, multi-classification, target1. shape(batch_size, 1)
        :param targets2: torch.Tensor. multitask, multi-classification, target2. shape(batch_size, 1)
        :param targets3: torch.Tensor. multitask, multi-classification, target3. shape(batch_size, 1)
        :param alpha: float, positive
        :return: resulting image, multi-target
        """
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]

    return data, targets
