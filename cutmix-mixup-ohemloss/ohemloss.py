def ohem_loss(rate, cls_pred, cls_target):

    batch_size = cls_pred.size(0)
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss
    

# loss
def cutmix_criterion(preds1, preds2, preds3, targets, rate=0.7):
    targets1, targets2, targets3, targets4, targets5, targets6, lam =\
        targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion = ohem_loss
    return lam * criterion(rate, preds1, targets1) + (1 - lam) * criterion(rate, preds1, targets2) + lam * criterion(rate, preds2, targets3) + (1 - lam) * criterion(rate, preds2, targets4) + lam * criterion(rate, preds3, targets5) + (1 - lam) * criterion(rate, preds3, targets6)


def mixup_criterion(preds1, preds2, preds3, targets, rate=0.7):
    targets1, targets2, targets3, targets4, targets5, targets6, lam =\
        targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion = ohem_loss
    return lam * criterion(rate, preds1, targets1) + (1 - lam) * criterion(rate, preds1, targets2) + lam * criterion(rate, preds2, targets3) + (1 - lam) * criterion(rate, preds2, targets4) + lam * criterion(rate, preds3, targets5) + (1 - lam) * criterion(rate, preds3, targets6)
