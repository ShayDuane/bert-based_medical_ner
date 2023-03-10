import torch

def acc(y_pred, y_true):
    y_pred = y_pred[0]
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = torch.sum(y_pred.eq(y_true)).item() / y_true.numel()
    return {'acc': acc}