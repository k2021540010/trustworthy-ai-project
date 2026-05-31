import torch
import torch.nn as nn


def fgsm_attack(model, x, y, epsilon, mask_indices, criterion, device):
    x        = x.to(device)
    x.requires_grad = True
    y_target = torch.zeros_like(y).to(device)

    model.eval()
    out  = model(x)
    loss = criterion(out, y_target)
    model.zero_grad()
    loss.backward()

    grad_sign = x.grad.data.sign()
    mask      = torch.zeros_like(x)
    mask[:, mask_indices] = 1.0
    x_adv = x - epsilon * grad_sign * mask

    return x_adv.detach()