import torch
import torch.nn as nn


def bim_attack(model, x, y, epsilon, alpha, steps, mask_indices, criterion, device):
    x        = x.to(device)
    x_adv    = x.clone().detach()
    y_target = torch.zeros_like(y).to(device)

    mask = torch.zeros_like(x)
    mask[:, mask_indices] = 1.0

    model.eval()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        out  = model(x_adv)
        loss = criterion(out, y_target)
        model.zero_grad()
        loss.backward()

        grad  = x_adv.grad.detach()
        x_adv = x_adv.detach() - alpha * grad.sign() * mask

        # mask 적용 피처에만 epsilon clamp
        x_adv_clipped = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = torch.where(mask.bool(), x_adv_clipped, x_adv)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()