import torch
from constraint import C3_EXCLUDED


def calc_asr(model, x_adv, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        out  = model(x_adv.to(device))
        pred = (out >= threshold).float().squeeze()
    return (pred == 0).sum().item() / len(pred)


def calc_asr_valid(model, x_adv, x_orig, device, threshold=0.5):
    model.eval()
    x_adv  = x_adv.clone().to(device)
    x_orig = x_orig.to(device)

    for idx in C3_EXCLUDED:
        x_adv[:, idx] = x_orig[:, idx]

    with torch.no_grad():
        out  = model(x_adv)
        pred = (out >= threshold).float().squeeze()

    return (pred == 0).sum().item() / len(pred)