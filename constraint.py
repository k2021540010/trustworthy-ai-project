import torch
import numpy as np
import warnings

C0_MODIFIABLE = list(range(0, 32))

C1_EXCLUDED   = [2, 5, 6, 7, 8, 9, 10, 11, 12]
C1_MODIFIABLE = [i for i in C0_MODIFIABLE if i not in C1_EXCLUDED]

C2_MODIFIABLE = C1_MODIFIABLE

C3_EXCLUDED   = [13, 14, 15, 16, 22, 23, 28, 29]
C3_MODIFIABLE = [i for i in C2_MODIFIABLE if i not in C3_EXCLUDED]

MASK = {
    'C0': C0_MODIFIABLE,
    'C1': C1_MODIFIABLE,
    'C2': C2_MODIFIABLE,
    'C3': C3_MODIFIABLE,
}

INT_FEATURES = [1, 3, 4, 13, 14, 22, 23]

def apply_c2_constraints(x_adv, x_orig, scaler):
    x_adv      = x_adv.clone()
    num_scaled = x_adv[:, :38].cpu().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num_orig     = scaler.inverse_transform(num_scaled)

        for idx in INT_FEATURES:
            num_orig[:, idx] = np.round(num_orig[:, idx])
            num_orig[:, idx] = np.clip(num_orig[:, idx], 0, None)

        num_rescaled = scaler.transform(num_orig)

    x_adv[:, :38] = torch.tensor(num_rescaled, dtype=torch.float32).to(x_adv.device)
    x_adv          = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

def apply_constraint(x_adv, x_orig, constraint, scaler):
    if constraint in ['C2', 'C3']:
        x_adv = apply_c2_constraints(x_adv, x_orig, scaler)
        if constraint == 'C3':
            for idx in C3_EXCLUDED:
                x_adv[:, idx] = x_orig[:, idx]
    return x_adv