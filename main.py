import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from model import MLP
from fgsm import fgsm_attack
from bim import bim_attack
from constraint import MASK, apply_constraint
from evaluate import calc_asr, calc_asr_valid


def run_experiment(
    data_dir  = 'data/',
    ckpt_path = 'checkpoints/best_model.pt',
    epsilons  = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
    bim_steps = 40,
    bim_alpha = 0.01,
    save_dir  = 'results/'
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # 모델 로드
    model = MLP(input_dim=116).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 데이터 로드
    X_fgsm = torch.tensor(np.load(os.path.join(data_dir, 'X_fgsm.npy')), dtype=torch.float32)
    y_fgsm = torch.tensor(np.load(os.path.join(data_dir, 'y_fgsm.npy')), dtype=torch.float32).unsqueeze(1)

    with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    criterion = nn.BCELoss()

    # DoS 정확히 탐지된 샘플만 필터링
    with torch.no_grad():
        pred = (model(X_fgsm.to(device)) >= 0.5).float().squeeze().cpu()
    correctly_detected = (pred == 1)
    n_total    = len(X_fgsm)
    n_detected = correctly_detected.sum().item()
    X_fgsm = X_fgsm[correctly_detected]
    y_fgsm = y_fgsm[correctly_detected]

    print(f"전체 샘플:               {n_total}개")
    print(f"DoS로 정확히 탐지된 샘플: {n_detected}개 (공격 대상)")
    print(f"애초에 오분류된 샘플:     {n_total - n_detected}개 (제외)\n")

    results = []

    for eps in epsilons:
        print(f"{'='*60}")
        print(f"Epsilon: {eps}")

        for constraint in ['C0', 'C1', 'C2', 'C3']:
            mask_indices = MASK[constraint]

            # FGSM
            x_fgsm = fgsm_attack(model, X_fgsm.clone(), y_fgsm.clone(),
                                  eps, mask_indices, criterion, device)
            x_fgsm = apply_constraint(x_fgsm, X_fgsm, constraint, scaler)
            asr_fgsm       = calc_asr(model, x_fgsm, device)
            asr_valid_fgsm = calc_asr_valid(model, x_fgsm, X_fgsm, device)

            # BIM
            x_bim = bim_attack(model, X_fgsm.clone(), y_fgsm.clone(),
                                eps, bim_alpha, bim_steps, mask_indices, criterion, device)
            x_bim = apply_constraint(x_bim, X_fgsm, constraint, scaler)
            asr_bim       = calc_asr(model, x_bim, device)
            asr_valid_bim = calc_asr_valid(model, x_bim, X_fgsm, device)

            print(f"  [{constraint}] "
                  f"FGSM ASR={asr_fgsm:.4f} valid={asr_valid_fgsm:.4f} | "
                  f"BIM  ASR={asr_bim:.4f}  valid={asr_valid_bim:.4f} | "
                  f"수정 가능: {len(mask_indices)}개")

            results.append({
                'epsilon':        eps,
                'constraint':     constraint,
                'fgsm_asr':       asr_fgsm,
                'fgsm_asr_valid': asr_valid_fgsm,
                'bim_asr':        asr_bim,
                'bim_asr_valid':  asr_valid_bim,
                'n_modifiable':   len(mask_indices)
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    print(f"\n결과 저장 → {save_dir}/results.csv")
    return df


if __name__ == '__main__':
    run_experiment()