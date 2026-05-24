import os
import numpy as np
import torch
import torch.nn as nn
import pickle

from model import MLP

# 제약 정의
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

# 정수형 Feature 제약
INT_FEATURES = [1, 3, 4, 13, 14, 22, 23]


# 정수형, 양수 제약
def apply_c2_constraints(x_adv, x_orig, scaler):
    x_adv = x_adv.clone()
    num_scaled = x_adv[:, :38].cpu().numpy()
    num_orig   = scaler.inverse_transform(num_scaled)

    for idx in INT_FEATURES:
        num_orig[:, idx] = np.round(num_orig[:, idx])
        num_orig[:, idx] = np.clip(num_orig[:, idx], 0, None)

    num_rescaled = scaler.transform(num_orig)
    x_adv[:, :38] = torch.tensor(num_rescaled, dtype=torch.float32).to(x_adv.device)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def fgsm_attack(model, x, y, epsilon, mask_indices, criterion, device):
    x = x.to(device)
    x.requires_grad = True

    # targeted attack: normal(0) 방향으로 공격
    y_target = torch.zeros_like(y).to(device)

    model.eval()
    out  = model(x)
    loss = criterion(out, y_target)
    model.zero_grad()
    loss.backward()

    grad_sign = x.grad.data.sign()

    mask = torch.zeros_like(x)
    mask[:, mask_indices] = 1.0

    # loss 최소화 방향 = normal(0)에 가까워지는 방향
    x_adv = x - epsilon * grad_sign * mask

    return x_adv.detach()


def calc_asr(model, x_adv, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        out  = model(x_adv.to(device))
        pred = (out >= threshold).float().squeeze()
    success = (pred == 0).sum().item()
    return success / len(pred)


def calc_asr_valid(model, x_adv, x_orig, device, threshold=0.5):
    """
    ASR_valid: x_adv에 C3 제약을 강제 적용(projection)한 뒤 ASR 계산
    → C3_EXCLUDED 피처를 원본값으로 복원한 샘플로 모델 평가
    """
    model.eval()
    x_adv  = x_adv.clone().to(device)
    x_orig = x_orig.to(device)

    # C3 피처 원본으로 강제 복원
    for idx in C3_EXCLUDED:
        x_adv[:, idx] = x_orig[:, idx]

    with torch.no_grad():
        out  = model(x_adv)
        pred = (out >= threshold).float().squeeze()

    success = (pred == 0).sum().item()
    return success / len(pred)


def run_experiment(
    data_dir  = 'data/',
    ckpt_path = 'checkpoints/best_model.pt',
    epsilons  = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
    save_dir  = 'results/'
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model = MLP(input_dim=116).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    X_fgsm = torch.tensor(np.load(os.path.join(data_dir, 'X_fgsm.npy')), dtype=torch.float32)
    y_fgsm = torch.tensor(np.load(os.path.join(data_dir, 'y_fgsm.npy')), dtype=torch.float32).unsqueeze(1)

    with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    criterion = nn.BCELoss()

    # 모델이 DoS(1)로 정확히 탐지한 샘플만 필터링
    with torch.no_grad():
        out  = model(X_fgsm.to(device))
        pred = (out >= 0.5).float().squeeze().cpu()

    correctly_detected = (pred == 1)
    n_total    = len(X_fgsm)
    n_detected = correctly_detected.sum().item()

    X_fgsm = X_fgsm[correctly_detected]
    y_fgsm = y_fgsm[correctly_detected]

    print(f"전체 샘플: {n_total}개")
    print(f"DoS로 정확히 탐지된 샘플: {n_detected}개 (공격 대상)")
    print(f"애초에 오분류된 샘플: {n_total - n_detected}개 (제외)\n")

    results = []

    for eps in epsilons:
        print(f"{'='*50}")
        print(f"Epsilon: {eps}")
        for constraint in ['C0', 'C1', 'C2', 'C3']:
            mask_indices = MASK[constraint]

            x_adv = fgsm_attack(
                model, X_fgsm.clone(), y_fgsm.clone(),
                epsilon=eps, mask_indices=mask_indices,
                criterion=criterion, device=device
            )

            if constraint in ['C2', 'C3']:
                x_adv = apply_c2_constraints(x_adv, X_fgsm, scaler)

                if constraint == 'C3':
                    for idx in C3_EXCLUDED:
                        x_adv[:, idx] = X_fgsm[:, idx]

            asr       = calc_asr(model, x_adv, device)
            asr_valid = calc_asr_valid(model, x_adv, X_fgsm, device)

            print(f"  {constraint} | ASR: {asr:.4f} | ASR_valid: {asr_valid:.4f} | 수정 가능 피처 수: {len(mask_indices)}")
            results.append({
                'epsilon': eps, 'constraint': constraint,
                'asr': asr, 'asr_valid': asr_valid,
                'n_modifiable': len(mask_indices)
            })

    import pandas as pd
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    print(f"\n결과 저장 → {save_dir}/results.csv")
    return df_results


if __name__ == '__main__':
    run_experiment()