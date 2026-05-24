import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import MLP


INPUT_DIM  = 116
BATCH_SIZE = 256
LR         = 1e-3
EPOCHS     = 50
PATIENCE   = 10
VAL_RATIO  = 0.2
DATA_DIR   = 'data/'
SAVE_DIR   = 'checkpoints/'


def load_data(data_dir: str):
    X_train_all = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train_all = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test      = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test      = np.load(os.path.join(data_dir, 'y_test.npy'))

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all,
        test_size=VAL_RATIO, random_state=42, stratify=y_train_all
    )

    print(f"Train: {len(X_train)} (normal={(y_train==0).sum()}, DoS={(y_train==1).sum()})")
    print(f"Val:   {len(X_val)}  (normal={(y_val==0).sum()},  DoS={(y_val==1).sum()})")
    print(f"Test:  {len(X_test)}  (normal={(y_test==0).sum()},  DoS={(y_test==1).sum()})")

    def to_loader(X, y, shuffle):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)
    test_loader  = to_loader(X_test,  y_test,  shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += loss.item() * len(y_batch)

            pred = (out >= 0.5).float()
            correct += (pred == y_batch).sum().item()
            total   += len(y_batch)

    return total_loss / total, correct / total


def train(data_dir: str = DATA_DIR, save_dir: str = SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    train_loader, val_loader, test_loader = load_data(data_dir)

    model     = MLP(input_dim=INPUT_DIM).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    patience_cnt  = 0

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader,   criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>8.4f} | {val_acc:>7.4f} | {current_lr:>8.6f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"\n학습 완료. Best val loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"모델 저장 → {save_dir}/best_model.pt")


if __name__ == '__main__':
    train()