import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pickle
import os

# 컬럼 정의

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

BINARY_COLS = [
    'land', 'logged_in', 'root_shell', 'su_attempted',
    'is_host_login', 'is_guest_login'
]

# 왜도 50 이상
LOG1P_COLS = [
    'num_compromised', 'num_root', 'src_bytes', 'urgent', 'dst_bytes',
    'num_failed_logins', 'num_shells', 'num_file_creations', 'num_access_files'
]

ALL_FEATURES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

CONTINUOUS_COLS = [
    c for c in ALL_FEATURES
    if c not in CATEGORICAL_COLS + BINARY_COLS
]


# 왜도 50 이상인 경우 log1p 변환
def apply_log1p(df):
    df = df.copy()
    df[LOG1P_COLS] = np.log1p(df[LOG1P_COLS])
    return df


# train 데이터 기준으로 fit -> encoder, scaler 반환
def fit_preprocessor(df_train):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(df_train[CATEGORICAL_COLS])

    scaler = MinMaxScaler()
    scaler.fit(df_train[CONTINUOUS_COLS + BINARY_COLS])

    return encoder, scaler


def transform(df, encoder, scaler):
    num_scaled = scaler.transform(df[CONTINUOUS_COLS + BINARY_COLS])
    cat_encoded = encoder.transform(df[CATEGORICAL_COLS])
    cat_feature_names = encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()

    X = np.hstack([num_scaled, cat_encoded])
    feature_names = CONTINUOUS_COLS + BINARY_COLS + cat_feature_names

    return X, feature_names


# 전체 파이프라인
def build_dataset(train_path, test_path, fgsm_path, save_dir='./data'):
    os.makedirs(save_dir, exist_ok=True)

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    df_fgsm  = pd.read_csv(fgsm_path)

    print(f"Train: {len(df_train)} (normal={(df_train['target']==0).sum()}, DoS={(df_train['target']==1).sum()})")
    print(f"Test:  {len(df_test)}  (normal={(df_test['target']==0).sum()},  DoS={(df_test['target']==1).sum()})")
    print(f"FGSM:  {len(df_fgsm)} (DoS only)")

    # log1p 변환
    df_train = apply_log1p(df_train)
    df_test  = apply_log1p(df_test)
    df_fgsm  = apply_log1p(df_fgsm)

    # fit (train 기준)
    encoder, scaler = fit_preprocessor(df_train)

    # transform
    X_train, feature_names = transform(df_train, encoder, scaler)
    X_test,  _             = transform(df_test,  encoder, scaler)
    X_fgsm,  _             = transform(df_fgsm,  encoder, scaler)

    y_train = df_train['target'].values
    y_test  = df_test['target'].values
    y_fgsm  = df_fgsm['target'].values

    print(f"\nInput dim: {X_train.shape[1]}")
    print(f"Feature names ({len(feature_names)}): {feature_names[:5]} ... {feature_names[-3:]}")

    # 저장
    np.save(os.path.join(save_dir, 'X_train.npy'),       X_train)
    np.save(os.path.join(save_dir, 'X_test.npy'),        X_test)
    np.save(os.path.join(save_dir, 'X_fgsm.npy'),        X_fgsm)
    np.save(os.path.join(save_dir, 'y_train.npy'),       y_train)
    np.save(os.path.join(save_dir, 'y_test.npy'),        y_test)
    np.save(os.path.join(save_dir, 'y_fgsm.npy'),        y_fgsm)
    np.save(os.path.join(save_dir, 'feature_names.npy'), np.array(feature_names))

    with open(os.path.join(save_dir, 'encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n저장 완료 → {save_dir}/")
    return X_train, X_test, X_fgsm, y_train, y_test, y_fgsm, feature_names


if __name__ == '__main__':
    build_dataset(
        train_path='KDDTrain_DoS_vs_Normal.csv',
        test_path ='KDDTest_DoS_vs_Normal.csv',
        fgsm_path ='FGSM_DoS_input_1000.csv',
        save_dir  ='data/'
    )