# batch_run_regression_perturbations.py

import subprocess
import pandas as pd
import numpy as np
import ipaddress
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -----------------------------
# 常量 & 辅助函数
# -----------------------------
COMMON_PORTS = {80, 443, 53, 110, 25, 22, 21, 23, 123, 445, 3306, 8080}

def is_private(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except:
        return False

def extract_ip_port_features(df: pd.DataFrame) -> pd.DataFrame:
    df['src_private'] = df['Source IP'].apply(is_private).astype(int)
    df['dst_private'] = df['Destination IP'].apply(is_private).astype(int)

    df['src_well_known_port'] = (df['Source Port'].astype(int) < 1024).astype(int)
    df['dst_well_known_port'] = (df['Destination Port'].astype(int) < 1024).astype(int)

    df['src_common_port'] = df['Source Port'].astype(int).isin(COMMON_PORTS).astype(int)
    df['dst_common_port'] = df['Destination Port'].astype(int).isin(COMMON_PORTS).astype(int)

    df['same_port'] = (df['Source Port'] == df['Destination Port']).astype(int)
    return df

# -----------------------------
# 调用 generate_perturbation_1.py，单一扰动
# -----------------------------
def generate_single_perturbed_data(perturb_type: str, level: float):
    params = {
        'noise_std':    0.0,
        'scale_low':    1.0,
        'scale_high':   1.0,
        'dropout_rate': 0.0,
        'permute_rate': 0.0,
    }
    if perturb_type == 'gaussian':
        params['noise_std'] = level
    elif perturb_type == 'random_sc':
        params['scale_low']  = 1.0 - level
        params['scale_high'] = 1.0 + level
    elif perturb_type == 'feature_d':
        params['dropout_rate'] = level
    elif perturb_type == 'feature_p':
        params['permute_rate'] = level
    else:
        raise ValueError(f"Unknown perturb_type: {perturb_type}")

    cmd = [
        'python', 'generate_perturbation_1.py',
        '--noise_std',    str(params['noise_std']),
        '--scale_low',    str(params['scale_low']),
        '--scale_high',   str(params['scale_high']),
        '--dropout_rate', str(params['dropout_rate']),
        '--permute_rate', str(params['permute_rate']),
    ]
    subprocess.run(cmd, check=True)

# -----------------------------
# 加载并预处理对抗样本，返回 X 和 y
# -----------------------------
def load_adv_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # 处理无穷/缺失
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # 增加 IP/Port 特征 & 丢弃原始列
    df = extract_ip_port_features(df)
    df.drop(columns=['Source IP', 'Destination IP', 'Source Port', 'Destination Port'], inplace=True)

    # 标签编码
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    X = df.drop(columns=['label'])
    y = df['label'].values
    return X, y

# -----------------------------
# 主流程
# -----------------------------
def main():
    model_names     = ['KNN', 'Random Forest', 'SVM']
    perturb_types   = ['gaussian', 'random_sc', 'feature_d', 'feature_p']
    levels          = [i * 0.01 for i in range(1, 26)]
    adv_csv         = '../Scenario-B-merged_5s_adversarial.csv'

    for name in model_names:
        # 加载已经包含预处理 Pipeline 的模型
        model = joblib.load(f'../model/{name}.pkl')

        records = []
        for p_type in perturb_types:
            for lvl in levels:
                lvl_str = f"{lvl:.2f}"
                print(f"[{name}] Testing {p_type} @ level={lvl_str}")
                generate_single_perturbed_data(p_type, lvl)

                # 读取刚生成的对抗样本
                X_adv, y_adv = load_adv_data(adv_csv)
                y_pred = model.predict(X_adv)
                acc    = accuracy_score(y_adv, y_pred)

                records.append({
                    'perturbation': p_type,
                    'level':        lvl_str,
                    'accuracy':     f"{acc:.4f}"
                })

        # 保存该模型的所有测试结果
        fname = name.replace(' ', '_') + '_perturbation_results.csv'
        pd.DataFrame(records).to_csv(fname, index=False)
        print(f"=> Saved results to {fname}\n")

if __name__ == '__main__':
    main()