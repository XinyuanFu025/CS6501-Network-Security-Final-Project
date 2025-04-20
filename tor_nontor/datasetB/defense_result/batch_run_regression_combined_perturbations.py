# batch_run_regression_combined_perturbations.py

import subprocess
import pandas as pd
import numpy as np
import ipaddress
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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

def generate_combined_perturbed_data(level: float):
    cmd = [
        'python', 'generate_perturbation_1.py',
        '--noise_std',    str(level),
        '--scale_low',    str(1.0 - level),
        '--scale_high',   str(1.0 + level),
        '--dropout_rate', str(level),
        '--permute_rate', str(level),
    ]
    subprocess.run(cmd, check=True)

def load_adv_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df = extract_ip_port_features(df)
    df.drop(columns=['Source IP', 'Destination IP', 'Source Port', 'Destination Port'], inplace=True)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    X = df.drop(columns=['label'])
    y = df['label'].values
    return X, y

def main():
    model_names = ['KNN', 'Random Forest', 'SVM']
    levels = [i * 0.01 for i in range(1, 26)]
    adv_csv = '../Scenario-B-merged_5s_adversarial.csv'

    for name in model_names:
        model = joblib.load(f'../model/{name}.pkl')
        records = []

        for lvl in levels:
            lvl_str = f"{lvl:.2f}"
            print(f"[{name}] Testing combined perturb @ level={lvl_str}")
            generate_combined_perturbed_data(lvl)
            X_adv, y_adv = load_adv_data(adv_csv)
            y_pred = model.predict(X_adv)
            acc = accuracy_score(y_adv, y_pred)
            print(f"  → Accuracy {acc:.4f}")
            records.append({'level': lvl_str, 'accuracy': f"{acc:.4f}"})

        out_file = name.replace(' ', '_') + '_combined_perturbation_results.csv'
        pd.DataFrame(records).to_csv(out_file, index=False)
        print(f"=> Saved results to {out_file}\n")

if __name__ == '__main__':
    main()