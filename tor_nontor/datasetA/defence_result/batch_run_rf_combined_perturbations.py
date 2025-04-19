import subprocess
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def generate_combined_perturbed_data(level):
    command = [
        'python', 'generate_perturbation_1.py',
        '--noise_std', str(level),
        '--scale_low', str(1.0 - level),
        '--scale_high', str(1.0 + level),
        '--dropout_rate', str(level),
        '--permute_rate', str(level),
    ]
    subprocess.run(command, check=True)

# === RF ===
def test_with_saved_rf(model_path='rf_model.pkl', csv_path='Scenario-A-merged_5s_adversarial.csv'):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    y = df['label']
    X = df.drop(columns=['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'label'])

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = joblib.load(model_path)
    y_pred = model.predict(X_scaled)
    acc = accuracy_score(y, y_pred)
    return acc


def main():
    levels = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]
    results = []

    for lvl in levels:
        print(f'测试 Combined Perturbation @ level={lvl:.2f}')
        generate_combined_perturbed_data(lvl)
        acc = test_with_saved_rf()
        print(f'Combined @ {lvl:.2f} => Accuracy: {acc:.4f}')
        results.append({
            'perturbation': 'combined_all',
            'level': lvl,
            'accuracy': acc
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv('rf_perturbation_all_combined.csv', index=False)
    print("\n所有结果已保存至 rf_perturbation_all_combined.csv")

if __name__ == '__main__':
    main()
