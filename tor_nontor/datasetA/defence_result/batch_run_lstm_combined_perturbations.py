import subprocess
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

# ===LSTM===
def test_with_saved_lstm(model_path='lstm_model.h5', csv_path='Scenario-A-merged_5s_adversarial.csv'):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    y_raw = df['label']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X = df.drop(columns=['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'label'])

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # reshape 
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    model = tf.keras.models.load_model(model_path)
    y_pred_prob = model.predict(X_lstm)
    y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

    acc = accuracy_score(y, y_pred)
    return acc


def main():
    levels = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]
    results = []

    for lvl in levels:
        print(f'正在测试 Combined Perturbation @ level={lvl:.2f}')
        generate_combined_perturbed_data(lvl)
        acc = test_with_saved_lstm()
        print(f'Combined @ {lvl:.2f} => Accuracy: {acc:.4f}')
        results.append({
            'perturbation': 'combined_all',
            'level': lvl,
            'accuracy': acc
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv('lstm_perturbation_all_combined.csv', index=False)
    print("\n所有结果已保存至 lstm_perturbation_all_combined.csv")

if __name__ == '__main__':
    main()
