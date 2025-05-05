import subprocess
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def generate_perturbed_data(perturb_type, level):
    args = {
        'noise_std': 0.0,
        'scale_low': 1.0,
        'scale_high': 1.0,
        'dropout_rate': 0.0,
        'permute_rate': 0.0
    }

    if perturb_type == 'gaussian':
        args['noise_std'] = level
    elif perturb_type == 'random_sc':
        args['scale_low'] = 1.0 - level
        args['scale_high'] = 1.0 + level
    elif perturb_type == 'feature_d':
        args['dropout_rate'] = level
    elif perturb_type == 'feature_p':
        args['permute_rate'] = level

    command = [
        'python', 'generate_perturbation_1.py',
        '--noise_std', str(args['noise_std']),
        '--scale_low', str(args['scale_low']),
        '--scale_high', str(args['scale_high']),
        '--dropout_rate', str(args['dropout_rate']),
        '--permute_rate', str(args['permute_rate']),
    ]
    subprocess.run(command, check=True)

#  KNN
def test_with_saved_knn(model_path='knn_model.pkl', csv_path='Scenario-A-merged_5s_adversarial.csv'):
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
    perturb_types = ['gaussian', 'random_sc', 'feature_d', 'feature_p']
    #levels = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]
    levels = [round(x, 2) for x in np.arange(0.01, 0.26, 0.01)]

    results = []

    for p_type in perturb_types:
        for lvl in levels:
            print(f'Testing {p_type} @ level={lvl:.2f}')
            generate_perturbed_data(p_type, lvl)
            acc = test_with_saved_knn()
            print(f'{p_type} @ {lvl:.2f} => Accuracy: {acc:.4f}')
            results.append({
                'perturbation': p_type,
                'level': lvl,
                'accuracy': acc
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv('knn_perturbation_results.csv', index=False)
    print("\nsaved knn_perturbation_results.csv")

if __name__ == '__main__':
    main()
