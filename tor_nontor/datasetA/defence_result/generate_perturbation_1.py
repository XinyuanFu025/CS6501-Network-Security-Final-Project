import pandas as pd
import numpy as np
import argparse

def add_gaussian_noise(df: pd.DataFrame, std: float) -> pd.DataFrame:
    noise = np.random.normal(0, std, df.shape)
    return df + noise

def random_scaling(df: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
    scales = np.random.uniform(low, high, size=(1, df.shape[1]))
    return df * scales

def feature_dropout(df: pd.DataFrame, dropout_rate: float) -> pd.DataFrame:
    mask = np.random.rand(*df.shape) < dropout_rate
    df_dropped = df.copy()
    df_dropped[mask] = 0
    return df_dropped

def feature_permutation(df: pd.DataFrame, permute_rate: float) -> pd.DataFrame:
    df_perm = df.copy()
    n_features = df.shape[1]
    n_permute = max(1, int(n_features * permute_rate))
    cols = df.columns.tolist()
    permuted_cols = np.random.choice(cols, size=n_permute, replace=False)
    for col in permuted_cols:
        df_perm[col] = np.random.permutation(df_perm[col].values)
    return df_perm

def generate_adversarial_data(
    df: pd.DataFrame,
    noise_std: float,
    scale_range: tuple,
    dropout_rate: float,
    permute_rate: float
) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    label_col = df.columns[-1]

    exclude_cols = [
        'Source IP', 'Source Port',
        'Destination IP', 'Destination Port',
        'Protocol'
    ]

    df_meta = df[exclude_cols].reset_index(drop=True) if all(col in df.columns for col in exclude_cols) else pd.DataFrame()
    df_label = df[[label_col]].reset_index(drop=True)
    df_feat = df.drop(columns=exclude_cols + [label_col], errors='ignore').reset_index(drop=True)

    perturbed = df_feat.copy()
    perturbed = add_gaussian_noise(perturbed, std=noise_std)
    perturbed = random_scaling(perturbed, low=scale_range[0], high=scale_range[1])
    perturbed = feature_dropout(perturbed, dropout_rate=dropout_rate)
    perturbed = feature_permutation(perturbed, permute_rate=permute_rate)

    df_adv = pd.concat([df_meta, perturbed, df_label], axis=1)
    return df_adv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate adversarial perturbations.')
    parser.add_argument('--noise_std', type=float, default=0.05, help='Standard deviation for Gaussian noise.')
    parser.add_argument('--scale_low', type=float, default=0.8, help='Lower bound for scaling.')
    parser.add_argument('--scale_high', type=float, default=1.2, help='Upper bound for scaling.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--permute_rate', type=float, default=0.1, help='Permutation rate.')

    args = parser.parse_args()

    input_path = 'Scenario-A-merged_5s.csv'
    output_path = 'Scenario-A-merged_5s_adversarial.csv'

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    df_adv = generate_adversarial_data(
        df,
        noise_std=args.noise_std,
        scale_range=(args.scale_low, args.scale_high),
        dropout_rate=args.dropout_rate,
        permute_rate=args.permute_rate
    )

    df_adv.to_csv(output_path, index=False)
    print(f'Adversarial data saved to {output_path}')
