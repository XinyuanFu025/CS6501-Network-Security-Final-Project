import pandas as pd
import numpy as np

def add_gaussian_noise(df: pd.DataFrame, std: float = 0.01) -> pd.DataFrame:
    """
    Add Gaussian noise to each feature value.
    :param df: DataFrame of feature columns.
    :param std: Standard deviation of the noise relative to feature scale.
    :return: DataFrame with added noise.
    """
    noise = np.random.normal(0, std, df.shape)
    return df + noise


def random_scaling(df: pd.DataFrame, low: float = 0.9, high: float = 1.1) -> pd.DataFrame:
    """
    Apply random scaling to each feature column independently.
    :param df: DataFrame of feature columns.
    :param low: Minimum scaling factor.
    :param high: Maximum scaling factor.
    :return: Scaled DataFrame.
    """
    scales = np.random.uniform(low, high, size=(1, df.shape[1]))
    return df * scales


def feature_dropout(df: pd.DataFrame, dropout_rate: float = 0.1) -> pd.DataFrame:
    """
    Randomly set a fraction of feature values to zero.
    :param df: DataFrame of feature columns.
    :param dropout_rate: Proportion of values to drop.
    :return: DataFrame with dropped values.
    """
    mask = np.random.rand(*df.shape) < dropout_rate
    df_dropped = df.copy()
    df_dropped[mask] = 0
    return df_dropped


def feature_permutation(df: pd.DataFrame, permute_rate: float = 0.1) -> pd.DataFrame:
    """
    Randomly permute values of a subset of feature columns across samples.
    :param df: DataFrame of feature columns.
    :param permute_rate: Proportion of columns to permute.
    :return: Permuted DataFrame.
    """
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
    noise_std: float = 0.01,
    scale_range: tuple = (0.9, 1.1),
    dropout_rate: float = 0.1,
    permute_rate: float = 0.1
) -> pd.DataFrame:
    """
    Create adversarially perturbed data from the original DataFrame.
    Excludes non-feature columns, applies all four perturbations, then reassembles.

    :param df: Original DataFrame including metadata and label.
    :param noise_std: Std for Gaussian noise.
    :param scale_range: (min, max) for random scaling.
    :param dropout_rate: Rate for feature dropout.
    :param permute_rate: Rate for feature permutation.
    :return: New DataFrame with adversarial perturbations.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    exclude_cols = [
        'Source IP', 'Source Port',
        'Destination IP', 'Destination Port',
        'Protocol', 'label'
    ]

    df_meta = df[exclude_cols].reset_index(drop=True)
    df_feat = df.drop(columns=exclude_cols).reset_index(drop=True)

    perturbed = df_feat.copy()
    perturbed = add_gaussian_noise(perturbed, std=noise_std)
    perturbed = random_scaling(perturbed, low=scale_range[0], high=scale_range[1])
    perturbed = feature_dropout(perturbed, dropout_rate=dropout_rate)
    perturbed = feature_permutation(perturbed, permute_rate=permute_rate)

    df_adv = pd.concat([df_meta, perturbed], axis=1)
    return df_adv


if __name__ == '__main__':
    # Example usage
    input_path = '../Scenario-A-merged_5s.csv'
    output_path = '../Scenario-A-merged_5s_adversarial.csv'

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    df_adv = generate_adversarial_data(
        df,
        noise_std=0.05,
        scale_range=(0.8, 1.2),
        dropout_rate=0.1,
        permute_rate=0.1
    )

    df_adv.to_csv(output_path, index=False)
    print(f'Adversarial data saved to {output_path}')
