# batch_run_lstm_perturbations.py

import subprocess
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# -- 定义 LSTM 分类器
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True
        )
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, F) -> (B, 1, F)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# -- 生成单一扰动数据
def generate_single_perturbed_data(perturb_type, level):
    # 默认不扰动的参数
    params = {
        'noise_std':    0.0,
        'scale_low':    1.0,
        'scale_high':   1.0,
        'dropout_rate': 0.0,
        'permute_rate': 0.0,
    }

    # 根据新命名映射到对应参数
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

# -- 测试函数：加载对抗样本并评估 LSTM 准确度
def test_with_saved_pytorch(
    model_path='../model/lstm.pth',
    csv_path='../Scenario-B-merged_5s_adversarial.csv'
):
    df = pd.read_csv(csv_path)
    df.drop(columns=['Source IP', 'Destination IP'], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    y = LabelEncoder().fit_transform(df['label'])
    X = df.drop(columns=['label'])
    X = StandardScaler().fit_transform(X)

    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(TensorDataset(tensor_X, tensor_y), batch_size=64)

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    input_dim = X.shape[1]
    num_classes = int(np.max(y) + 1)
    model = LSTMClassifier(input_dim, hidden_dim=64, output_dim=num_classes).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            labels.extend(yb.cpu().numpy())
    return accuracy_score(labels, preds)

# -- 主流程
def main():
    perturb_types = ['gaussian', 'random_sc', 'feature_d', 'feature_p']
    levels = [i * 0.01 for i in range(1, 26)]
    records = []

    for p_type in perturb_types:
        for lvl in levels:
            lvl_str = f"{lvl:.2f}"
            print(f"Testing {p_type} @ level={lvl_str}")
            generate_single_perturbed_data(p_type, lvl)
            acc = test_with_saved_pytorch()
            print(f"  → {p_type} {lvl_str} => Accuracy {acc:.4f}")
            records.append({
                'perturbation': p_type,
                'level':        lvl_str,
                'accuracy':     f"{acc:.4f}"
            })

    # 保存到 CSV
    df_res = pd.DataFrame(records)
    df_res.to_csv('lstm_perturbation_results.csv', index=False)
    print('All results saved to lstm_perturbation_results.csv')

if __name__ == '__main__':
    main()