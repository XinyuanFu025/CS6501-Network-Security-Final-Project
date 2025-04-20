# batch_run_lstm_combined_perturbations.py

import subprocess
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# -- 生成组合扰动数据
def generate_combined_perturbed_data(level):
    cmd = [
        'python', 'generate_perturbation_1.py',
        '--noise_std',    str(level),
        '--scale_low',    str(1.0 - level),
        '--scale_high',   str(1.0 + level),
        '--dropout_rate', str(level),
        '--permute_rate', str(level),
    ]
    subprocess.run(cmd, check=True)

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
        # x: (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# -- 加载数据并做推理，返回准确度
def test_with_saved_pytorch(
    model_path='../model/lstm.pth',
    csv_path='../Scenario-B-merged_5s_adversarial.csv'
):
    # 读取并清洗数据
    df = pd.read_csv(csv_path)
    df.drop(columns=['Source IP', 'Destination IP'], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 标签编码 & 特征缩放
    y = LabelEncoder().fit_transform(df['label'])
    X = df.drop(columns=['label'])
    X = StandardScaler().fit_transform(X)

    # 转为 DataLoader
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(tensor_X, tensor_y)
    loader = DataLoader(dataset, batch_size=64)

    # 设备
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    # 构建模型并加载权重
    input_dim = X.shape[1]
    num_classes = int(np.max(y) + 1)
    model = LSTMClassifier(input_dim, hidden_dim=64, output_dim=num_classes).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 推理
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
    levels = [i * 0.01 for i in range(1, 26)]  # 0.01, 0.02, ..., 0.25
    results = []
    for lvl in levels:
        lvl_str = f"{lvl:.2f}"
        print(f"Testing LSTM combined perturbation @ level={lvl_str}")
        generate_combined_perturbed_data(lvl)
        acc = test_with_saved_pytorch()
        print(f"Level {lvl_str} => Accuracy {acc:.4f}\n")
        results.append({'level': lvl_str, 'accuracy': acc})

    # 保存结果
    pd.DataFrame(results).to_csv('lstm_combined_perturbation_results.csv', index=False)
    print('Results saved to lstm_combined_perturbation_results.csv')

if __name__ == '__main__':
    main()