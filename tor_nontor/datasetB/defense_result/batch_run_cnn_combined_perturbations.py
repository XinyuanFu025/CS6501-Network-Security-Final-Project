# batch_run_cnn_combined_perturbations.py

import subprocess
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

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

# -- 定义模型结构

class DFCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(DFCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 32, kernel_size=8)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=5)
        self.relu2 = torch.nn.ReLU()
        self.pool  = torch.nn.AdaptiveMaxPool1d(1)
        self.flatten = torch.nn.Flatten()
        self.fc    = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

# -- 加载模型并推理

def test_with_saved_pytorch(
    model_path='../model/cnn.pth',
    csv_path='../Scenario-B-merged_5s_adversarial.csv'
):
    # 读取并清洗数据
    df = pd.read_csv(csv_path)
    df.drop(columns=['Source IP', 'Destination IP'], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 编码和缩放
    y = LabelEncoder().fit_transform(df['label'])
    X = df.drop(columns=['label'])
    X = StandardScaler().fit_transform(X)

    # 转张量
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=64)

    # 选择设备（MPS 或 CPU）并加载模型
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    num_classes = int(max(y)+1)
    model = DFCNN(num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 推理并返回准确度
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(yb.cpu().numpy())
    return accuracy_score(labels, preds)

# -- 主流程

def main():
    levels = [i * 0.01 for i in range(1, 26)]
    results = []
    for lvl in levels:
        lvl_str = f"{lvl:.2f}"
        print(f"Testing combined perturbation @ level={lvl_str}")
        generate_combined_perturbed_data(lvl)
        acc = test_with_saved_pytorch()
        print(f"Level {lvl_str} => Accuracy {acc:.4f}\n")
        results.append({'level': lvl_str, 'accuracy': acc})

    pd.DataFrame(results).to_csv('cnn_combined_perturbation_results.csv', index=False)
    print('Results saved to cnn_combined_perturbation_results.csv')

if __name__ == '__main__':
    main()
