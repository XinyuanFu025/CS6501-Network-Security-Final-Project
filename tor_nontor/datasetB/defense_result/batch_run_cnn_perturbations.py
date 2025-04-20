# batch_run_cnn_perturbations.py

import subprocess
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# -------------------------
# (1) 扰动数据生成函数
# -------------------------
def generate_perturbed_data(perturb_type, level):
    args = {
        'noise_std':    0.0,
        'scale_low':    1.0,
        'scale_high':   1.0,
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

    cmd = [
        'python', 'generate_perturbation_1.py',
        '--noise_std', str(args['noise_std']),
        '--scale_low', str(args['scale_low']),
        '--scale_high', str(args['scale_high']),
        '--dropout_rate', str(args['dropout_rate']),
        '--permute_rate', str(args['permute_rate']),
    ]
    subprocess.run(cmd, check=True)

# -------------------------
# (2) 定义 PyTorch CNN 模型结构
# -------------------------
class DFCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 32, kernel_size=8)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=5)
        self.relu2 = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveMaxPool1d(1)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

# -------------------------
# (3) 测试函数
# -------------------------
def test_with_saved_pytorch(model_path='../model/cnn.pth',
                            csv_path='../Scenario-B-merged_5s_adversarial.csv'):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Source IP', 'Destination IP'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    y = LabelEncoder().fit_transform(df['label'])
    X = df.drop(columns=['label'])
    X_scaled = StandardScaler().fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor, y_tensor),
        batch_size=64
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(np.unique(y))
    model = DFCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

# -------------------------
# (4) 主流程
# -------------------------
def main():
    perturb_types = ['gaussian', 'random_sc', 'feature_d', 'feature_p']
    # 从0.01开始，每次加0.01，直到0.25
    levels = [i * 0.01 for i in range(1, 26)]  # 0.01, 0.02, ..., 0.25
    results = []

    for p_type in perturb_types:
        for lvl in levels:
            print(f'正在测试 {p_type} @ level={lvl:.2f}')
            generate_perturbed_data(p_type, lvl)
            acc = test_with_saved_pytorch()
            print(f'{p_type} @ {lvl:.2f} => Accuracy: {acc}\n')
            results.append({
                'perturbation': p_type,
                'level':        f"{lvl:.2f}",
                'accuracy':     acc
            })

    # 保存结果：level 保留两位小数字符串，accuracy 原样输出
    df_res = pd.DataFrame(results)
    df_res.to_csv('cnn_perturbation_results.csv', index=False)
    print('所有结果已保存至 cnn_perturbation_results.csv')

if __name__ == '__main__':
    main()
