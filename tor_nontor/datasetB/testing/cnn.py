import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Load CSV
df = pd.read_csv("../Scenario-B-merged_5s.csv")

# Drop IP columns and handle infinite values
df = df.drop(columns=['Source IP', 'Destination IP'])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Encode categorical columns
df['label'] = LabelEncoder().fit_transform(df['label'])

# Features and labels
X = df.drop(columns=['label'])
y = df['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-val-test split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# PyTorch dataset
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TabularDataset(X_train, y_train)
val_dataset = TabularDataset(X_val, y_val)
test_dataset = TabularDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# DF-CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DFCNN(nn.Module):
    def __init__(self, num_classes):
        super(DFCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # shape: (B, 1, Features)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = DFCNN(num_classes=len(np.unique(y))).to(device)

# Training function with best model save
def train_model(model, train_loader, val_loader, num_epochs, device, save_path="best_model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        all_preds, all_labels = [], []
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1} with val acc: {val_acc:.4f}")

# Evaluation
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

# Train
# train_model(model, train_loader, val_loader, num_epochs=50, device=device, save_path="../model/cnn.pth")

# Load best model
model.load_state_dict(torch.load("../model/cnn.pth"))
model.to(device)
model.eval()

# Final test accuracy
test_acc = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.4f}")
