# Traffic Fingerprinting Binary Classification - Scenario A Dataset (SVM)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load dataset
df = pd.read_csv("Scenario-A-merged_5s.csv")

# Step 2: Clean column names
df.columns = df.columns.str.strip()

# Step 3: Use existing 'label' column (TOR vs nonTOR)
y = df['label']

# Step 4: Drop irrelevant columns
X = df.drop(columns=['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'label'])

# Step 5: Handle inf and NaNs
print("Missing values (before):", X.isnull().sum().sum())
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)
print("Missing values (after):", X.isnull().sum().sum())

# Step 6: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 8: Train SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale')  # 可调参数
model.fit(X_train, y_train)

# Step 9: Save the trained model
joblib.dump(model, "svm_model.pkl")
print("SVM model saved as svm_model.pkl")

# Step 10: Evaluate
y_pred = model.predict(X_test)
print("\n================ Evaluation ================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Pause
input("\nPress Enter to exit...")
