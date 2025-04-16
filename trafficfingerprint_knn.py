# Traffic Fingerprinting Binary Classification - Scenario A Dataset (KNN)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load dataset
df = pd.read_csv("Scenario-A-merged_5s.csv")
df.columns = df.columns.str.strip()

# Step 2: Labels
y = df['label']

# Step 3: Drop irrelevant columns
X = df.drop(columns=['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'label'])

# Step 4: Handle inf & NaNs
print("Missing values (before):", X.isnull().sum().sum())
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)
print("Missing values (after):", X.isnull().sum().sum())

# Step 5: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 7: Train KNN model
model = KNeighborsClassifier(n_neighbors=5)  # 默认取最近5个邻居
model.fit(X_train, y_train)

# Step 8: Save model
joblib.dump(model, "knn_model.pkl")
print("KNN model saved as knn_model.pkl")

# Step 9: Evaluate
y_pred = model.predict(X_test)
print("\n================ Evaluation ================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Pause
input("\nPress Enter to exit...")
