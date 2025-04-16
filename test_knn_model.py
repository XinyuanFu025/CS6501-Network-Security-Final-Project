# Test saved vs. random vs. untrained KNN model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import random

# Step 1: Load dataset
df = pd.read_csv("tor_nontor/datasetA/Scenario-A-merged_5s.csv")
df.columns = df.columns.str.strip()

# Step 2: Extract features and labels
y = df['label']
X = df.drop(columns=['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'label'])

# Step 3: Handle missing/infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Try untrained KNN (should error)
try:
    untrained_model = KNeighborsClassifier(n_neighbors=5)
    untrained_model.predict(X_test)
    print("Unexpected: Untrained KNN made predictions!")
except Exception as e:
    print("Untrained model can't predict (as expected):", e)

# Step 7: Random guess baseline
random.seed(42)
random_preds = np.random.choice(y_test.unique(), size=len(y_test))

print("\n===== Accuracy of random guessing (baseline) =====")
print("Accuracy:", accuracy_score(y_test, random_preds))
print("Classification Report:")
print(classification_report(y_test, random_preds))

# Step 8: Load trained KNN model
knn_model = joblib.load("knn_model.pkl")
y_pred = knn_model.predict(X_test)

print("\n===== Accuracy of loaded (pretrained) KNN model =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Pause
input("\nPress Enter to exit...")
