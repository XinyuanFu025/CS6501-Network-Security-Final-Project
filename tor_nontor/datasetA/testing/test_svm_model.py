# Test saved vs. random vs. untrained SVM model

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import random

# Step 1: Load data
df = pd.read_csv("../Scenario-A-merged_5s.csv")
df.columns = df.columns.str.strip()

# Step 2: Label & features
y = df['label']
X = df.drop(columns=['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'label'])

# Step 3: Clean
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Step 4: Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Try untrained SVM prediction
untrained_model = SVC()
try:
    untrained_model.predict(X_test)
    print("Unexpected: Untrained SVM made predictions!")
except Exception as e:
    print("Untrained model can't predict (as expected):", e)

# Step 7: Random guess baseline
random.seed(42)
random_preds = np.random.choice(y_test.unique(), size=len(y_test))

print("\n===== Accuracy of random guessing (baseline) =====")
print("Accuracy:", accuracy_score(y_test, random_preds))
print("Classification Report:")
print(classification_report(y_test, random_preds))

# Step 8: Load trained SVM model
svm_model = joblib.load("../model/svm_model.pkl")
y_pred = svm_model.predict(X_test)

print("\n===== Accuracy of loaded (pretrained) SVM model =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Pause
input("\nPress Enter to exit...")
