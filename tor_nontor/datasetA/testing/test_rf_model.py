# Test saved vs. untrained Random Forest model

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load data
df = pd.read_csv("../Scenario-A-merged_5s.csv")
df.columns = df.columns.str.strip()

# Step 2: Label & features
y = df['label']
X = df.drop(columns=['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'label'])

# Step 3: Clean data
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Step 4: Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Create an untrained RF model and test it (just to see bad result)
untrained_model = RandomForestClassifier(n_estimators=100, random_state=42)
try:
    y_pred_dummy = untrained_model.predict(X_test)
    print("You shouldn't see this, because model isn't trained!")
except Exception as e:
    print("Untrained model can't predict (as expected):", e)

# Step 7: Train a dummy model just to compare
dummy_model = RandomForestClassifier(n_estimators=100, random_state=42)
dummy_model.fit(X_train, y_train)
y_dummy_pred = dummy_model.predict(X_test)
print("\n===== Accuracy of new (just trained) model =====")
print("Accuracy:", accuracy_score(y_test, y_dummy_pred))

# Step 8: Load saved model
loaded_model = joblib.load("../models/rf_model.pkl")
y_loaded_pred = loaded_model.predict(X_test)
print("\n===== Accuracy of loaded (pretrained) model =====")
print("Accuracy:", accuracy_score(y_test, y_loaded_pred))
print("Classification Report:")
print(classification_report(y_test, y_loaded_pred))


# Step 9: Random Guessing Baseline
import random
random.seed(42)

unique_labels = y_test.unique()
random_preds = np.random.choice(unique_labels, size=len(y_test))

print("\n===== Accuracy of random guessing (baseline) =====")
print("Accuracy:", accuracy_score(y_test, random_preds))
print("Classification Report:")
print(classification_report(y_test, random_preds))


# Pause
input("\nPress Enter to exit...")
