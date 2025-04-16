# Test saved LSTM model vs random guessing (baseline)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
import random

# Step 1: Load dataset
df = pd.read_csv("../Scenario-A-merged_5s.csv")
df.columns = df.columns.str.strip()

# Step 2: Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])  # TOR=1, nonTOR=0

# Step 3: Drop irrelevant columns
X = df.drop(columns=['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'label'])

# Step 4: Clean and scale
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Reshape for LSTM: (samples, timesteps=1, features)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Random guess baseline
random.seed(42)
random_preds = np.random.choice(np.unique(y_test), size=len(y_test))

print("\n===== Accuracy of random guessing (baseline) =====")
print("Accuracy:", accuracy_score(y_test, random_preds))
print("Classification Report:")
print(classification_report(y_test, random_preds, target_names=label_encoder.classes_))

# Step 8: Load trained LSTM model
model = load_model("../models/lstm_model.h5")
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

print("\n===== Accuracy of loaded (pretrained) LSTM model =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Pause
input("\nPress Enter to exit...")
