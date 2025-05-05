# Traffic Fingerprinting Binary Classification - Scenario A Dataset (CNN)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

df = pd.read_csv("../Scenario-A-merged_5s.csv")
df.columns = df.columns.str.strip()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])  # TOR=1, nonTOR=0

X = df.drop(columns=['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'label'])

print("Missing values (before):", X.isnull().sum().sum())
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)
print("Missing values (after):", X.isnull().sum().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled[..., np.newaxis]  # reshape to (samples, features, 1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42, stratify=y)


model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=2)

y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
print("\n================ Evaluation ================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

model.save("cnn_model.h5")
print("CNN model saved as cnn_model.h5")

input("\nPress Enter to exit...")
