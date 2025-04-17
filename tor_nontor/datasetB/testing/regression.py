import pandas as pd
import numpy as np
import ipaddress
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------
# 
# -----------------------------
COMMON_PORTS = {80, 443, 53, 110, 25, 22, 21, 23, 123, 445, 3306, 8080}

def is_private(ip):
    try:
        return ipaddress.ip_address(ip).is_private
    except:
        return False

def extract_ip_port_features(df):
    df['src_private'] = df['Source IP'].apply(is_private).astype(int)
    df['dst_private'] = df['Destination IP'].apply(is_private).astype(int)

    df['src_well_known_port'] = df['Source Port'].apply(lambda x: int(x) < 1024).astype(int)
    df['dst_well_known_port'] = df['Destination Port'].apply(lambda x: int(x) < 1024).astype(int)

    df['src_common_port'] = df['Source Port'].apply(lambda x: int(x) in COMMON_PORTS).astype(int)
    df['dst_common_port'] = df['Destination Port'].apply(lambda x: int(x) in COMMON_PORTS).astype(int)

    df['same_port'] = (df['Source Port'] == df['Destination Port']).astype(int)
    return df

# -----------------------------
# 
# -----------------------------
df = pd.read_csv("../Scenario-B-merged_5s.csv")

# 
# df.columns = df.columns.str.strip().str.replace(r'[^\w\s]', '', regex=True).str.replace(' ', '_').str.lower()

# -----------------------------
#
# -----------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# 
# -----------------------------
df = extract_ip_port_features(df)

# 
df.drop(columns=['Source IP', 'Destination IP', 'Source Port', 'Destination Port'], inplace=True)


le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])



X = df.drop(columns=['label'])
y = df['label']


categorical_features = ['Protocol']
numeric_features = [col for col in X.columns if col not in categorical_features]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),   # 处理缺失值
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


for name, model in models.items():
    
    model = joblib.load(f'../model/{name}.pkl')
   
    
    y_pred = model.predict(X_test)
    print(f"\n{name} Classification Report:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\n", classification_report(y_test, y_pred, target_names=le.classes_))

    
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
