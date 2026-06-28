# import library
import os
import kagglehub
import shutil

import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# download dataset
print(f"="*50)
print("Download Dataset")
print(f"="*50)

path = kagglehub.dataset_download("gauravtopre/bank-customer-churn-dataset")

if os.path.exists("./datasets"):
    shutil.rmtree("./datasets")

shutil.copytree(path, "./datasets")
filename = os.listdir("./datasets")

# load dataset
print(f"="*50)
print("Load Dataset")
print(f"="*50)
df = pd.read_csv('./datasets/'+filename[0])

# hapus customer_id
df = df.drop(columns=['customer_id'])

# mengubah gender ke biner
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Preprocessing
# membuat object One-Hot-Encoder
print(f"="*50)
print("Preprocessing Dataset")
print(f"="*50)
encoderOHE = OneHotEncoder(sparse_output=False, drop='first')
country_encode = encoderOHE.fit_transform(df[['country']])
nama_kolom = encoderOHE.get_feature_names_out(['country'])
df_country_encode = pd.DataFrame(country_encode, columns=nama_kolom)

# gabungkan data
df = pd.concat([df, df_country_encode], axis=1)
# hapus kolom 'country'
df = df.drop('country', axis=1)

# split dataset
X = df.drop(columns=['churn'])
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.3
)

# melakukan scaling terhadap fitur numeric
numeric_columns = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "estimated_salary"
]

scaler = StandardScaler()

X_train[numeric_columns] = scaler.fit_transform(
    X_train[numeric_columns]
)

X_test[numeric_columns] = scaler.transform(
    X_test[numeric_columns]
)

# melakukan hyper parameter tunning
param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
    "class_weight": [None, "balanced"]
}

# membuat model
print(f"="*50)
print("Training Model")
print(f"="*50)
model = LogisticRegression(
    C=0.01,
    class_weight='balanced',
    penalty='l1',
    solver='liblinear',
)

# training model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
# evaluasi
print(f"="*60)
print("Logistic Regression")
print(f"="*60)
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print(f"="*50)
print("Simpan Model")
print(f"="*50)
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model_churn.pkl')

print("Model berhasil disimpan di folder models!")
