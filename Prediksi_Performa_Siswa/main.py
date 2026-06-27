# import library
import os
import kagglehub
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# download dataset
path = kagglehub.dataset_download("nikhil7280/student-performance-multiple-linear-regression")

if os.path.exists("./datasets"):
    shutil.rmtree("./datasets")

shutil.copytree(path, "./datasets")
filename = os.listdir("./datasets")[0]
path = os.path.join("./datasets", filename)

# load dataset
df = pd.read_csv(path)

# ubah extracurricular activities menjadi biner
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# menambahkan fitur baru
# mengalikan hours studied dengan previous score
df['study_score_interaction'] = df['Hours Studied'] * df['Previous Scores']
# Ditambah 0.1 di pembagi agar tidak terjadi error pembagian dengan angka nol
df['sleep_to_study_ratio'] = df['Sleep Hours'] / (df['Hours Studied'] + 0.1)
df['hours_studied_squared'] = df['Hours Studied'] ** 2


cols = [
    'Hours Studied',
    'Previous Scores',
    'Extracurricular Activities',
    'Sleep Hours',
    'Sample Question Papers Practiced',
    'study_score_interaction',
    'sleep_to_study_ratio',
    'hours_studied_squared',  
    'Performance Index'
]

df = df[cols].sort_values(by='Performance Index')

# split dataset
X = df[['Previous Scores', 'study_score_interaction', 'Hours Studied', 'sleep_to_study_ratio']]
y = df['Performance Index']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.3, # 30% data test, dan 70% data training
)
# training model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluasi
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mse_train = mean_squared_error(
    y_train,
    y_train_pred
)

mae_train = mean_absolute_error(
    y_train,
    y_train_pred
)

r2_train = r2_score(
    y_train,
    y_train_pred
)

mse_test = mean_squared_error(
    y_test,
    y_test_pred
)

mae_test = mean_absolute_error(
    y_test,
    y_test_pred
)

r2_test = r2_score(
    y_test,
    y_test_pred
)

print(f"Mean Squared Error test: {mse_test:.2f}")
print(f"Mean Absolute Error test: {mae_test:.2f}")
print(f"R2 score test: {r2_test:.0%}")

print(f"Mean Squared Error train: {mse_train:.2f}")
print(f"Mean Absolute Error train: {mae_train:.2f}")
print(f"R2 score train: {r2_train:.0%}")

# simpan model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/student_performance_model.pkl')

print("Model berhasil disimpan di folder models!")
