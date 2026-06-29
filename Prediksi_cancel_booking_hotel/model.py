# import library
import os
import kagglehub
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# download dataset
print(f"Download Dataset")
path = kagglehub.dataset_download("youssefaboelwafa/hotel-booking-cancellation-prediction")

if os.path.exists("./datasets"):
    shutil.rmtree("./datasets")

shutil.copytree(path, "./datasets")
filename = os.listdir("./datasets")

# load dataset
print("Load Dataset")
df = pd.read_csv('./datasets/'+filename[0])

# EDA
print("EDA")
# hapus Booking_ID dan date_of_reservation
df = df.drop(columns=['Booking_ID', 'date of reservation'])
# ubah status booking ke biner
df['booking status'] = df['booking status'].map({'Not_Canceled': 0, 'Canceled': 1})

# preprocessing
# pisahkan target dan fitur
print("Preprocessing")
X = df.drop(columns=['booking status'])
y = df['booking status']

kolom_robust = ['lead time', 'average price']
kolom_kategorikal = ['type of meal', 'room type', 'market segment type']
kolom_lainnya = [col for col in X.columns if col not in kolom_robust + kolom_kategorikal]

preprocessor = ColumnTransformer(
    transformers=[
        ('skala_robust', RobustScaler(), kolom_robust),
        
        ('kategori_encode', OneHotEncoder(drop='first', sparse_output=False), kolom_kategorikal)
    ],
    remainder='passthrough' 
)

# split dataset
print("Split dataset")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
# eksekusi preprocessing
X_train_ready = preprocessor.fit_transform(X_train)

X_test_ready = preprocessor.transform(X_test)

kolom_kategori_baru = preprocessor.named_transformers_['kategori_encode'].get_feature_names_out(kolom_kategorikal)

semua_kolom_baru = kolom_robust + list(kolom_kategori_baru) + kolom_lainnya

X_train_ready_df = pd.DataFrame(X_train_ready, columns=semua_kolom_baru)

# membuat baseline model
print("Baseline model")
knn = KNeighborsClassifier()

# Hyperparameter Tuning
param_grid = {
    # Mencoba nilai K ganjil dari 3 sampai 15 untuk menghindari hasil voting seri
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    
    # 'uniform': semua tetangga punya bobot sama. 
    # 'distance': tetangga yang lebih dekat punya pengaruh lebih besar (bagus untuk data dengan outliers)
    'weights': ['uniform', 'distance'],
    
    # 'euclidean': jarak garis lurus standar
    # 'manhattan': jarak blok (terkadang lebih baik untuk data berdimensi banyak / campuran)
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(
    estimator=knn, 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=2,
    verbose=1
)

print("Training model, ini akan memakan waktu")
grid_search.fit(X_train_ready, y_train)

print("\n=== HASIL TUNING ===")
print(f"Kombinasi Parameter Terbaik : {grid_search.best_params_}")
print(f"Akurasi Terbaik (CV Score)  : {grid_search.best_score_ * 100:.2f}%")

best_knn_model = grid_search.best_estimator_

# Prediksi ke data test yang belum pernah dilihat model
y_pred = best_knn_model.predict(X_test_ready)

print("\n=== EVALUASI PADA DATA TESTING ===")
print(f"Akurasi Akhir Data Test : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))