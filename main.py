#main.py
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# Load dataset
df = pd.read_csv('Lung Cancer Dataset.csv')


# === FUNCTION: Preprocessing Data ===
def preprocess_data(df):
    df = df.drop_duplicates().dropna()

    le = LabelEncoder()
    df['PULMONARY_DISEASE_encoded'] = le.fit_transform(df['PULMONARY_DISEASE'])

    scaler = MinMaxScaler(feature_range=(0.0, 0.1))
    df[['ENERGY_LEVEL', 'OXYGEN_SATURATION']] = scaler.fit_transform(df[['ENERGY_LEVEL', 'OXYGEN_SATURATION']])

    df['ENERGY_LEVEL_KATEGORI'] = df['ENERGY_LEVEL'].apply(lambda x: 1 if x > 0.05 else 0)
    df['OXYGEN_SATURATION_KATEGORI'] = df['OXYGEN_SATURATION'].apply(lambda x: 1 if x > 0.05 else 0)

    df.drop(columns=['ENERGY_LEVEL', 'OXYGEN_SATURATION'], inplace=True)
    return df


# === PREPROCESSING ===
df_processed = preprocess_data(df)
X = df_processed.drop(['PULMONARY_DISEASE', 'PULMONARY_DISEASE_encoded'], axis=1)
y = df_processed['PULMONARY_DISEASE_encoded']

joblib.dump(X.columns.tolist(), "feature_columns.pkl")

# === SPLIT DATA TRAIN-TEST ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === GRID SEARCH CV ===
print("\n=== Grid Search untuk Hyperparameter Tuning ===")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# === EVALUASI MODEL TERBAIK DI TEST SET ===
best_model = grid_search.best_estimator_

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

print(f"\nTraining Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# === SIMPAN MODEL ===
joblib.dump(best_model, "model_lung_cancer.pkl")

# === FEATURE IMPORTANCE ===
importances = best_model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

print("\nFeature Importance:")
print(feature_importance_df)
