# Entraîne le meilleur modèle (Random Forest) et le sauvegarde

import pandas as pd
import numpy as np
import pickle
import os
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("models", exist_ok=True)

SEED = 42

# Chargement et nettoyage
df = pd.read_csv("bostonhousing-693729dedb019653836667.csv")

# Décisions éthiques + nettoyage
df = df.drop(columns=["b"])
df = df[df["medv"] < 50]
Q1, Q3 = df["crim"].quantile(0.25), df["crim"].quantile(0.75)
df["crim"] = df["crim"].clip(upper=Q3 + 3*(Q3-Q1))

X = df.drop(columns=["medv"])
y = df["medv"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

#MLflow tracking
mlflow.set_experiment("fastia_boston_housing")

with mlflow.start_run(run_name="random_forest_final"):

    rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)

    #cross-validation
    kfold   = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rmse = np.sqrt(-cross_val_score(rf, X_train_s, y_train, cv=kfold,
                                        scoring="neg_mean_squared_error"))
    cv_r2   = cross_val_score(rf, X_train_s, y_train, cv=kfold, scoring="r2")

    # Entraînement final
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    # Log MLflow
    mlflow.log_params({
        "model":         "RandomForest",
        "n_estimators":  200,
        "random_state":  SEED,
        "variable_b_supprimee": True,
        "medv_50_supprime":     True,
    })
    mlflow.log_metrics({
        "cv_rmse_mean": round(cv_rmse.mean(), 3),
        "cv_r2_mean":   round(cv_r2.mean(), 3),
        "test_rmse":    round(rmse, 3),
        "test_mae":     round(mae, 3),
        "test_r2":      round(r2, 3),
    })
    mlflow.sklearn.log_model(rf, "random_forest_model")

    print(f"✔  Test RMSE : {rmse:.3f}")
    print(f"✔  Test MAE  : {mae:.3f}")
    print(f"✔  Test R²   : {r2:.3f}")

#sauvegarde
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("✔  Modèle sauvegardé : models/random_forest.pkl")
print("✔  Scaler sauvegardé : models/scaler.pkl")
print("👉 Lance : mlflow ui  pour voir les résultats")