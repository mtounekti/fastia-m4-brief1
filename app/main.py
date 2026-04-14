# API FastAPI – Exposition du modèle Random Forest Boston Housing

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os

# chargement du modèle au démarrage
MODEL_PATH   = "models/random_forest.pkl"
SCALER_PATH  = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    MODEL_CHARGE = True
    print(f"✔  Modèle chargé : {MODEL_PATH}")
except Exception as e:
    MODEL_CHARGE = False
    print(f"❌ Erreur : {e}")

# app
app = FastAPI(
    title       = "FastIA – Prédiction Prix Immobilier Boston",
    description = """
API REST exposant le modèle Random Forest pour prédire
les prix de l'immobilier à Boston.

## Performances
- **R² = 0.882** sur le jeu de test
- **RMSE = 2.452** (milliers de $)

## Note éthique
La variable `b` (proportion de résidents noirs) a été **exclue**
conformément aux principes d'équité en IA et au RGPD Art. 9.
    """,
    version = "1.0.0",
)


# schémas
class DemandePredict(BaseModel):
    crim:    float = Field(..., ge=0,   description="Taux de criminalité")
    zn:      float = Field(..., ge=0,   description="% terrains résidentiels > 25k m²")
    indus:   float = Field(..., ge=0,   description="% zones industrielles")
    chas:    int   = Field(..., ge=0, le=1, description="Charles River (0/1)")
    nox:     float = Field(..., ge=0,   description="Concentration NOx")
    rm:      float = Field(..., ge=1,   description="Nb moyen de pièces")
    age:     float = Field(..., ge=0, le=100, description="% logements avant 1940")
    dis:     float = Field(..., ge=0,   description="Distance centres emploi")
    rad:     int   = Field(..., ge=1,   description="Accessibilité autoroutes")
    tax:     float = Field(..., ge=0,   description="Taux imposition foncière")
    ptratio: float = Field(..., ge=0,   description="Ratio élèves/enseignant")
    lstat:   float = Field(..., ge=0,   description="% population faibles revenus")

    class Config:
        json_schema_extra = {
            "example": {
                "crim": 0.006, "zn": 18.0, "indus": 2.31,
                "chas": 0, "nox": 0.538, "rm": 6.575,
                "age": 65.2, "dis": 4.09, "rad": 1,
                "tax": 296, "ptratio": 15.3, "lstat": 4.98
            }
        }


class ReponsePrediction(BaseModel):
    prix_median_predit_k: float
    prix_median_predit_dollars: int
    intervalle_confiance: dict
    modele: str


@app.get("/", tags=["Accueil"])
def accueil():
    return {
        "message":       "FastIA – API Prix Immobilier Boston 🏠",
        "modele":        "Random Forest (200 estimateurs)",
        "performance":   {"R2": 0.882, "RMSE": 2.452, "MAE": 1.855},
        "modele_charge": MODEL_CHARGE,
        "note_ethique":  "Variable b (race) exclue – RGPD Art. 9",
        "docs":          "/docs",
    }


@app.get("/health", tags=["Santé"])
def health():
    return {
        "status":        "ok" if MODEL_CHARGE else "degraded",
        "modele_charge": MODEL_CHARGE,
    }


@app.post("/predict", response_model=ReponsePrediction, tags=["Prédiction"])
def predire(demande: DemandePredict):
    """
    Prédit le prix médian d'un logement en milliers de dollars.
    La variable `b` (race) est volontairement exclue du modèle.
    """
    if not MODEL_CHARGE:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    try:
        # Construction du vecteur de features dans le bon ordre
        X = np.array([[
            demande.crim, demande.zn, demande.indus, demande.chas,
            demande.nox, demande.rm, demande.age, demande.dis,
            demande.rad, demande.tax, demande.ptratio, demande.lstat
        ]])

        # Scaling
        X_scaled = scaler.transform(X)

        # prédiction + intervalle (std des arbres)
        predictions_arbres = np.array([
            tree.predict(X_scaled) for tree in model.estimators_
        ])
        prix_moyen = float(predictions_arbres.mean())
        prix_std   = float(predictions_arbres.std())

        return ReponsePrediction(
            prix_median_predit_k       = round(prix_moyen, 2),
            prix_median_predit_dollars = int(prix_moyen * 1000),
            intervalle_confiance       = {
                "borne_basse_k": round(prix_moyen - 2*prix_std, 2),
                "borne_haute_k": round(prix_moyen + 2*prix_std, 2),
            },
            modele = "Random Forest – R²=0.882"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))