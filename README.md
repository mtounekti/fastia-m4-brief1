# M4 Brief 1 – Benchmark de Modèles de Régression
### Projet FastIA – Dataset Boston Housing

---

## Description

FastIA souhaite développer un modèle capable de prédire les prix de l'immobilier.
Ce projet benchmark trois modèles de régression sur le dataset Boston Housing,
avec une analyse éthique rigoureuse et un pipeline de préparation des données complet.

---
## Structure du projet

```
fastia-m4-brief1/
├── benchmark_boston.ipynb     # Notebook principal – pipeline complet
├── graphiques/
│   ├── ethique_variable_b.png       # Analyse variable sensible
│   ├── distribution_target.png      # Distribution de medv
│   ├── distributions.png            # Toutes les features
│   ├── correlation.png              # Matrice de corrélation
│   ├── comparaison_modeles.png      # Benchmark RMSE/MAE/R²
│   ├── predictions_vs_realite.png   # Prédictions vs réalité
│   └── feature_importance.png       # Importance des features
├── requirements.txt
└── README.md
```
---

## Description du dataset

| Propriété | Valeur |
|---|---|
| Source | Boston Housing (Harrison & Rubinfeld, 1978) |
| Lignes | 506 |
| Colonnes | 14 (13 features + 1 target) |
| Target | `medv` – valeur médiane des logements (k$) |
| NaN | 0 |

### Variables

| Variable | Description |
|---|---|
| `crim` | Taux de criminalité par habitant |
| `zn` | Proportion de terrains résidentiels > 25 000 m² |
| `indus` | Proportion de zones industrielles |
| `chas` | Indicatrice Charles River (0/1) |
| `nox` | Concentration en oxydes nitriques |
| `rm` | Nombre moyen de pièces par logement |
| `age` | Proportion de logements construits avant 1940 |
| `dis` | Distance aux centres d'emploi |
| `rad` | Accessibilité aux autoroutes |
| `tax` | Taux d'imposition foncière |
| `ptratio` | Ratio élèves/enseignant |
| ~~`b`~~ | ❌ **Supprimée** – discrimination raciale |
| `lstat` | % population à faibles revenus |
| `medv` |  **Target** – prix médian (k$) |

---

## Analyse Éthique

### Controverse du Dataset Boston Housing

Ce dataset a été **retiré de scikit-learn en 2023** en raison de la variable `b` :

> La colonne `b` encode `1000(Bk - 0.63)²` où `Bk` est la proportion de résidents noirs.
> Son utilisation dans un modèle de prédiction de prix immobiliers constitue une
> **discrimination raciale directe**, interdite par le RGPD (Art. 9) et contraire
> aux principes d'équité en IA.

**Décision : suppression de `b` avant tout entraînement.**

### Autres risques éthiques résiduels

- `lstat` (% pauvres) et `crim` peuvent être des **proxies de discrimination** indirecte
- Les **valeurs censurées à 50k$** introduisent un biais sur les logements chers
- Dataset de **1978** — les patterns socio-économiques ont évolué
- Ce modèle **ne doit pas être utilisé** pour des décisions réelles sans audit éthique

---

## Pipeline de traitement

1. **Suppression** de `b` (discrimination raciale)
2. **Suppression** des lignes `medv == 50` (valeurs censurées)
3. **Winsorisation** de `crim` (IQR × 3, distribution très asymétrique)
4. **Split** train/test (80/20, random_state=42)
5. **Normalisation** StandardScaler

---

## Benchmark des modèles

### Résultats (5-fold cross-validation + test set)

| Modèle | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| Régression Linéaire | 3.52 | 2.662 | 0.757 |
| Random Forest | **2.452** | **1.855** | **0.882** |
| LightGBM | 2.609 | 1.890 | 0.867 |

### Interprétation

- **Random Forest** obtient les meilleures performances sur toutes les métriques
- **LightGBM** est très proche (R²=0.867) avec un temps d'entraînement plus rapide
- La **Régression Linéaire** (baseline) confirme que les relations sont non-linéaires
- Les features les plus prédictives : **`lstat`** (% pauvres) et **`rm`** (nb pièces)

### Limites du modèle

- R² de 0.88 → 12% de variance non expliquée
- Performances dégradées sur les prix extrêmes (effet du plafond à 50k$)
- Modèle entraîné sur des données de 1978 → non généralisable

---

## Installation et utilisation

```bash
# Cloner le repo
git clone https://github.com/mtounekti/fastia-m4-brief1.git
cd fastia-m4-brief1

# Environnement virtuel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Installer libomp (Mac uniquement, pour LightGBM)
brew install libomp

# Lancer le notebook
jupyter notebook benchmark_boston.ipynb

# Ou exécuter en ligne de commande
jupyter nbconvert --to notebook --execute benchmark_boston.ipynb --output benchmark_boston.ipynb
```
---

## 🔗 Références

- [Harrison & Rubinfeld (1978) – Dataset original](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
- [Carlini et al. (2021) – Critique éthique du dataset](https://fairlearn.org/main/user_guide/fairness_in_machine_learning.html)
- [scikit-learn – Retrait du dataset (2023)](https://github.com/scikit-learn/scikit-learn/issues/16155)

---

*Brief M4 – Benchmark Régression | FastIA 2025*