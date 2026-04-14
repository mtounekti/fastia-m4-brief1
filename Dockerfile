# Conteneurisation de l'API FastIA – Boston Housing
# Build  : docker build -t fastia-boston .
# Run    : docker run -p 8000:8000 fastia-boston
# Docs   : http://localhost:8000/docs

FROM python:3.9-slim

# Métadonnées
LABEL maintainer="FastIA"
LABEL description="API prédiction prix immobilier Boston – Random Forest"
LABEL version="1.0.0"

# Répertoire de travail
WORKDIR /app

# copie des dépendances en premier
COPY requirements.txt .

# installation des dépendances
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copie du code source
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY bostonhousing-693729dedb019653836667.csv .

# entraînement et sauvegarde du modèle au build
RUN mkdir -p models && \
    python3 scripts/train_and_save.py

# port exposé
EXPOSE 8000

# lancement de l'API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]