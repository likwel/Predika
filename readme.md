# 🚀 AutoML - Entraînement et Prédiction Automatisés

Un projet **AutoML** permettant de charger un jeu de données, de prétraiter les colonnes (dates, valeurs numériques, catégorielles), d’entraîner plusieurs modèles automatiquement, et de générer des prédictions avec suivi des logs et des performances.

---

## 📦 Installation

### 1. Cloner le projet
```bash
git clone https://github.com/likwel/Predika.git
cd automl
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## ⚙️ Utilisation

### 1. Lancer l’application Streamlit
```bash
streamlit run pdv.py
```

### 2. Charger vos données
- Fichier CSV, Excel ou parquet.  
- Sélectionner la colonne cible (variable à prédire).  
- Les colonnes de type **date** sont automatiquement converties en `datetime`.

### 3. Prétraitement automatique
- Gestion des valeurs manquantes.  
- Encodage des variables catégorielles.  
- Normalisation des données numériques.  

### 4. Entraînement des modèles
- Plusieurs modèles (RandomForest, XGBoost, etc.) sont entraînés.  
- AutoML choisit le **meilleur modèle** selon les métriques (RMSE, MAE, Accuracy…).  
- Suivi en temps réel : *AutoML — Suivi & logs*.

### 5. Générer des prédictions
- Fournir un nouveau dataset ou des valeurs en entrée.  
- Le modèle sélectionné produit la prédiction.  
- Résultats affichés sous forme de tableau et graphiques.

---

## 📊 Exemple de code (prédiction rapide)

```python
from automl import AutoML

# Initialiser AutoML
automl = AutoML(target="price")

# Charger et entraîner
automl.fit("data/train.csv")

# Prédire sur de nouvelles données
pred = automl.predict("data/test.csv")
print(pred.head())
```

---

## ✅ Fonctionnalités
- Conversion automatique des colonnes dates.  
- Gestion des données tabulaires.  
- AutoML avec logs et suivi en temps réel.  
- Interface utilisateur via **Streamlit**.  

---

## 📝 Licence
Projet open-source — libre à utiliser et modifier.
