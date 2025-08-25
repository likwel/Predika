# app_prevision_vente_final_v23.py
# -------------------------------------------------------------------
# - Navigation stable (pending_nav + rerun)
# - Schéma du workflow (image) en étape "Workflow" via Matplotlib
# - Import: Fichier / API / SQL / MongoDB (retours (None,None) si pas cliqué)
# - Détection auto + choix manuel du type de données (dans Import)
# - TS -> régression sur lags ; Tabulaire -> régression ou classification
# - Projets .pdv, snapshots CSV, historique, restauration, suppression
# - TOUS les graphiques (EDA, éval, ROC, TS) en Chart.js (sauf workflow)
# - Couleur des prédictions personnalisable (sidebar)
# - Légendes d'axes X/Y ajoutées partout
# - Descriptions sous chaque graphique pour guider l'utilisateur
# - ANALYSE : retirer des colonnes du dataset (persistant dans le .pdv)
# - PROJET : choisir un dataset existant du projet et passer l'import
# -------------------------------------------------------------------

import os
import io
import json
import shutil
from pathlib import Path
from datetime import datetime
import time, traceback

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from streamlit.components.v1 import html as st_html

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Régression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# ========= Encoders cat. (sans dépendances externes) =========
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode chaque catégorie par sa fréquence d'apparition (entre 0 et 1)."""
    def __init__(self, columns=None):
        self.columns = columns
        self.freqs_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        cols = self.columns or X.columns.tolist()
        self.columns_ = cols
        self.freqs_ = {}
        for c in cols:
            vc = X[c].astype(str).value_counts(normalize=True)
            self.freqs_[c] = vc.to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.columns_:
            freq_map = self.freqs_.get(c, {})
            X[c] = X[c].astype(str).map(freq_map).fillna(0.0)
        return X

class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """
    Mean target encoding avec lissage simple. 
    ⚠️ Recommandé: régression ou classification BINAIRE.
    Pour multiclasses => fallback auto à la fréquence.
    """
    def __init__(self, columns=None, smoothing=10.0):
        self.columns = columns
        self.smoothing = float(smoothing)
        self.global_mean_ = None
        self.maps_ = {}
        self.binary_ = True

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        cols = self.columns or X.columns.tolist()
        self.columns_ = cols

        y = pd.Series(y)
        # Détecter binaire
        self.binary_ = (y.nunique() == 2)
        if not self.binary_:
            # On ne gère pas le multiclasses ici => signaler via map vide
            self.maps_ = {}
            return self

        # Pour classification binaire: transformer y en {0,1} si besoin
        if y.dtype.kind not in "biufc":
            y = y.astype(str)
        # si non num, on essaie de map binaire automatiquement
        if y.dtype.kind not in "biufc":
            pos = y.unique()[0]
            y = (y == pos).astype(float)

        self.global_mean_ = float(y.mean())
        self.maps_ = {}
        for c in cols:
            g = X[c].astype(str).groupby(X[c].astype(str))
            stats = g.apply(lambda s: y.loc[s.index]).agg(['mean','count']).T
            # stats ici n’est pas triviale, on refait proprement:
            dfc = pd.DataFrame({
                'cat': X[c].astype(str).values,
                'y': y.values
            })
            agg = dfc.groupby('cat')['y'].agg(['mean','count'])
            # lissage
            m = agg['mean']
            n = agg['count']
            smooth = (n * m + self.smoothing * self.global_mean_) / (n + self.smoothing)
            self.maps_[c] = smooth.to_dict()
        return self

    def transform(self, X):
        if not self.binary_ or not self.maps_:
            # fallback: fréquences si multiclasses
            return FrequencyEncoder(self.columns_).fit(X).transform(X)

        X = pd.DataFrame(X).copy()
        for c in self.columns_:
            mp = self.maps_.get(c, {})
            X[c] = X[c].astype(str).map(mp).fillna(self.global_mean_ if self.global_mean_ is not None else 0.0)
        return X

# ----------- CSS -------------
st.markdown("""
    <style>
        .st-emotion-cache-9ko04w {
            border: 1px solid darkgrey;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            width:100%;
        }
        #prevision-de-vente-auto-ml{
            padding : 0 1rem;
        }
        .chart-desc{
            margin: -35px 0 18px;
            color:#6c757d;
            font-size:0.95rem;
            border-bottom: 1px solid darkgray;
            padding-bottom: 1rem;
            font-style: italic
        }
        .stMainBlockContainer{
            padding: 2rem 2rem 2rem 3rem;
        }
        [data-testid="stSidebarHeader"]{
            margin-top: 1rem;
            margin-bottom : 0;
            height : 1rem !important;
        }
        button[kind="header"] svg {
            fill: #ff5722 !important;  /* orange exemple */
        }
        [data-testid="collapsedControl"] svg {
            fill: #4CAF50 !important; /* vert exemple */
        }
    </style>
""", unsafe_allow_html=True)

# --- Badge "Projet actif" aligné à droite (près de Deploy / Settings)
project_title = st.session_state.get("project_name") or "Aucun projet"

# --- Scroll-to-top infra ---
ss = st.session_state
ss.setdefault('__scroll_to_top__', False)

def inject_scroll_if_needed():
    if ss.get('__scroll_to_top__'):
        st.markdown(
            """
            <script>
            // remet en haut dès le chargement du run suivant
            window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
            </script>
            """,
            unsafe_allow_html=True
        )
        ss['__scroll_to_top__'] = False


# ---------- Config ----------
st.set_page_config(page_title="Prévision de Vente AutoML", layout="wide")

inject_scroll_if_needed()

# st.title("Prévision de Vente AutoML ✅")

BASE_DIR = Path.cwd()
PROJECTS_DIR = BASE_DIR / "projets"
PROJECTS_DIR.mkdir(exist_ok=True)

# Étapes
steps = ["Workflow", "Projet", "Import données", "Analyse", "AutoML", "Prédiction", "Historique"]
button_titles = {
    "Workflow": "Démarrer ▶️",
    "Projet": "Passer à l'import ▶️",
    "Import données": "Passer à l'analyse ▶️",
    "Analyse": "Lancer l’apprentissage des modèles ▶️",
    "AutoML": "Passer à la prédiction ▶️",
    "Prédiction": "Passer à l'historique ▶️",
    "Historique": "Fin ▶️"
}

ss.setdefault('nav_step', steps[0])
ss.setdefault('step_done', {s: False for s in steps})

# ---------- IMPORTANT: appliquer la navigation en attente AVANT de créer les widgets ----------
if 'pending_nav' in ss:
    ss['nav_step'] = ss.pop('pending_nav')

# Projet
ss.setdefault('project_name', None)
ss.setdefault('project_slug', None)
ss.setdefault('project_dir', None)
ss.setdefault('project_file', None)
ss.setdefault('created_at', None)

# Données & config
ss.setdefault('df', None)
ss.setdefault('data_file', None)
ss.setdefault('source_type', 'file')        # file | api | sql | nosql
ss.setdefault('source_details', {})         # méta (url, query, etc.)
ss.setdefault('auto_data_type', "tabular")
ss.setdefault('data_type', "tabular")
ss.setdefault('date_col', None)
ss.setdefault('target_column', None)
ss.setdefault('num_cols', [])
ss.setdefault('cat_cols', [])
ss.setdefault('X_columns', [])
ss.setdefault('drop_cols', [])  # colonnes à exclure du dataset (persisté)
ss.setdefault("target_nan_strategy_reg", "drop")  # "drop", "mean", "median", "ffill", "bfill", "zero"

# Tâche & AutoML
ss.setdefault('task_type', "regression")  # regression | classification
ss.setdefault('models_selected', [])
ss.setdefault('models', {})
ss.setdefault('metrics', {})
ss.setdefault('predictions', {})
ss.setdefault('y_test', None)
ss.setdefault('X_test', None)

# Prétraitement & évaluation
ss.setdefault('scale_numeric', True)
ss.setdefault('encode_categorical', True)
ss.setdefault('impute_strategy_num', "mean")
ss.setdefault('impute_strategy_cat', "most_frequent")
ss.setdefault('use_date_features', True)
ss.setdefault('test_size', 0.2)
ss.setdefault('show_graphs', True)
ss.setdefault('n_lags', 6)
ss.setdefault('horizon', 7)

# -- Etats pour le choix de type après sélection d'un dataset du projet
ss.setdefault('pending_project_ds', None)     # chemin Path() du dataset choisi mais pas encore chargé
ss.setdefault('await_dtype_choice', False)    # si True, on doit afficher le choix "tabular" / "time_series"
ss.setdefault('proj_dtype_choice', None)      # mémoire du choix de l'utilisateur

ss.setdefault('is_training', False)     # True pendant l'entraînement
ss.setdefault('automl_logs', [])        # persistance des logs (optionnel mais pratique)

# ---------- Utils ----------
def theme_primary():
    try:
        c = st.get_option("theme.primaryColor")
    except Exception:
        c = None
    return c or "#2C7BE5"

PRIMARY = theme_primary()
# Couleur utilisée pour les tracés de PRÉDICTION
ss.setdefault('pred_color', '#FF7F0E')  # orange par défaut

# primary = theme_primary() if 'theme_primary' in globals() else "#2C7BE5"
# --- Titre fixe en haut-gauche de la navbar Streamlit
st.markdown(f"""
<style>
  .custom-title {{
    background: {PRIMARY}1A; /* léger fond translucide */
    backdrop-filter: blur(4px);
    border: 1px solid {PRIMARY}55;
    font-size: 16px;
    font-weight: 600;
    color: #0f172a;
    padding:10px 20px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    # color:white;
    border-radius:8px;
    margin-bottom:15px;
  }}
  .custom-title .dot {{
    width: 8px; 
    height: 8px; 
    border-radius: 50%;
    background: {PRIMARY};
    box-shadow: 0 0 0 0 {PRIMARY}66;
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0% {{ box-shadow: 0 0 0 0 {PRIMARY}66; }}
    70% {{ box-shadow: 0 0 0 8px rgba(0,0,0,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(0,0,0,0); }}
  }}

</style>
""", unsafe_allow_html=True)

# Supposons que tu as stocké le nom du projet actif :
project_title = st.session_state.get("project_name", "Aucun projet actif")

st.markdown(
    f"""
    <div class="custom-title">
        <div style="font-weight:bold;">🚀 Prévision des Ventes</div>
        <div class="project" style="font-size:16px;"> 📂 Projet actif : <b style="color:#22bb33">{project_title} </b></div>
    </div>
    """,
    unsafe_allow_html=True
)

def sanitize_target_for_regression(df: pd.DataFrame, target_col: str, strategy: str = "drop"):
    """
    Nettoie la colonne cible (y) pour la régression.
    - Convertit en numérique (coerce -> NaN).
    - Stratégies:
        * drop  : supprime les lignes où y est NaN (par défaut)
        * mean  : impute par la moyenne
        * median: impute par la médiane
        * ffill : impute par propagation vers l'avant
        * bfill : impute par propagation vers l'arrière
        * zero  : impute par 0
    Retourne (df_clean, info).
    """
    df2 = df.copy()
    df2[target_col] = pd.to_numeric(df2[target_col], errors="coerce")
    y = df2[target_col]
    n_total = len(y)
    n_nan = int(y.isna().sum())
    info = {"n_total": n_total, "n_nan": n_nan, "strategy": strategy}

    if n_nan > 0:
        if strategy == "mean":
            fill_val = float(y.mean())
            df2[target_col] = y.fillna(fill_val)
            info.update({"strategy_used": "mean", "fill_val": fill_val, "n_imputed": n_nan})
        elif strategy == "median":
            fill_val = float(y.median())
            df2[target_col] = y.fillna(fill_val)
            info.update({"strategy_used": "median", "fill_val": fill_val, "n_imputed": n_nan})
        elif strategy == "ffill":
            df2[target_col] = y.ffill()
            # si début NaN, complète avec bfill
            df2[target_col] = df2[target_col].bfill()
            info.update({"strategy_used": "ffill+bfill", "n_imputed": n_nan})
        elif strategy == "bfill":
            df2[target_col] = y.bfill()
            df2[target_col] = df2[target_col].ffill()
            info.update({"strategy_used": "bfill+ffill", "n_imputed": n_nan})
        elif strategy == "zero":
            df2[target_col] = y.fillna(0.0)
            info.update({"strategy_used": "zero", "fill_val": 0.0, "n_imputed": n_nan})
        else:  # "drop"
            df2 = df2.loc[y.notna()].copy()
            info.update({"strategy_used": "drop", "n_dropped": n_nan})
    else:
        info["strategy_used"] = "none"

    # sécurité finale
    if df2[target_col].isna().any():
        # Si l'imputation n'a pas tout couvert, remove hard
        df2 = df2.loc[df2[target_col].notna()].copy()
        info["post_cleanup_dropped"] = int(df2[target_col].isna().sum())

    return df2, info

def sanitize_target_for_classification(df: pd.DataFrame, target_col: str, strategy: str = "drop"):
    """
    Nettoie la colonne cible pour la classification.
    - Remplace '' / ' ' / 'NaN' strings par NaN.
    - Soit DROP les lignes où y est NaN, soit IMPUTE par la modalité majoritaire.
    Retourne df_clean (copie), info dict.
    """
    df2 = df.copy()
    # Normaliser quelques valeurs vides fréquentes
    df2[target_col] = (
        df2[target_col]
        .replace(["", " ", "NaN", "nan", "None"], np.nan)
    )
    y = df2[target_col]
    n_total = len(y)
    n_nan = int(y.isna().sum())
    info = {"n_total": n_total, "n_nan": n_nan, "strategy": strategy}

    if n_nan > 0:
        if strategy == "mode":
            mode_vals = y.mode(dropna=True)
            if len(mode_vals) == 0:
                # rien à imputer -> fallback drop
                df2 = df2.loc[y.notna()].copy()
                info["strategy_used"] = "drop"
                info["n_dropped"] = n_nan
            else:
                fill_val = mode_vals.iloc[0]
                df2[target_col] = y.fillna(fill_val)
                info["strategy_used"] = "mode"
                info["fill_val"] = fill_val
                info["n_imputed"] = n_nan
        else:
            # drop (par défaut)
            df2 = df2.loc[y.notna()].copy()
            info["strategy_used"] = "drop"
            info["n_dropped"] = n_nan
    else:
        info["strategy_used"] = "none"

    # Après nettoyage : vérifier le nb de classes
    y_clean = df2[target_col]
    info["n_classes"] = int(pd.Series(y_clean).nunique(dropna=True))
    return df2, info
    
def scroll_top():
    st.markdown(
        """
        <script>
        window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
def chart_desc(text: str):
    st.markdown(f"<div class='chart-desc'>ℹ️ <b>À quoi ça sert cette graphe?</b> {text}</div>", unsafe_allow_html=True)

def _json_default(o):
    # sérialisation robuste pour numpy/pandas
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)

def grid_radio(label: str, options: list[str], *, n_cols: int = 3, key: str, default_value: str = None):
    """
    Radio exclusif affiché en grille avec boutons alignés à gauche.
    - options : liste d'options (tu peux y mettre '-- Aucune --')
    - default_value : None (aucune sélection), ou une valeur des options
    Retourne la valeur sélectionnée (str) ou None si aucune.
    """
    st.markdown(f"**{label}**")

    # Init de l'état
    if key not in st.session_state:
        st.session_state[key] = default_value

    selected = st.session_state[key]

    rows = [options[i:i+n_cols] for i in range(0, len(options), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for i, opt in enumerate(row):
            with cols[i]:
                is_sel = (opt == selected)
                icon = "◯" if not is_sel else "⭕"  # creux (gris via CSS thème)
                if st.button(f"{icon} {opt}", key=f"{key}_{opt}"):
                    st.session_state[key] = opt
                    st.rerun()

    return st.session_state[key]

def _deep_update(dst: dict, src: dict):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

# --- encoder numpy -> JSON propre ---
def _np_encoder(o):
    import numpy as _np
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    return o

def _chartjs(chart_type: str, data: dict, options: dict | None = None,
             height: int = 360, extra_scripts: list[str] | None = None):
    """
    Render Chart.js inside Streamlit with *no outer padding* and clean sizing.
    """
    options = options or {}
    extra_scripts = extra_scripts or []
    chart_id = f"chartjs_{np.random.randint(0, 1_000_000)}"

    data_json = json.dumps(data, default=_np_encoder, ensure_ascii=False)
    opts_json = json.dumps(options, default=_np_encoder, ensure_ascii=False)
    scripts = "\n".join([f'<script src="{u}"></script>' for u in extra_scripts])

    html = f"""
    <style>
      html, body {{ margin:0; padding:0; height:100%; }}
      #wrap-{chart_id} {{
        margin:0; padding:0; width:100%; height:{height}px; position:relative;
      }}
      #wrap-{chart_id} canvas {{
        display:block; width:100% !important; height:100% !important;
      }}
    </style>
    <div id="wrap-{chart_id}">
      <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
      {scripts}
      <canvas id="{chart_id}"></canvas>
      <script>
        const ctx = document.getElementById("{chart_id}").getContext('2d');
        const cfg = {{
          type: '{chart_type}',
          data: {data_json},
          options: Object.assign({{
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            layout: {{ padding: {{ top: 6, right: 6, bottom: 28, left: 8 }} }},
            plugins: {{
              legend: {{ display: true, position: 'top' }},
              tooltip: {{ enabled: true }},
              title: {{ display: true}}
            }},
          }}, {opts_json})
        }};
        const chart = new Chart(ctx, cfg);
        new ResizeObserver(() => chart.resize()).observe(document.getElementById("wrap-{chart_id}"));
      </script>
    </div>
    """
    st_html(html, height=height, scrolling=False)

def _chartjs_corr_heatmap(corr_df: pd.DataFrame, x_label="Variables", y_label="Variables", height: int = 420):
    """
    Heatmap de corrélation avec chartjs-chart-matrix.
    """
    labels = list(corr_df.columns)
    mat = corr_df.values.tolist()

    chart_id = f"corrhm_{np.random.randint(0, 1_000_000)}"
    labels_json = json.dumps(labels, ensure_ascii=False)
    values_json = json.dumps(mat)

    html = f"""
    <style>
      html, body {{ margin:0; padding:0; height:100%; }}
      #wrap-{chart_id} {{
        margin:0; padding:0; width:100%; height:{height}px; position:relative;
      }}
      #wrap-{chart_id} canvas {{
        display:block; width:100% !important; height:100% !important;
      }}
    </style>
    <div id="wrap-{chart_id}">
      <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
      <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@2"></script>
      <canvas id="{chart_id}"></canvas>
      <script>
        const labels = {labels_json};
        const mat = {values_json};

        const dataPoints = [];
        for (let i = 0; i < labels.length; i++) {{
          for (let j = 0; j < labels.length; j++) {{
            dataPoints.push({{ x: labels[j], y: labels[i], v: mat[i][j] }});
          }}
        }}

        const lerp = (a,b,t) => a + (b - a) * t;
        function colorFor(v) {{
            const t = (v + 1) / 2; // [-1,1] -> [0,1]
            const r = Math.round(lerp(173, 13, t));
            const g = Math.round(lerp(216, 110, t));
            const b = Math.round(lerp(230, 253, t));
            return `rgba(${{r}}, ${{g}}, ${{b}}, 0.95)`;
        }}

        const ctx = document.getElementById("{chart_id}").getContext('2d');
        const chart = new Chart(ctx, {{
          type: 'matrix',
          data: {{
            datasets: [{{
              label: 'Corrélation',
              data: dataPoints,
              borderWidth: 1,
              borderColor: 'rgba(0,0,0,0.08)',
              backgroundColor: (c) => colorFor(c.raw.v),
              width: (c) => {{
                const a = c.chart.chartArea;
                if (!a) return 10;
                return Math.max(6, (a.width / Math.max(1, labels.length)) - 2);
              }},
              height: (c) => {{
                const a = c.chart.chartArea;
                if (!a) return 10;
                return Math.max(6, (a.height / Math.max(1, labels.length)) - 2);
              }},
            }}]
          }},
          options: {{
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            layout: {{ padding: {{ top: 6, right: 6, bottom: 36, left: 6 }} }},
            plugins: {{
              legend: {{ display: false }},
              tooltip: {{
                callbacks: {{
                  title: (items) => items && items[0] ? (items[0].raw.y + ' × ' + items[0].raw.x) : '',
                  label: (it) => 'corr: ' + Number(it.raw.v).toFixed(2)
                }}
              }}
            }},
            scales: {{
              x: {{
                type: 'category',
                labels: labels,
                grid: {{ display: false }},
                title: {{ display: true, text: '{x_label}' }},
                ticks: {{ autoSkip: false, maxRotation: 45, minRotation: 45 }}
              }},
              y: {{
                type: 'category',
                labels: labels,
                reverse: true,
                grid: {{ display: false }},
                title: {{ display: true, text: '{y_label}' }},
                ticks: {{ autoSkip: false }}
              }}
            }}
          }}
        }});
        new ResizeObserver(() => chart.resize()).observe(document.getElementById("wrap-{chart_id}"));
      </script>
    </div>
    """
    st_html(html, height=height, scrolling=False)

def ds_line(label, values, color, fill=False):
    return {
        "type": "line",
        "label": label,
        "data": [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in values],
        "borderColor": color,
        "backgroundColor": color,
        "borderWidth": 2,
        "pointRadius": 0,
        "tension": 0.4,
        "fill": fill,
        "spanGaps": True,
    }

def format_dates_for_labels(dates: pd.Series) -> list[str]:
    """Formate les dates pour affichage Chart.js : 
       - sans HHMMSS si toutes les heures == 00:00:00
       - sinon avec HHMMSS.
    """
    dates = pd.to_datetime(dates, errors="coerce")
    if dates.dt.hour.nunique() == 1 and dates.dt.hour.iloc[0] == 0 \
       and dates.dt.minute.nunique() == 1 and dates.dt.minute.iloc[0] == 0 \
       and dates.dt.second.nunique() == 1 and dates.dt.second.iloc[0] == 0:
        # toutes les heures sont minuit → afficher que la date
        return dates.dt.strftime("%Y-%m-%d").tolist()
    else:
        return dates.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

def ds_scatter(label, xy_pairs, color):
    return {
        "type": "scatter",
        "label": label,
        "data": [{"x": float(p["x"]), "y": float(p["y"])} for p in xy_pairs],
        "borderColor": color,
        "backgroundColor": color,
        "pointRadius": 3,
        "showLine": False,
    }

def _rgba_from_hex(hex_color: str, alpha: float):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    alpha = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},{b},{alpha})"

def _color_corr(v: float):
    t = min(1.0, max(0.0, abs(float(v))))
    if v >= 0:
        r, g, b = 239, 68, 68
    else:
        r, g, b = 59, 130, 246
    alpha = 0.15 + 0.85*t
    return f"rgba({r},{g},{b},{alpha})"

def normalize_sql_host_user(host_input: str, user_input: str):
    host = (host_input or "").strip()
    user = (user_input or "").strip()
    if "@" in host:
        maybe_user, maybe_host = host.split("@", 1)
        if not user:
            user = maybe_user.strip()
        host = maybe_host.strip()
    host = host.strip()
    user = user.strip()
    return host, user

def slugify(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_ " else "-" for c in name).strip().replace(" ", "_")
    return s.lower() or f"projet_{int(datetime.now().timestamp())}"

def list_projects():
    return sorted([p for p in PROJECTS_DIR.glob("*.pdv")], key=lambda p: p.stat().st_mtime, reverse=True)

def list_project_datasets(project_dir: Path):
    data_dir = Path(project_dir) / "data"
    if not data_dir.exists():
        return []
    files = []
    for ext in ("*.csv", "*.txt", "*.xls", "*.xlsx", "*.parquet"):
        files += list(data_dir.glob(ext))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

def save_uploaded_into_project(project_dir: Path, uploaded_file) -> Path:
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)
    dest = data_dir / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest

def save_snapshot_to_project(project_dir: Path, df: pd.DataFrame, hint: str = "snapshot") -> Path:
    data_dir = Path(project_dir) / "data"
    data_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = data_dir / f"{hint}_{ts}.csv"
    df.to_csv(dest, index=False)
    return dest

def datetime_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

def detect_data_type_simple(df: pd.DataFrame) -> str:
    return "time_series" if len(datetime_columns(df)) > 0 else "tabular"

def build_reg_models(names):
    avail = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42),
        "SVR": SVR()
    }
    return {k: avail[k] for k in names if k in avail}

def build_clf_models(names):
    avail = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=300, random_state=42),
        "SVC": SVC(probability=True)
    }
    return {k: avail[k] for k in names if k in avail}

def make_ts_lags_safe(y: pd.Series, lags: int):
    y = pd.Series(y).copy()
    n = len(y.dropna())
    if n < 3:
        return None, None
    max_lags = max(1, min(lags, n - 2))
    X = pd.concat({f"lag_{i}": y.shift(i) for i in range(1, max_lags+1)}, axis=1)
    df_xy = pd.concat([X, y.rename("target")], axis=1).dropna()
    if df_xy.empty:
        return None, None
    return df_xy.drop(columns=["target"]), df_xy["target"]

def coerce_numeric_inplace(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def expand_date_features(df: pd.DataFrame, cols):
    added = []
    for c in cols:
        if c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce", utc=False)
            if parsed.notna().mean() >= 0.8:
                df[f"{c}_year"] = parsed.dt.year
                df[f"{c}_month"] = parsed.dt.month
                df[f"{c}_dow"] = parsed.dt.dayofweek
                df.drop(columns=[c], inplace=True)
                added += [f"{c}_year", f"{c}_month", f"{c}_dow"]
    return added

def safe_split(X, y, test_size=0.2, stratify=None):
    n = len(y)
    if n < 2:
        return None
    n_test = max(1, int(round(n * test_size)))
    if n - n_test < 1:
        n_test = n - 1
    ts = n_test / n
    try:
        return train_test_split(X, y, test_size=ts, random_state=42, stratify=stratify)
    except ValueError:
        return train_test_split(X, y, test_size=ts, random_state=42, stratify=None)

def train_test_split_ts_safe(X, y, test_size=0.2):
    n = len(y)
    if n < 2:
        return None
    n_test = max(1, int(round(n * test_size)))
    if n - n_test < 1:
        n_test = n - 1
    return X.iloc[:-n_test, :], X.iloc[-n_test:, :], y.iloc[:-n_test], y.iloc[-n_test:]

def forecast_with_lags(model, history: pd.Series, steps: int, lags: int):
    hist = history.copy().tolist()
    preds = []
    l = max(1, lags)
    for _ in range(steps):
        if len(hist) < l:
            break
        feats = np.array([hist[-i] for i in range(1, l+1)]).reshape(1, -1)
        yhat = model.predict(feats)[0]
        preds.append(yhat)
        hist.append(yhat)
    return preds

def save_project_config():
    if not ss.get('project_dir'):
        return
    cfg = {
        "project_name": ss.get('project_name'),
        "project_slug": ss.get('project_slug'),
        "created_at": ss.get('created_at'),
        "updated_at": datetime.now().isoformat(),
        "data_file": ss.get('data_file'),
        "source_type": ss.get('source_type'),
        "source_details": ss.get('source_details'),
        "data_type": ss.get('data_type'),
        "date_col": ss.get('date_col'),
        "target_column": ss.get('target_column'),
        "num_cols": ss.get('num_cols'),
        "cat_cols": ss.get('cat_cols'),
        "task_type": ss.get('task_type'),
        "models_selected": ss.get('models_selected'),
        "scale_numeric": ss.get('scale_numeric'),
        "encode_categorical": ss.get('encode_categorical'),
        "impute_strategy_num": ss.get('impute_strategy_num'),
        "impute_strategy_cat": ss.get('impute_strategy_cat'),
        "use_date_features": ss.get('use_date_features'),
        "test_size": ss.get('test_size'),
        "show_graphs": ss.get('show_graphs'),
        "n_lags": ss.get('n_lags'),
        "horizon": ss.get('horizon'),
        "drop_cols": ss.get('drop_cols'),
    }
    pdv_path = Path(ss['project_file'])
    with open(pdv_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def load_project_config(pdv_path: Path):
    with open(pdv_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    df = None
    data_path = Path(cfg.get("data_file")) if cfg.get("data_file") else None
    if data_path and data_path.exists():
        try:
            if data_path.suffix.lower() in (".csv", ".txt"):
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() in (".xls", ".xlsx"):
                df = pd.read_excel(data_path)
            elif data_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)
        except Exception:
            df = None

    # appliquer drop_cols avant l'affectation finale
    drop_cols_cfg = cfg.get("drop_cols", [])
    if df is not None and drop_cols_cfg:
        df = df.drop(columns=[c for c in drop_cols_cfg if c in df.columns], errors="ignore")

    ss['project_name'] = cfg.get("project_name")
    ss['project_slug'] = cfg.get("project_slug")
    ss['project_dir'] = str((PROJECTS_DIR / ss['project_slug']).resolve())
    ss['project_file'] = str(pdv_path.resolve())
    ss['created_at'] = cfg.get("created_at", datetime.now().isoformat())
    ss['data_file'] = cfg.get("data_file")
    ss['source_type'] = cfg.get("source_type", "file")
    ss['source_details'] = cfg.get("source_details", {})
    ss['df'] = df
    ss['data_type'] = cfg.get("data_type", "tabular")
    ss['date_col'] = cfg.get("date_col")
    ss['target_column'] = cfg.get("target_column")
    ss['num_cols'] = cfg.get("num_cols", [])
    ss['cat_cols'] = cfg.get("cat_cols", [])
    ss['task_type'] = cfg.get("task_type", "regression")
    ss['models_selected'] = cfg.get("models_selected", [])
    ss['scale_numeric'] = cfg.get("scale_numeric", True)
    ss['encode_categorical'] = cfg.get("encode_categorical", True)
    ss['impute_strategy_num'] = cfg.get("impute_strategy_num", "mean")
    ss['impute_strategy_cat'] = cfg.get("impute_strategy_cat", "most_frequent")
    ss['use_date_features'] = cfg.get("use_date_features", True)
    ss['test_size'] = float(cfg.get("test_size", 0.2))
    ss['show_graphs'] = bool(cfg.get("show_graphs", True))
    ss['n_lags'] = int(cfg.get("n_lags", 6))
    ss['horizon'] = int(cfg.get("horizon", 7))
    ss['drop_cols'] = drop_cols_cfg
    ss['step_done'].update({s: True for s in ["Projet", "Import données", "Analyse"]})

# ---------- Navigation helpers ----------
def go_to_step(step_name: str, *, request_rerun: bool = False):
    # On enregistre seulement la nav demandée
    ss['pending_nav'] = step_name
    ss['__scroll_to_top__'] = True
    # ⚠️ Ne JAMAIS appeler st.rerun() ici si on est dans un callback
    if request_rerun:
        st.rerun()

def nav_controls(next_enabled: bool, next_label: str, save_before_next: bool = False,
                 show_prev: bool = True, show_next: bool = True, key_prefix: str | None = None):
    # identifiant unique basé sur l'étape + timestamp
    import uuid
    kp = key_prefix or f"nav_{ss.get('nav_step','unknown')}"
    uid = str(uuid.uuid4())[:6]   # petit suffixe unique
    prev_key = f"{kp}_prev_btn_{uid}"
    next_key = f"{kp}_next_btn_{uid}"

    col_prev, col_next = st.columns(2)
    with col_prev:
        if show_prev:
            st.button(
                "◀️ Retour",
                key=prev_key,
                on_click=lambda: go_to_step(steps[max(0, steps.index(ss['nav_step']) - 1)])
            )
    with col_next:
        if show_next:
            st.button(
                next_label,
                key=next_key,
                disabled=not next_enabled,
                on_click=(lambda: (save_project_config() if save_before_next else None,
                                   go_to_step(steps[min(len(steps)-1, steps.index(ss['nav_step']) + 1)])))
            )

# ---------- Sidebar: Navigation ----------
st.sidebar.title("Navigation")
_ = st.sidebar.radio("Étapes", steps, key="nav_step", index=steps.index(ss['nav_step']))

# ---------- Sidebar: Paramètres généraux ----------
st.sidebar.title("Paramètres")
with st.sidebar.expander("Modèles AutoML", expanded=True):
    reg_options = ["LinearRegression", "Ridge", "Lasso", "RandomForest", "GradientBoosting", "ExtraTrees", "SVR"]
    clf_options = ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier", "ExtraTreesClassifier", "SVC"]
    reg_default = ["LinearRegression", "RandomForest", "GradientBoosting"]
    clf_default = ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"]

    if ss['data_type'] == "time_series" or ss['task_type'] == "regression":
        prev = [m for m in ss['models_selected'] if m in reg_options]
        default_list = prev or reg_default
        ss['models_selected'] = st.multiselect("Modèles régression", reg_options, default=default_list)
    else:
        prev = [m for m in ss['models_selected'] if m in clf_options]
        default_list = prev or clf_default
        ss['models_selected'] = st.multiselect("Modèles classification", clf_options, default=default_list)

with st.sidebar.expander("Prétraitement", expanded=False):
    ss['target_nan_strategy_reg'] = st.multiselect("Remplacer NaN par", ["drop", "mean", "median", "ffill", "bfill", "zero"], default="drop")
    ss['scale_numeric'] = st.checkbox("Standardiser les numériques", value=ss['scale_numeric'])
    ss['encode_categorical'] = st.checkbox("One-Hot Encode (qualitatives)", value=ss['encode_categorical'])
    ss['impute_strategy_num'] = st.selectbox("Imputation numériques", ["mean", "median"],
                                             index=0 if ss['impute_strategy_num']=="mean" else 1)
    ss['impute_strategy_cat'] = st.selectbox("Imputation qualitatives", ["most_frequent", "constant"],
                                             index=0 if ss['impute_strategy_cat']=="most_frequent" else 1)
    ss['use_date_features'] = st.checkbox("Dériver year/month/dow à partir des colonnes date (tabulaire)",
                                          value=ss['use_date_features'])

with st.sidebar.expander("Évaluation & Graphes", expanded=False):
    ss['test_size'] = st.slider("Taille test (%)", 10, 40, int(ss['test_size']*100)) / 100
    ss['show_graphs'] = st.checkbox("Afficher les graphiques", value=ss['show_graphs'])
    if ss['data_type'] == "time_series":
        ss['n_lags'] = st.slider("Nombre de lags (TS)", 1, 24, ss['n_lags'])
        ss['horizon'] = st.slider("Horizon de prévision (pas)", 1, 30, ss['horizon'])
    else:
        st.caption("ℹ️ Lags/horizon visibles en mode série temporelle uniquement.")

with st.sidebar.expander("Apparence", expanded=False):
    ss['pred_color'] = st.color_picker("Couleur des prédictions", ss['pred_color'])

# ------------------ Étape -1 : Workflow ------------------
def render_workflow_diagram():
    labels = ["Projet", "Importer", "Analyse", "Choix de Modèle", "Prédiction", "Historique"]
    n = len(labels)
    fig_w = min(2 + 2.2*n, 18)
    fig, ax = plt.subplots(figsize=(fig_w, 2.8))
    ax.axis('off')
    x0, y0 = 0.06, 0.45
    step_w = (0.88) / n
    for i, lab in enumerate(labels):
        x = x0 + i*step_w
        box = FancyBboxPatch((x, y0), step_w*0.8, 0.32,
                             boxstyle="round,pad=0.02,rounding_size=0.04",
                             linewidth=1.5, edgecolor=PRIMARY, facecolor="white")
        ax.add_patch(box)
        ax.text(x + step_w*0.4, y0 + 0.16, lab, ha='center', va='center', fontsize=11)
        if i < n-1:
            start = (x + step_w*0.8, y0 + 0.16)
            end = (x + step_w*0.98, y0 + 0.16)
            arr = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12, linewidth=1.4, color=PRIMARY)
            ax.add_patch(arr)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    return fig

step = ss['nav_step']
if step == "Workflow":
    st.header("🔰 Workflow de la prévision")
    st.markdown("""
1. **Projet** : crée/charge un projet `.pdv`.  
2. **Importer** : Fichier, **API**, **SQL**, **MongoDB** → aperçu, détection auto (**tabulaire** / **série temporelle**), puis **validation manuelle**.  
   - *TS* : choisir/convertir la **colonne date**.  
3. **Analyse** : EDA (stats, manquants, corrélations), choisir la **cible**, typer **quantitatives/qualitatives**.  
4. **AutoML** : choisir les **modèles** (TS=régression sur lags ; Tabulaire=régression ou classification).  
5. **Prédiction** : entrée des features (tabulaire) ou prévisions futures (TS).  
6. **Historique** : journal des actions et restauration / suppression de projets.
""")
    st.subheader("Schéma du workflow")
    fig = render_workflow_diagram()
    st.pyplot(fig); plt.close(fig)

    st.divider()
    if st.button("▶️ Démarrer"):
        go_to_step("Projet", request_rerun=True)

    nav_controls(True, button_titles[step], save_before_next=False, show_prev=False, show_next=False, key_prefix=step)

# ------------------ Étape 0 : Projet ------------------
def save_project_then_next(pname: str):
    if not pname.strip():
        st.warning("Donnez un nom de projet.")
        return
    slug = slugify(pname)
    proj_dir = PROJECTS_DIR / slug
    proj_dir.mkdir(exist_ok=True)
    ss['project_name'] = pname.strip()
    ss['project_slug'] = slug
    ss['project_dir'] = str(proj_dir.resolve())
    ss['project_file'] = str((PROJECTS_DIR / f"{slug}.pdv").resolve())
    ss['created_at'] = datetime.now().isoformat()
    save_project_config()
    st.success(f"Projet créé : {pname} → {ss['project_file']}")
    ss['step_done']['Projet'] = True
    # Rester sur Projet pour pouvoir choisir un dataset existant directement
    go_to_step("Projet", request_rerun=True)

def use_project_dataset(file_path: Path):
    """Utiliser un fichier du dossier du projet comme dataset et sauter l'étape Import."""
    try:
        if file_path.suffix.lower() in (".csv", ".txt"):
            df_loaded = pd.read_csv(file_path) if file_path.suffix.lower()==".csv" else pd.read_csv(file_path, sep="\t")
        elif file_path.suffix.lower() in (".xls", ".xlsx"):
            df_loaded = pd.read_excel(file_path)
        elif file_path.suffix.lower() == ".parquet":
            df_loaded = pd.read_parquet(file_path)
        else:
            df_loaded = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Lecture impossible : {e}")
        return

    # appliquer drop_cols déjà enregistrées
    if ss.get('drop_cols'):
        df_loaded = df_loaded.drop(columns=[c for c in ss['drop_cols'] if c in df_loaded.columns], errors="ignore")

    ss['df'] = df_loaded.copy()
    
    # --- NEW: si TS, appliquer la conversion datetime ici aussi ---
    if ss.get('data_type') == "time_series" and ss.get('date_col'):
        try:
            col = ss['date_col']
            if col in df_loaded.columns and not pd.api.types.is_datetime64_any_dtype(df_loaded[col]):
                df_loaded[col] = pd.to_datetime(df_loaded[col], errors='coerce', utc=False)
            # remettre df après conversion
            ss['df'] = df_loaded.copy()
            # petit contrôle
            if df_loaded[ss['date_col']].notna().sum() < 2:
                st.warning("La colonne date ne contient pas assez de valeurs valides pour une TS.")
        except Exception as e:
            st.warning(f"Conversion datetime impossible sur '{ss.get('date_col')}' : {e}")

    ss['data_file'] = str(file_path.resolve())
    ss['source_type'] = "file"
    ss['source_details'] = {"kind":"file","name":file_path.name}
    ss['auto_data_type'] = detect_data_type_simple(df_loaded)
    if ss['data_type'] not in ("tabular", "time_series"):
        ss['data_type'] = ss['auto_data_type']
    st.success(f"Dataset chargé depuis le projet : {file_path.name}")
    ss['step_done'].update({"Import données": True, "Analyse": True})
    save_project_config()
    go_to_step("Analyse", request_rerun=True)

if step == "Projet":
    st.header("0️⃣ Projet")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Créer un projet")
        proj_name = st.text_input("Nom du projet", value=ss.get('project_name') or "")
        st.button("Créer le projet", on_click=lambda: save_project_then_next(proj_name))

    with col2:
        st.subheader("Charger un projet existant")
        pdv_files = list_projects()
        if pdv_files:
            sel = st.selectbox("Sélectionner un .pdv", pdv_files, format_func=lambda p: p.name)
            if st.button("Charger ce projet"):
                load_project_config(sel)
                st.success(f"Projet restauré : {ss['project_name']}")
                ss['step_done']['Projet'] = True
                # rester sur Projet pour afficher les datasets du projet et pouvoir en choisir un
                go_to_step("Projet", request_rerun=True)
        else:
            st.info("Aucun projet pour l’instant. Créez-en un à gauche.")

    # --- Jeux de données déjà présents dans le projet courant ---
    if ss.get('project_dir'):
        awaiting = bool(ss.get('await_dtype_choice'))
        pending_ds = ss.get('pending_project_ds')

        # CAS 1 : on attend le choix -> afficher radio (+ bloc TS si time_series) + confirmer/annuler
        if awaiting and pending_ds:
            
            #begin colonne cible

            st.subheader("Sélection de la colonne cible (à prédire)")

            df_t = ss['pending_df_tmp']

            all_cols = df_t.columns.tolist()

            # Ajout d’une option factice
            options = ["-- Aucune --"] + all_cols

            # choice = st.radio(
            #     "Sélectionner la colonne cible (target)",
            #     options,
            #     index=0,  # Toujours démarrer sur "Aucune"
            #     key="target_choice"
            # )

            choice = grid_radio(
                "Sélectionner la colonne cible (target)",
                options=options,
                n_cols=3,
                key="target_choice"
            )
                            
            # Stocker seulement si une vraie colonne est choisie
            ss['target_column'] = None if choice == "-- Aucune --" else choice

            # Vérification obligatoire
            if ss['target_column'] is None:
                st.warning("⚠️ Veuillez sélectionner une colonne cible avant de continuer.")
                st.stop()   # bloque l'exécution de l'étape
            else :
                st.success(f"✅ Colonne cible sélectionée : **{choice}** ")
                
            #End colonne cible

            st.subheader("⚙️ Choix utilisateur (type de données)")
            current_auto = ss.get('auto_data_type', 'tabular')
            # radio type de données
            dtype_choice = st.radio(
                "Type de données (valider/modifier)",
                options=["tabular", "time_series"],
                index=0 if (ss.get('proj_dtype_choice') or current_auto) == "tabular" else 1,
                horizontal=True,
            )
            ss['proj_dtype_choice'] = dtype_choice

            # Charger/tenir un df temporaire en mémoire pour le formulaire TS
            if 'pending_df_tmp' not in ss or ss['pending_df_tmp'] is None:
                try:
                    p = Path(pending_ds)
                    if p.suffix.lower() in (".csv", ".txt"):
                        ss['pending_df_tmp'] = pd.read_csv(p) if p.suffix.lower()==".csv" else pd.read_csv(p, sep="\t")
                    elif p.suffix.lower() in (".xls", ".xlsx"):
                        ss['pending_df_tmp'] = pd.read_excel(p)
                    elif p.suffix.lower() == ".parquet":
                        ss['pending_df_tmp'] = pd.read_parquet(p)
                    else:
                        ss['pending_df_tmp'] = pd.read_csv(p)
                except Exception as e:
                    ss['pending_df_tmp'] = None
                    st.error(f"Lecture temporaire impossible : {e}")

            # --- Si time_series: montrer le sous-formulaire TS (conversion + date_col) ---
            chosen_date_col = ss.get('pending_date_col')  # mémoriser temporairement avant confirmation
            if dtype_choice == "time_series" and ss.get('pending_df_tmp') is not None:
                df_tmp = ss['pending_df_tmp']

                st.caption("ℹ️ La tâche sera traitée en **régression sur lags**.")
                # détection colonnes datetime actuelles
                dt_cols = [c for c in df_tmp.columns if pd.api.types.is_datetime64_any_dtype(df_tmp[c])]
                if not dt_cols:
                    st.warning("Aucune colonne *dtype datetime* détectée. Convertissez si nécessaire.")
                # sélection colonnes à convertir
                conv_cols = st.multiselect(
                    "Colonnes à convertir en datetime (optionnel)",
                    options=df_tmp.columns.tolist(),
                    key=f"proj_ts_conv_cols_{Path(pending_ds).name}"
                )
                if st.button("Convertir en datetime",
                            key=f"proj_ts_convert_btn_{Path(pending_ds).name}"):
                    for c in conv_cols:
                        try:
                            df_tmp[c] = pd.to_datetime(df_tmp[c], errors='coerce', utc=False)
                        except Exception:
                            pass
                    ss['pending_df_tmp'] = df_tmp  # sauver modifié
                    st.success(f"✅ {','.join(conv_cols)} transformée en objet datetime.")

                # recalcul des colonnes datetime
                dt_cols = [c for c in df_tmp.columns if pd.api.types.is_datetime64_any_dtype(df_tmp[c])]
                if dt_cols:
                    chosen_date_col = st.selectbox(
                        "Sélectionner la colonne de date",
                        options=dt_cols,
                        index=0 if (chosen_date_col not in dt_cols) else dt_cols.index(chosen_date_col),
                        key=f"proj_ts_date_col_{Path(pending_ds).name}"
                    )
                    ss['pending_date_col'] = chosen_date_col
                else:
                    ss['pending_date_col'] = None
                    st.info("Aucune colonne de date valide après conversion.")

            # Boutons Confirmer / Annuler
            cok, ccancel = st.columns([0.4, 0.6])
            with cok:
                if st.button("✅ Confirmer et charger ce dataset",
                            key=f"btn_confirm_load_{Path(pending_ds).name}"):
                    # Appliquer les choix
                    ss['data_type'] = ss['proj_dtype_choice'] or current_auto or "tabular"
                    if ss['data_type'] == "time_series":
                        # imposer le choix de la date côté session pour le chargement
                        ss['date_col'] = ss.get('pending_date_col')

                    try:
                        use_project_dataset(Path(pending_ds))  # fait la lecture finale + go_to_step('Analyse')
                    finally:
                        # Nettoyage
                        ss['await_dtype_choice'] = False
                        ss['pending_project_ds'] = None
                        ss['pending_df_tmp'] = None
                        ss['pending_date_col'] = None

            with ccancel:
                if st.button("❌ Annuler",
                            key=f"btn_cancel_load_{Path(pending_ds).name}"):
                    ss['await_dtype_choice'] = False
                    ss['pending_project_ds'] = None
                    ss['pending_df_tmp'] = None
                    ss['pending_date_col'] = None
                    go_to_step("Projet", request_rerun=True)

        # CAS 2 : pas d’attente -> afficher la liste
        else:
            st.subheader("📦 Jeux de données dans ce projet")
            files = list_project_datasets(Path(ss['project_dir']))
            if files:
                file_sel = st.selectbox(
                    "Sélectionner un dataset du projet",
                    files,
                    format_func=lambda p: p.name,
                    key="proj_ds_sel"
                )
                c1, c2 = st.columns([0.5, 0.5])
                with c1:
                    if st.button("Utiliser ce dataset maintenant ▶️", key=f"btn_use_project_ds_{file_sel.name}"):
                        ss['pending_project_ds'] = str(file_sel)
                        # détection auto pour présélection radio
                        try:
                            if file_sel.suffix.lower() in (".csv", ".txt"):
                                df_tmp = pd.read_csv(file_sel) if file_sel.suffix.lower()==".csv" else pd.read_csv(file_sel, sep="\t")
                            elif file_sel.suffix.lower() in (".xls", ".xlsx"):
                                df_tmp = pd.read_excel(file_sel)
                            elif file_sel.suffix.lower() == ".parquet":
                                df_tmp = pd.read_parquet(file_sel)
                            else:
                                df_tmp = pd.read_csv(file_sel)
                            ss['auto_data_type'] = detect_data_type_simple(df_tmp)
                        except Exception:
                            ss['auto_data_type'] = "tabular"
                            df_tmp = None

                        # garder un df temporaire si besoin (pour le sous-formulaire TS)
                        ss['pending_df_tmp'] = df_tmp
                        ss['proj_dtype_choice'] = ss.get('data_type') or ss['auto_data_type'] or "tabular"
                        ss['await_dtype_choice'] = True
                        go_to_step("Projet", request_rerun=True)  # rester sur la page Projet pour afficher le choix

                with c2:
                    st.caption(f"Chemin : `{file_sel}`")
            else:
                st.info("Aucun dataset dans le dossier `data` du projet pour le moment.")

    nav_controls(ss['step_done']['Projet'], button_titles[step], save_before_next=False,
        show_prev=True, show_next=True)

# ------------------ Étape 1 : Import ------------------
def parse_json_field(txt: str):
    txt = (txt or "").strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception as e:
        st.warning(f"JSON invalide : {e}")
        return None

def load_from_api_ui():
    st.subheader("Source : API (REST)")
    url = st.text_input("URL", placeholder="https://api.exemple.com/ressource")
    method = st.selectbox("Méthode", ["GET", "POST"])
    headers_txt = st.text_area("Headers (JSON)", value="", placeholder='{"Authorization":"Bearer ..."}')
    params_txt = st.text_area("Params / Query (JSON)", value="", placeholder='{"q":"term","page":1}')
    body_txt = st.text_area("Body (JSON, POST seulement)", value="", placeholder='{"from":"2024-01-01"}')

    if st.button("Charger depuis l'API"):
        if not url:
            st.warning("Veuillez saisir l’URL.")
            return (None, None)
        try:
            import requests
        except Exception:
            st.error("Le paquet `requests` est requis. Ajoutez-le à votre environnement.")
            return (None, None)
        headers = parse_json_field(headers_txt) or {}
        params = parse_json_field(params_txt) or {}
        json_body = parse_json_field(body_txt) if method == "POST" else None
        try:
            resp = requests.request(method, url, headers=headers, params=params, json=json_body, timeout=60)
            resp.raise_for_status()
            try:
                data = resp.json()
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                elif isinstance(data, dict):
                    if 'data' in data and isinstance(data['data'], (list, dict)):
                        df = pd.json_normalize(data['data'])
                    else:
                        df = pd.json_normalize(data)
                else:
                    df = pd.read_csv(io.StringIO(resp.text))
            except ValueError:
                df = pd.read_csv(io.StringIO(resp.text))
            return (df, {"kind":"api","url":url,"method":method,"headers":bool(headers),"params":bool(params)})
        except Exception as e:
            st.error(f"Erreur API : {e}")
            return (None, None)
    return (None, None)

def safe_read_sql(engine, query: str):
    from sqlalchemy import text
    try:
        return pd.read_sql_query(text(query), con=engine)
    except Exception as e1:
        try:
            return pd.read_sql_query(query, con=engine)
        except Exception as e2:
            try:
                raw = engine.raw_connection()
                try:
                    cur = raw.cursor()
                    cur.execute("SELECT 1")
                    cur.close()
                except Exception:
                    pass
                df = pd.read_sql_query(query, con=raw)
                raw.close()
                return df
            except Exception as e3:
                raise RuntimeError(f"read_sql failed: {e1} ; {e2} ; {e3}")

def load_from_sql_ui():
    st.subheader("Source : SQL (MySQL / PostgreSQL)")
    db_kind = st.radio("Base", ["MySQL", "PostgreSQL"], horizontal=True)
    host_in = st.text_input("Hôte", value="localhost", help="Ex: localhost (ne mettez pas user@host ici)")
    port = st.number_input("Port", value=3306 if db_kind=="MySQL" else 5432, step=1)
    database = st.text_input("Base de données")
    user_in = st.text_input("Utilisateur")
    password = st.text_input("Mot de passe", type="password")
    query = st.text_area("Requête SQL", value="SELECT 1")

    colA, colB = st.columns(2)
    with colA:
        force_ipv4 = st.checkbox("Forcer 127.0.0.1 si 'localhost'", value=False)
    with colB:
        show_debug = st.checkbox("Afficher debug de connexion (sans mot de passe)", value=False)

    host, user = normalize_sql_host_user(host_in, user_in)
    if force_ipv4 and host.lower() == "localhost":
        host = "127.0.0.1"
    if show_debug:
        st.caption(f"🔧 Hôte nettoyé = `{host}`, Utilisateur = `{user or '(vide)'}`")

    if st.button("Exécuter la requête"):
        try:
            from sqlalchemy import create_engine, text
            try:
                from sqlalchemy.engine import URL as SA_URL
                def make_url(drivername, username, password, host, port, database):
                    return SA_URL.create(
                        drivername=drivername,
                        username=(username or None),
                        password=(password or None),
                        host=(host or None),
                        port=(int(port) if port else None),
                        database=(database or None),
                    )
            except Exception:
                from urllib.parse import quote_plus
                def make_url(drivername, username, password, host, port, database):
                    u = quote_plus(username or "")
                    p = quote_plus(password or "")
                    h = (host or "")
                    d = (database or "")
                    prt = f":{int(port)}" if port else ""
                    cred = f"{u}:{p}@" if (username or password) else ""
                    return f"{drivername}://{cred}{h}{prt}/{d}"
        except Exception:
            st.error("Le paquet `sqlalchemy` est requis. Installez-le (ex: `pip install sqlalchemy`).")
            return (None, None)

        if db_kind == "MySQL":
            try:
                import pymysql  # noqa: F401
                driver = "mysql+pymysql"
            except Exception:
                driver = "mysql"
        else:
            try:
                import psycopg2  # noqa: F401
                driver = "postgresql+psycopg2"
            except Exception:
                driver = "postgresql"

        conn_url = make_url(driver, user, password, host, port, database)

        try:
            engine = create_engine(conn_url, pool_pre_ping=True)
            with engine.begin() as conn:
                conn.execute(text("SELECT 1"))
            df = safe_read_sql(engine, query)
            return (df, {"kind":"sql","db":db_kind,"host":host,"port":int(port),"database":database})
        except Exception as e:
            st.error(f"Erreur SQL : {e}")
            st.info("Vérifiez : service, host/port, firewall, droits, et évitez `user@host` dans Hôte. Essayez 127.0.0.1.")
            return (None, None)

    return (None, None)

def load_from_mongo_ui():
    st.subheader("Source : NoSQL (MongoDB)")
    uri = st.text_input("URI MongoDB", "mongodb://localhost:27017")
    database = st.text_input("Base de données", "")
    collection = st.text_input("Collection", "")
    find_query_txt = st.text_area("Filtre (JSON)", "{}", help='Ex: {"status": "active"}')
    limit = st.number_input("Limite de documents", value=100, step=10, min_value=1)

    if st.button("Charger depuis MongoDB"):
        try:
            from pymongo import MongoClient
        except Exception:
            st.error("Le paquet `pymongo` est requis (`pip install pymongo`).")
            return (None, None)
        try:
            query = parse_json_field(find_query_txt) or {}
            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            db = client[database]
            col = db[collection]
            docs = list(col.find(query).limit(int(limit)))
            if not docs:
                st.warning("Aucun document renvoyé par la requête.")
                return (None, None)
            for d in docs:
                d.pop("_id", None)
            df = pd.json_normalize(docs)
            return (df, {"kind":"nosql","db":"MongoDB","uri":uri,"database":database,"collection":collection})
        except Exception as e:
            st.error(f"Erreur MongoDB : {e}")
            return (None, None)
    return (None, None)

if step == "Import données":
    st.header("1️⃣ Importer les données")
    if not ss.get('project_dir'):
        st.info("Veuillez d’abord créer ou charger un projet (étape Projet).")
    else:
        source_option = st.radio(
            "Source de données :",
            ["Fichier (CSV/Excel/TXT)", "API (REST)", "SQL (MySQL/PostgreSQL)", "NoSQL (MongoDB)"],
            horizontal=True
        )

        df_loaded, source_meta = None, None

        if source_option == "Fichier (CSV/Excel/TXT)":
            up = st.file_uploader("Importer un fichier", type=["csv", "xls", "xlsx", "txt"])
            if up:
                dest = save_uploaded_into_project(Path(ss['project_dir']), up)
                ss['data_file'] = str(dest.resolve())
                if dest.suffix.lower() in [".csv", ".txt"]:
                    sep = "," if dest.suffix.lower()==".csv" else "\t"
                    df_loaded = pd.read_csv(dest, sep=sep)
                else:
                    df_loaded = pd.read_excel(dest)
                source_meta = {"kind":"file","name":up.name}

        elif source_option == "API (REST)":
            df_loaded, source_meta = load_from_api_ui()
            if df_loaded is not None:
                snapshot = save_snapshot_to_project(ss['project_dir'], df_loaded, hint="api")
                ss['data_file'] = str(snapshot.resolve())

        elif source_option == "SQL (MySQL/PostgreSQL)":
            df_loaded, source_meta = load_from_sql_ui()
            if df_loaded is not None:
                snapshot = save_snapshot_to_project(ss['project_dir'], df_loaded, hint="sql")
                ss['data_file'] = str(snapshot.resolve())

        else:  # NoSQL
            df_loaded, source_meta = load_from_mongo_ui()
            if df_loaded is not None:
                snapshot = save_snapshot_to_project(ss['project_dir'], df_loaded, hint="nosql")
                ss['data_file'] = str(snapshot.resolve())

        if df_loaded is not None:
            st.write("Aperçu (10 premières lignes) :")
            st.dataframe(df_loaded.head(10))

            # Détection auto
            ss['auto_data_type'] = detect_data_type_simple(df_loaded)
            st.info(f"🔎 Détection automatique : **{ss['auto_data_type']}**")

            #begin colonne cible

            st.subheader("Sélection de la colonne cible (à prédire)")

            df_t = df_loaded

            all_cols = df_t.columns.tolist()

            # Ajout d’une option factice
            options = ["-- Aucune --"] + all_cols

            # choice = st.radio(
            #     "Sélectionner la colonne cible (target)",
            #     options,
            #     index=0,  # Toujours démarrer sur "Aucune"
            #     key="target_choice"
            # )

            choice = grid_radio(
                "Sélectionner la colonne cible (target)",
                options=options,
                n_cols=3,
                key="target_choice"
            )
                            
            # Stocker seulement si une vraie colonne est choisie
            ss['target_column'] = None if choice == "-- Aucune --" else choice

            # Vérification obligatoire
            if ss['target_column'] is None:
                st.warning("⚠️ Veuillez sélectionner une colonne cible avant de continuer.")
                st.stop()   # bloque l'exécution de l'étape
            else :
                st.success(f"✅ Colonne cible sélectionée : **{choice}** ")
                
            #End colonne cible

            st.subheader("⚙️ Choix utilisateur (type de données)")
            # Choix utilisateur
            ss['data_type'] = st.radio(
                "Type de données (valider/modifier)",
                options=["tabular", "time_series"],
                index=0 if ss['auto_data_type']=="tabular" else 1,
                horizontal=True
            )

            # Si TS : gérer colonne de date + conversion
            if ss['data_type'] == "time_series":
                st.caption("ℹ️ La tâche sera traitée en **régression sur lags**.")
                dt_cols = datetime_columns(df_loaded)
                if not dt_cols:
                    st.warning("Aucune colonne *dtype datetime* détectée. Convertissez si nécessaire.")
                conv_cols = st.multiselect("Colonnes à convertir en datetime (optionnel)", options=df_loaded.columns.tolist())
                if st.button("Convertir en datetime"):
                    for c in conv_cols:
                        df_loaded[c] = pd.to_datetime(df_loaded[c], errors='coerce', utc=False)
                    dt_cols = datetime_columns(df_loaded)
                    # st.success("✅ Conversion effectuée.")
                    st.success(f"✅ {','.join(conv_cols)} transformée en objet datetime.")

                if dt_cols:
                    ss['date_col'] = st.selectbox("Sélectionner la colonne de date", options=dt_cols)
                else:
                    ss['date_col'] = None

                if ss['date_col']:
                    if not pd.api.types.is_datetime64_any_dtype(df_loaded[ss['date_col']]):
                        df_loaded[ss['date_col']] = pd.to_datetime(df_loaded[ss['date_col']], errors='coerce', utc=False)
                    if df_loaded[ss['date_col']].notna().sum() < 2:
                        st.warning("La colonne date ne contient pas assez de valeurs valides.")
                ss['task_type'] = "regression"
            else:
                ss['date_col'] = None

            # Finaliser
            ss['df'] = df_loaded.copy()
            ss['source_type'] = (source_meta or {}).get("kind", "file")
            ss['source_details'] = source_meta or {}
            st.success("Données chargées et configuration mise à jour.")
            ss['step_done']['Import données'] = True
            save_project_config()

    nav_controls(ss['step_done']['Import données'], button_titles[step], save_before_next=True,
                 show_prev=True, show_next=True, key_prefix=step)

# ------------------ Étape 2 : Analyse ------------------
if step == "Analyse":
    st.header("2️⃣ Analyse exploratoire & descriptive")
    if ss.get('df') is None:
        st.info("Veuillez importer des données.")
    else:
        # --- Retirer des colonnes du dataset (persistant) ---
        st.subheader("🧹 Exclure des colonnes du dataset")
        current_cols = ss['df'].columns.tolist()

        # Colonnes protégées (ex: colonne date) qu'on ne peut pas retirer
        protected = set([ss.get('date_col')]) - {None}
        drop_candidates = [c for c in current_cols if c not in protected]

        # Pré‑sélection initiale = ce qui est déjà dans ss['drop_cols']
        preselected = [c for c in ss.get('drop_cols', []) if c in drop_candidates]

        # Petit filtre pour les listes longues
        filter_txt = st.text_input("Filtrer les colonnes…", value="", placeholder="Tapez pour filtrer")
        filtered = [c for c in drop_candidates if filter_txt.lower() in c.lower()]

        # Actions groupées
        def _set_all_dropcols(val: bool):
            for col in filtered:
                st.session_state[f"dropcol_{col}"] = val

        col_tools1, col_tools2 = st.columns([0.5, 0.5])
        with col_tools1:
            st.button("Tout cocher (filtré)", on_click=lambda: _set_all_dropcols(True))
        with col_tools2:
            st.button("Tout décocher (filtré)", on_click=lambda: _set_all_dropcols(False))

        st.caption("Astuce : utilisez le filtre pour cocher/décocher en masse un sous‑ensemble.")

        # Rendu des checkboxes (clés stables)
        # Valeur par défaut = déjà selectionné auparavant
        ncols = 3  # mise en page en 3 colonnes
        cols_grid = st.columns(ncols)
        selected_now = []

        for i, col in enumerate(filtered):
            key = f"dropcol_{col}"
            default_val = col in preselected
            # Si la clé existe déjà, Streamlit garde la valeur courante ; sinon on injecte le défaut
            if key not in st.session_state:
                st.session_state[key] = default_val
            with cols_grid[i % ncols]:
                if st.checkbox(col, key=key):
                    selected_now.append(col)

        # Mettre à jour la sélection courante dans la session (persiste pour les prochains runs)
        ss['drop_cols'] = selected_now

        # Boutons Appliquer / Ré-initialiser
        col_apply, col_reset = st.columns([0.4, 0.6])

        with col_apply:
            if st.button("Appliquer la suppression"):
                if ss.get('target_column') in ss['drop_cols']:
                    st.warning(f"La cible `{ss['target_column']}` ne peut pas être supprimée.")
                else:
                    ss['df'] = ss['df'].drop(columns=ss['drop_cols'], errors="ignore")
                    ss['num_cols'] = [c for c in ss.get('num_cols', []) if c in ss['df'].columns]
                    ss['cat_cols'] = [c for c in ss.get('cat_cols', []) if c in ss['df'].columns]
                    if ss.get('date_col') and ss['date_col'] not in ss['df'].columns:
                        ss['date_col'] = None
                    save_project_config()
                    msg = f"Colonnes supprimées : {', '.join(ss['drop_cols'])}" if ss['drop_cols'] else "Aucune colonne sélectionnée."
                    st.success(msg)

        with col_reset:
            if st.button("Ré‑initialiser (annuler toutes les suppressions)"):
                # Réinitialiser l'état des checkboxes
                for col in drop_candidates:
                    st.session_state[f"dropcol_{col}"] = False
                ss['drop_cols'] = []
                try:
                    if ss.get('data_file'):
                        p = Path(ss['data_file'])
                        if p.exists():
                            if p.suffix.lower() in (".csv", ".txt"):
                                try:
                                    ss['df'] = pd.read_csv(p)
                                except Exception:
                                    ss['df'] = pd.read_csv(p, sep="\t")
                            elif p.suffix.lower() in (".xls", ".xlsx"):
                                ss['df'] = pd.read_excel(p)
                            elif p.suffix.lower() == ".parquet":
                                ss['df'] = pd.read_parquet(p)
                            else:
                                ss['df'] = pd.read_csv(p)
                        else:
                            st.warning("Fichier source introuvable, réinitialisation partielle uniquement.")
                    else:
                        st.warning("Pas de fichier source enregistré, réinitialisation partielle uniquement.")
                except Exception as e:
                    st.error(f"Réinitialisation impossible : {e}")
                save_project_config()
                st.success("Colonnes rétablies.")

        # --- Analyse des qualitatives & Choix d'encodage ---
        st.subheader("Encodage des variables qualitatives")

        # (1) Petit récap sur les colonnes catégorielles restantes
        df_preview = ss['df']
        cat_cols_all = [c for c in df_preview.columns if c != ss.get('target_column') and not pd.api.types.is_numeric_dtype(df_preview[c])]
        if ss.get('date_col') and ss['date_col'] in cat_cols_all:
            cat_cols_all.remove(ss['date_col'])

        if len(cat_cols_all) == 0:
            st.info("Aucune variable qualitative détectée (hors date/cible).")
        else:
            # tableau: nb de modalités et 5 plus fréquentes
            recs = []
            for c in cat_cols_all:
                vc = df_preview[c].astype(str).value_counts()
                top = ", ".join([f"{k} ({v})" for k, v in vc.head(5).items()])
                recs.append({"colonne": c, "niveaux": int(vc.shape[0]), "top5": top})
            st.write(pd.DataFrame(recs))

        # (2) Choix global + overrides par colonne
        ss.setdefault('cat_encoding_global', 'onehot')             # 'onehot' | 'ordinal' | 'freq' | 'target'
        ss.setdefault('cat_encoding_map', {})                      # dict colonne -> encodage spécifique
        enc_help = {
            "onehot": "One‑Hot (par défaut) : + de colonnes, mais sans ordre artificiel.",
            "ordinal": "Ordinal : entier par catégorie (OK pour arbres ; attention à l'ordre artificiel).",
            "freq": "Fréquence : encode par la fréquence d'apparition (compact & robuste).",
            "target": "Target‑mean : encode par la moyenne de la cible (éviter fuite de cible ; binaire/ régression conseillé)."
        }
        st.markdown("**Stratégie globale d'encodage :**")
        enc_global = st.selectbox(
            "Méthode par défaut pour toutes les colonnes qualitatives",
            options=["onehot", "ordinal", "freq", "target"],
            format_func=lambda x: f"{x} — {enc_help[x]}",
            index=["onehot","ordinal","freq","target"].index(ss['cat_encoding_global'])
        )
        ss['cat_encoding_global'] = enc_global

        if cat_cols_all:
            st.markdown("**Overrides par colonne (optionnel) :**")
            with st.expander("Choisir une méthode différente pour certaines colonnes"):
                for c in cat_cols_all:
                    cur = ss['cat_encoding_map'].get(c, ss['cat_encoding_global'])
                    sel = st.selectbox(
                        f"{c}",
                        ["(hériter)", "onehot", "ordinal", "freq", "target"],
                        index=["(hériter)","onehot","ordinal","freq","target"].index(cur if cur in ["onehot","ordinal","freq","target"] else "(hériter)")
                    )
                    if sel == "(hériter)":
                        ss['cat_encoding_map'].pop(c, None)
                    else:
                        ss['cat_encoding_map'][c] = sel

        # (3) Mémo UX
        st.caption("ℹ️ L'encodage sera appliqué automatiquement dans l'étape **AutoML** selon vos choix (global + overrides). "
                "En classification **multiclasse**, le *target‑mean* repasse en **fréquence** pour éviter la fuite de cible.")
        save_project_config()
            
        df = ss['df']  # travailler sur la version nettoyée

        st.subheader("Aperçu")
        st.dataframe(df.head(10))

        st.subheader("Statistiques descriptives")
        st.write(df.describe(include='all').transpose())

        st.subheader("Types de colonnes")
        st.write(df.dtypes.value_counts())

        # st.subheader("Proportion manquante de chaque colonne")

        def plot_missingness_js(df_):
            miss = df_.isna().mean().sort_values(ascending=False)
            if (miss > 0).any():
                data = {
                    "labels": miss.index.tolist(),
                    "datasets": [{
                        "label": "Taux manquant",
                        "data": [float(x) for x in miss.values],
                        "borderColor": "rgba(13, 110, 253, 1)",
                        "backgroundColor": "rgba(13, 110, 253, 0.5)",
                        "borderWidth": 2,
                        "borderRadius": 5,
                        "borderSkipped": False,
                    }]
                }
                options = {
                    "plugins": {"legend": {"display": False}, "title": {"display": True, "text": "Proportion manquante"}},
                    "scales": {
                        "x": {"title": {"display": True, "text": "Colonnes"}, "ticks": {"maxRotation": 45, "minRotation": 45}},
                        "y": {"min": 0, "max": 1, "title": {"display": True, "text": "Proportion manquante"}}
                    }
                }
                _chartjs("bar", data, options, height=int(160 + 24*len(miss)))
                chart_desc("Identifier les colonnes fortement incomplètes pour décider d’une imputation, d’un filtrage ou d’une suppression.")

        plot_missingness_js(df)

        # all_cols = df.columns.tolist()
        # target_idx = 0 if all_cols else 0
        # ss['target_column'] = st.selectbox("Sélectionner la colonne cible (target)", all_cols, index=target_idx)

        # auto num/cat (excluant date_col et target)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]
        if ss['date_col'] and ss['date_col'] in cat_cols:
            cat_cols.remove(ss['date_col'])
        if ss['target_column'] in num_cols:
            num_cols.remove(ss['target_column'])
        if ss['target_column'] in cat_cols:
            cat_cols.remove(ss['target_column'])

        st.subheader("Typage des variables (override manuel)")
        num_options = [c for c in df.columns if c != ss['target_column']]
        ss['num_cols'] = st.multiselect("Colonnes quantitatives", num_options,
                                        default=[c for c in num_cols if c in num_options])
        cat_options = [c for c in df.columns if c not in ss['num_cols'] and c != ss['target_column']]
        ss['cat_cols'] = st.multiselect("Colonnes qualitatives", cat_options,
                                        default=[c for c in cat_cols if c in cat_options])

        # Graphes Chart.js
        if ss['show_graphs']:
            if ss['num_cols']:
                st.subheader("Distributions (quantitatives)")
                for c in ss['num_cols'][:6]:
                    vals = df[c].dropna().astype(float).values
                    if len(vals) == 0:
                        continue
                    hist, bin_edges = np.histogram(vals, bins=30)
                    data = {
                        "labels": [f"{bin_edges[i]:.2f}" for i in range(len(bin_edges)-1)],
                        "datasets": [{
                            "label": c,
                            "data": hist.tolist(),
                            "borderColor": "rgba(13, 110, 253, 1)",
                            "backgroundColor": "rgba(13, 110, 253, 0.5)",
                            "borderWidth": 2,
                            "borderRadius": 5,
                            "borderSkipped": False,
                        }]
                    }
                    options = {
                        "plugins": {"legend": {"display": False}, "title": {"display": True, "text": c}},
                        "scales": {
                            "x": {"title": {"display": True, "text": "Valeurs (bins)" }},
                            "y": {"title": {"display": True, "text": "Fréquence"}}
                        }
                    }
                    _chartjs("bar", data, options, height=450)
                    chart_desc("Comprendre la forme de la distribution (asymétrie, dispersion, valeurs extrêmes).")

            if ss['cat_cols']:
                st.subheader("Fréquences (qualitatives)")
                for c in ss['cat_cols'][:6]:
                    vc = df[c].astype(str).value_counts().head(12)
                    data = {
                        "labels": vc.index.tolist(),
                        "datasets": [{
                            "label": c,
                            "data": [float(x) for x in vc.values],
                            "borderColor": "rgba(13, 110, 253, 1)",
                            "backgroundColor": "rgba(13, 110, 253, 0.5)",
                            "borderWidth": 2,
                            "borderRadius": 5,
                            "borderSkipped": False,
                        }]
                    }
                    options = {
                        "plugins": {"legend": {"display": False}, "title": {"display": True, "text": c}},
                        "scales": {
                            "x": {"title": {"display": True, "text": "Catégories"}, "ticks": {"maxRotation": 45, "minRotation": 45}},
                            "y": {"title": {"display": True, "text": "Comptes"}}
                        }
                    }
                    _chartjs("bar", data, options, height=450)
                    chart_desc("Repérer les catégories dominantes/rares pour orienter l’encodage ou un regroupement.")

            if len(ss['num_cols']) >= 2:
                st.subheader("Matrice de corrélation")
                corr = df[ss['num_cols']].corr().clip(-1, 1).round(2)
                _chartjs_corr_heatmap(corr, x_label="Variables", y_label="Variables", height=480)
                chart_desc("Mesurer la corrélation (Pearson). Des |corr| élevés (>0,8) suggèrent de gérer la colinéarité.")
            else:
                st.info("Ajoutez au moins deux colonnes quantitatives pour afficher la matrice de corrélation.")

        # Tâche (TS ⇒ régression)
        if ss['data_type'] == "time_series":
            ss['task_type'] = "regression"
            st.info("Mode **série temporelle** : tâche fixée à **régression**.")
        else:
            task_lbl = st.selectbox("Type de variable cible", ["Quantitative (régression)", "Qualitative (classification)"],
                                    index=0 if ss['task_type']=="regression" else 1)
            ss['task_type'] = "regression" if task_lbl.startswith("Quantitative") else "classification"

        ss['step_done']['Analyse'] = True
        save_project_config()

    nav_controls(ss['step_done']['Analyse'], button_titles[step], save_before_next=True,
                 show_prev=True, show_next=True, key_prefix=step)

# ------------------ Étape 3 : AutoML ------------------
if step == "AutoML":
    
    st.header("3️⃣ AutoML")
    if ss.get('df') is None or ss.get('target_column') is None:
        st.info("Veuillez compléter l'analyse et choisir la cible.")
    else:
        df = ss['df']
        target = ss['target_column']

        # 1) Sélection des modèles selon la tâche
        if ss['data_type'] == "time_series" or ss['task_type'] == "regression":
            models = build_reg_models(ss['models_selected'])
        else:
            models = build_clf_models(ss['models_selected'])
        if not models:
            st.warning("Veuillez sélectionner au moins un modèle dans Paramètres > Modèles AutoML.")
            st.stop()

        # === Préparation des données (branche selon data_type) ===
        # Les variables suivantes DOIVENT être définies avant l'entraînement :
        # X_train, X_test, y_train, y_test, preprocessor, ss['X_columns']
        if ss['data_type'] == "tabular":
            # 1) Construction X, y bruts
            X = df.drop(columns=[target]).copy()
            y = df[target].copy()

            # 2) Nettoyage de la cible selon la tâche (avant split)
            if ss['task_type'] == "classification":
                # Choix de stratégie via session (ou fixe "drop")
                cls_strategy = ss.get("target_nan_strategy_cls", "drop")  # "drop" ou "mode"
                df_clean, y_info = sanitize_target_for_classification(df[[*X.columns, target]], target, strategy=cls_strategy)

                # Feedback UI
                if y_info["strategy_used"] == "drop" and y_info["n_nan"] > 0:
                    st.warning(f"🧹 Cible (clf): {y_info['n_nan']} NaN supprimés ({y_info['n_total']} lignes au départ).")
                elif y_info["strategy_used"] == "mode":
                    st.info(f"🧩 Cible (clf): {y_info['n_nan']} NaN imputés par la classe majoritaire: '{y_info.get('fill_val')}'.")
                else:
                    st.caption("✅ Cible (clf) sans NaN.")

                if y_info["n_classes"] < 2:
                    st.error("La cible ne comporte pas au moins 2 classes distinctes après nettoyage. Impossible d'entraîner un classifieur.")
                    st.stop()

                X = df_clean.drop(columns=[target]).copy()
                y = df_clean[target].copy()

            else:  # regression
                reg_strategy = ss.get("target_nan_strategy_reg", "drop")  # "drop", "mean", "median", "ffill", "bfill", "zero"
                df_clean, y_info = sanitize_target_for_regression(df[[*X.columns, target]], target, strategy=reg_strategy)

                if y_info["strategy_used"] == "drop" and y_info["n_nan"] > 0:
                    st.warning(f"🧹 Cible (reg): {y_info['n_nan']} NaN supprimés sur {y_info['n_total']} lignes.")
                elif y_info["strategy_used"] in ("mean", "median", "zero", "ffill+bfill", "bfill+ffill"):
                    msg = f"🧩 Cible (reg): {y_info['n_nan']} NaN imputés (méthode={y_info['strategy_used']}"
                    if "fill_val" in y_info:
                        msg += f", valeur={y_info['fill_val']:.5g}"
                    msg += ")."
                    st.info(msg)
                else:
                    st.caption("✅ Cible (reg) sans NaN.")

                X = df_clean.drop(columns=[target]).copy()
                y = df_clean[target].copy()

            # 3) Dérivation de features dates si demandé (sur X)
            if ss['use_date_features'] and len(ss.get('cat_cols', [])) > 0:
                date_like = []
                for c in list(ss['cat_cols']):
                    parsed = pd.to_datetime(X[c], errors='coerce', utc=False)
                    if parsed.notna().mean() >= 0.8:
                        date_like.append(c)
                if date_like:
                    added = expand_date_features(X, date_like)  # côté utilisateur
                    ss['cat_cols'] = [c for c in ss['cat_cols'] if c not in date_like]
                    ss['num_cols'] = list(dict.fromkeys(ss.get('num_cols', []) + added))

            # 4) Sélection des colonnes num/cat
            num_cols = [c for c in ss.get('num_cols', []) if c in X.columns]
            cat_cols = [c for c in ss.get('cat_cols', []) if c in X.columns]

            if not num_cols and not cat_cols:
                X['index'] = np.arange(len(X))
                num_cols = ['index']
                st.warning("Aucune feature sélectionnée — fallback sur une feature 'index'.")

            # 5) Cast numérique (sur num_cols)
            coerce_numeric_inplace(X, num_cols)  # côté utilisateur

            # 6) Split train/test (stratify que si pertinent)
            strat = None
            if ss['task_type'] == "classification":
                n_classes = y.nunique(dropna=True)
                if n_classes >= 2 and n_classes < 40:
                    strat = y

            split_res = safe_split(
                X, y, test_size=ss['test_size'],
                stratify=strat
            )
            if split_res is None:
                st.error("Jeu de données trop petit pour un split train/test.")
                st.stop()
            X_train, X_test, y_train, y_test = split_res

            # 7) Garde‑fous post‑split
            if pd.isna(y_train).any() or pd.isna(y_test).any():
                st.error("La cible (y) contient des NaN après nettoyage/split. Change la stratégie de nettoyage de la cible.")
                st.stop()
            if ss['task_type'] == "classification":
                # au moins 2 classes côté train
                if pd.Series(y_train).nunique(dropna=True) < 2:
                    st.error("Moins de 2 classes dans y_train après split. Rééquilibrer les classes ou ajuster le split.")
                    st.stop()

            # 8) Préprocesseur (num/cat)
            transformers = []
            if num_cols:
                num_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy=ss['impute_strategy_num'])),
                    ("scaler", StandardScaler() if ss['scale_numeric'] else "passthrough")
                ])
                transformers.append(("num", num_pipe, num_cols))

            if cat_cols and ss['encode_categorical']:
                def split_cats_by_method(cat_cols, global_method, overrides):
                    groups = {"onehot": [], "ordinal": [], "freq": [], "target": []}
                    for c in cat_cols:
                        m = overrides.get(c, global_method)
                        if m not in groups:
                            m = global_method
                        groups[m].append(c)
                    return groups

                global_m = ss.get('cat_encoding_global', 'onehot')
                overrides = ss.get('cat_encoding_map', {})
                groups = split_cats_by_method(cat_cols, global_m, overrides)

                if groups["onehot"]:
                    cat_pipe_onehot = Pipeline([
                        ("imputer", SimpleImputer(strategy=ss['impute_strategy_cat'], fill_value="NA")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                    ])
                    transformers.append(("cat_onehot", cat_pipe_onehot, groups["onehot"]))

                if groups["ordinal"]:
                    cat_pipe_ord = Pipeline([
                        ("imputer", SimpleImputer(strategy=ss['impute_strategy_cat'], fill_value="NA")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
                    ])
                    transformers.append(("cat_ordinal", cat_pipe_ord, groups["ordinal"]))

                if groups["freq"]:
                    cat_pipe_freq = Pipeline([
                        ("imputer", SimpleImputer(strategy=ss['impute_strategy_cat'], fill_value="NA")),
                        ("freq", FrequencyEncoder(columns=groups["freq"]))
                    ])
                    transformers.append(("cat_freq", cat_pipe_freq, groups["freq"]))

                if groups["target"]:
                    cat_pipe_tgt = Pipeline([
                        ("imputer", SimpleImputer(strategy=ss['impute_strategy_cat'], fill_value="NA")),
                        ("tgt", TargetMeanEncoder(columns=groups["target"], smoothing=10.0))
                    ])
                    transformers.append(("cat_target", cat_pipe_tgt, groups["target"]))

            preprocessor = ColumnTransformer(transformers, remainder="drop") if transformers else "passthrough"
            ss['X_columns'] = num_cols + cat_cols

        else:
            # === time_series ===
            if not ss.get('date_col'):
                st.error("Mode série temporelle : la colonne de date est obligatoire.")
                st.stop()

            # 1) Trier et parser la colonne date
            df_sorted = df.sort_values(by=ss['date_col']).reset_index(drop=True)
            dates_all = pd.to_datetime(df_sorted[ss['date_col']], errors='coerce')
            if dates_all.isna().any():
                n_bad = int(dates_all.isna().sum())
                st.warning(f"⚠️ {n_bad} dates invalides détectées → lignes supprimées.")
                df_sorted = df_sorted.loc[dates_all.notna()].reset_index(drop=True)
                dates_all = pd.to_datetime(df_sorted[ss['date_col']], errors='coerce')

            # Vérifier l'ordre strictement croissant pour éviter les surprises
            if not dates_all.is_monotonic_increasing:
                st.info("Ré-ordonnancement chronologique effectué sur la colonne de date.")
                df_sorted = df_sorted.sort_values(by=ss['date_col']).reset_index(drop=True)
                dates_all = pd.to_datetime(df_sorted[ss['date_col']], errors='coerce')

            # 2) Cible numérique + petite stratégie d'imputation
            #    (tu peux exposer ss['ts_target_strategy'] = "mean|median|ffill|bfill|zero|drop")
            ts_strategy = ss.get("ts_target_strategy", "mean")
            y_full = pd.to_numeric(df_sorted[target], errors='coerce')

            if y_full.isna().all():
                st.error("Toutes les valeurs de la cible sont NaN après conversion numérique.")
                st.stop()

            if ts_strategy == "median":
                y_full = y_full.fillna(y_full.median())
            elif ts_strategy == "ffill":
                y_full = y_full.ffill().bfill()
            elif ts_strategy == "bfill":
                y_full = y_full.bfill().ffill()
            elif ts_strategy == "zero":
                y_full = y_full.fillna(0.0)
            elif ts_strategy == "drop":
                keep = y_full.notna()
                drop_n = int((~keep).sum())
                if drop_n > 0:
                    st.warning(f"🧹 {drop_n} lignes supprimées (y NaN) avant construction des lags.")
                y_full = y_full.loc[keep].reset_index(drop=True)
                dates_all = dates_all.loc[keep].reset_index(drop=True)
            else:  # "mean" (défaut)
                y_full = y_full.fillna(y_full.mean())

            y_full = pd.Series(y_full.values)  # propre

            # 3) Lags
            X_lag, y_lag = make_ts_lags_safe(y_full, ss['n_lags'])
            if X_lag is None or y_lag is None:
                st.error("Données insuffisantes pour construire des lags (≥3 points requis).")
                st.stop()

            # IMPORTANT : aligner les dates avec les lags (on perd n_lags premières dates)
            dates_lag = dates_all.iloc[int(ss['n_lags']):].reset_index(drop=True)
            if len(dates_lag) != len(y_lag):
                st.error("Incohérence longueur dates/valeurs après lags.")
                st.stop()

            # 4) Split temporel
            split_res = train_test_split_ts_safe(X_lag, y_lag, test_size=ss['test_size'])
            if split_res is None:
                st.error("Données insuffisantes pour un split temporel.")
                st.stop()
            X_train, X_test, y_train, y_test = split_res

            # Découper les dates dans la même proportion/longueur
            n_train = len(X_train)
            dates_train = dates_lag.iloc[:n_train]
            dates_test  = dates_lag.iloc[n_train:]

            # 5) Garde‑fous post‑split
            if pd.isna(y_train).any() or pd.isna(y_test).any():
                st.error("NaN détectés dans y (time_series) après préparation. Vérifie la stratégie d'imputation/suppression.")
                st.stop()

            # 6) Sauvegardes session (pour graphes à dates réelles)
            ss['ts_dates_all'] = dates_all
            ss['ts_dates_lag'] = dates_lag
            ss['ts_dates_train'] = dates_train
            ss['ts_dates_test']  = dates_test

            # Fréquence utile pour générer les futures dates (prévision)
            freq = pd.infer_freq(dates_all)
            if freq is None:
                diffs = dates_all.diff().dropna()
                ss['ts_freq'] = diffs.median() if not diffs.empty else pd.Timedelta(days=1)
            else:
                ss['ts_freq'] = freq

            # 7) Préprocesseur & colonnes
            preprocessor = "passthrough"
            ss['X_columns'] = X_lag.columns.tolist()

        # === 2) Entraînement & métriques via @st.dialog (COMMUN tabulaire / TS) ===

        ss.setdefault('automl_logs', [])
        ss['is_training'] = True  # flag visible dans la page

        @st.dialog("🚀 Monitoring et suivi des logs — AutoML", width="large")
        def run_automl_dialog(models, preprocessor, X_train, y_train, X_test, y_test, task_type):
            predictions, metrics, fitted = {}, {}, {}
            total_models = max(1, len(models))

            # reset logs visibles dans la modale
            ss['automl_logs'] = []
            log_lines = ss['automl_logs']

            # UI dans la modale
            status = st.status("⚙️ Préparation…", expanded=True)
            prog = st.progress(0, text="Initialisation de l'entraînement…")
            log_box = st.empty()
            dl_box  = st.empty()

            def log(msg: str):
                ts = datetime.now().strftime("%H:%M:%S")
                line = f"[{ts}] {msg}"
                log_lines.append(line)
                ss['automl_logs'] = log_lines
                log_box.code("\n".join(log_lines[-400:]), language="bash")

            # --- Contexte & départ
            log("Préprocesseur généré et split train/test OK.")
            log(f"Contexte: task={task_type} | X_train={getattr(X_train,'shape',None)} | "
                f"X_test={getattr(X_test,'shape',None)} | y_train={getattr(y_train,'shape',None)} | "
                f"y_test={getattr(y_test,'shape',None)}")
            status.update(label="🚀 Démarrage des modèles AutoML", state="running")

            t_global0 = time.time()

            # --- Entraînement
            for idx, (name, model) in enumerate(models.items(), start=1):
                step_label = f"({idx}/{total_models}) {name}"
                prog.progress((idx-1)/total_models, text=f"Entraînement {step_label}…")
                log(f"→ Entraînement du modèle: {name}")

                try:
                    pipe = Pipeline([("prep", preprocessor), ("model", model)])

                    t0 = time.time()
                    pipe.fit(X_train, y_train)
                    fit_dt = time.time() - t0
                    log(f"   ✓ Fit terminé en {fit_dt:.2f}s")

                    y_pred = pipe.predict(X_test)
                    predictions[name] = y_pred

                    if task_type == "regression":
                        m = {
                            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                            "MAE": float(mean_absolute_error(y_test, y_pred)),
                            "R2": float(r2_score(y_test, y_pred))
                        }
                    else:
                        acc = accuracy_score(y_test, y_pred)
                        f1w = f1_score(y_test, y_pred, average="weighted")
                        m = {"Accuracy": float(acc), "F1_weighted": float(f1w)}
                        try:
                            proba = pipe.predict_proba(X_test)
                            if getattr(proba, "ndim", 1) == 2 and proba.shape[1] == 2:
                                auc = roc_auc_score(y_test, proba[:, 1])
                                m["ROC_AUC"] = float(auc)
                        except Exception as e:
                            log(f"   (info) Pas de predict_proba ou AUC non applicable: {e}")

                    metrics[name] = m
                    fitted[name] = pipe
                    log(f"   ✓ Métriques: {m}")

                except Exception as e:
                    log(f"   ✗ Erreur sur {name}: {e}")
                    for ln in traceback.format_exc().splitlines():
                        log(f"      {ln}")

                prog.progress(idx/total_models, text=f"Entraînement {step_label} terminé.")

            # --- Persistes dans la session (lisible hors modale)
            ss['models'] = fitted
            ss['metrics'] = metrics
            ss['predictions'] = predictions
            ss['y_test'] = y_test
            ss['X_test'] = X_test

            # --- Fin
            t_total = time.time() - t_global0
            if len(fitted):
                status.update(label=f"✅ Apprentissage des modèles AutoML terminé en {t_total:.2f}s", state="complete")
                log(f"AutoML terminé avec {len(fitted)}/{total_models} modèles. Durée totale: {t_total:.2f}s")
            else:
                status.update(label="⚠️ Aucun modèle entraîné (voir logs)", state="error")
                log("AutoML terminé sans modèle entraîné.")

            # Télécharger les logs
            dl_box.download_button(
                "📥 Télécharger les logs",
                data="\n".join(ss['automl_logs']).encode("utf-8"),
                file_name=f"automl_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

            st.divider()
            st.button("Fermer la fenêtre")

        # 👉 Lance la modale et l'entraînement
        run_automl_dialog(models, preprocessor, X_train, y_train, X_test, y_test, ss['task_type'])

        # Fin du training
        ss['is_training'] = False


        # 3) Affichage des métriques + visualisations
        if not ss.get('is_training', False):
            st.subheader("Métriques des modèles")
            if ss.get('metrics'):
                st.dataframe(pd.DataFrame(ss['metrics']).T)
            else:
                st.info("Aucune métrique disponible (voir logs ci‑dessus).")

            if ss['show_graphs'] and ss.get('metrics'):
                # meilleur modèle
                if ss['task_type'] == "regression":
                    scorer = lambda k: ss['metrics'][k].get("R2", -1e9)
                else:
                    scorer = lambda k: ss['metrics'][k].get("Accuracy", -1e9)
                best_name = max(ss['metrics'], key=scorer)
                best_model = ss['models'][best_name]
                y_pred_best = ss['predictions'][best_name]
                st.markdown(f"**Meilleur modèle :** `{best_name}`")

                if ss['task_type'] == "regression":
                    if ss['data_type'] == "time_series":
                        # === 1) Prévision (réel vs prédit) - Test ===
                        st.subheader("Prévision (réel vs prédit) - Test")

                        if ss['data_type'] == "time_series":
                            # dates de la série ordonnées
                            df_sorted = df.sort_values(by=ss['date_col']).reset_index(drop=True)
                            # Quand on fabrique des lags, on “perd” les n_lags premières dates.
                            # On reconstitue l’index temporel aligné avec y_lag => dates_lag
                            n_lags = int(ss.get('n_lags', 0))
                            dates_lag = pd.to_datetime(df_sorted[ss['date_col']], errors="coerce").iloc[n_lags:].reset_index(drop=True)

                            # Les dates de TEST = les dernières len(y_test) dates de dates_lag
                            test_len = len(ss['y_test'])
                            test_dates = dates_lag.iloc[-test_len:]
                            # labels_test = test_dates.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                            labels_test = format_dates_for_labels(test_dates)
                        else:
                            # Tabulaire : on reste sur des indices (ou mets ici des dates si tu en as)
                            labels_test = list(range(len(ss['y_test'])))

                        data = {
                            "labels": labels_test,
                            "datasets": [
                                ds_line("Réel", np.asarray(ss['y_test']).ravel().tolist(), PRIMARY),
                                ds_line("Prédit", np.asarray(y_pred_best).ravel().tolist(), ss['pred_color']),
                            ],
                        }
                        options = {
                            "plugins": {"legend": {"display": True}, "title": {"display": True, "text": "Prévision (réel vs prédit) — Test"}},
                            "interaction": {"mode": "index", "intersect": False},
                            "scales": {
                                "x": {"title": {"display": True, "text": "Date" if ss['data_type']=="time_series" else "Index test"}},
                                "y": {"title": {"display": True, "text": ss['target_column'] or "Valeur"}},
                            },
                        }
                        _chartjs("line", data, options, height=450)
                        chart_desc("Vérifier que le modèle suit la dynamique passée.")

                        # === 2) Prévision future (dates réelles) ===
                        st.subheader(f"Prévision future de {target}")

                        # historique propre du target (ordre chronologique)
                        series_sorted = pd.to_numeric(
                            df.sort_values(by=ss['date_col'])[ss['target_column']],
                            errors='coerce'
                        )
                        # impute moyenne simple
                        series_sorted = pd.Series(
                            SimpleImputer(strategy="mean").fit_transform(np.array(series_sorted).reshape(-1,1)).ravel()
                        )
                        future_vals = forecast_with_lags(best_model, series_sorted, steps=ss['horizon'], lags=ss['n_lags'])

                        if future_vals:
                            future_vals = list(map(float, future_vals))

                            # queue de l'historique pour contexte (au moins 30 points ou n_lags)
                            tail_len = max(30, int(ss['n_lags']))
                            hist_tail = series_sorted.iloc[-tail_len:].astype(float).tolist()

                            # === Labels par dates ===
                            # 1) dates historiques (queue)
                            full_dates = pd.to_datetime(df.sort_values(by=ss['date_col'])[ss['date_col']], errors="coerce")
                            hist_dates_tail = full_dates.iloc[-tail_len:]

                            # 2) fréquence
                            freq = pd.infer_freq(full_dates)
                            if freq is None:
                                # fallback: médiane des deltas (peut être Timedelta)
                                diffs = full_dates.diff().dropna()
                                if not diffs.empty:
                                    freq = diffs.median()
                                else:
                                    # dernier recours: 1 jour
                                    freq = pd.Timedelta(days=1)

                            # 3) dates futures, à partir de la dernière date connue
                            last_date = full_dates.iloc[-1]
                            if isinstance(freq, pd.Timedelta):
                                # date_range n'accepte pas Timedelta direct en freq; on incrémente à la main
                                future_dates = [last_date + (i+1)*freq for i in range(ss['horizon'])]
                                future_dates = pd.to_datetime(future_dates)
                            else:
                                future_dates = pd.date_range(start=last_date, periods=ss['horizon']+1, freq=freq)[1:]

                            # 4) labels finaux = hist_tail dates + futures dates
                            # labels_future = hist_dates_tail.dt.strftime("%Y-%m-%d %H:%M:%S").tolist() + \
                            #                 pd.to_datetime(future_dates).strftime("%Y-%m-%d %H:%M:%S").tolist()

                            labels_future = format_dates_for_labels(hist_dates_tail) + format_dates_for_labels(pd.Series(future_dates))

                            # datasets (on “décale” avec None pour aligner visuellement)
                            hist_ds = ds_line(f"{target} Historique (fin)", hist_tail + [None]*len(future_vals), PRIMARY)
                            pred_ds = ds_line(f"Prévision (+{ss['horizon']})", [None]*len(hist_tail) + future_vals, ss['pred_color'])
                            hist_ds.update({"pointStyle":"circle","pointRadius":5,"pointHoverRadius":10})
                            pred_ds.update({"pointStyle":"circle","pointRadius":5,"pointHoverRadius":10})

                            data = {"labels": labels_future, "datasets": [hist_ds, pred_ds]}
                            options = {
                                "plugins": {"legend": {"display": True}, "title": {"display": True, "text": "Prévision future"}},
                                "scales": {
                                    "x": {"title": {"display": True, "text": "Date"}},
                                    "y": {"title": {"display": True, "text": ss['target_column'] or "Valeur"}},
                                },
                            }
                            _chartjs("line", data, options, height=450)
                            chart_desc("Projection du modèle sur les prochaines dates réelles. ⚠️ Pas d’intervalle d’incertitude.")
                    else:
                        st.subheader("Réel vs Prédit (test)")
                        pts = [{"x": float(a), "y": float(b)} for a, b in zip(ss['y_test'], y_pred_best)]
                        minv = float(min(np.min(ss['y_test']), np.min(y_pred_best)))
                        maxv = float(max(np.max(ss['y_test']), np.max(y_pred_best)))
                        data = {
                            "datasets": [
                                ds_scatter("Points", pts, ss['pred_color']),
                                {"type": "line", "label": "y = x",
                                "data": [{"x": minv, "y": minv}, {"x": maxv, "y": maxv}],
                                "borderColor": "rgba(13, 110, 253, 1)", "borderDash": [6, 6],
                                "pointRadius": 0, "tension": 0}
                            ]
                        }
                        options = {
                            "plugins": {"legend": {"display": True}, "title": {"display": True, "text": "Réel vs Prédit (test)"}},
                            "scales": {"x": {"title": {"display": True, "text": "Réel"}},
                                    "y": {"title": {"display": True, "text": "Prédit"}}}
                        }
                        _chartjs("scatter", data, options, height=450)
                        chart_desc("Plus les points sont proches de la diagonale, meilleur est le modèle.")

                        st.subheader("Résidus (test)")
                        resid = (np.asarray(ss['y_test']).ravel() - np.asarray(y_pred_best).ravel()).astype(float)
                        data = {
                            "labels": list(range(len(resid))),
                            "datasets": [ds_line("Résidu", resid.tolist(), PRIMARY),
                                        ds_line("0", [0.0]*len(resid), "#FF7F0E")]
                        }
                        options = {"plugins": {"legend": {"display": True}, "title": {"display": True, "text": "Résidus (test)"}},
                                "scales": {"x": {"title": {"display": True, "text": "Index"}},
                                            "y": {"title": {"display": True, "text": "Résidu"}}}}
                        _chartjs("line", data, options, height=450)
                        chart_desc("Les résidus doivent osciller autour de 0 sans structure apparente.")
                else:
                    st.subheader("Matrice de confusion (meilleur modèle)")
                    cm = confusion_matrix(ss['y_test'], y_pred_best)
                    classes = [str(x) for x in np.unique(ss['y_test'])]
                    cells, mx = [], int(cm.max()) if cm.size else 1
                    for yi, row in enumerate(cm):
                        for xi, v in enumerate(row):
                            t = (int(v) / mx) if mx else 0.0
                            alpha = 0.15 + 0.85 * t
                            cells.append({"x": classes[xi], "y": classes[yi], "v": int(v),
                                        "bg": _rgba_from_hex(PRIMARY, alpha)})
                    data = {"datasets": [{
                        "label": "Confusion",
                        "data": [{"x": c["x"], "y": c["y"], "v": c["v"]} for c in cells],
                        "backgroundColor": [c["bg"] for c in cells],
                        "borderWidth": 1, "borderColor": "#FFFFFF", "width": 28, "height": 28
                    }]}
                    options = {"plugins": {"legend": {"display": False}, "title": {"display": True, "text": "Matrice de confusion"}},
                            "scales": {"x": {"type": "category", "labels": classes, "title": {"display": True, "text": "Prédit"}},
                                        "y": {"type": "category", "labels": classes, "reverse": True, "title": {"display": True, "text": "Réel"}}}}
                    _chartjs("matrix", data, options, height=int(200 + 30*len(classes)),
                            extra_scripts=["https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@2"])
                    chart_desc("Analyser la qualité par classe : la diagonale = bonnes prédictions ; hors-diagonale = erreurs.")

                    try:
                        if hasattr(best_model, "predict_proba"):
                            proba = best_model.predict_proba(ss['X_test'])
                            if proba.shape[1] == 2:
                                st.subheader("Courbe ROC (binaire)")
                                fpr, tpr, _ = roc_curve(ss['y_test'], proba[:, 1])
                                data = {"labels": [float(x) for x in fpr],
                                        "datasets": [ds_line("ROC", [float(x) for x in tpr], ss['pred_color']),
                                                    ds_line("Aléatoire", [float(x) for x in fpr], "#888888")]}
                                options = {"plugins": {"legend": {"display": True}, "title": {"display": True, "text": "Courbe ROC (binaire)"}},
                                        "scales": {"x": {"min": 0, "max": 1, "title": {"display": True, "text": "FPR"}},
                                                    "y": {"min": 0, "max": 1, "title": {"display": True, "text": "TPR"}}}}
                                _chartjs("line", data, options, height=360)
                                chart_desc("Plus la courbe est proche du coin supérieur gauche et l’AUC élevée, meilleur est le modèle.")
                    except Exception:
                        pass

        st.success("AutoML terminé !")
        st.toast("Traitement de AutoML terminé !", icon="✅")
        ss['step_done']['AutoML'] = True
        save_project_config()

    nav_controls(ss['step_done']['AutoML'], button_titles[step], save_before_next=False, show_prev=True, show_next=True, key_prefix=step)

# ------------------ Étape 4 : Prédiction ------------------
if step == "Prédiction":
    st.header("4️⃣ Entrée des paramètres pour prédiction")
    if ss.get('df') is None or not ss.get('models') or ss.get('target_column') is None:
        st.info("Veuillez lancer AutoML avant la prédiction.")
    else:
        df = ss['df']
        target = ss['target_column']
        model_name = st.selectbox("Choisir un modèle", list(ss['models'].keys()), index=0)
        model = ss['models'][model_name]

        if ss['data_type'] == "time_series" and ss['task_type'] == "regression":
            st.subheader("Prévision future")
            h_pred = st.slider("Horizon (pas)", 1, 60, ss['horizon'])

            # Historique (trié par date) et série nettoyée
            df_sorted = df.sort_values(by=ss['date_col']).reset_index(drop=True)
            full_dates = pd.to_datetime(df_sorted[ss['date_col']], errors="coerce")

            last_hist = pd.to_numeric(df_sorted[target], errors='coerce')
            last_hist = pd.Series(SimpleImputer(strategy="mean")
                                .fit_transform(np.array(last_hist).reshape(-1,1)).ravel())

            future_vals = forecast_with_lags(model, last_hist, steps=h_pred, lags=ss['n_lags'])

            if future_vals:
                future_vals = list(map(float, future_vals))
                st.success(f"Première prévision future de {target} : {future_vals[0]:.4f}")

                # Queue historique pour le contexte
                tail_len = max(30, int(ss['n_lags']))
                hist_tail_vals = last_hist.iloc[-tail_len:].astype(float).tolist()
                hist_tail_dates = full_dates.iloc[-tail_len:]

                # Déterminer la fréquence des dates
                freq = pd.infer_freq(full_dates)
                if freq is None:
                    diffs = full_dates.diff().dropna()
                    if not diffs.empty:
                        freq = diffs.median()  # Timedelta
                    else:
                        freq = pd.Timedelta(days=1)  # dernier recours

                # Générer les futures dates à partir de la dernière date connue
                last_date = full_dates.iloc[-1]
                if isinstance(freq, pd.Timedelta):
                    fut_dates = [last_date + (i+1)*freq for i in range(h_pred)]
                    fut_dates = pd.to_datetime(fut_dates)
                else:
                    fut_dates = pd.date_range(start=last_date, periods=h_pred+1, freq=freq)[1:]

                # Labels = dates réelles formatées (masque HH:MM:SS si minuit)
                labels = format_dates_for_labels(hist_tail_dates) + format_dates_for_labels(pd.Series(fut_dates))

                # Datasets (décalage visuel avec None)
                hist_ds = ds_line(f"{target} Historique (fin)", hist_tail_vals + [None]*len(future_vals), PRIMARY)
                pred_ds = ds_line(f"Prévision (+{h_pred})", [None]*len(hist_tail_vals) + future_vals, ss['pred_color'])
                pred_ds.update({"pointStyle":"circle","pointRadius":5,"pointHoverRadius":10})
                hist_ds.update({"pointStyle":"circle","pointRadius":5,"pointHoverRadius":10})

                data = {"labels": labels, "datasets": [hist_ds, pred_ds]}
                options = {
                    "plugins": {"legend": {"display": True}, "title": {"display": True, "text": "Prévision future"},
                                "tooltip": {"enabled": True}},
                    "interaction": {"mode": "index", "intersect": False},
                    "scales": {
                        "x": {"title": {"display": True, "text": "Date"}},
                        "y": {"title": {"display": True, "text": ss['target_column'] or "Valeur"}},
                    },
                }
                _chartjs("line", data, options, height=450)
                chart_desc("Projeter la série sur l’horizon choisi et comparer la transition entre historique et futur prévu.")

                # Sauvegarde historique d'action (avec date de première prévision)
                proj_hist = Path(ss['project_dir']) / "history.csv"
                first_pred_date = pd.to_datetime(fut_dates[0]).isoformat() if len(fut_dates) else None
                row = pd.DataFrame([{
                    "timestamp": datetime.now().isoformat(),
                    "action": "forecast",
                    "model": model_name,              # conserve ta variable en amont
                    "horizon": h_pred,
                    "prediction_first": future_vals[0] if future_vals else None,
                    "prediction_first_date": first_pred_date
                }])
                if proj_hist.exists():
                    pd.concat([pd.read_csv(proj_hist), row], ignore_index=True).to_csv(proj_hist, index=False)
                else:
                    row.to_csv(proj_hist, index=False)

            else:
                st.warning("Pas assez d'historique pour générer des prévisions avec les lags actuels.")

        else:
            features = ss['X_columns'] if ss['X_columns'] else ['index']
            st.subheader("Paramètres d'entrée")
            user_input = {}
            for col in features:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    default_val = float(df[col].dropna().mean() if df[col].notna().any() else 0.0)
                    user_input[col] = st.number_input(f"{col}", value=default_val)
                elif col in df.columns:
                    default_val = str(df[col].dropna().mode()[0]) if df[col].notna().any() else "NA"
                    user_input[col] = st.text_input(f"{col} (catégorie)", value=default_val)
                else:
                    user_input[col] = st.number_input(f"{col}", value=0.0)

            if st.button("Prédire"):
                X_input = pd.DataFrame([user_input], columns=features)
                coerce_numeric_inplace(X_input, [c for c in features if c in ss['num_cols'] or c == 'index'])
                y_pred_input = model.predict(X_input)[0]
                target_name = ss.get('target_column', "Résultat")

                if ss['task_type'] == "regression":
                    st.success(f"✅ Prédiction de **{target_name}** : {float(y_pred_input):.4f}")
                else:
                    st.success(f"✅ Prédiction de **{target_name}** : {y_pred_input}")
                    try:
                        proba = model.predict_proba(X_input)[0]
                        classes = getattr(model, "classes_", np.arange(len(proba)))
                        proba_map = {str(c): float(p) for c, p in zip(classes, proba)}
                        st.subheader("Probabilités")
                        st.json(proba_map)
                    except Exception:
                        pass

                proj_hist = Path(ss['project_dir']) / "history.csv"
                row = pd.DataFrame([{
                    "timestamp": datetime.now().isoformat(),
                    "action": "predict",
                    "model": model_name,
                    "inputs": json.dumps(user_input, ensure_ascii=False),
                    "prediction": y_pred_input
                }])
                if proj_hist.exists():
                    pd.concat([pd.read_csv(proj_hist), row], ignore_index=True).to_csv(proj_hist, index=False)
                else:
                    row.to_csv(proj_hist, index=False)

        ss['step_done']['Prédiction'] = True
        save_project_config()

    nav_controls(ss['step_done']['Prédiction'], button_titles[step], save_before_next=False,
                 show_prev=True, show_next=True, key_prefix=step)

# ------------------ Étape 5 : Historique ------------------
def delete_project(slug: str, pdv_path: Path):
    """Supprimer le .pdv et le dossier du projet."""
    proj_dir = PROJECTS_DIR / slug
    try:
        if pdv_path.exists():
            pdv_path.unlink()
        if proj_dir.exists():
            shutil.rmtree(proj_dir, ignore_errors=True)
        # si on supprime le projet courant, nettoyer l'état minimal
        if ss.get('project_slug') == slug:
            for k in ['project_name','project_slug','project_dir','project_file','df','data_file']:
                ss[k] = None
        st.success(f"Projet '{slug}' supprimé.")
    except Exception as e:
        st.error(f"Suppression impossible : {e}")

if step == "Historique":
    st.header("5️⃣ Historique & Restauration de projets")
    pdv_files = list_projects()
    if not pdv_files:
        st.info("Aucun projet sauvegardé.")
    else:
        st.write("Projets disponibles :")
        for pdv in pdv_files:
            cols = st.columns([0.5, 1, 1, 0.35, 0.35])
            with open(pdv, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            pname = cfg.get("project_name", pdv.stem)
            pslug = cfg.get("project_slug", pdv.stem)
            pcreated = cfg.get("created_at", "")
            ptype = cfg.get("data_type", "")
            cols[0].markdown(f"**{pname}**  \n`{pslug}`")
            cols[1].markdown(f"Créé : {pcreated[:19]}")
            cols[2].markdown(f"Type : **{ptype}**")
            if cols[3].button("Restaurer", key=f"restore_{pslug}"):
                load_project_config(pdv)
                st.success(f"Projet restauré : {ss['project_name']}")
                go_to_step("Analyse", request_rerun=True)
            if cols[4].button("🗑️ Supprimer", key=f"delete_{pslug}"):
                delete_project(pslug, pdv)

        if ss.get('project_dir'):
            proj_hist = Path(ss['project_dir']) / "history.csv"
            if proj_hist.exists():
                st.subheader(f"Historique du projet : {ss['project_name']}")
                st.dataframe(pd.read_csv(proj_hist))
            else:
                st.info("Aucune entrée d'historique pour le projet courant.")

    nav_controls(True, button_titles[step], save_before_next=False,
                 show_prev=True, show_next=False, key_prefix=step)
