# app_prevision_vente_final_v23.py
# -------------------------------------------------------------------
# - FIX navigation: plus d'erreur "nav_step cannot be modified..."
#   -> go_to_step() écrit dans ss['pending_nav'] puis st.rerun()
#   -> avant de créer le radio, on applique pending_nav -> nav_step
# - Schéma du workflow (image) en étape "Workflow" via Matplotlib
# - Import: Fichier / API / SQL / MongoDB (retours (None,None) si pas cliqué)
# - Détection auto + choix manuel du type de données (dans Import)
# - TS -> régression sur lags ; Tabulaire -> régression ou classification
# - Projets .pdv, snapshots CSV, historique, restauration
# - Graphes à la couleur du thème + légendes ; navigation stable
# -------------------------------------------------------------------

import os
import io
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, RocCurveDisplay

# ----------- CSS -------------
# CSS personnalisé
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
    </style>
""", unsafe_allow_html=True)

# ---------- Config ----------
st.set_page_config(page_title="Prévision de Vente AutoML", layout="wide")
st.title("Prévision de Vente AutoML ✅")

BASE_DIR = Path.cwd()
PROJECTS_DIR = BASE_DIR / "projets"
PROJECTS_DIR.mkdir(exist_ok=True)

# Étapes (avec "Workflow" en premier)
steps = ["Workflow", "Projet", "Import données", "Analyse", "AutoML", "Prédiction", "Historique"]
button_titles = {
    "Workflow": "Démarrer ▶️",
    "Projet": "Créer/Charger puis passer à l'import ▶️",
    "Import données": "Importer et passer à l'analyse ▶️",
    "Analyse": "Analyser et passer à AutoML ▶️",
    "AutoML": "Lancer AutoML et passer à Prédiction ▶️",
    "Prédiction": "Prédire et passer à Historique ▶️",
    "Historique": "Fin ▶️"
}

ss = st.session_state
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

def _chartjs(chart_type: str, data: dict, options: dict | None = None,
             height: int = 360, extra_scripts: list[str] | None = None):
    """
    Render a Chart.js chart inside Streamlit.
    chart_type: 'line' | 'bar' | 'scatter' | etc.
    data: dict Chart.js (labels, datasets...)
    options: dict Chart.js options
    height: canvas height
    extra_scripts: CDN URLs for plugins (optional)
    """
    options = options or {}
    extra_scripts = extra_scripts or []
    chart_id = f"chartjs_{np.random.randint(0, 1_000_000)}"
    # Assure une sérialisation JSON propre (numpy -> python natif)
    data_json = json.dumps(data, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o))
    opts_json = json.dumps(options, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o))

    scripts = "\n".join([f'<script src="{u}"></script>' for u in extra_scripts])
    html = f"""
    <div style="width: 100%;">
      <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
      {scripts}
      <canvas id="{chart_id}"></canvas>
      <script>
        const ctx = document.getElementById("{chart_id}").getContext('2d');
        const cfg = {{
          type: '{chart_type}',
          data: {data_json},
          options: {opts_json}
        }};
        // eslint-disable-next-line no-unused-vars
        const chart_{chart_id} = new Chart(ctx, cfg);
      </script>
    </div>
    """
    st_html(html, height=height)

def ds_line(label, values, color, fill=False):
    return {
        "type": "line",
        "label": label,
        "data": list(map(lambda v: None if pd.isna(v) else float(v), values)),
        "borderColor": color,
        "backgroundColor": color,
        "pointRadius": 0,
        "tension": 0.15,
        "fill": fill,
    }

def ds_scatter(label, xy_pairs, color):
    # xy_pairs = [{"x": x, "y": y}, ...]
    return {
        "type": "scatter",
        "label": label,
        "data": [{"x": float(p["x"]), "y": float(p["y"])} for p in xy_pairs],
        "borderColor": color,
        "backgroundColor": color,
        "pointRadius": 3,
        "showLine": False,
    }

def normalize_sql_host_user(host_input: str, user_input: str):
    host = (host_input or "").strip()
    user = (user_input or "").strip()
    if "@" in host:
        maybe_user, maybe_host = host.split("@", 1)
        if not user:
            user = maybe_user.strip()
        host = maybe_host.strip()
    # anti-pièges "localhost " etc.
    host = host.strip()
    user = user.strip()
    return host, user

def slugify(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_ " else "-" for c in name).strip().replace(" ", "_")
    return s.lower() or f"projet_{int(datetime.now().timestamp())}"

def list_projects():
    return sorted([p for p in PROJECTS_DIR.glob("*.pdv")], key=lambda p: p.stat().st_mtime, reverse=True)

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
        "horizon": ss.get('horizon')
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
            df = pd.read_csv(data_path)
        except Exception:
            try:
                df = pd.read_excel(data_path)
            except Exception:
                df = None
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
    ss['step_done'].update({s: True for s in ["Projet", "Import données", "Analyse"]})

# ---------- Navigation helpers ----------
def go_to_step(step_name: str):
    # IMPORTANT: ne modifie pas nav_step directement (widget déjà instancié)
    ss['pending_nav'] = step_name
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

def nav_controls(next_enabled: bool, next_label: str, save_before_next: bool = False,
                 show_prev: bool = True, show_next: bool = True):
    col_prev, col_next = st.columns(2)
    with col_prev:
        if show_prev:
            st.button("◀️ Retour",
                      on_click=lambda: go_to_step(steps[max(0, steps.index(ss['nav_step']) - 1)]))
    with col_next:
        if show_next:
            st.button(next_label, disabled=not next_enabled,
                      on_click=(lambda: (save_project_config() if save_before_next else None,
                                         go_to_step(steps[min(len(steps)-1, steps.index(ss['nav_step']) + 1)]))))

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
        # 🔧 garde uniquement les modèles valides comme default
        prev = [m for m in ss['models_selected'] if m in reg_options]
        default_list = prev or reg_default
        ss['models_selected'] = st.multiselect(
            "Modèles régression",
            reg_options,
            default=default_list
        )
    else:
        prev = [m for m in ss['models_selected'] if m in clf_options]
        default_list = prev or clf_default
        ss['models_selected'] = st.multiselect(
            "Modèles classification",
            clf_options,
            default=default_list
        )

with st.sidebar.expander("Prétraitement", expanded=False):
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
    labels = ["Projet", "Importer", "Analyse", "AutoML", "Prédiction", "Historique"]
    n = len(labels)
    fig_w = min(2 + 2.2*n, 18)
    fig, ax = plt.subplots(figsize=(fig_w, 2.8))
    ax.axis('off')
    x0, y0 = 0.06, 0.45
    step_w = (0.88) / n
    for i, lab in enumerate(labels):
        x = x0 + i*step_w
        # box
        box = FancyBboxPatch((x, y0), step_w*0.8, 0.32,
                             boxstyle="round,pad=0.02,rounding_size=0.04",
                             linewidth=1.5, edgecolor=PRIMARY, facecolor="white")
        ax.add_patch(box)
        ax.text(x + step_w*0.4, y0 + 0.16, lab, ha='center', va='center', fontsize=11)
        # arrow
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
6. **Historique** : journal des actions et restauration de projets.
""")
    st.subheader("Schéma du workflow")
    fig = render_workflow_diagram()
    st.pyplot(fig); plt.close(fig)

    st.divider()
    if st.button("▶️ Démarrer"):
        go_to_step("Projet")

    # Première étape: pas de bouton "Retour", pas de "Suivant"
    nav_controls(True, button_titles[step], save_before_next=False, show_prev=False, show_next=False)

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
    go_to_step("Import données")

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
                go_to_step("Import données")
        else:
            st.info("Aucun projet pour l’instant. Créez-en un à gauche.")

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

def load_from_sql_ui():
    st.subheader("Source : SQL (MySQL / PostgreSQL)")
    db_kind = st.radio("Base", ["MySQL", "PostgreSQL"], horizontal=True)
    host_in = st.text_input("Hôte", value="localhost", help="Ex: localhost (ne mettez pas user@host ici)")
    port = st.number_input("Port", value=3306 if db_kind=="MySQL" else 5432, step=1)
    database = st.text_input("Base de données")
    user_in = st.text_input("Utilisateur")
    password = st.text_input("Mot de passe", type="password")
    query = st.text_area("Requête SQL", value="SELECT 1")

    # Options pratiques
    colA, colB = st.columns(2)
    with colA:
        force_ipv4 = st.checkbox("Forcer 127.0.0.1 si 'localhost'", value=False)
    with colB:
        show_debug = st.checkbox("Afficher debug de connexion (sans mot de passe)", value=False)

    # Normalisation user@host
    host, user = normalize_sql_host_user(host_in, user_in)
    if force_ipv4 and host.lower() == "localhost":
        host = "127.0.0.1"

    if show_debug:
        st.caption(f"🔧 Hôte nettoyé = `{host}`, Utilisateur = `{user or '(vide)'}`")

    if st.button("Exécuter la requête"):
        try:
            from sqlalchemy import create_engine, text
            try:
                # URL.create = encodage sûr des identifiants & host
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
                # Fallback si vieille version de SQLAlchemy
                from urllib.parse import quote_plus
                def make_url(drivername, username, password, host, port, database):
                    u = quote_plus(username or "")
                    p = quote_plus(password or "")
                    h = (host or "")
                    d = (database or "")
                    prt = f":{int(port)}" if port else ""
                    cred = f"{u}:{p}@" if username or password else ""
                    return f"{drivername}://{cred}{h}{prt}/{d}"
        except Exception:
            st.error("Le paquet `sqlalchemy` est requis. Installez-le (ex: `pip install sqlalchemy`).")
            return (None, None)

        # Driver
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

        # Construit l’URL **sécurisée**
        conn_url = make_url(driver, user, password, host, port, database)

        # Connexion & lecture
        try:
            engine = create_engine(conn_url, pool_pre_ping=True)
            with engine.begin() as conn:
                # ping rapide
                conn.execute(text("SELECT 1"))
                df = pd.read_sql(query, conn)
            return (df, {"kind":"sql","db":db_kind,"host":host,"port":int(port),"database":database})
        except Exception as e:
            st.error(f"Erreur SQL : {e}")
            st.info("Vérifiez : service démarré, host/port corrects, firewall, droits utilisateur, et que "
                    "vous n’avez pas mis `user@host` dans le champ Hôte. Essayez aussi Hôte=127.0.0.1.")
            return (None, None)

    return (None, None)

def safe_read_sql(engine, query: str):
    """
    Lecture SQL robuste pour SQLAlchemy 1.4/2.0 + Pandas < 2.0 :
    1) Essaye pd.read_sql_query(text(query), con=engine)
    2) Sinon pd.read_sql_query(query, con=engine)
    3) Sinon fallback DB-API: engine.raw_connection()
    """
    from sqlalchemy import text
    # a) SQLAlchemy-friendly
    try:
        return pd.read_sql_query(text(query), con=engine)
    except Exception as e1:
        # b) Chemin classique (certaines versions Pandas préfèrent une str)
        try:
            return pd.read_sql_query(query, con=engine)
        except Exception as e2:
            # c) Fallback DB-API (a .cursor()) pour Pandas très ancien
            try:
                raw = engine.raw_connection()
                try:
                    # ping léger pour lever tôt les erreurs d’auth/réseau
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

def normalize_sql_host_user(host_input: str, user_input: str):
    host = (host_input or "").strip()
    user = (user_input or "").strip()
    if "@" in host:
        maybe_user, maybe_host = host.split("@", 1)
        if not user:
            user = maybe_user.strip()
        host = maybe_host.strip()
    return host, user

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
                # Fallback encodé si SQLAlchemy trop ancien
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
            st.error("Le paquet `sqlalchemy` est requis. Installez-le (`pip install sqlalchemy`).")
            return (None, None)

        # Driver
        if db_kind == "MySQL":
            try:
                import pymysql  # noqa
                driver = "mysql+pymysql"
            except Exception:
                driver = "mysql"
        else:
            try:
                import psycopg2  # noqa
                driver = "postgresql+psycopg2"
            except Exception:
                driver = "postgresql"

        conn_url = make_url(driver, user, password, host, port, database)

        try:
            engine = create_engine(conn_url, pool_pre_ping=True, future=True)

            # Ping rapide
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            # ★ LECTURE ROBUSTE (pas de Connection → .cursor): on passe l’ENGINE
            df = safe_read_sql(engine, query)

            return (df, {"kind":"sql","db":db_kind,"host":host,"port":int(port),"database":database})

        except Exception as e:
            st.error(f"Erreur SQL : {e}")
            st.info("Vérifiez service/host/port, firewall, droits, identifiants. "
                    "Évitez `user@host` dans Hôte ; essayez 127.0.0.1 si besoin.")
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
                    st.success("Conversion effectuée.")

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
                 show_prev=True, show_next=True)

# ------------------ Étape 2 : Analyse ------------------
if step == "Analyse":
    st.header("2️⃣ Analyse exploratoire & descriptive")
    if ss.get('df') is None:
        st.info("Veuillez importer des données.")
    else:
        df = ss['df']
        st.subheader("Aperçu")
        st.dataframe(df.head(10))

        st.subheader("Statistiques descriptives")
        st.write(df.describe(include='all').transpose())

        st.subheader("Types de colonnes")
        st.write(df.dtypes.value_counts())

        st.subheader("Qualité des données")
        st.write(f"🔁 Lignes dupliquées : **{df.duplicated().sum()}**")
        st.write(f"❓ Valeurs manquantes (total) : **{int(df.isna().sum().sum())}**")

        def plot_missingness(df_):
            miss = df_.isna().mean().sort_values(ascending=False)
            if (miss > 0).any():
                fig, ax = plt.subplots(figsize=(8, max(3, 0.35*len(miss))))
                sns.barplot(x=miss.values, y=miss.index, ax=ax, color=PRIMARY)
                ax.set_xlabel("Proportion manquante"); ax.set_ylabel("Colonnes")
                ax.legend(["Taux manquant"])
                st.pyplot(fig); plt.close(fig)
        plot_missingness(df)

        all_cols = df.columns.tolist()
        target_idx = 0 if all_cols else 0
        ss['target_column'] = st.selectbox("Sélectionner la colonne cible (target)", all_cols, index=target_idx)

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

        # Graphes
        if ss['show_graphs']:
            if ss['num_cols']:
                st.subheader("Distributions (quantitatives)")
                for c in ss['num_cols'][:6]:
                    fig, ax = plt.subplots(figsize=(6,3))
                    ax.hist(df[c].dropna(), bins=30, edgecolor='black', color=PRIMARY, label=c)
                    ax.set_title(c)
                    ax.legend()
                    st.pyplot(fig); plt.close(fig)

            if ss['cat_cols']:
                st.subheader("Fréquences (qualitatives)")
                for c in ss['cat_cols'][:6]:
                    vc = df[c].astype(str).value_counts().head(12)
                    fig, ax = plt.subplots(figsize=(7,3))
                    ax.bar(vc.index, vc.values, color=PRIMARY, label=c)
                    ax.set_title(c); ax.set_xticklabels(vc.index, rotation=45, ha='right')
                    ax.legend()
                    st.pyplot(fig); plt.close(fig)

            if len(ss['num_cols']) >= 2:
                st.subheader("Matrice de corrélation")
                corr = df[ss['num_cols']].corr()
                fig, ax = plt.subplots(figsize=(min(12, 1+0.8*len(ss['num_cols'])), min(10, 1+0.8*len(ss['num_cols']))))
                cmap = sns.light_palette(PRIMARY, as_cmap=True)
                sns.heatmap(corr, annot=True, cmap=cmap, fmt=".2f", ax=ax, cbar_kws={'label':'corr'})
                ax.set_title("Corrélations (quantitatives)")
                st.pyplot(fig); plt.close(fig)

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
                 show_prev=True, show_next=True)

# ------------------ Étape 3 : AutoML ------------------
if step == "AutoML":
    st.header("3️⃣ AutoML")
    if ss.get('df') is None or ss.get('target_column') is None:
        st.info("Veuillez compléter l'analyse et choisir la cible.")
    else:
        df = ss['df']
        target = ss['target_column']

        # Sélection modèles
        if ss['data_type'] == "time_series" or ss['task_type'] == "regression":
            models = build_reg_models(ss['models_selected'])
        else:
            models = build_clf_models(ss['models_selected'])
        if not models:
            st.warning("Veuillez sélectionner au moins un modèle dans Paramètres > Modèles AutoML.")
            st.stop()

        # Construction X, y
        if ss['data_type'] == "tabular":
            X = df.drop(columns=[target]).copy()
            y = df[target].copy()

            if ss['use_date_features'] and ss['cat_cols']:
                date_like = []
                for c in list(ss['cat_cols']):
                    parsed = pd.to_datetime(X[c], errors='coerce', utc=False)
                    if parsed.notna().mean() >= 0.8:
                        date_like.append(c)
                if date_like:
                    added = expand_date_features(X, date_like)
                    ss['cat_cols'] = [c for c in ss['cat_cols'] if c not in date_like]
                    ss['num_cols'] = list(dict.fromkeys(ss['num_cols'] + added))

            num_cols = [c for c in ss['num_cols'] if c in X.columns]
            cat_cols = [c for c in ss['cat_cols'] if c in X.columns]

            if not num_cols and not cat_cols:
                X['index'] = np.arange(len(X))
                num_cols = ['index']
                st.warning("Aucune feature sélectionnée — fallback sur une feature 'index'.")

            coerce_numeric_inplace(X, num_cols)

            split_res = safe_split(
                X, y, test_size=ss['test_size'],
                stratify=(y if ss['task_type']=="classification" and y.nunique() < 40 else None)
            )
            if split_res is None:
                st.error("Jeu de données trop petit pour un split train/test.")
                st.stop()
            X_train, X_test, y_train, y_test = split_res

            transformers = []
            if num_cols:
                num_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy=ss['impute_strategy_num'])),
                    ("scaler", StandardScaler() if ss['scale_numeric'] else "passthrough")
                ])
                transformers.append(("num", num_pipe, num_cols))
            if cat_cols and ss['encode_categorical']:
                cat_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy=ss['impute_strategy_cat'], fill_value="NA")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ])
                transformers.append(("cat", cat_pipe, cat_cols))
            preprocessor = ColumnTransformer(transformers, remainder="drop") if transformers else "passthrough"
            ss['X_columns'] = num_cols + cat_cols

        else:  # time_series
            if not ss.get('date_col'):
                st.error("Mode série temporelle : la colonne de date est obligatoire.")
                st.stop()
            df_sorted = df.sort_values(by=ss['date_col']).reset_index(drop=True)
            y_full = pd.to_numeric(df_sorted[target], errors='coerce')
            y_full = pd.Series(SimpleImputer(strategy="mean").fit_transform(y_full.values.reshape(-1,1)).ravel())

            X_lag, y_lag = make_ts_lags_safe(y_full, ss['n_lags'])
            if X_lag is None or y_lag is None:
                st.error("Données insuffisantes pour construire des lags (≥3 points requis).")
                st.stop()

            split_res = train_test_split_ts_safe(X_lag, y_lag, test_size=ss['test_size'])
            if split_res is None:
                st.error("Données insuffisantes pour un split temporel.")
                st.stop()
            X_train, X_test, y_train, y_test = split_res
            preprocessor = "passthrough"
            ss['X_columns'] = X_lag.columns.tolist()

        # Entraînement & métriques
        predictions, metrics, fitted = {}, {}, {}
        for name, model in models.items():
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            fitted[name] = pipe
            y_pred = pipe.predict(X_test)
            predictions[name] = y_pred

            if ss['task_type'] == "regression":
                metrics[name] = {
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "MAE": float(mean_absolute_error(y_test, y_pred)),
                    "R2": float(r2_score(y_test, y_pred))
                }
            else:
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                row = {"Accuracy": float(acc), "F1_weighted": float(f1)}
                try:
                    if hasattr(pipe, "predict_proba"):
                        proba = pipe.predict_proba(X_test)
                        if proba.shape[1] == 2:
                            auc = roc_auc_score(y_test, proba[:,1])
                            row["ROC_AUC"] = float(auc)
                except Exception:
                    pass
                metrics[name] = row

        ss['models'] = fitted
        ss['metrics'] = metrics
        ss['predictions'] = predictions
        ss['y_test'] = y_test
        ss['X_test'] = X_test

        st.subheader("Métriques des modèles")
        st.dataframe(pd.DataFrame(metrics).T)

        # Graphes d'évaluation / prévision
        if ss['show_graphs'] and len(metrics):
            best_name = max(metrics, key=lambda k: (metrics[k].get("R2", -1e9) if ss['task_type']=="regression"
                                                    else metrics[k].get("Accuracy", -1e9)))
            best_model = fitted[best_name]
            y_pred_best = predictions[best_name]
            st.markdown(f"**Meilleur modèle :** `{best_name}`")

            if ss['task_type'] == "regression":
                if ss['data_type'] == "time_series":
                    st.subheader("Prévision (réel vs prédit) - Test")
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(range(len(y_test)), y_test.values, label="Réel", linewidth=2)
                    ax.plot(range(len(y_pred_best)), y_pred_best, label="Prédit", linewidth=2, color=ss['pred_color'])
                    ax.legend()
                    st.pyplot(fig); plt.close(fig)

                    st.subheader("Prévision future")
                    last_hist = pd.to_numeric(df.sort_values(by=ss['date_col'])[ss['target_column']], errors='coerce')
                    last_hist = pd.Series(SimpleImputer(strategy="mean").fit_transform(np.array(last_hist).reshape(-1,1)).ravel())
                    future = forecast_with_lags(best_model, last_hist, steps=ss['horizon'], lags=ss['n_lags'])
                    if future:
                        fig, ax = plt.subplots(figsize=(10,4))
                        hist_tail = last_hist.iloc[-max(30, ss['n_lags']):].tolist()
                        ax.plot(range(len(hist_tail)), hist_tail, label="Historique (fin)", linewidth=2)
                        ax.plot(range(len(hist_tail), len(hist_tail)+len(future)), future,
                                label=f"Prévision (+{ss['horizon']})", linewidth=2, color=ss['pred_color'])
                        ax.legend()
                        st.pyplot(fig); plt.close(fig)
                else:
                    st.subheader("Réel vs Prédit (test)")
                    fig, ax = plt.subplots(figsize=(5,5))
                    ax.scatter(y_test, y_pred_best, alpha=0.6, color=ss['pred_color'], label="Points")
                    minv = float(min(np.min(y_test), np.min(y_pred_best)))
                    maxv = float(max(np.max(y_test), np.max(y_pred_best)))
                    ax.plot([minv,maxv],[minv,maxv],'r--', label="y=x")
                    ax.set_xlabel("Réel"); ax.set_ylabel("Prédit")
                    ax.legend()
                    st.pyplot(fig); plt.close(fig)

                    st.subheader("Résidus (test)")
                    resid = y_test.values - y_pred_best
                    fig, ax = plt.subplots(figsize=(8,3))
                    ax.plot(resid, marker='o', linestyle='-', alpha=0.8, color=PRIMARY, label="Résidu")
                    ax.axhline(0, color='gray', linestyle='--', label="0")
                    ax.legend()
                    st.pyplot(fig); plt.close(fig)

            else:
                st.subheader("Matrice de confusion (meilleur modèle)")
                cm = confusion_matrix(y_test, y_pred_best)
                fig, ax = plt.subplots(figsize=(5,4))
                cmap = sns.light_palette(PRIMARY, as_cmap=True)
                sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, cbar_kws={'label':'comptes'})
                ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
                st.pyplot(fig); plt.close(fig)

                try:
                    if hasattr(best_model, "predict_proba"):
                        proba = best_model.predict_proba(ss['X_test'])
                        if proba.shape[1] == 2:
                            st.subheader("Courbe ROC (binaire)")
                            fig, ax = plt.subplots(figsize=(6,4))
                            RocCurveDisplay.from_predictions(y_test, proba[:,1], ax=ax, name="ROC")
                            for line in ax.get_lines():
                                line.set_color(PRIMARY)
                            st.pyplot(fig); plt.close(fig)
                except Exception:
                    pass

        st.success("AutoML terminé !")
        ss['step_done']['AutoML'] = True
        save_project_config()

    nav_controls(ss['step_done']['AutoML'], button_titles[step], save_before_next=False,
                 show_prev=True, show_next=True)

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
            last_hist = pd.to_numeric(df.sort_values(by=ss['date_col'])[target], errors='coerce')
            last_hist = pd.Series(SimpleImputer(strategy="mean").fit_transform(np.array(last_hist).reshape(-1,1)).ravel())
            future = forecast_with_lags(model, last_hist, steps=h_pred, lags=ss['n_lags'])
            if future:
                st.success(f"Première prévision future: {future[0]:.4f}")
                fig, ax = plt.subplots(figsize=(10,4))
                hist_tail = last_hist.iloc[-max(30, ss['n_lags']):].tolist()
                ax.plot(range(len(hist_tail)), hist_tail, label="Historique (fin)", linewidth=2)
                ax.plot(range(len(hist_tail), len(hist_tail)+len(future)), future,
                        label=f"Prévision (+{h_pred})", linewidth=2, color=ss['pred_color'])
                ax.legend()
                st.pyplot(fig); plt.close(fig)
            else:
                st.warning("Pas assez d'historique pour générer des prévisions avec les lags actuels.")

            proj_hist = Path(ss['project_dir']) / "history.csv"
            row = pd.DataFrame([{
                "timestamp": datetime.now().isoformat(),
                "action": "forecast",
                "model": model_name,
                "horizon": h_pred,
                "prediction_first": future[0] if future else None
            }])
            if proj_hist.exists():
                pd.concat([pd.read_csv(proj_hist), row], ignore_index=True).to_csv(proj_hist, index=False)
            else:
                row.to_csv(proj_hist, index=False)

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
                # Nom de la colonne cible
                target_name = ss.get('target_column', "Résultat")

                if ss['task_type'] == "regression":
                    # st.success(f"✅ Prédiction : {float(y_pred_input):.4f}")
                    st.success(f"✅ Prédiction de **{target_name}** : {float(y_pred_input):.4f}")
                else:
                    # st.success(f"✅ Prédiction : {y_pred_input}")
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
                 show_prev=True, show_next=True)

# ------------------ Étape 5 : Historique ------------------
if step == "Historique":
    st.header("5️⃣ Historique & Restauration de projets")
    pdv_files = list_projects()
    if not pdv_files:
        st.info("Aucun projet sauvegardé.")
    else:
        st.write("Projets disponibles :")
        for pdv in pdv_files:
            cols = st.columns([0.5, 1, 1, 0.5])
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
                go_to_step("Analyse")

        if ss.get('project_dir'):
            proj_hist = Path(ss['project_dir']) / "history.csv"
            if proj_hist.exists():
                st.subheader(f"Historique du projet : {ss['project_name']}")
                st.dataframe(pd.read_csv(proj_hist))
            else:
                st.info("Aucune entrée d'historique pour le projet courant.")

    nav_controls(True, button_titles[step], save_before_next=False,
                 show_prev=True, show_next=False)
