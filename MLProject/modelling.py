"""
modelling.py
============
Melatih model machine learning untuk klasifikasi gempa bumi Indonesia.
Menggunakan MLflow autolog untuk mencatat semua metrik dan artefak secara otomatis.
 
Dataset  : earthquake_indonesia_preprocessing.csv (output dari automate_*.py)
Task     : Binary Classification (significant: 0 / 1)
Model    : RandomForestClassifier (Scikit-Learn)
Logging  : MLflow autolog (Basic)
 
Usage:
    python modelling.py
    python modelling.py --data path/to/earthquake_indonesia_preprocessing.csv
"""

import os
import argparse
import logging
 
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'Latitude', 'Longitude', 'Depth', 'Magnitude',
    'depth_category', 'year', 'month', 'day_of_week', 'decade'
]
TARGET_COL   = 'significant'
TEST_SIZE    = 0.2
RANDOM_STATE = 42
 
# MLflow
EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "Earthquake-Indonesia-Classification"
)
RUN_NAME        = "RandomForest-Autolog"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data(filepath: str):
    """
    Memuat dataset preprocessing yang sudah siap dilatih.
 
    Args:
        filepath: Path ke file CSV hasil preprocessing
 
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {filepath}")
 
    df = pd.read_csv(filepath)
    log.info("Dataset loaded   : %s baris × %s kolom", f"{df.shape[0]:,}", df.shape[1])
 
    # Validasi kolom
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")
 
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
 
    log.info("Distribusi target: %s", y.value_counts().sort_index().to_dict())
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
 
    log.info(
        "Train-test split : train=%s | test=%s",
        f"{len(X_train):,}", f"{len(X_test):,}"
    )
 
    return X_train, X_test, y_train, y_test

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X_train, X_test, y_train, y_test):
    """
    Melatih RandomForestClassifier dengan MLflow autolog.
 
    Tracking URI dan Run ID dikelola sepenuhnya oleh `mlflow run` (MLflow Project).
    Script ini TIDAK memanggil set_tracking_uri / set_experiment / start_run
    agar tidak konflik dengan run yang sudah di-inject oleh MLflow Project.
    """
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    # Aktifkan autolog sebelum fit() — akan log ke run aktif yang di-inject mlflow run
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
        log_datasets=True,
    )
 
    log.info("Memulai training dengan MLflow autolog...")
    log.info("Tracking URI : %s", mlflow.get_tracking_uri())
    log.info("Active Run ID: %s", os.environ.get("MLFLOW_RUN_ID", "tidak ada (lokal)"))
 
    # Definisi model
    with mlflow.start_run(run_name=RUN_NAME) as run:
        log.info("MLflow Run ID: %s", run.info.run_id)
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    
        # Training — autolog otomatis log ke run aktif
        model.fit(X_train, y_train)
    
        # Evaluasi manual (untuk console log)
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    
        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_pred_prob)
        cm        = confusion_matrix(y_test, y_pred)
    
        log.info("=" * 50)
        log.info("HASIL EVALUASI MODEL")
        log.info("=" * 50)
        log.info("  Accuracy  : %.4f", acc)
        log.info("  Precision : %.4f", precision)
        log.info("  Recall    : %.4f", recall)
        log.info("  F1-Score  : %.4f", f1)
        log.info("  ROC-AUC   : %.4f", roc_auc)
        log.info("  Confusion Matrix:\n%s", cm)
        log.info("=" * 50)
    
        # Tag tambahan ke run aktif
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("dataset", "earthquake_indonesia")
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("triggered_by", os.environ.get("GITHUB_EVENT_NAME", "manual"))
 
    log.info("Training selesai! Artefak tersimpan di MLflow.")
 
    return model

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Modelling — Gempa Bumi Indonesia (MLflow Autolog)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "earthquake_indonesia_preprocessing",
            "earthquake_indonesia_preprocessing.csv"
        ),
        help="Path ke file CSV hasil preprocessing"
    )
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
 
    log.info("=" * 60)
    log.info("  MODELLING — GEMPA BUMI INDONESIA")
    log.info("  Mode: MLflow Autolog (Basic)")
    log.info("=" * 60)
    log.info("  Dataset: %s", args.data)
 
    X_train, X_test, y_train, y_test = load_data(args.data)
    model = train_model(X_train, X_test, y_train, y_test)
 
    log.info("Selesai. Buka MLflow UI: mlflow ui --port 5000")