"""
Model training, hyperparameter tuning, model comparison, and serialization pipeline.
"""

import logging
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.config import (
    DEFAULT_MODEL_PATH,
    ALT_MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
)
from src.data_preprocessing import prepare_training_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_model_search_space() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Define model candidates and hyperparameter grids for tuning.
    """
    search_space = {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10],
                "min_samples_split": [2, 5],
            },
        ),
    }

    if HAS_XGBOOST:
        search_space["XGBoost"] = (
            XGBClassifier(
                eval_metric="logloss",
                random_state=42,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        )
    else:
        search_space["Gradient Boosting"] = (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        )

    return search_space


def train_and_evaluate(
    data_path: str = None,
    save_models: bool = True,
) -> Tuple[Any, pd.DataFrame]:
    """
    Execute full training lifecycle:
    - Preprocess and split dataset
    - Perform GridSearchCV across candidate algorithms
    - Rank models by test accuracy & F1 score
    - Save best model, scaler, and label encoder
    """
    logger.info("Preparing data...")
    X_train, X_test, y_train, y_test, scaler, le = prepare_training_data(data_path)

    search_space = get_model_search_space()
    results = []
    trained_estimators = {}

    for name, (estimator, param_grid) in search_space.items():
        logger.info(f"Tuning {name} with 5-fold cross-validation...")
        grid = GridSearchCV(
            estimator,
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_est = grid.best_estimator_
        y_pred = best_est.predict(X_test)

        if hasattr(best_est, "predict_proba"):
            y_prob = best_est.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": roc_auc,
                "Best Parameters": grid.best_params_,
            }
        )
        trained_estimators[name] = best_est

    comparison_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = trained_estimators[best_model_name]

    logger.info("\n=== MODEL COMPARISON RESULTS ===\n" + comparison_df.to_string(index=False))
    logger.info(f"Top Performer: {best_model_name}")

    if save_models:
        logger.info("Saving best model and preprocessors...")
        joblib.dump(best_model, DEFAULT_MODEL_PATH)
        joblib.dump(best_model, ALT_MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(le, ENCODER_PATH)
        logger.info("Model artifacts saved successfully.")

    return best_model, comparison_df


if __name__ == "__main__":
    train_and_evaluate()
