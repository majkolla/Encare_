from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.preprocess import fit_preprocessor, transform_for_model
from src.utils.types import Schema


def make_real_vs_syn_dataset(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: Schema,
) -> tuple[pd.DataFrame, np.ndarray]:
    combined = pd.concat([real_df, syn_df], axis=0, ignore_index=True)
    preprocessor = fit_preprocessor(combined, schema)
    encoded = transform_for_model(combined, preprocessor, model_name="discriminator")
    y = np.concatenate([np.ones(len(real_df)), np.zeros(len(syn_df))]).astype(int)
    return encoded, y


def train_discriminator(X: pd.DataFrame, y: np.ndarray, model: str = "random_forest") -> float:
    auc, _ = train_discriminator_with_details(X, y, model=model)
    return auc


def train_discriminator_with_details(
    X: pd.DataFrame,
    y: np.ndarray,
    model: str = "random_forest",
) -> tuple[float, dict[str, object]]:
    if len(np.unique(y)) < 2:
        return 0.5, {"feature_importances": []}

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    if model != "random_forest":
        raise ValueError(f"Unsupported discriminator model: {model}")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    probabilities = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probabilities))
    feature_importances = []
    for feature_name, importance in zip(X.columns.tolist(), clf.feature_importances_.tolist()):
        feature_importances.append(
            {
                "feature": feature_name,
                "importance": float(importance),
            }
        )
    ranked_importances = sorted(
        feature_importances,
        key=lambda item: item["importance"],
        reverse=True,
    )
    return auc, {"feature_importances": ranked_importances}


def compute_discriminator_auc(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: Schema,
) -> dict[str, float]:
    X, y = make_real_vs_syn_dataset(real_df, syn_df, schema)
    auc = train_discriminator(X, y)
    score = max(0.0, 1.0 - abs(auc - 0.5) / 0.5)
    return {"auc": auc, "score": score}


def compute_discriminator_diagnostics(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: Schema,
    top_n: int = 25,
) -> dict[str, float | list[dict[str, object]]]:
    X, y = make_real_vs_syn_dataset(real_df, syn_df, schema)
    auc, details = train_discriminator_with_details(X, y)
    score = max(0.0, 1.0 - abs(auc - 0.5) / 0.5)
    return {
        "auc": auc,
        "score": score,
        "top_features": details["feature_importances"][:top_n],
    }
