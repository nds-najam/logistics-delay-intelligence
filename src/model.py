"""
Delay-prediction model: feature engineering, training/comparison of multiple
classifiers, model selection, persistence, and single-shipment prediction.

Feature selection note (important): the component "process time" columns
(Warehouse_Processing_Time_Hours, Pickup_Delay_Hours, Transit_Time_Hours,
Sorting_Time_Hours, Last_Mile_Time_Hours, Customs_Clearance_Time_Hours) are
deliberately EXCLUDED from the predictive feature set. Those are only known
*after* a shipment has largely completed its journey -- including them would
let the model trivially reconstruct Delay_Flag rather than genuinely predict
risk from information available at dispatch time. They are used for root
cause analysis (src/delay_analysis.py), not for prediction.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

MODEL_PATH = "models/delay_risk_model.joblib"

NUMERIC_FEATURES = [
    "Distance_KM",
    "Number_of_Handoffs",
    "Package_Weight",
    "Package_Volume",
    "Warehouse_Capacity_Utilization",
    "Driver_Availability",
    "Vehicle_Availability",
]
CATEGORICAL_FEATURES = [
    "Origin_Region",
    "Destination_Region",
    "Shipping_Mode",
    "Carrier",
    "Service_Type",
    "Package_Type",
    "Weather_Condition",
    "Traffic_Level",
    "Address_Quality",
    "Customer_Priority",
    "Payment_Status",
]
BINARY_FEATURES = ["Holiday_Flag", "Weekend_Flag", "Peak_Season_Flag", "Is_Cross_Region"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

RISK_THRESHOLDS = {"LOW": 0.0, "MEDIUM": 0.25, "HIGH": 0.5, "CRITICAL": 0.75}


def risk_tier(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if prob >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    if prob >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the feature frame used by the model (input can be raw shipment
    rows, e.g. from a single manually entered shipment in the UI)."""
    out = df.copy()
    out["Is_Cross_Region"] = (out["Origin_Region"] != out["Destination_Region"]).astype(int)
    return out[ALL_FEATURES]


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ]
    )


def _candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=42,
        ),
        "HistGradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, max_depth=8, learning_rate=0.08, random_state=42,
        ),
    }


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = list(NUMERIC_FEATURES)
    ohe: OneHotEncoder = preprocessor.named_transformers_["categorical"]
    names += list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    names += list(BINARY_FEATURES)
    return names


@dataclass
class TrainingResult:
    model_name: str
    pipeline: Pipeline
    metrics: dict
    confusion: np.ndarray


@dataclass
class ModelBundle:
    pipeline: Pipeline
    model_name: str
    feature_names: list
    metrics: dict
    comparison: list
    trained_at: str
    n_train: int
    n_test: int
    base_delay_rate: float
    delay_hours_by_tier: dict = field(default_factory=dict)
    sample_X: pd.DataFrame = None
    sample_y: pd.Series = None


def train_and_select(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> ModelBundle:
    X = build_features(df)
    y = df["Delay_Flag"].astype(int)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=test_size, random_state=random_state, stratify=y
    )

    results: list[TrainingResult] = []
    for name, clf in _candidate_models().items():
        pipe = Pipeline(steps=[("preprocessor", _build_preprocessor()), ("classifier", clf)])
        if name == "HistGradient Boosting":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            pipe.fit(X_train, y_train, classifier__sample_weight=sample_weight)
        else:
            pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        cm = confusion_matrix(y_test, preds)
        results.append(TrainingResult(model_name=name, pipeline=pipe, metrics=metrics, confusion=cm))

    base_rate = float(y_train.mean())
    # Selection rule: recall on the delayed class is the priority (a missed
    # at-risk shipment is costlier than a false alarm -- see README/Model
    # Governance), but gate out degenerate high-recall/low-precision models
    # (e.g. "always predict delayed") by requiring precision meaningfully
    # above the base delay rate.
    precision_floor = max(0.35, base_rate * 1.15)
    qualifying = [r for r in results if r.metrics["precision"] >= precision_floor]
    pool = qualifying if qualifying else results
    best = max(pool, key=lambda r: r.metrics["recall"])

    comparison = [
        {"model_name": r.model_name, **r.metrics, "confusion_matrix": r.confusion.tolist()}
        for r in results
    ]

    preprocessor = best.pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)

    # Delay-hours calibration by risk tier, computed on the held-out test set
    # so it reflects genuine out-of-sample behavior, not training-set fit.
    test_proba = best.pipeline.predict_proba(X_test)[:, 1]
    tiers = pd.Series([risk_tier(p) for p in test_proba], index=X_test.index)
    delay_hours = df.loc[idx_test, "Delay_Hours"]
    delay_hours_by_tier = delay_hours.groupby(tiers).mean().to_dict()
    for t in RISK_THRESHOLDS:
        delay_hours_by_tier.setdefault(t, float(delay_hours.mean()) if len(delay_hours) else 0.0)

    # Held-out sample retained for model-agnostic global importance
    # (permutation importance) and for explainability background data.
    sample_size = min(3000, len(X_test))
    sample_idx = X_test.sample(n=sample_size, random_state=random_state).index
    sample_X = X_test.loc[sample_idx].reset_index(drop=True)
    sample_y = y_test.loc[sample_idx].reset_index(drop=True)

    return ModelBundle(
        pipeline=best.pipeline,
        model_name=best.model_name,
        feature_names=feature_names,
        metrics=best.metrics,
        comparison=comparison,
        trained_at=dt.datetime.now().isoformat(timespec="seconds"),
        n_train=len(X_train),
        n_test=len(X_test),
        base_delay_rate=base_rate,
        delay_hours_by_tier={k: float(v) for k, v in delay_hours_by_tier.items()},
        sample_X=sample_X,
        sample_y=sample_y,
    )


def save_model(bundle: ModelBundle, path: str = MODEL_PATH) -> None:
    joblib.dump(bundle, path)


def load_model(path: str = MODEL_PATH) -> ModelBundle:
    return joblib.load(path)


def predict_shipment(bundle: ModelBundle, shipment: dict) -> dict:
    """Predict delay probability / risk tier / estimated delay hours for a
    single shipment described as a dict of raw field values."""
    row = pd.DataFrame([shipment])
    X = build_features(row)
    proba = float(bundle.pipeline.predict_proba(X)[0, 1])
    tier = risk_tier(proba)
    est_hours = bundle.delay_hours_by_tier.get(tier, 0.0)
    return {
        "delay_probability": proba,
        "risk_tier": tier,
        "estimated_delay_hours": est_hours,
    }


def get_global_feature_importance(bundle: ModelBundle, top_n: int = 15) -> pd.DataFrame:
    """Model-agnostic global importance via permutation importance on the raw
    (pre-encoding) feature columns, run through the whole pipeline. Works
    identically regardless of which model won the comparison, and gives one
    human-readable row per real-world feature rather than per one-hot dummy."""
    result = permutation_importance(
        bundle.pipeline, bundle.sample_X, bundle.sample_y,
        n_repeats=5, random_state=42, scoring="roc_auc", n_jobs=1,
    )
    out = pd.DataFrame({
        "Feature": bundle.sample_X.columns,
        "Importance": result.importances_mean,
    })
    out["Importance"] = out["Importance"].clip(lower=0)
    out = out.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)
    return out
