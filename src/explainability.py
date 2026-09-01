"""
Explainability: per-shipment "why is this at risk" explanations, and an
optional SHAP-based global summary when the trained model is tree-based.

Primary method (used everywhere in the app) is a simple, always-available,
model-agnostic local explanation: for a given shipment, replace one raw
feature at a time with its "normal" (population median/mode) value and
measure how much the predicted delay probability moves. This works
identically regardless of which model won the comparison, needs no fragile
plumbing through the one-hot encoding, and produces plain-language factors a
non-technical reader can follow -- satisfying the brief's explicit fallback
requirement ("If SHAP creates dependency issues, implement model-agnostic
feature importance or another explainability method").

SHAP (TreeExplainer) is used, where practical, purely as a supplementary
global feature-importance chart on the Delay Prediction page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model import ALL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features

FEATURE_LABELS = {
    "Distance_KM": "Shipment distance",
    "Number_of_Handoffs": "Number of handoffs",
    "Package_Weight": "Package weight",
    "Package_Volume": "Package volume",
    "Warehouse_Capacity_Utilization": "Warehouse capacity utilization",
    "Driver_Availability": "Driver availability",
    "Vehicle_Availability": "Vehicle availability",
    "Origin_Region": "Origin region",
    "Destination_Region": "Destination region",
    "Shipping_Mode": "Shipping mode",
    "Carrier": "Carrier",
    "Service_Type": "Service type",
    "Package_Type": "Package type",
    "Weather_Condition": "Weather condition",
    "Traffic_Level": "Traffic level",
    "Address_Quality": "Address quality",
    "Customer_Priority": "Customer priority",
    "Payment_Status": "Payment status",
    "Holiday_Flag": "Holiday",
    "Weekend_Flag": "Weekend dispatch",
    "Peak_Season_Flag": "Peak season",
    "Is_Cross_Region": "Cross-region shipment",
}


def _reference_values(reference_df: pd.DataFrame) -> dict:
    """'Normal' value for each model feature (including derived ones, e.g.
    Is_Cross_Region), used as the ablation baseline."""
    feature_df = build_features(reference_df)
    ref = {}
    for col in NUMERIC_FEATURES + BINARY_FEATURES:
        ref[col] = float(feature_df[col].median())
    for col in CATEGORICAL_FEATURES:
        ref[col] = feature_df[col].mode(dropna=True).iloc[0]
    return ref


def _values_equal(a, b) -> bool:
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9
    return a == b


def explain_shipment(bundle, shipment: dict, reference_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Local, model-agnostic explanation for one shipment. For each feature,
    substitute its population-normal value and measure the change in
    predicted delay probability. A large positive impact means that
    feature's actual value is pushing this shipment's risk up; negative
    means it's pulling risk down."""
    reference = _reference_values(reference_df)
    base_X = build_features(pd.DataFrame([shipment]))
    # Widen numeric dtypes up front so substituting a reference float value
    # below never hits pandas' "incompatible dtype" assignment warning/error.
    for col in NUMERIC_FEATURES + BINARY_FEATURES:
        base_X[col] = base_X[col].astype("float64")
    base_proba = float(bundle.pipeline.predict_proba(base_X)[0, 1])
    actual_features = base_X.iloc[0].to_dict()

    contributions = []
    for col in ALL_FEATURES:
        actual_value = actual_features[col]
        normal_value = reference[col]
        if _values_equal(actual_value, normal_value):
            continue
        altered_X = base_X.copy()
        altered_X.at[0, col] = normal_value
        alt_proba = float(bundle.pipeline.predict_proba(altered_X)[0, 1])
        impact = base_proba - alt_proba  # positive = this factor raises risk
        contributions.append({
            "feature": col,
            "label": FEATURE_LABELS.get(col, col),
            "value": actual_value,
            "normal_value": normal_value,
            "impact": impact,
        })

    contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
    return contributions[:top_n]


def factor_sentence(factor: dict) -> str:
    label, value = factor["label"], factor["value"]
    direction = "increases" if factor["impact"] > 0 else "decreases"
    value_str = f"{value:,.1f}" if isinstance(value, float) else str(value)
    return f"{label}: {value_str} ({direction} risk by {abs(factor['impact']) * 100:.1f} pts vs. typical)"


def try_shap_global_summary(bundle, top_n: int = 15):
    """Best-effort SHAP TreeExplainer global summary for tree-based winners.
    Returns a DataFrame (Feature, Mean_Abs_SHAP) or None if not applicable or
    unavailable -- callers must handle None gracefully. Permutation
    importance in model.py always covers global importance regardless, so
    this is purely a supplementary chart, never a hard dependency."""
    try:
        import shap
    except ImportError:
        return None

    clf = bundle.pipeline.named_steps["classifier"]
    if clf.__class__.__name__ not in (
        "RandomForestClassifier", "HistGradientBoostingClassifier",
        "GradientBoostingClassifier", "DecisionTreeClassifier",
    ):
        return None

    try:
        preprocessor = bundle.pipeline.named_steps["preprocessor"]
        X_sample = bundle.sample_X.sample(n=min(300, len(bundle.sample_X)), random_state=42)
        X_transformed = preprocessor.transform(X_sample)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        out = pd.DataFrame({"Feature": bundle.feature_names, "Mean_Abs_SHAP": mean_abs})
        return out.sort_values("Mean_Abs_SHAP", ascending=False).head(top_n).reset_index(drop=True)
    except Exception:
        return None
