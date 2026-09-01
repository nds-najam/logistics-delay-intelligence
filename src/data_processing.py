"""
Data loading, quality assessment, and cleaning for the DataQ logistics dataset.

Two responsibilities live here, deliberately kept separate:

1. assess_data_quality(df) -- produces a DQ report on the RAW data (used by the
   "Data Quality" section of the app so the client can see what was wrong).
2. clean_pipeline(df) -- returns a cleaned copy of the data, plus a report of
   what was changed, ready for EDA / modeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RAW_DATA_PATH = "data/synthetic_logistics_data.csv"

DATE_COLUMNS = ["Order_Date", "Pickup_Date", "Expected_Delivery_Date", "Actual_Delivery_Date"]

# Columns where injected outliers live (see data_generator._inject_data_quality_issues)
OUTLIER_CHECK_COLUMNS = ["Package_Weight", "Delay_Hours", "Distance_KM"]

# Canonical spelling for categorical columns with injected inconsistencies.
# Keyed by column -> {stripped.lower() variant: canonical value}
CATEGORICAL_CANONICAL = {
    "Shipping_Mode": {
        "road": "Road",
        "air": "Air",
        "rail": "Rail",
        "sea": "Sea",
    },
    "Carrier": {
        "carrier a": "Carrier A",
        "carrier-a": "Carrier A",
        "carrier b": "Carrier B",
        "carrier c": "Carrier C",
        "carrier d": "Carrier D",
        "carrier e": "Carrier E",
    },
}


@dataclass
class DataQualityReport:
    n_records: int
    n_columns: int
    missing_by_column: dict
    missing_total: int
    duplicate_records: int
    outliers_by_column: dict
    completeness_pct: float
    inconsistent_categorical_values: dict = field(default_factory=dict)


@dataclass
class CleaningReport:
    rows_before: int
    rows_after: int
    duplicates_removed: int
    categorical_values_standardized: int
    missing_values_imputed: dict
    outliers_capped: dict


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=DATE_COLUMNS)
    return df


def _iqr_outlier_count(series: pd.Series, k: float = 3.0) -> int:
    s = series.dropna()
    if s.empty:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return int(((s < lower) | (s > upper)).sum())


def assess_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """Data-quality report computed on the raw (uncleaned) dataframe."""
    n_records, n_columns = df.shape

    missing_by_column = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    missing_total = int(sum(missing_by_column.values()))

    duplicate_records = int(df.duplicated(keep=False).sum())

    outliers_by_column = {
        col: _iqr_outlier_count(df[col]) for col in OUTLIER_CHECK_COLUMNS if col in df.columns
    }

    inconsistent = {}
    for col, canonical_map in CATEGORICAL_CANONICAL.items():
        if col not in df.columns:
            continue
        non_canonical_values = sorted(
            v for v in df[col].dropna().unique() if v not in canonical_map.values()
        )
        if non_canonical_values:
            inconsistent[col] = non_canonical_values

    total_cells = n_records * n_columns
    completeness_pct = 100.0 * (1 - missing_total / total_cells) if total_cells else 100.0

    return DataQualityReport(
        n_records=n_records,
        n_columns=n_columns,
        missing_by_column=missing_by_column,
        missing_total=missing_total,
        duplicate_records=duplicate_records,
        outliers_by_column=outliers_by_column,
        completeness_pct=round(completeness_pct, 2),
        inconsistent_categorical_values=inconsistent,
    )


def _standardize_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    changed = 0
    for col, canonical_map in CATEGORICAL_CANONICAL.items():
        if col not in df.columns:
            continue
        normalized = df[col].astype(str).str.strip().str.lower().map(canonical_map)
        mask = normalized.notna() & (df[col] != normalized)
        changed += int(mask.sum())
        df.loc[normalized.notna(), col] = normalized[normalized.notna()]
    return df, changed


def _remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    return df, before - len(df)


def _impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    imputed = {}

    numeric_cols = ["Package_Weight", "Driver_Availability", "Customs_Clearance_Time_Hours"]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if col == "Customs_Clearance_Time_Hours":
            # Missing customs time most plausibly means "not applicable" -> 0,
            # not a population median (which would be dominated by non-customs 0s anyway).
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = df[col].fillna(df[col].median())
        imputed[col] = n_missing

    categorical_cols = ["Address_Quality", "Weather_Condition"]
    for col in categorical_cols:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        mode_val = df[col].mode(dropna=True)
        df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")
        imputed[col] = n_missing

    return df, imputed


def _cap_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Winsorize extreme outliers (99.5th percentile cap) rather than dropping
    rows, so shipment volume is preserved for downstream aggregation."""
    df = df.copy()
    capped = {}
    for col in OUTLIER_CHECK_COLUMNS:
        if col not in df.columns:
            continue
        upper = df[col].quantile(0.995)
        mask = df[col] > upper
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = upper
            capped[col] = n
    return df, capped


def clean_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline and return (clean_df, report)."""
    rows_before = len(df)

    df, n_standardized = _standardize_categoricals(df)
    df, n_duplicates = _remove_duplicates(df)
    df, imputed = _impute_missing(df)
    df, capped = _cap_outliers(df)

    report = CleaningReport(
        rows_before=rows_before,
        rows_after=len(df),
        duplicates_removed=n_duplicates,
        categorical_values_standardized=n_standardized,
        missing_values_imputed=imputed,
        outliers_capped=capped,
    )
    return df, report


def load_and_clean(path: str = RAW_DATA_PATH) -> tuple[pd.DataFrame, DataQualityReport, CleaningReport]:
    raw = load_raw_data(path)
    dq_report = assess_data_quality(raw)
    clean_df, cleaning_report = clean_pipeline(raw)
    return clean_df, dq_report, cleaning_report
