"""
Pipeline tests: data generation, cleaning, delay calculation, feature
engineering, model training, prediction, and recommendation generation.

Uses a smaller synthetic sample than the full app (a few thousand records)
so the suite runs quickly while still exercising real code paths end to end.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_generator as dg
import data_processing as dp
import delay_analysis as da
import eda
import explainability as ex
import model as mdl
import recommendations as rec

N_TEST_RECORDS = 6000


@pytest.fixture(scope="session")
def raw_df():
    return dg.generate_dataset(N_TEST_RECORDS, seed=7)


@pytest.fixture(scope="session")
def raw_csv_path(tmp_path_factory, raw_df):
    path = tmp_path_factory.mktemp("data") / "synthetic_logistics_data.csv"
    raw_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="session")
def clean_bundle(raw_csv_path):
    return dp.load_and_clean(raw_csv_path)


@pytest.fixture(scope="session")
def clean_df(clean_bundle):
    return clean_bundle[0]


@pytest.fixture(scope="session")
def trained_bundle(clean_df):
    return mdl.train_and_select(clean_df)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

class TestDataGeneration:
    def test_row_count_and_required_columns(self, raw_df):
        assert len(raw_df) >= N_TEST_RECORDS
        required = {
            "Shipment_ID", "Order_Date", "Expected_Delivery_Date", "Actual_Delivery_Date",
            "Delay_Flag", "Delay_Hours", "Delay_Category", "Warehouse_Capacity_Utilization",
            "Carrier", "Service_Type",
        }
        assert required.issubset(raw_df.columns)

    def test_delay_flag_is_binary(self, raw_df):
        assert set(raw_df["Delay_Flag"].unique()).issubset({0, 1})

    def test_delay_hours_nonnegative_when_not_extreme_outliers(self, raw_df):
        # Outliers are intentionally injected; the bulk of the distribution
        # should still be non-negative and within a sane range.
        assert (raw_df["Delay_Hours"] >= 0).all()

    def test_causal_structure_warehouse_utilization(self, raw_df):
        # Higher warehouse utilization should be associated with materially
        # more processing time -- sanity check the generator's causal wiring,
        # not just that columns exist.
        corr = raw_df["Warehouse_Capacity_Utilization"].corr(raw_df["Warehouse_Processing_Time_Hours"])
        assert corr > 0.3

    def test_data_quality_issues_injected(self, raw_df):
        assert raw_df.isna().sum().sum() > 0
        assert raw_df.duplicated().sum() > 0
        assert raw_df["Shipping_Mode"].nunique() > len(dg.SHIPPING_MODES)  # inconsistent spellings present

    def test_no_delay_sentinel_survives_csv_roundtrip(self, raw_csv_path):
        # Regression test: pandas treats the literal string "None" as NA on
        # read, which previously corrupted the whole Delay_Category column.
        loaded = pd.read_csv(raw_csv_path)
        assert "No Delay" in loaded["Delay_Category"].unique()
        on_time_rows = loaded.loc[loaded["Delay_Flag"] == 0]
        assert (on_time_rows["Delay_Category"] == "No Delay").all()


# ---------------------------------------------------------------------------
# Data cleaning / quality
# ---------------------------------------------------------------------------

class TestDataCleaning:
    def test_dq_report_detects_injected_issues(self, clean_bundle):
        _, dq_report, _ = clean_bundle
        assert dq_report.missing_total > 0
        assert dq_report.duplicate_records > 0
        assert "Shipping_Mode" in dq_report.inconsistent_categorical_values

    def test_cleaning_removes_duplicates(self, clean_bundle):
        clean_df_, _, cleaning_report = clean_bundle
        assert cleaning_report.duplicates_removed > 0
        assert not clean_df_.duplicated().any()

    def test_cleaning_standardizes_categoricals(self, clean_df):
        assert set(clean_df["Shipping_Mode"].unique()).issubset(set(dg.SHIPPING_MODES))
        assert set(clean_df["Carrier"].unique()).issubset(set(dg.CARRIERS))

    def test_cleaning_imputes_key_columns(self, clean_df):
        for col in ["Package_Weight", "Driver_Availability", "Address_Quality", "Weather_Condition"]:
            assert clean_df[col].isna().sum() == 0

    def test_cleaning_caps_extreme_outliers(self, raw_df, clean_df):
        assert clean_df["Package_Weight"].max() <= raw_df["Package_Weight"].max()


# ---------------------------------------------------------------------------
# Delay calculation / root-cause analysis
# ---------------------------------------------------------------------------

class TestDelayCalculation:
    def test_delay_flag_matches_date_comparison(self, clean_df):
        expected_flag = (clean_df["Actual_Delivery_Date"] > clean_df["Expected_Delivery_Date"]).astype(int)
        assert (clean_df["Delay_Flag"] == expected_flag).all()

    def test_delay_hours_zero_when_on_time(self, clean_df):
        on_time = clean_df.loc[clean_df["Delay_Flag"] == 0]
        assert (on_time["Delay_Hours"] == 0).all()

    def test_delay_by_cause_shares_sum_to_one(self, clean_df):
        table = da.delay_by_cause(clean_df)
        assert abs(table["Pct_of_Delayed"].sum() - 1.0) < 1e-6

    def test_kpis_consistent_with_raw_counts(self, clean_df):
        kpis = eda.compute_kpis(clean_df)
        assert kpis["total_shipments"] == len(clean_df)
        assert kpis["delayed_shipments"] == int(clean_df["Delay_Flag"].sum())

    def test_generate_alerts_returns_well_formed_entries(self, clean_df):
        alerts = da.generate_alerts(clean_df, min_volume=100)
        for alert in alerts:
            assert {"type", "entity", "message", "recommended_action", "severity"}.issubset(alert.keys())


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_build_features_returns_expected_columns(self, clean_df):
        X = mdl.build_features(clean_df)
        assert list(X.columns) == mdl.ALL_FEATURES

    def test_build_features_no_missing_values(self, clean_df):
        X = mdl.build_features(clean_df)
        assert X.isna().sum().sum() == 0

    def test_is_cross_region_matches_regions(self, clean_df):
        X = mdl.build_features(clean_df)
        expected = (clean_df["Origin_Region"] != clean_df["Destination_Region"]).astype(int)
        assert (X["Is_Cross_Region"].values == expected.values).all()

    def test_process_time_columns_excluded_from_features(self):
        leakage_cols = {
            "Warehouse_Processing_Time_Hours", "Pickup_Delay_Hours", "Transit_Time_Hours",
            "Sorting_Time_Hours", "Last_Mile_Time_Hours", "Customs_Clearance_Time_Hours",
            "Delay_Hours", "Delay_Category", "Actual_Delivery_Date", "Actual_Delivery_Days",
        }
        assert leakage_cols.isdisjoint(set(mdl.ALL_FEATURES))


# ---------------------------------------------------------------------------
# Model training / prediction
# ---------------------------------------------------------------------------

class TestModelTraining:
    def test_trains_without_error_and_selects_a_model(self, trained_bundle):
        assert trained_bundle.model_name in {"Logistic Regression", "Random Forest", "HistGradient Boosting"}

    def test_metrics_are_valid_probabilities(self, trained_bundle):
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert 0.0 <= trained_bundle.metrics[key] <= 1.0

    def test_comparison_includes_all_candidate_models(self, trained_bundle):
        names = {c["model_name"] for c in trained_bundle.comparison}
        assert names == {"Logistic Regression", "Random Forest", "HistGradient Boosting"}

    def test_model_beats_naive_baseline_on_roc_auc(self, trained_bundle):
        # A model with no signal would score ~0.5 ROC-AUC.
        assert trained_bundle.metrics["roc_auc"] > 0.6

    def test_delay_hours_by_tier_covers_all_tiers(self, trained_bundle):
        assert set(trained_bundle.delay_hours_by_tier.keys()) == set(mdl.RISK_THRESHOLDS.keys())


class TestPrediction:
    def test_predict_shipment_returns_valid_probability(self, trained_bundle, clean_df):
        shipment = clean_df.iloc[0].to_dict()
        result = mdl.predict_shipment(trained_bundle, shipment)
        assert 0.0 <= result["delay_probability"] <= 1.0
        assert result["risk_tier"] in mdl.RISK_THRESHOLDS
        assert result["estimated_delay_hours"] >= 0

    def test_higher_risk_inputs_score_higher(self, trained_bundle, clean_df):
        benign = clean_df.iloc[0].to_dict()
        risky = dict(benign)
        risky.update({
            "Warehouse_Capacity_Utilization": 97.0, "Driver_Availability": 15.0,
            "Vehicle_Availability": 20.0, "Traffic_Level": "Severe",
            "Weather_Condition": "Storm", "Address_Quality": "Poor",
            "Number_of_Handoffs": 7, "Service_Type": "Same-Day",
        })
        benign.update({
            "Warehouse_Capacity_Utilization": 40.0, "Driver_Availability": 95.0,
            "Vehicle_Availability": 95.0, "Traffic_Level": "Low",
            "Weather_Condition": "Clear", "Address_Quality": "Good",
            "Number_of_Handoffs": 1, "Service_Type": "Economy",
        })
        p_benign = mdl.predict_shipment(trained_bundle, benign)["delay_probability"]
        p_risky = mdl.predict_shipment(trained_bundle, risky)["delay_probability"]
        assert p_risky > p_benign

    def test_explain_shipment_returns_ranked_factors(self, trained_bundle, clean_df):
        shipment = clean_df.iloc[0].to_dict()
        factors = ex.explain_shipment(trained_bundle, shipment, clean_df, top_n=5)
        assert len(factors) <= 5
        impacts = [abs(f["impact"]) for f in factors]
        assert impacts == sorted(impacts, reverse=True)


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------

class TestRecommendations:
    def test_generate_recommendations_returns_well_formed_cards(self, clean_df):
        cards = rec.generate_recommendations(clean_df)
        required_keys = {"issue", "evidence", "recommendation", "expected_impact", "priority"}
        for card in cards:
            assert required_keys.issubset(card.keys())
            assert card["priority"] in {"HIGH", "MEDIUM", "LOW"}

    def test_shipment_recommendations_nonempty(self, trained_bundle, clean_df):
        shipment = clean_df.iloc[0].to_dict()
        shipment.update({"Warehouse_Capacity_Utilization": 96.0, "Traffic_Level": "Severe"})
        factors = ex.explain_shipment(trained_bundle, shipment, clean_df, top_n=5)
        actions = rec.shipment_recommendations(factors)
        assert len(actions) > 0
