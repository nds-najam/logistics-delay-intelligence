"""
DataQ | Logistics Delay Intelligence
--------------------------------------
Client-facing proof-of-concept: identify why packages are delayed, quantify
the major causes, predict which shipments are at risk, explain why, and
recommend operational actions -- on fully local, synthetic data.

Run with: streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import data_generator as dg
import data_processing as dp
import delay_analysis as da
import eda
import explainability as ex
import model as mdl
import recommendations as rec

# ---------------------------------------------------------------------------
# Page config + brand styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DataQ | Logistics Delay Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0F766E"
PRIMARY_DARK = "#0B4F49"
ACCENT = "#2563EB"
DANGER = "#DC2626"
WARNING = "#D97706"
SUCCESS = "#16A34A"
SLATE = "#334155"
PALETTE = ["#0F766E", "#2563EB", "#7C3AED", "#DB2777", "#D97706", "#65A30D", "#0891B2", "#94A3B8"]
RISK_COLORS = {"LOW": "#16A34A", "MEDIUM": "#D97706", "HIGH": "#EA580C", "CRITICAL": "#DC2626"}

st.markdown(f"""
<style>
    .block-container {{ padding-top: 1.5rem; max-width: 1400px; }}
    .dataq-header {{
        display: flex; align-items: baseline; gap: 0.6rem;
        border-bottom: 3px solid {PRIMARY}; padding-bottom: 0.6rem; margin-bottom: 1.2rem;
    }}
    .dataq-brand {{ font-size: 1.9rem; font-weight: 800; color: {PRIMARY_DARK}; letter-spacing: 0.5px; }}
    .dataq-subtitle {{ font-size: 1.05rem; color: {SLATE}; font-weight: 500; }}
    .dataq-page-title {{ font-size: 1.4rem; font-weight: 700; color: {PRIMARY_DARK}; margin-bottom: 0.2rem; }}
    .dataq-page-desc {{ color: #64748B; margin-bottom: 1.1rem; }}
    .kpi-card {{
        background: white; border: 1px solid #E2E8F0; border-top: 4px solid {PRIMARY};
        border-radius: 8px; padding: 0.9rem 1rem; height: 100%;
    }}
    .kpi-label {{ font-size: 0.78rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.4px; font-weight: 600; }}
    .kpi-value {{ font-size: 1.65rem; font-weight: 800; color: {SLATE}; margin-top: 0.15rem; }}
    .kpi-sub {{ font-size: 0.78rem; color: #94A3B8; margin-top: 0.1rem; }}
    .insight-card {{
        background: #F0FDFA; border-left: 4px solid {PRIMARY}; border-radius: 6px;
        padding: 0.65rem 0.9rem; margin-bottom: 0.5rem; color: {SLATE}; font-size: 0.92rem;
    }}
    .alert-card {{
        border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; border-left: 5px solid;
    }}
    .alert-high {{ background: #FEF2F2; border-color: {DANGER}; }}
    .alert-medium {{ background: #FFFBEB; border-color: {WARNING}; }}
    .rec-card {{
        background: white; border: 1px solid #E2E8F0; border-radius: 8px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;
    }}
    .priority-badge {{
        display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; color: white; letter-spacing: 0.3px;
    }}
    .risk-badge {{
        display: inline-block; padding: 0.3rem 0.9rem; border-radius: 6px;
        font-size: 1.1rem; font-weight: 800; color: white; letter-spacing: 0.5px;
    }}
    .disclaimer {{ font-size: 0.8rem; color: #94A3B8; font-style: italic; margin-top: 1rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data / model loading (cached)
# ---------------------------------------------------------------------------

DATA_PATH = "data/synthetic_logistics_data.csv"
MODEL_PATH = mdl.MODEL_PATH


@st.cache_data(show_spinner="Loading and cleaning shipment data...")
def get_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        raw = dg.generate_dataset(120_000)
        raw.to_csv(DATA_PATH, index=False)
    clean_df, dq_report, cleaning_report = dp.load_and_clean(DATA_PATH)
    return clean_df, dq_report, cleaning_report


@st.cache_resource(show_spinner="Training delay-prediction models (first run only)...")
def get_model_bundle(_df: pd.DataFrame):
    if os.path.exists(MODEL_PATH):
        try:
            return mdl.load_model(MODEL_PATH)
        except Exception:
            pass
    bundle = mdl.train_and_select(_df)
    os.makedirs("models", exist_ok=True)
    mdl.save_model(bundle, MODEL_PATH)
    return bundle


clean_df, dq_report, cleaning_report = get_data()
model_bundle = get_model_bundle(clean_df)


# ---------------------------------------------------------------------------
# Sidebar: brand, navigation, filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f'<div class="dataq-brand">DATAQ</div>'
        f'<div class="dataq-subtitle" style="margin-bottom:0.8rem;">Logistics Delay Intelligence</div>',
        unsafe_allow_html=True,
    )

    PAGES = [
        "Executive Overview",
        "Root Cause Analysis",
        "Delay Prediction",
        "Shipment Risk",
        "Recommendations",
        "What-If Analysis",
        "Route Analytics",
        "Warehouse Analytics",
        "Carrier Analytics",
        "Business Impact",
        "Operational Alerts",
        "Data Quality",
        "Model Governance",
    ]
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")

    st.divider()
    st.markdown("**Filters**")

    min_date = clean_df["Order_Date"].min().date()
    max_date = clean_df["Order_Date"].max().date()
    date_range = st.date_input("Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    def ms(label, col, default_all=True):
        options = sorted(clean_df[col].dropna().unique().tolist())
        return st.multiselect(label, options, default=options if default_all else [])

    f_region = ms("Region (origin)", "Origin_Region")
    f_origin = ms("Origin city", "Origin_City")
    f_destination = ms("Destination city", "Destination_City")
    f_carrier = ms("Carrier", "Carrier")
    f_mode = ms("Shipping mode", "Shipping_Mode")
    f_service = ms("Service type", "Service_Type")
    f_warehouse = ms("Origin warehouse", "Origin_Warehouse")
    f_delay_cat = ms("Delay category", "Delay_Category")
    f_priority = ms("Customer priority", "Customer_Priority")
    f_weather = ms("Weather", "Weather_Condition")

    st.divider()
    st.caption("DataQ Logistics Delay Intelligence -- proof of concept. All data is synthetic.")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        mask &= (df["Order_Date"].dt.date >= start) & (df["Order_Date"].dt.date <= end)
    mask &= df["Origin_Region"].isin(f_region)
    mask &= df["Origin_City"].isin(f_origin)
    mask &= df["Destination_City"].isin(f_destination)
    mask &= df["Carrier"].isin(f_carrier)
    mask &= df["Shipping_Mode"].isin(f_mode)
    mask &= df["Service_Type"].isin(f_service)
    mask &= df["Origin_Warehouse"].isin(f_warehouse)
    mask &= df["Delay_Category"].isin(f_delay_cat)
    mask &= df["Customer_Priority"].isin(f_priority)
    mask &= df["Weather_Condition"].isin(f_weather)
    return df.loc[mask]


filtered_df = apply_filters(clean_df)

st.markdown(
    '<div class="dataq-header"><span class="dataq-brand">DATAQ</span>'
    '<span class="dataq-subtitle">Logistics Delay Intelligence</span></div>',
    unsafe_allow_html=True,
)

if filtered_df.empty:
    st.warning("No shipments match the current filter selection. Adjust filters in the sidebar.")
    st.stop()

st.caption(f"Showing {len(filtered_df):,} of {len(clean_df):,} shipments based on current filters.")


# ---------------------------------------------------------------------------
# Small render helpers
# ---------------------------------------------------------------------------

def page_header(title: str, description: str):
    st.markdown(f'<div class="dataq-page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dataq-page-desc">{description}</div>', unsafe_allow_html=True)


def kpi_card(col, label, value, sub=None):
    # `col` may be a st.columns() entry, a container, or the `st` module
    # itself (for a full-width card) -- all support .markdown() directly.
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{f"<div class=\'kpi-sub\'>{sub}</div>" if sub else ""}</div>',
        unsafe_allow_html=True,
    )


def insight_cards(insights: list[str]):
    for text in insights:
        st.markdown(f'<div class="insight-card">💡 {text}</div>', unsafe_allow_html=True)


def priority_color(p):
    return {"HIGH": DANGER, "MEDIUM": WARNING, "LOW": SUCCESS}.get(p, SLATE)


def recommendation_card(card: dict):
    color = priority_color(card["priority"])
    st.markdown(f"""
    <div class="rec-card">
        <span class="priority-badge" style="background:{color};">{card['priority']} PRIORITY</span>
        <div style="font-weight:700; margin-top:0.5rem; color:{SLATE};">{card['issue']}</div>
        <div style="margin-top:0.35rem;"><b>Evidence:</b> {card['evidence']}</div>
        <div style="margin-top:0.35rem;"><b>Recommendation:</b> {card['recommendation']}</div>
        <div style="margin-top:0.35rem; color:#64748B;"><b>Expected Impact:</b> {card['expected_impact']}</div>
    </div>
    """, unsafe_allow_html=True)


def alert_card(alert: dict):
    css_class = "alert-high" if alert["severity"] == "HIGH" else "alert-medium"
    st.markdown(f"""
    <div class="alert-card {css_class}">
        <b>⚠ {alert['severity']} -- {alert['type']}</b><br/>
        {alert['message']}<br/>
        <span style="color:#64748B;"><b>Recommended action:</b> {alert['recommended_action']}</span>
    </div>
    """, unsafe_allow_html=True)


def risk_badge(tier: str):
    color = RISK_COLORS.get(tier, SLATE)
    st.markdown(f'<span class="risk-badge" style="background:{color};">{tier}</span>', unsafe_allow_html=True)


# ===========================================================================
# PAGE 1 -- EXECUTIVE OVERVIEW
# ===========================================================================

if page == "Executive Overview":
    page_header("Executive Overview", "Network-level shipment volume, delay performance, and dynamically generated key insights.")

    kpis = eda.compute_kpis(filtered_df)
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Total Shipments", f"{kpis['total_shipments']:,}")
    kpi_card(c2, "Delayed Shipments", f"{kpis['delayed_shipments']:,}")
    kpi_card(c3, "Delay Rate", f"{kpis['delay_rate']*100:.1f}%")
    kpi_card(c4, "On-Time Delivery %", f"{kpis['on_time_pct']*100:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    kpi_card(c5, "Avg Delay Hours", f"{kpis['avg_delay_hours']:.1f}h", "among delayed shipments")
    kpi_card(c6, "Median Delay Hours", f"{kpis['median_delay_hours']:.1f}h", "among delayed shipments")
    kpi_card(c7, "Avg Delivery Time", f"{kpis['avg_delivery_days']:.1f} days")
    kpi_card(c8, "Est. Delay Impact", f"${kpis['estimated_delay_cost']:,.0f}", f"assumes ${kpis['cost_per_delay_assumption']:.0f}/delayed shipment (estimate)")

    st.markdown("####")
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Shipment Volume Over Time")
        vol = eda.shipment_volume_over_time(filtered_df, freq="W")
        fig = px.line(vol, x="Order_Date", y="Shipments", color_discrete_sequence=[PRIMARY])
        fig.update_layout(margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.subheader("Delay Rate Over Time")
        rate = eda.delay_rate_over_time(filtered_df, freq="W")
        fig = px.line(rate, x="Order_Date", y="Delay_Rate", color_discrete_sequence=[DANGER])
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("On-Time vs Delayed")
        status = eda.on_time_vs_delayed(filtered_df)
        fig = px.pie(status, names="Status", values="Shipments", hole=0.55,
                     color="Status", color_discrete_map={"On-Time": SUCCESS, "Delayed": DANGER})
        fig.update_layout(margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        st.subheader("Delay Hours Distribution")
        dist = eda.delay_hours_distribution(filtered_df)
        if len(dist):
            fig = px.histogram(dist, nbins=40, color_discrete_sequence=[ACCENT])
            fig.update_layout(margin=dict(t=10, b=10), height=320, showlegend=False, xaxis_title="Delay Hours", yaxis_title="Shipments")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No delayed shipments in the current filter selection.")

    st.subheader("DataQ Key Insights")
    insight_cards(eda.generate_key_insights(filtered_df))


# ===========================================================================
# PAGE 2 -- ROOT CAUSE ANALYSIS
# ===========================================================================

elif page == "Root Cause Analysis":
    page_header("Root Cause Analysis", "Why are packages getting delayed -- across the network, and within any segment you select.")

    st.subheader("Delay by Cause")
    cause_table = da.delay_by_cause(filtered_df)
    if cause_table.empty:
        st.info("No delayed shipments in the current filter selection.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(cause_table, x="Delayed_Shipments", y="Delay_Category", orientation="h",
                         color_discrete_sequence=[PRIMARY], title="Delayed Shipments by Cause")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(cause_table, x="Total_Delay_Hours", y="Delay_Category", orientation="h",
                         color_discrete_sequence=[ACCENT], title="Total Delay Hours by Cause")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top contributors (share of delayed shipments):**")
        top = cause_table.head(5).copy()
        top["Share"] = (top["Pct_of_Delayed"] * 100).round(1).astype(str) + "%"
        st.dataframe(top[["Delay_Category", "Delayed_Shipments", "Share"]], hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Root Cause Decomposition")
    st.caption("Select a segment to see its dominant delay causes vs. the network.")

    seg_col1, seg_col2 = st.columns(2)
    seg_dimension = seg_col1.selectbox(
        "Segment by", ["Carrier", "Origin_Region", "Origin_Warehouse", "Shipping_Mode", "Service_Type"]
    )
    seg_options = sorted(filtered_df[seg_dimension].dropna().unique().tolist())
    seg_value = seg_col2.selectbox("Value", seg_options) if seg_options else None

    if seg_value is not None:
        decomposition = da.segment_decomposition(filtered_df, seg_dimension, seg_value)
        if decomposition.get("segment_size", 0) > 0:
            for sentence in decomposition["narrative"]:
                st.markdown(f'<div class="insight-card">📍 {sentence}</div>', unsafe_allow_html=True)
            if not decomposition["cause_table"].empty:
                fig = px.bar(decomposition["cause_table"].head(6), x="Pct_of_Delay_Hours", y="Delay_Category",
                             orientation="h", color_discrete_sequence=[PRIMARY])
                fig.update_xaxes(tickformat=".0%")
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=320, margin=dict(t=10),
                                   title=f"{seg_value}: Share of Delay Hours by Cause")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough shipments in this segment under the current filters.")

    st.divider()
    st.subheader("Driver Analysis")
    st.caption("Operational factors associated with delay. Wording reflects association, not proven causation.")

    driver_corr = da.driver_correlations(filtered_df)
    if not driver_corr.empty:
        fig = px.bar(driver_corr, x="Correlation", y="Driver", orientation="h", color="Correlation",
                     color_continuous_scale=["#2563EB", "#E2E8F0", "#DC2626"], range_color=[-0.3, 0.3])
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=340, margin=dict(t=10),
                           title="Correlation with Delay_Flag")
        st.plotly_chart(fig, use_container_width=True)

    driver_col = st.selectbox(
        "Explore a driver", ["Warehouse_Capacity_Utilization", "Number_of_Handoffs", "Distance_KM",
                              "Driver_Availability", "Vehicle_Availability", "Pickup_Delay_Hours"],
        format_func=lambda c: da.DRIVER_LABELS.get(c, c),
    )
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(filtered_df, x="Delay_Flag", y=driver_col, color="Delay_Flag",
                     color_discrete_map={0: SUCCESS, 1: DANGER},
                     labels={"Delay_Flag": "Delayed (1) vs On-Time (0)"})
        fig.update_layout(height=340, margin=dict(t=10), showlegend=False, title=f"{da.DRIVER_LABELS.get(driver_col, driver_col)} by Outcome")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        bucketed = da.delay_rate_by_bucket(filtered_df, driver_col)
        if not bucketed.empty:
            fig = px.bar(bucketed, x=driver_col, y="Delay_Rate", color_discrete_sequence=[PRIMARY])
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(height=340, margin=dict(t=10), title="Delay Rate by Range")
            st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 3 -- DELAY PREDICTION
# ===========================================================================

elif page == "Delay Prediction":
    page_header("Delay Prediction", "Machine-learning models trained to predict Delay_Flag, compared and selected on recall and ROC-AUC.")

    st.info(
        "**Why recall matters most here:** a shipment the model fails to flag as at-risk (a false negative) "
        "gives operations no chance to intervene -- the customer simply experiences the delay. A false alarm "
        "(flagging an on-time shipment as at-risk) costs a wasted review, which is far cheaper. The model "
        "below is selected primarily on **recall for delayed shipments**, with ROC-AUC and precision reported "
        "for full transparency."
    )

    st.subheader("Model Comparison")
    comp_df = pd.DataFrame(model_bundle.comparison)
    display_cols = ["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"]
    styled = comp_df[display_cols].rename(columns={
        "model_name": "Model", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
    })
    for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        styled[c] = (styled[c] * 100).round(1)
    st.dataframe(styled, hide_index=True, use_container_width=True)
    st.success(f"**Selected model: {model_bundle.model_name}** (recall-prioritized selection, gated on minimum precision).")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrix")
        cm = np.array(next(c["confusion_matrix"] for c in model_bundle.comparison if c["model_name"] == model_bundle.model_name))
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted On-Time", "Predicted Delayed"], y=["Actual On-Time", "Actual Delayed"],
            colorscale=[[0, "#F0FDFA"], [1, PRIMARY]], text=cm, texttemplate="%{text:,}", showscale=False,
        ))
        fig.update_layout(height=340, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Global Feature Importance")
        importance = mdl.get_global_feature_importance(model_bundle, top_n=12)
        fig = px.bar(importance, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[PRIMARY])
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=340, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Permutation importance (model-agnostic): drop in ROC-AUC when a feature is shuffled, evaluated through the full pipeline.")

    shap_summary = ex.try_shap_global_summary(model_bundle)
    if shap_summary is not None:
        st.subheader("SHAP Global Summary (supplementary)")
        fig = px.bar(shap_summary, x="Mean_Abs_SHAP", y="Feature", orientation="h", color_discrete_sequence=[ACCENT])
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 4 -- SHIPMENT RISK
# ===========================================================================

elif page == "Shipment Risk":
    page_header("Shipment Risk", "Predict delay probability for a shipment and see the factors driving that risk.")

    source = st.radio("Shipment source", ["Select existing shipment", "Enter shipment manually"], horizontal=True)

    if source == "Select existing shipment":
        options = filtered_df["Shipment_ID"].head(3000).tolist()
        shipment_id = st.selectbox("Shipment ID", options)
        shipment = filtered_df.loc[filtered_df["Shipment_ID"] == shipment_id].iloc[0].to_dict()
    else:
        with st.form("manual_shipment_form"):
            c1, c2, c3 = st.columns(3)
            distance = c1.slider("Distance (KM)", 15, 2500, 300)
            handoffs = c2.slider("Number of handoffs", 1, 8, 2)
            weight = c3.slider("Package weight (kg)", 0.1, 200.0, 5.0)
            c4, c5, c6 = st.columns(3)
            warehouse_util = c4.slider("Warehouse capacity utilization (%)", 5, 100, 65)
            driver_avail = c5.slider("Driver availability (%)", 5, 100, 78)
            vehicle_avail = c6.slider("Vehicle availability (%)", 5, 100, 80)
            c7, c8, c9 = st.columns(3)
            carrier = c7.selectbox("Carrier", dg.CARRIERS)
            mode = c8.selectbox("Shipping mode", dg.SHIPPING_MODES)
            service = c9.selectbox("Service type", dg.SERVICE_TYPES)
            c10, c11, c12 = st.columns(3)
            weather = c10.selectbox("Weather condition", dg.WEATHER_CONDITIONS)
            traffic = c11.selectbox("Traffic level", dg.TRAFFIC_LEVELS)
            address = c12.selectbox("Address quality", dg.ADDRESS_QUALITY)
            c13, c14, c15 = st.columns(3)
            origin_region = c13.selectbox("Origin region", dg.REGIONS)
            dest_region = c14.selectbox("Destination region", dg.REGIONS, index=1)
            priority = c15.selectbox("Customer priority", dg.CUSTOMER_PRIORITY)
            c16, c17 = st.columns(2)
            package_type = c16.selectbox("Package type", dg.PACKAGE_TYPES)
            payment = c17.selectbox("Payment status", dg.PAYMENT_STATUS)
            c18, c19, c20 = st.columns(3)
            peak = c18.checkbox("Peak season")
            weekend = c19.checkbox("Weekend dispatch")
            holiday = c20.checkbox("Holiday")
            submitted = st.form_submit_button("Assess Risk", type="primary")

        shipment = {
            "Distance_KM": distance, "Number_of_Handoffs": handoffs, "Package_Weight": weight,
            "Package_Volume": max(0.01, weight * 0.02),
            "Warehouse_Capacity_Utilization": warehouse_util, "Driver_Availability": driver_avail,
            "Vehicle_Availability": vehicle_avail, "Carrier": carrier, "Shipping_Mode": mode,
            "Service_Type": service, "Weather_Condition": weather, "Traffic_Level": traffic,
            "Address_Quality": address, "Origin_Region": origin_region, "Destination_Region": dest_region,
            "Customer_Priority": priority, "Package_Type": package_type, "Payment_Status": payment,
            "Peak_Season_Flag": int(peak), "Weekend_Flag": int(weekend), "Holiday_Flag": int(holiday),
        }
        if not submitted:
            st.stop()

    pred = mdl.predict_shipment(model_bundle, shipment)
    factors = ex.explain_shipment(model_bundle, shipment, clean_df, top_n=6)
    actions = rec.shipment_recommendations(factors)

    st.markdown("####")
    c1, c2, c3 = st.columns(3)
    kpi_card(c1, "Delay Probability", f"{pred['delay_probability']*100:.0f}%")
    with c2:
        st.markdown('<div class="kpi-label" style="margin-bottom:0.3rem;">RISK LEVEL</div>', unsafe_allow_html=True)
        risk_badge(pred["risk_tier"])
    kpi_card(c3, "Estimated Delay", f"{pred['estimated_delay_hours']:.1f}h", "historical avg. for this risk tier")

    st.markdown("####")
    st.subheader("Top Contributing Factors")
    if factors:
        factor_df = pd.DataFrame([{"Factor": f["label"], "Impact (probability pts)": f["impact"] * 100} for f in factors])
        fig = px.bar(factor_df, x="Impact (probability pts)", y="Factor", orientation="h",
                     color="Impact (probability pts)", color_continuous_scale=["#2563EB", "#E2E8F0", "#DC2626"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=320, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        for i, f in enumerate(factors, 1):
            st.markdown(f"{i}. {ex.factor_sentence(f)}")
    else:
        st.info("This shipment's characteristics are all close to network-typical values.")

    st.subheader("Recommended Actions")
    for a in actions:
        st.markdown(f'<div class="insight-card">✅ {a}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="disclaimer">Predictions indicate risk based on historical patterns and should support, '
        'not replace, operational decision-making.</div>', unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 5 -- RECOMMENDATIONS
# ===========================================================================

elif page == "Recommendations":
    page_header("Recommendations", "What should the logistics company do? Prioritized, data-driven recommendations.")

    cards = rec.generate_recommendations(filtered_df)
    if not cards:
        st.info("No significant issues detected above threshold under the current filters.")
    else:
        for card in cards:
            recommendation_card(card)
    st.markdown(
        '<div class="disclaimer">Expected impact is an estimate based on closing the gap to the network average; '
        'it is not a guaranteed outcome or a financial projection.</div>', unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 6 -- WHAT-IF ANALYSIS
# ===========================================================================

elif page == "What-If Analysis":
    page_header("What-If Analysis", "Adjust operational parameters and see how predicted delay probability responds.")
    st.warning("**MODEL-BASED SCENARIO** -- this shows how the trained model's prediction changes, which reflects "
               "learned association, not guaranteed causality.")

    options = filtered_df["Shipment_ID"].head(3000).tolist()
    shipment_id = st.selectbox("Base shipment", options)
    base_shipment = filtered_df.loc[filtered_df["Shipment_ID"] == shipment_id].iloc[0].to_dict()
    base_pred = mdl.predict_shipment(model_bundle, base_shipment)

    st.markdown("**Adjust parameters:**")
    c1, c2 = st.columns(2)
    with c1:
        util = st.slider("Warehouse capacity utilization (%)", 5, 100, int(base_shipment["Warehouse_Capacity_Utilization"]))
        driver = st.slider("Driver availability (%)", 5, 100, int(base_shipment["Driver_Availability"]))
        handoffs = st.slider("Number of handoffs", 1, 8, int(base_shipment["Number_of_Handoffs"]))
    with c2:
        traffic = st.selectbox("Traffic level", dg.TRAFFIC_LEVELS, index=dg.TRAFFIC_LEVELS.index(base_shipment["Traffic_Level"]))
        weather = st.selectbox("Weather condition", dg.WEATHER_CONDITIONS, index=dg.WEATHER_CONDITIONS.index(base_shipment["Weather_Condition"]))

    scenario_shipment = dict(base_shipment)
    scenario_shipment.update({
        "Warehouse_Capacity_Utilization": util, "Driver_Availability": driver,
        "Number_of_Handoffs": handoffs, "Traffic_Level": traffic, "Weather_Condition": weather,
    })
    scenario_pred = mdl.predict_shipment(model_bundle, scenario_shipment)

    st.markdown("####")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Current**")
        kpi_card(st, "Delay Probability", f"{base_pred['delay_probability']*100:.0f}%")
        risk_badge(base_pred["risk_tier"])
    with c2:
        st.markdown("**What-If Scenario**")
        kpi_card(st, "Delay Probability", f"{scenario_pred['delay_probability']*100:.0f}%")
        risk_badge(scenario_pred["risk_tier"])

    delta = (scenario_pred["delay_probability"] - base_pred["delay_probability"]) * 100
    direction = "reduces" if delta < 0 else "increases"
    st.markdown(f"#### This scenario {direction} predicted delay probability by {abs(delta):.1f} points.")


# ===========================================================================
# PAGE 7 -- ROUTE ANALYTICS
# ===========================================================================

elif page == "Route Analytics":
    page_header("Route Analytics", "Origin -> Destination performance. High volume + high delay rate routes are priority areas.")

    routes = da.route_analytics(filtered_df, min_volume=30)
    routes_display = routes.copy()
    routes_display["Route"] = routes_display["Origin_City"] + " -> " + routes_display["Destination_City"]
    routes_display["Delay_Rate"] = (routes_display["Delay_Rate"] * 100).round(1)
    routes_display["On_Time_Pct"] = (routes_display["On_Time_Pct"] * 100).round(1)

    fig = px.scatter(
        routes_display, x="Shipments", y="Delay_Rate", size="Avg_Delay_Hours", color="Delay_Rate",
        hover_name="Route", color_continuous_scale=["#16A34A", "#D97706", "#DC2626"],
        labels={"Delay_Rate": "Delay Rate (%)"}, title="Volume vs. Delay Rate (bubble size = avg delay hours)",
    )
    fig.update_layout(height=420, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Worst-Performing Routes (min. 30 shipments)")
    worst = routes_display.sort_values("Delay_Rate", ascending=False).head(15)
    st.dataframe(
        worst[["Route", "Shipments", "Delay_Rate", "Avg_Delay_Hours", "On_Time_Pct", "Avg_Transit_Hours"]]
        .rename(columns={"Delay_Rate": "Delay Rate (%)", "Avg_Delay_Hours": "Avg Delay (h)",
                          "On_Time_Pct": "On-Time (%)", "Avg_Transit_Hours": "Avg Transit (h)"}),
        hide_index=True, use_container_width=True,
    )


# ===========================================================================
# PAGE 8 -- WAREHOUSE ANALYTICS
# ===========================================================================

elif page == "Warehouse Analytics":
    page_header("Warehouse Analytics", "Processing performance and capacity by origin warehouse.")

    wh = da.warehouse_analytics(filtered_df)
    wh_display = wh.copy()
    wh_display["Delay_Rate"] = (wh_display["Delay_Rate"] * 100).round(1)
    wh_display["Avg_Capacity_Utilization"] = wh_display["Avg_Capacity_Utilization"].round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(wh_display.sort_values("Delay_Rate", ascending=False), x="Delay_Rate", y="Warehouse",
                     orientation="h", color_discrete_sequence=[DANGER], title="Delay Rate by Warehouse (%)")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(wh_display.sort_values("Avg_Capacity_Utilization", ascending=False), x="Avg_Capacity_Utilization",
                     y="Warehouse", orientation="h", color_discrete_sequence=[PRIMARY], title="Avg Capacity Utilization (%)")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    high_risk = wh_display[(wh_display["Shipments_Processed"] >= wh_display["Shipments_Processed"].median()) &
                            (wh_display["Delay_Rate"] >= wh_display["Delay_Rate"].median())]
    if not high_risk.empty:
        st.markdown("**High volume + high delay warehouses:** " + ", ".join(high_risk["Warehouse"].tolist()))

    st.dataframe(
        wh_display.rename(columns={
            "Shipments_Processed": "Shipments Processed", "Avg_Processing_Time_Hours": "Avg Processing Time (h)",
            "Avg_Capacity_Utilization": "Avg Capacity Utilization (%)", "Delay_Rate": "Delay Rate (%)",
            "Avg_Delay_Hours": "Avg Delay (h)",
        }),
        hide_index=True, use_container_width=True,
    )


# ===========================================================================
# PAGE 9 -- CARRIER ANALYTICS
# ===========================================================================

elif page == "Carrier Analytics":
    page_header("Carrier Analytics", f"Carrier comparison. Rankings require at least {da.MIN_RANKING_VOLUME:,} shipments for a fair comparison.")

    car = da.carrier_analytics(filtered_df)
    car_display = car.copy()
    car_display["Delay_Rate"] = (car_display["Delay_Rate"] * 100).round(1)
    car_display["On_Time_Pct"] = (car_display["On_Time_Pct"] * 100).round(1)

    fig = px.bar(car_display.sort_values("Delay_Rate", ascending=False), x="Delay_Rate", y="Carrier", orientation="h",
                 color="Sufficient_Volume", color_discrete_map={True: PRIMARY, False: "#CBD5E1"},
                 title="Delay Rate by Carrier (%) -- grey bars have insufficient volume for fair ranking")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        car_display.rename(columns={
            "Delay_Rate": "Delay Rate (%)", "Avg_Delay_Hours": "Avg Delay (h)", "On_Time_Pct": "On-Time (%)",
            "Avg_Transit_Hours": "Avg Transit (h)", "Sufficient_Volume": "Sufficient Volume for Ranking",
        }),
        hide_index=True, use_container_width=True,
    )


# ===========================================================================
# PAGE 10 -- BUSINESS IMPACT
# ===========================================================================

elif page == "Business Impact":
    page_header("Business Impact", "Estimated operational and financial impact of delays. Assumptions are stated explicitly.")

    cost_per_delay = st.slider("Assumed cost per delayed shipment ($, estimate)", 10, 300, 75, step=5)
    impact = da.business_impact(filtered_df, cost_per_delayed_shipment=cost_per_delay)

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Delayed Shipments", f"{impact['delayed_shipments']:,}")
    kpi_card(c2, "Total Delay Hours", f"{impact['total_delay_hours']:,.0f}h")
    kpi_card(c3, "Est. SLA Breaches", f"{impact['estimated_sla_breaches']:,}", f">{impact['sla_breach_threshold_hours']:.0f}h delay")
    kpi_card(c4, "Customers Affected", f"{impact['estimated_customers_affected']:,}", "estimate: unique customers")

    st.markdown("####")
    kpi_card(st, "Estimated Delay Cost", f"${impact['estimated_delay_cost']:,.0f}",
              f"ASSUMPTION: ${impact['cost_per_delayed_shipment_assumption']:.0f} per delayed shipment x delayed shipments")
    st.markdown(
        '<div class="disclaimer">All financial figures on this page are estimates based on the explicit assumption '
        'above, adjustable via the slider. They are not derived from real cost data.</div>', unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 11 -- OPERATIONAL ALERTS
# ===========================================================================

elif page == "Operational Alerts":
    page_header("Operational Alerts", "Automatically detected anomalies in routes, warehouses, and carriers.")

    alerts = da.generate_alerts(filtered_df, min_volume=100)
    if not alerts:
        st.success("No active alerts under the current filter selection.")
    else:
        for a in alerts:
            alert_card(a)

    st.divider()
    st.subheader("DataQ Intelligent Insights")
    insight_cards(da.generate_intelligent_insights(filtered_df))


# ===========================================================================
# PAGE 12 -- DATA QUALITY
# ===========================================================================

elif page == "Data Quality":
    page_header("Data Quality", "Raw data-quality assessment and the cleaning pipeline applied before analysis.")

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Records (raw)", f"{dq_report.n_records:,}")
    kpi_card(c2, "Missing Values", f"{dq_report.missing_total:,}")
    kpi_card(c3, "Duplicate Records", f"{dq_report.duplicate_records:,}")
    kpi_card(c4, "Completeness", f"{dq_report.completeness_pct:.2f}%")

    st.markdown("####")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Missing Values by Column (raw)")
        if dq_report.missing_by_column:
            miss_df = pd.DataFrame(list(dq_report.missing_by_column.items()), columns=["Column", "Missing"])
            fig = px.bar(miss_df.sort_values("Missing"), x="Missing", y="Column", orientation="h", color_discrete_sequence=[WARNING])
            fig.update_layout(height=320, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing values detected.")
    with c2:
        st.subheader("Outliers Detected (IQR method)")
        out_df = pd.DataFrame(list(dq_report.outliers_by_column.items()), columns=["Column", "Outliers"])
        fig = px.bar(out_df.sort_values("Outliers"), x="Outliers", y="Column", orientation="h", color_discrete_sequence=[DANGER])
        fig.update_layout(height=320, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    if dq_report.inconsistent_categorical_values:
        st.subheader("Inconsistent Categorical Spellings (raw)")
        for col, values in dq_report.inconsistent_categorical_values.items():
            st.write(f"**{col}:** {', '.join(repr(v) for v in values)}")

    st.divider()
    st.subheader("Cleaning Pipeline -- What Was Fixed")
    c1, c2, c3 = st.columns(3)
    kpi_card(c1, "Duplicates Removed", f"{cleaning_report.duplicates_removed:,}")
    kpi_card(c2, "Categorical Values Standardized", f"{cleaning_report.categorical_values_standardized:,}")
    kpi_card(c3, "Rows After Cleaning", f"{cleaning_report.rows_after:,}")

    st.markdown("**Missing values imputed:**")
    if cleaning_report.missing_values_imputed:
        st.dataframe(pd.DataFrame(list(cleaning_report.missing_values_imputed.items()), columns=["Column", "Values Imputed"]),
                     hide_index=True, use_container_width=True)
    st.markdown("**Outliers capped (99.5th percentile winsorization):**")
    if cleaning_report.outliers_capped:
        st.dataframe(pd.DataFrame(list(cleaning_report.outliers_capped.items()), columns=["Column", "Values Capped"]),
                     hide_index=True, use_container_width=True)


# ===========================================================================
# PAGE 13 -- MODEL GOVERNANCE
# ===========================================================================

elif page == "Model Governance":
    page_header("Model Governance", "Model provenance, training details, and known limitations.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        **Model used:** {model_bundle.model_name}
        **Trained:** {model_bundle.trained_at}
        **Training records:** {model_bundle.n_train:,}
        **Test records:** {model_bundle.n_test:,}
        **Base delay rate (training set):** {model_bundle.base_delay_rate*100:.1f}%
        **Features used:** {len(mdl.ALL_FEATURES)}
        """)
    with c2:
        st.markdown("**Test-set performance:**")
        for k, v in model_bundle.metrics.items():
            st.markdown(f"- **{k.replace('_',' ').upper()}:** {v*100:.1f}%")

    st.markdown("**Feature list:**")
    st.code(", ".join(mdl.ALL_FEATURES), language=None)

    st.subheader("Limitations")
    st.markdown("""
    - Trained entirely on **synthetic** data; real-world performance will depend on how closely actual
      operational patterns match the causal structure encoded in this proof-of-concept.
    - Only features known at or before dispatch are used (process-time columns such as transit/sorting hours
      are intentionally excluded as they are only known after the fact and would leak the outcome).
    - Delay-hour estimates are historical averages per risk tier, not a dedicated regression model, and should
      be read as a rough magnitude, not a precise forecast.
    - The model reflects statistical association learned from historical patterns, not a causal or mechanistic
      model of logistics operations.
    """)

    st.markdown(
        '<div class="disclaimer"><b>Disclaimer:</b> Predictions indicate risk based on historical patterns and '
        'should support, not replace, operational decision-making.</div>', unsafe_allow_html=True,
    )
