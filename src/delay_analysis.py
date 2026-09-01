"""
Root-cause analysis, segment decomposition, driver association analysis,
route/warehouse/carrier analytics, business impact, and operational alerts.

This module answers "WHERE and WHY are packages delayed?" -- it is pure
analysis (no ML), operating on the cleaned dataframe. Everything here is
correlational; wording deliberately avoids causal claims ("associated with",
not "causes").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SEGMENT_SIZE = 150
MIN_RANKING_VOLUME = 1000  # minimum shipments for a carrier/warehouse/route to be "fairly" ranked

DRIVER_COLUMNS = [
    "Warehouse_Capacity_Utilization",
    "Number_of_Handoffs",
    "Distance_KM",
    "Driver_Availability",
    "Vehicle_Availability",
    "Customs_Clearance_Time_Hours",
    "Pickup_Delay_Hours",
]

DRIVER_LABELS = {
    "Warehouse_Capacity_Utilization": "Warehouse capacity utilization",
    "Number_of_Handoffs": "Number of handoffs",
    "Distance_KM": "Distance",
    "Driver_Availability": "Driver availability",
    "Vehicle_Availability": "Vehicle availability",
    "Customs_Clearance_Time_Hours": "Customs clearance time",
    "Pickup_Delay_Hours": "Pickup delay",
}


# ---------------------------------------------------------------------------
# Root cause: delay by category
# ---------------------------------------------------------------------------

def delay_by_cause(df: pd.DataFrame) -> pd.DataFrame:
    delayed = df.loc[df["Delay_Flag"] == 1]
    if delayed.empty:
        return pd.DataFrame(columns=["Delay_Category", "Delayed_Shipments", "Pct_of_Delayed", "Total_Delay_Hours", "Pct_of_Delay_Hours"])

    counts = delayed["Delay_Category"].value_counts()
    hours = delayed.groupby("Delay_Category")["Delay_Hours"].sum()

    out = pd.DataFrame({
        "Delay_Category": counts.index,
        "Delayed_Shipments": counts.values,
    })
    out["Pct_of_Delayed"] = out["Delayed_Shipments"] / out["Delayed_Shipments"].sum()
    out["Total_Delay_Hours"] = out["Delay_Category"].map(hours).fillna(0.0)
    out["Pct_of_Delay_Hours"] = out["Total_Delay_Hours"] / out["Total_Delay_Hours"].sum()
    return out.sort_values("Delayed_Shipments", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Segment decomposition (root cause for a selected segment)
# ---------------------------------------------------------------------------

def segment_decomposition(df: pd.DataFrame, column: str, value) -> dict:
    """Compare a segment (e.g. Carrier == 'Carrier A') against the network."""
    network_rate = df["Delay_Flag"].mean() if len(df) else 0.0
    segment = df.loc[df[column] == value]
    if segment.empty:
        return {"segment_size": 0}

    segment_rate = segment["Delay_Flag"].mean()
    cause_table = delay_by_cause(segment)
    top_cause = cause_table.iloc[0] if not cause_table.empty else None

    # Strongest associated numeric driver: correlate driver values with Delay_Flag within segment
    driver_corr = {}
    for col in DRIVER_COLUMNS:
        if col in segment.columns and segment[col].notna().sum() > 30:
            corr = segment[[col, "Delay_Flag"]].corr().iloc[0, 1]
            if pd.notna(corr):
                driver_corr[col] = corr
    strongest_driver = max(driver_corr, key=lambda k: abs(driver_corr[k])) if driver_corr else None

    sentences = []
    if network_rate > 0:
        sentences.append(
            f"{value} has a {segment_rate * 100:.1f}% delay rate compared with a network average of "
            f"{network_rate * 100:.1f}%."
        )
    if top_cause is not None:
        sentences.append(
            f"{top_cause['Delay_Category']} contributes {top_cause['Pct_of_Delay_Hours'] * 100:.0f}% of "
            f"{value}'s total delay hours."
        )
    if strongest_driver:
        sentences.append(
            f"{DRIVER_LABELS.get(strongest_driver, strongest_driver)} is the operational factor most "
            f"strongly associated with delay in this segment."
        )

    return {
        "segment_size": len(segment),
        "segment_rate": segment_rate,
        "network_rate": network_rate,
        "cause_table": cause_table,
        "top_cause": top_cause,
        "driver_correlations": driver_corr,
        "strongest_driver": strongest_driver,
        "narrative": sentences,
    }


# ---------------------------------------------------------------------------
# Driver association analysis
# ---------------------------------------------------------------------------

def driver_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Point-biserial-style correlation of each numeric driver with Delay_Flag,
    plus a magnitude-ranked table for the Driver Analysis page."""
    rows = []
    for col in DRIVER_COLUMNS:
        if col not in df.columns:
            continue
        valid = df[[col, "Delay_Flag"]].dropna()
        if len(valid) < 30:
            continue
        corr = valid[col].corr(valid["Delay_Flag"])
        rows.append({"Driver": DRIVER_LABELS.get(col, col), "Column": col, "Correlation": corr})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Abs_Correlation"] = out["Correlation"].abs()
    return out.sort_values("Abs_Correlation", ascending=False).reset_index(drop=True)


def delay_rate_by_bucket(df: pd.DataFrame, column: str, n_buckets: int = 5) -> pd.DataFrame:
    """Bucket a numeric driver into quantile bins and compute delay rate per bin
    -- used for 'delay rate by category' style bar charts."""
    valid = df[[column, "Delay_Flag"]].dropna()
    if len(valid) < n_buckets * 10:
        return pd.DataFrame(columns=[column, "Delay_Rate", "Shipments"])
    try:
        bins = pd.qcut(valid[column], q=n_buckets, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=[column, "Delay_Rate", "Shipments"])
    grouped = valid.groupby(bins, observed=True)["Delay_Flag"].agg(["mean", "count"])
    grouped = grouped.rename(columns={"mean": "Delay_Rate", "count": "Shipments"})
    grouped.index = grouped.index.astype(str)
    return grouped.reset_index().rename(columns={"index": column})


# ---------------------------------------------------------------------------
# Route / Warehouse / Carrier analytics
# ---------------------------------------------------------------------------

def route_analytics(df: pd.DataFrame, min_volume: int = 0) -> pd.DataFrame:
    g = df.groupby(["Origin_City", "Destination_City"]).agg(
        Shipments=("Shipment_ID", "count"),
        Delay_Rate=("Delay_Flag", "mean"),
        Avg_Delay_Hours=("Delay_Hours", "mean"),
        Avg_Transit_Hours=("Transit_Time_Hours", "mean"),
    ).reset_index()
    g["On_Time_Pct"] = 1 - g["Delay_Rate"]
    g = g[g["Shipments"] >= min_volume]
    return g.sort_values("Shipments", ascending=False).reset_index(drop=True)


def warehouse_analytics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Origin_Warehouse").agg(
        Shipments_Processed=("Shipment_ID", "count"),
        Avg_Processing_Time_Hours=("Warehouse_Processing_Time_Hours", "mean"),
        Avg_Capacity_Utilization=("Warehouse_Capacity_Utilization", "mean"),
        Delay_Rate=("Delay_Flag", "mean"),
        Avg_Delay_Hours=("Delay_Hours", "mean"),
    ).reset_index().rename(columns={"Origin_Warehouse": "Warehouse"})
    return g.sort_values("Delay_Rate", ascending=False).reset_index(drop=True)


def carrier_analytics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Carrier").agg(
        Shipments=("Shipment_ID", "count"),
        Delay_Rate=("Delay_Flag", "mean"),
        Avg_Delay_Hours=("Delay_Hours", "mean"),
        Avg_Transit_Hours=("Transit_Time_Hours", "mean"),
    ).reset_index()
    g["On_Time_Pct"] = 1 - g["Delay_Rate"]
    g["Sufficient_Volume"] = g["Shipments"] >= MIN_RANKING_VOLUME
    return g.sort_values("Delay_Rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Business impact
# ---------------------------------------------------------------------------

def business_impact(
    df: pd.DataFrame,
    cost_per_delayed_shipment: float = 75.0,
    sla_breach_threshold_hours: float = 24.0,
) -> dict:
    delayed = df.loc[df["Delay_Flag"] == 1]
    n_delayed = len(delayed)
    total_delay_hours = float(delayed["Delay_Hours"].sum())
    sla_breaches = int((delayed["Delay_Hours"] > sla_breach_threshold_hours).sum())
    customers_affected = int(delayed["Customer_ID"].nunique())

    return {
        "delayed_shipments": n_delayed,
        "total_delay_hours": total_delay_hours,
        "sla_breach_threshold_hours": sla_breach_threshold_hours,
        "estimated_sla_breaches": sla_breaches,
        "estimated_customers_affected": customers_affected,
        "cost_per_delayed_shipment_assumption": cost_per_delayed_shipment,
        "estimated_delay_cost": n_delayed * cost_per_delayed_shipment,
    }


# ---------------------------------------------------------------------------
# Operational alerts
# ---------------------------------------------------------------------------

def generate_alerts(df: pd.DataFrame, min_volume: int = 300) -> list[dict]:
    alerts = []
    network_rate = df["Delay_Flag"].mean() if len(df) else 0.0

    # Warehouse capacity / delay alerts
    wh = warehouse_analytics(df)
    wh = wh[wh["Shipments_Processed"] >= min_volume]
    for _, row in wh.iterrows():
        if row["Avg_Capacity_Utilization"] >= 80 or row["Delay_Rate"] >= network_rate * 1.2:
            alerts.append({
                "type": "Warehouse Congestion",
                "entity": row["Warehouse"],
                "capacity_utilization": row["Avg_Capacity_Utilization"],
                "delay_rate": row["Delay_Rate"],
                "network_rate": network_rate,
                "message": f"Warehouse {row['Warehouse']} is running at {row['Avg_Capacity_Utilization']:.0f}% "
                           f"average capacity utilization with a {row['Delay_Rate']*100:.1f}% delay rate "
                           f"(network: {network_rate*100:.1f}%).",
                "recommended_action": "Redistribute incoming shipments to an alternate warehouse or add temporary processing capacity.",
                "severity": "HIGH" if row["Avg_Capacity_Utilization"] >= 90 or row["Delay_Rate"] >= network_rate * 1.4 else "MEDIUM",
            })

    # Carrier deteriorating performance
    car = carrier_analytics(df)
    car = car[car["Shipments"] >= min_volume]
    for _, row in car.iterrows():
        if row["Delay_Rate"] >= network_rate * 1.12:
            alerts.append({
                "type": "Carrier Performance",
                "entity": row["Carrier"],
                "delay_rate": row["Delay_Rate"],
                "network_rate": network_rate,
                "message": f"{row['Carrier']} has a {row['Delay_Rate']*100:.1f}% delay rate versus a "
                           f"{network_rate*100:.1f}% network average across {int(row['Shipments']):,} shipments.",
                "recommended_action": "Review carrier SLAs and consider shifting volume to better-performing carriers on affected routes.",
                "severity": "HIGH" if row["Delay_Rate"] >= network_rate * 1.3 else "MEDIUM",
            })

    # Abnormal routes
    routes = route_analytics(df, min_volume=min_volume)
    for _, row in routes.iterrows():
        if row["Delay_Rate"] >= network_rate * 1.25:
            alerts.append({
                "type": "Route Anomaly",
                "entity": f"{row['Origin_City']} → {row['Destination_City']}",
                "delay_rate": row["Delay_Rate"],
                "network_rate": network_rate,
                "shipments": row["Shipments"],
                "message": f"Route {row['Origin_City']} → {row['Destination_City']} has a "
                           f"{row['Delay_Rate']*100:.1f}% delay rate across {int(row['Shipments']):,} shipments "
                           f"(network: {network_rate*100:.1f}%).",
                "recommended_action": "Investigate route-specific causes (carrier mix, customs, handoffs) and consider rerouting or carrier reassignment.",
                "severity": "HIGH" if row["Delay_Rate"] >= network_rate * 1.6 else "MEDIUM",
            })

    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    alerts.sort(key=lambda a: severity_order.get(a["severity"], 3))
    return alerts


# ---------------------------------------------------------------------------
# "DataQ Intelligent Insights" -- broader programmatic narrative used on the
# insights section beyond the Executive Overview's shorter Key Insights list.
# ---------------------------------------------------------------------------

def generate_intelligent_insights(df: pd.DataFrame) -> list[str]:
    insights = []
    if df.empty:
        return ["No shipments match the current filters."]

    network_rate = df["Delay_Flag"].mean()

    cause_table = delay_by_cause(df)
    if not cause_table.empty:
        top = cause_table.iloc[0]
        insights.append(f"{top['Delay_Category']} is currently the largest contributor to delay, "
                         f"responsible for {top['Pct_of_Delay_Hours']*100:.0f}% of total delay hours.")

    routes = route_analytics(df, min_volume=MIN_SEGMENT_SIZE).sort_values("Avg_Delay_Hours", ascending=False)
    if len(routes) >= 3:
        top3_hours = (routes.head(3)["Avg_Delay_Hours"] * routes.head(3)["Shipments"]).sum()
        total_hours = (routes["Avg_Delay_Hours"] * routes["Shipments"]).sum()
        if total_hours > 0:
            share = top3_hours / total_hours
            insights.append(f"The top 3 highest-impact routes account for {share*100:.0f}% of all delay hours.")

    carriers = carrier_analytics(df)
    qualified = carriers[carriers["Shipments"] >= MIN_RANKING_VOLUME]
    if not qualified.empty:
        worst = qualified.iloc[0]
        if worst["Delay_Rate"] > network_rate * 1.1:
            insights.append(
                f"{worst['Carrier']} has the highest delay rate among carriers handling more than "
                f"{MIN_RANKING_VOLUME:,} shipments ({worst['Delay_Rate']*100:.1f}% vs {network_rate*100:.1f}% network average)."
            )

    peak = df.loc[df["Peak_Season_Flag"] == 1, "Delay_Flag"]
    non_peak = df.loc[df["Peak_Season_Flag"] == 0, "Delay_Flag"]
    if len(peak) >= MIN_SEGMENT_SIZE and len(non_peak) >= MIN_SEGMENT_SIZE and non_peak.mean() > 0:
        ratio = peak.mean() / non_peak.mean()
        if ratio >= 1.1:
            insights.append(f"Peak-season shipments have a {ratio:.1f}x higher probability of delay than off-peak shipments.")

    wh = warehouse_analytics(df)
    congested = wh[wh["Avg_Capacity_Utilization"] >= 90]
    if not congested.empty:
        insights.append(
            f"{len(congested)} warehouse(s) are operating above 90% capacity utilization, "
            "a level strongly associated with elevated processing delays."
        )

    return insights
