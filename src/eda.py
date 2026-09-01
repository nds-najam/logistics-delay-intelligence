"""
Executive-overview KPIs, time-series aggregations, and dynamically generated
"Key Insights" for the Executive Overview page.

Every insight is computed from the (filtered) dataframe at call time -- nothing
here is a hard-coded number. If a segment doesn't have enough volume to support
a reliable insight, that insight is simply skipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Assumption used for the "Estimated Delay Impact" KPI -- explicitly labeled as
# an assumption everywhere it surfaces in the UI. See src/delay_analysis.py's
# business_impact() for the fuller breakdown used on the Business Impact page.
DEFAULT_COST_PER_DELAYED_SHIPMENT = 75.0  # USD

MIN_SEGMENT_SIZE = 200
MIN_CARRIER_VOLUME_FOR_RANKING = 2000


def compute_kpis(df: pd.DataFrame, cost_per_delay: float = DEFAULT_COST_PER_DELAYED_SHIPMENT) -> dict:
    n = len(df)
    delayed = df.loc[df["Delay_Flag"] == 1]
    n_delayed = len(delayed)
    delay_rate = (n_delayed / n) if n else 0.0

    return {
        "total_shipments": n,
        "delayed_shipments": n_delayed,
        "delay_rate": delay_rate,
        "avg_delay_hours": float(delayed["Delay_Hours"].mean()) if n_delayed else 0.0,
        "median_delay_hours": float(delayed["Delay_Hours"].median()) if n_delayed else 0.0,
        "on_time_pct": 1 - delay_rate,
        "avg_delivery_days": float(df["Actual_Delivery_Days"].mean()) if n else 0.0,
        "estimated_delay_cost": n_delayed * cost_per_delay,
        "cost_per_delay_assumption": cost_per_delay,
    }


def shipment_volume_over_time(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    s = df.set_index("Order_Date").resample(freq).size().rename("Shipments")
    return s.reset_index()


def delay_rate_over_time(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    g = df.set_index("Order_Date").resample(freq)["Delay_Flag"].mean().rename("Delay_Rate")
    return g.reset_index()


def delay_hours_distribution(df: pd.DataFrame) -> pd.Series:
    return df.loc[df["Delay_Flag"] == 1, "Delay_Hours"]


def on_time_vs_delayed(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["Delay_Flag"].map({0: "On-Time", 1: "Delayed"}).value_counts()
    return counts.rename_axis("Status").reset_index(name="Shipments")


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def generate_key_insights(df: pd.DataFrame, max_insights: int = 6) -> list[str]:
    """Rule-based, data-driven insight sentences for the Executive Overview page."""
    insights: list[str] = []
    if df.empty:
        return ["No shipments match the current filters."]

    network_rate = df["Delay_Flag"].mean()

    # 1. Peak season effect
    peak = df.loc[df["Peak_Season_Flag"] == 1, "Delay_Flag"]
    non_peak = df.loc[df["Peak_Season_Flag"] == 0, "Delay_Flag"]
    if len(peak) >= MIN_SEGMENT_SIZE and len(non_peak) >= MIN_SEGMENT_SIZE and non_peak.mean() > 0:
        peak_rate, non_peak_rate = peak.mean(), non_peak.mean()
        change_pct = (peak_rate - non_peak_rate) / non_peak_rate * 100
        direction = "higher" if change_pct >= 0 else "lower"
        insights.append(
            f"Delay rate is {abs(change_pct):.1f}% {direction} during peak season "
            f"({_pct(peak_rate)} vs {_pct(non_peak_rate)} off-peak)."
        )

    # 2. Dominant delay category
    delayed = df.loc[df["Delay_Flag"] == 1]
    if len(delayed) >= MIN_SEGMENT_SIZE:
        cat_share = delayed["Delay_Category"].value_counts(normalize=True)
        cat_share = cat_share[cat_share.index != "Other"]
        if not cat_share.empty:
            top_cat, top_share = cat_share.index[0], cat_share.iloc[0]
            insights.append(
                f"{top_cat} accounts for {_pct(top_share)} of all delayed shipments, "
                "the single largest contributor."
            )

    # 3. Worst-performing route (min volume so we don't cite a route with 3 shipments)
    route_stats = (
        df.groupby(["Origin_City", "Destination_City"])
        .agg(n=("Shipment_ID", "count"), rate=("Delay_Flag", "mean"))
        .query("n >= @MIN_SEGMENT_SIZE")
    )
    if not route_stats.empty and network_rate > 0:
        worst = route_stats["rate"].idxmax()
        worst_rate = route_stats.loc[worst, "rate"]
        ratio = worst_rate / network_rate
        if ratio >= 1.15:
            insights.append(
                f"Route {worst[0]} → {worst[1]} has a {ratio:.1f}x higher delay rate "
                f"than the network average ({_pct(worst_rate)} vs {_pct(network_rate)})."
            )

    # 4. Carrier standout among carriers with meaningful volume
    carrier_stats = (
        df.groupby("Carrier")
        .agg(n=("Shipment_ID", "count"), rate=("Delay_Flag", "mean"))
    )
    volume_threshold = min(MIN_CARRIER_VOLUME_FOR_RANKING, carrier_stats["n"].max() * 0.3 if not carrier_stats.empty else 0)
    qualified = carrier_stats[carrier_stats["n"] >= max(volume_threshold, MIN_SEGMENT_SIZE)]
    if not qualified.empty:
        worst_carrier = qualified["rate"].idxmax()
        worst_carrier_rate = qualified.loc[worst_carrier, "rate"]
        if worst_carrier_rate > network_rate * 1.1:
            insights.append(
                f"{worst_carrier} has the highest delay rate ({_pct(worst_carrier_rate)}) among carriers "
                f"handling more than {int(qualified.loc[worst_carrier, 'n']):,} shipments."
            )

    # 5. Weekend effect
    weekend = df.loc[df["Weekend_Flag"] == 1, "Delay_Flag"]
    weekday = df.loc[df["Weekend_Flag"] == 0, "Delay_Flag"]
    if len(weekend) >= MIN_SEGMENT_SIZE and len(weekday) >= MIN_SEGMENT_SIZE and weekday.mean() > 0:
        ratio = weekend.mean() / weekday.mean()
        if ratio >= 1.1 or ratio <= 0.9:
            direction = "higher" if ratio >= 1 else "lower"
            insights.append(f"Weekend shipments see a {ratio:.1f}x {direction} delay rate than weekday shipments.")

    # 6. Address quality effect
    poor_addr = df.loc[df["Address_Quality"] == "Poor", "Delay_Flag"]
    good_addr = df.loc[df["Address_Quality"] == "Good", "Delay_Flag"]
    if len(poor_addr) >= MIN_SEGMENT_SIZE // 2 and len(good_addr) >= MIN_SEGMENT_SIZE and good_addr.mean() > 0:
        ratio = poor_addr.mean() / good_addr.mean()
        if ratio >= 1.15:
            insights.append(
                f"Shipments with poor address quality are delayed {ratio:.1f}x more often than "
                "those with good address quality."
            )

    return insights[:max_insights] if insights else ["Not enough data in the current filter selection to generate insights."]
