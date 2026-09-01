"""
Recommendation engine: turns detected root causes into prioritized,
actionable recommendation cards (Issue / Evidence / Recommended Action /
Expected Impact / Priority), plus lightweight per-shipment recommendations
driven by that shipment's top explainability factors.

All evidence is computed from the (filtered) dataframe at call time. Expected
impact is always phrased as an estimate, never a fabricated number.
"""

from __future__ import annotations

import pandas as pd

import delay_analysis as da

MIN_SEGMENT_VOLUME = 300


def _priority(ratio: float) -> str:
    if ratio >= 1.5:
        return "HIGH"
    if ratio >= 1.2:
        return "MEDIUM"
    return "LOW"


def generate_recommendations(df: pd.DataFrame, max_recommendations: int = 8) -> list[dict]:
    if df.empty:
        return []

    network_rate = df["Delay_Flag"].mean()
    cards: list[dict] = []

    # --- Warehouse capacity ---------------------------------------------
    wh = da.warehouse_analytics(df)
    wh = wh[wh["Shipments_Processed"] >= MIN_SEGMENT_VOLUME]
    for _, row in wh.sort_values("Avg_Capacity_Utilization", ascending=False).head(3).iterrows():
        if row["Avg_Capacity_Utilization"] < 78:
            continue
        ratio = row["Delay_Rate"] / network_rate if network_rate else 1.0
        cards.append({
            "issue": f"Warehouse {row['Warehouse']} capacity utilization averaging {row['Avg_Capacity_Utilization']:.0f}%",
            "evidence": f"Delay rate = {row['Delay_Rate']*100:.1f}% vs. network average {network_rate*100:.1f}% "
                        f"({int(row['Shipments_Processed']):,} shipments).",
            "recommendation": "Redistribute a portion of incoming shipments to an alternative warehouse "
                               "or add temporary processing capacity at this site.",
            "expected_impact": "Estimate: bringing utilization closer to the network norm would be expected "
                                "to reduce warehouse-processing-related delays at this site.",
            "priority": _priority(max(ratio, row["Avg_Capacity_Utilization"] / 78)),
        })

    # --- Carrier performance ---------------------------------------------
    car = da.carrier_analytics(df)
    car = car[car["Shipments"] >= MIN_SEGMENT_VOLUME]
    for _, row in car.sort_values("Delay_Rate", ascending=False).head(2).iterrows():
        ratio = row["Delay_Rate"] / network_rate if network_rate else 1.0
        if ratio < 1.1:
            continue
        cards.append({
            "issue": f"{row['Carrier']} delay rate is elevated relative to the network",
            "evidence": f"Delay rate = {row['Delay_Rate']*100:.1f}% vs. network average {network_rate*100:.1f}% "
                        f"across {int(row['Shipments']):,} shipments.",
            "recommendation": f"Review {row['Carrier']}'s service-level agreement and route assignments; "
                               "consider shifting volume on affected routes to better-performing carriers.",
            "expected_impact": "Estimate: aligning this carrier's delay rate with the network average would "
                                "reduce total delayed shipments proportionally to its volume share.",
            "priority": _priority(ratio),
        })

    # --- Driver / vehicle availability ------------------------------------
    low_driver = df.loc[df["Driver_Availability"] < 50]
    if len(low_driver) >= MIN_SEGMENT_VOLUME:
        rate = low_driver["Delay_Flag"].mean()
        ratio = rate / network_rate if network_rate else 1.0
        if ratio >= 1.1:
            cards.append({
                "issue": "Low driver availability (<50%) is associated with elevated delay",
                "evidence": f"Delay rate = {rate*100:.1f}% when driver availability is below 50%, vs. "
                            f"{network_rate*100:.1f}% network average ({len(low_driver):,} shipments).",
                "recommendation": "Increase driver allocation for the affected delivery regions, particularly "
                                   "on weekends and during peak season.",
                "expected_impact": "Estimate: improving driver availability in constrained regions would be "
                                    "expected to reduce pickup and last-mile delays.",
                "priority": _priority(ratio),
            })

    # --- Traffic --------------------------------------------------------
    heavy_traffic = df.loc[df["Traffic_Level"].isin(["High", "Severe"])]
    if len(heavy_traffic) >= MIN_SEGMENT_VOLUME:
        rate = heavy_traffic["Delay_Flag"].mean()
        ratio = rate / network_rate if network_rate else 1.0
        if ratio >= 1.1:
            cards.append({
                "issue": "High/Severe traffic conditions are associated with elevated delay",
                "evidence": f"Delay rate = {rate*100:.1f}% under High/Severe traffic vs. {network_rate*100:.1f}% "
                            f"network average ({len(heavy_traffic):,} shipments).",
                "recommendation": "Prioritize dispatch during lower-traffic windows where service commitments "
                                   "allow, or dynamically reroute last-mile deliveries around congestion.",
                "expected_impact": "Estimate: shifting a portion of dispatch timing away from peak congestion "
                                    "would be expected to reduce traffic-attributed delay hours.",
                "priority": _priority(ratio),
            })

    # --- Handoffs ---------------------------------------------------------
    high_handoff = df.loc[df["Number_of_Handoffs"] >= 4]
    if len(high_handoff) >= MIN_SEGMENT_VOLUME:
        rate = high_handoff["Delay_Flag"].mean()
        ratio = rate / network_rate if network_rate else 1.0
        if ratio >= 1.1:
            cards.append({
                "issue": "Shipments with 4+ handoffs show elevated delay rates",
                "evidence": f"Delay rate = {rate*100:.1f}% at 4+ handoffs vs. {network_rate*100:.1f}% network "
                            f"average ({len(high_handoff):,} shipments).",
                "recommendation": "Evaluate route consolidation opportunities to reduce the number of handoffs "
                                   "on multi-leg routes.",
                "expected_impact": "Estimate: each avoided handoff on a high-handoff route would be expected "
                                    "to reduce sorting-related delay for that shipment.",
                "priority": _priority(ratio),
            })

    # --- Address quality ----------------------------------------------
    poor_addr = df.loc[df["Address_Quality"] == "Poor"]
    if len(poor_addr) >= MIN_SEGMENT_VOLUME // 2:
        rate = poor_addr["Delay_Flag"].mean()
        ratio = rate / network_rate if network_rate else 1.0
        if ratio >= 1.1:
            cards.append({
                "issue": "Poor address quality is associated with elevated last-mile delay",
                "evidence": f"Delay rate = {rate*100:.1f}% for Poor address quality vs. {network_rate*100:.1f}% "
                            f"network average ({len(poor_addr):,} shipments).",
                "recommendation": "Trigger address verification / customer confirmation before dispatch for "
                                   "shipments flagged with incomplete or low-confidence addresses.",
                "expected_impact": "Estimate: reducing the Poor-address-quality share would be expected to "
                                    "reduce last-mile delay incidents.",
                "priority": _priority(ratio),
            })

    # --- Customs ------------------------------------------------------
    customs_shipments = df.loc[df["Customs_Clearance_Time_Hours"] > 0]
    if len(customs_shipments) >= MIN_SEGMENT_VOLUME // 2:
        rate = customs_shipments["Delay_Flag"].mean()
        ratio = rate / network_rate if network_rate else 1.0
        if ratio >= 1.1:
            cards.append({
                "issue": "Customs-cleared shipments show elevated delay rates",
                "evidence": f"Delay rate = {rate*100:.1f}% for shipments requiring customs clearance vs. "
                            f"{network_rate*100:.1f}% network average ({len(customs_shipments):,} shipments).",
                "recommendation": "Pre-clear eligible shipments where possible and flag customs-sensitive "
                                   "shipments earlier in the pipeline for proactive handling.",
                "expected_impact": "Estimate: earlier customs flagging would be expected to reduce customs-"
                                    "attributed delay hours on cross-region shipments.",
                "priority": _priority(ratio),
            })

    # --- Peak season capacity -------------------------------------------
    peak = df.loc[df["Peak_Season_Flag"] == 1]
    non_peak = df.loc[df["Peak_Season_Flag"] == 0]
    if len(peak) >= MIN_SEGMENT_VOLUME and len(non_peak) >= MIN_SEGMENT_VOLUME and non_peak["Delay_Flag"].mean() > 0:
        ratio = peak["Delay_Flag"].mean() / non_peak["Delay_Flag"].mean()
        if ratio >= 1.15:
            cards.append({
                "issue": "Peak-season shipments show a materially higher delay rate",
                "evidence": f"Delay rate = {peak['Delay_Flag'].mean()*100:.1f}% in peak season vs. "
                            f"{non_peak['Delay_Flag'].mean()*100:.1f}% off-peak ({ratio:.1f}x).",
                "recommendation": "Plan temporary warehouse capacity and driver staffing increases ahead of "
                                   "known peak-season windows.",
                "expected_impact": "Estimate: pre-scaling capacity for peak season would be expected to narrow "
                                    "the peak vs. off-peak delay-rate gap.",
                "priority": _priority(ratio),
            })

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    cards.sort(key=lambda c: priority_order.get(c["priority"], 3))
    return cards[:max_recommendations]


# ---------------------------------------------------------------------------
# Per-shipment recommendations, driven by that shipment's explainability
# factors (see src/explainability.py). Maps a risk-raising factor to a
# concrete suggested action.
# ---------------------------------------------------------------------------

_SHIPMENT_RULES = [
    (lambda f: f["feature"] == "Warehouse_Capacity_Utilization" and f["impact"] > 0,
     "Consider redistributing this shipment to an alternative warehouse or expediting processing given high capacity utilization."),
    (lambda f: f["feature"] == "Driver_Availability" and f["impact"] > 0,
     "Flag for priority driver assignment given low driver availability in this region."),
    (lambda f: f["feature"] == "Vehicle_Availability" and f["impact"] > 0,
     "Confirm vehicle assignment early given constrained vehicle availability."),
    (lambda f: f["feature"] == "Traffic_Level" and f["impact"] > 0,
     "Consider adjusting dispatch timing or last-mile routing to avoid current traffic conditions."),
    (lambda f: f["feature"] == "Weather_Condition" and f["impact"] > 0,
     "Monitor weather along the route and consider a routing or timing adjustment if conditions worsen."),
    (lambda f: f["feature"] == "Number_of_Handoffs" and f["impact"] > 0,
     "Evaluate whether this route's handoff count can be reduced through consolidation."),
    (lambda f: f["feature"] == "Address_Quality" and f["impact"] > 0,
     "Trigger address verification with the customer before dispatch."),
    (lambda f: f["feature"] == "Is_Cross_Region" and f["impact"] > 0,
     "Flag as customs-sensitive and pre-clear where eligible."),
    (lambda f: f["feature"] == "Service_Type" and f["impact"] > 0,
     "This service tier's promised window is tight relative to current conditions -- consider proactive customer notification."),
    (lambda f: f["feature"] == "Peak_Season_Flag" and f["impact"] > 0,
     "Apply peak-season handling priority given elevated seasonal load."),
]


def shipment_recommendations(factors: list[dict]) -> list[str]:
    """Map a shipment's top explainability factors to concrete actions."""
    actions = []
    for factor in factors:
        for condition, action in _SHIPMENT_RULES:
            if condition(factor) and action not in actions:
                actions.append(action)
                break
    if not actions:
        actions.append("No significant risk-raising factors detected; standard handling is appropriate.")
    return actions
