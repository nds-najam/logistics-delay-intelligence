"""
Synthetic logistics dataset generator for DataQ Logistics Delay Intelligence PoC.

Generates a realistic shipment-level dataset with causal structure baked in:
operational stressors (warehouse utilization, traffic, weather, handoffs,
driver availability, address quality, customs, peak season, distance) drive
component delay times, which combine into an overall delay outcome. Controlled
random noise is added throughout so the relationships are strong but not
deterministic -- a downstream ML model has genuine signal to learn rather than
simple lookup rules.

Intentional data-quality issues (missing values, duplicates, outliers,
inconsistent categorical spellings) are injected at the end so the app's
data-quality / cleaning pipeline has real work to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42

CITIES = {
    # region -> list of cities
    "North": ["Riyadh", "Buraidah", "Hail", "Tabuk"],
    "South": ["Jazan", "Abha", "Najran", "Khamis Mushait"],
    "East": ["Dammam", "Khobar", "Dhahran", "Jubail"],
    "West": ["Jeddah", "Makkah", "Madinah", "Yanbu"],
    "Central": ["Riyadh", "Kharj", "Dawadmi"],
}

REGIONS = list(CITIES.keys())

WAREHOUSES = {
    "North": ["WH-N1", "WH-N2"],
    "South": ["WH-S1"],
    "East": ["WH-E1", "WH-E2"],
    "West": ["WH-W1", "WH-W2", "WH-W3"],
    "Central": ["WH-C1"],
}

# Structural baseline utilization per warehouse (percentage points added to the
# per-shipment draw) -- some sites are chronically closer to capacity than
# others, independent of season, so warehouse-level analytics/alerts have real
# signal to surface rather than every site looking statistically identical.
WAREHOUSE_BASELINE_UTILIZATION = {
    "WH-N1": 3, "WH-N2": -4,
    "WH-S1": 18,   # chronically constrained single site for the whole South region
    "WH-E1": -2, "WH-E2": 10,
    "WH-W1": -6, "WH-W2": 8, "WH-W3": 0,
    "WH-C1": 12,
}

CARRIERS = ["Carrier A", "Carrier B", "Carrier C", "Carrier D", "Carrier E"]
# Baseline reliability multiplier per carrier (1.0 = network average risk)
CARRIER_RISK = {
    "Carrier A": 1.35,
    "Carrier B": 0.85,
    "Carrier C": 1.0,
    "Carrier D": 0.70,
    "Carrier E": 1.15,
}

SHIPPING_MODES = ["Road", "Air", "Rail", "Sea"]
SHIPPING_MODE_WEIGHTS = [0.62, 0.18, 0.10, 0.10]
SHIPPING_MODE_SPEED_KMH = {"Road": 55, "Air": 650, "Rail": 70, "Sea": 30}

SERVICE_TYPES = ["Standard", "Express", "Same-Day", "Economy"]
SERVICE_TYPE_WEIGHTS = [0.50, 0.28, 0.07, 0.15]
SERVICE_PROMISED_DAYS = {"Standard": 4, "Express": 2, "Same-Day": 1, "Economy": 6}

PACKAGE_TYPES = ["Document", "Small Parcel", "Standard Box", "Bulky Item", "Fragile", "Perishable"]
PACKAGE_TYPE_WEIGHTS = [0.12, 0.28, 0.30, 0.12, 0.10, 0.08]

WEATHER_CONDITIONS = ["Clear", "Rain", "Storm", "Fog", "Extreme Heat"]
WEATHER_WEIGHTS = [0.62, 0.18, 0.06, 0.08, 0.06]
WEATHER_SEVERITY = {"Clear": 0.0, "Rain": 0.35, "Fog": 0.4, "Extreme Heat": 0.3, "Storm": 0.9}

TRAFFIC_LEVELS = ["Low", "Medium", "High", "Severe"]
TRAFFIC_WEIGHTS = [0.30, 0.38, 0.24, 0.08]
TRAFFIC_SEVERITY = {"Low": 0.0, "Medium": 0.3, "High": 0.65, "Severe": 1.0}

ADDRESS_QUALITY = ["Good", "Fair", "Poor"]
ADDRESS_QUALITY_WEIGHTS = [0.72, 0.20, 0.08]

CUSTOMER_PRIORITY = ["Standard", "Priority", "VIP"]
CUSTOMER_PRIORITY_WEIGHTS = [0.65, 0.27, 0.08]

PAYMENT_STATUS = ["Paid", "Cash on Delivery", "Pending"]
PAYMENT_STATUS_WEIGHTS = [0.70, 0.25, 0.05]

DELAY_CATEGORIES = [
    "Warehouse Processing", "Pickup", "Transit", "Sorting", "Customs",
    "Last Mile", "Weather", "Traffic", "Address Issue", "Capacity",
    "Driver Availability", "Other",
]


def _rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _choice(rng, options, weights, size):
    return rng.choice(options, size=size, p=weights)


def generate_dataset(n_records: int = 120_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = _rng(seed)
    n = n_records

    # ---------------------------------------------------------------
    # Geography
    # ---------------------------------------------------------------
    origin_region = rng.choice(REGIONS, size=n)
    dest_region = rng.choice(REGIONS, size=n)

    origin_city = np.array([rng.choice(CITIES[r]) for r in origin_region])
    dest_city = np.array([rng.choice(CITIES[r]) for r in dest_region])

    origin_warehouse = np.array([rng.choice(WAREHOUSES[r]) for r in origin_region])
    dest_warehouse = np.array([rng.choice(WAREHOUSES[r]) for r in dest_region])

    is_customs = (origin_region != dest_region) & (rng.random(n) < 0.18)
    # cross-region shipments are naturally further; add base distance by region pair
    same_region = origin_region == dest_region
    base_distance = np.where(same_region, rng.normal(180, 90, n), rng.normal(650, 300, n))
    distance_km = np.clip(base_distance, 15, 2500)

    # ---------------------------------------------------------------
    # Carrier / shipping mode / service
    # ---------------------------------------------------------------
    carrier = rng.choice(CARRIERS, size=n)
    shipping_mode = _choice(rng, SHIPPING_MODES, SHIPPING_MODE_WEIGHTS, n)
    # Sea/Air only sensible for long distance / cross region; nudge via resampling
    long_haul = distance_km > 400
    forced_air = long_haul & (rng.random(n) < 0.25)
    shipping_mode = np.where(forced_air, "Air", shipping_mode)

    service_type = _choice(rng, SERVICE_TYPES, SERVICE_TYPE_WEIGHTS, n)
    package_type = _choice(rng, PACKAGE_TYPES, PACKAGE_TYPE_WEIGHTS, n)

    package_weight = np.clip(rng.lognormal(mean=1.1, sigma=0.9, size=n), 0.1, 200)
    package_volume = np.clip(package_weight * rng.uniform(0.01, 0.05, n) + rng.normal(0, 0.02, n), 0.005, 8)

    # ---------------------------------------------------------------
    # Dates
    # ---------------------------------------------------------------
    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2024-12-31")
    total_days = (end_date - start_date).days
    order_offset = rng.integers(0, total_days, n)
    order_date = start_date + pd.to_timedelta(order_offset, unit="D")
    order_date = order_date + pd.to_timedelta(rng.integers(0, 24 * 60, n), unit="m")

    month = order_date.month.to_numpy()
    dow = order_date.dayofweek.to_numpy()
    weekend_flag = (dow >= 5).astype(int)

    # Peak season: Nov-Dec (holiday shopping) + Ramadan proxy (varies, approx Mar-Apr)
    peak_season_flag = (np.isin(month, [11, 12]) | np.isin(month, [3, 4])).astype(int)
    # Holiday flag: sprinkle randomly, boosted near peak season
    holiday_base_p = 0.03 + 0.05 * peak_season_flag
    holiday_flag = (rng.random(n) < holiday_base_p).astype(int)

    # ---------------------------------------------------------------
    # Operational stressors (root causes)
    # ---------------------------------------------------------------
    weather_condition = _choice(rng, WEATHER_CONDITIONS, WEATHER_WEIGHTS, n)
    weather_severity = np.array([WEATHER_SEVERITY[w] for w in weather_condition])

    traffic_level = _choice(rng, TRAFFIC_LEVELS, TRAFFIC_WEIGHTS, n)
    traffic_severity = np.array([TRAFFIC_SEVERITY[t] for t in traffic_level])

    address_quality = _choice(rng, ADDRESS_QUALITY, ADDRESS_QUALITY_WEIGHTS, n)
    address_bad = np.array([1.0 if a == "Poor" else (0.4 if a == "Fair" else 0.0) for a in address_quality])

    customer_priority = _choice(rng, CUSTOMER_PRIORITY, CUSTOMER_PRIORITY_WEIGHTS, n)
    payment_status = _choice(rng, PAYMENT_STATUS, PAYMENT_STATUS_WEIGHTS, n)

    # Warehouse capacity utilization: per-warehouse structural baseline + peak
    # season boost + noise (see WAREHOUSE_BASELINE_UTILIZATION)
    warehouse_baseline = np.array([WAREHOUSE_BASELINE_UTILIZATION[w] for w in origin_warehouse])
    warehouse_capacity_utilization = np.clip(
        rng.normal(60, 12, n) + warehouse_baseline + peak_season_flag * rng.normal(15, 5, n), 5, 100
    )

    # Driver / vehicle availability (%): lower on weekends/holidays and high traffic regions
    driver_availability = np.clip(
        rng.normal(78, 14, n) - weekend_flag * rng.normal(8, 4, n)
        - holiday_flag * rng.normal(10, 5, n) - traffic_severity * rng.normal(10, 5, n),
        5, 100,
    )
    vehicle_availability = np.clip(
        rng.normal(80, 12, n) - peak_season_flag * rng.normal(8, 4, n), 5, 100
    )

    number_of_handoffs = np.clip(
        np.round(
            1 + (distance_km / 500) + is_customs.astype(int) * 1.5
            + (shipping_mode == "Sea").astype(int) * 1.5
            + rng.poisson(0.6, n)
        ),
        1, 8,
    ).astype(int)

    carrier_risk = np.array([CARRIER_RISK[c] for c in carrier])

    # Expedited service tiers get operationally prioritized handling (skip-the-
    # queue processing, priority pickup, priority sorting) so that Same-Day /
    # Express promises are achievable under normal conditions. Without this,
    # every Same-Day shipment would be near-certain to miss its 24h promise
    # regardless of operational conditions, which would swamp the genuine
    # operational causes (warehouse load, traffic, weather, handoffs) that are
    # the actual point of this dataset. Economy is deprioritized in exchange
    # for its generous promise window.
    SERVICE_SPEED_FACTOR = {"Same-Day": 0.35, "Express": 0.70, "Standard": 1.0, "Economy": 1.15}
    service_speed = np.array([SERVICE_SPEED_FACTOR[s] for s in service_type])

    # ---------------------------------------------------------------
    # Component delay drivers (all in hours), each with its own causal formula + noise
    # ---------------------------------------------------------------
    warehouse_processing_time = np.clip(
        (4 + (warehouse_capacity_utilization / 100) * 22
         + peak_season_flag * rng.normal(4, 2, n)
         + rng.normal(0, 3, n)) * service_speed,
        0.5, 96,
    )

    pickup_delay_hours = np.clip(
        (1 + (100 - driver_availability) / 100 * 10
         + (100 - vehicle_availability) / 100 * 6
         + weekend_flag * rng.normal(2, 1, n)
         + (carrier_risk - 1.0) * rng.normal(2.5, 1, n)
         + rng.normal(0, 2, n)) * service_speed,
        0, 60,
    )

    base_transit_speed = SHIPPING_MODE_SPEED_KMH
    mode_speed = np.array([base_transit_speed[m] for m in shipping_mode])
    transit_time_hours = np.clip(
        (distance_km / mode_speed) * (1 + weather_severity * 0.8 + traffic_severity * 0.3) * carrier_risk
        + rng.normal(0, 3, n),
        0.5, 400,
    )

    sorting_time_hours = np.clip(
        (1 + number_of_handoffs * rng.normal(1.4, 0.5, n)
         + (warehouse_capacity_utilization / 100) * 5
         + rng.normal(0, 1.5, n)) * service_speed,
        0.2, 60,
    )

    last_mile_time_hours = np.clip(
        (2 + traffic_severity * 14 + address_bad * 10
         + (100 - driver_availability) / 100 * 8
         + rng.normal(0, 2, n)) * service_speed,
        0.5, 80,
    )

    customs_clearance_time_hours = np.where(
        is_customs,
        np.clip(rng.normal(18, 10, n) + peak_season_flag * rng.normal(5, 3, n), 0, 120),
        0.0,
    )

    # ---------------------------------------------------------------
    # Promised / actual delivery days and dates
    # ---------------------------------------------------------------
    promised_delivery_days = np.array([SERVICE_TYPE_PROMISED := SERVICE_PROMISED_DAYS[s] for s in service_type])

    pickup_date = order_date + pd.to_timedelta(warehouse_processing_time + pickup_delay_hours, unit="h")

    total_operational_hours = (
        transit_time_hours + sorting_time_hours + last_mile_time_hours + customs_clearance_time_hours
    )
    actual_delivery_days_raw = (
        warehouse_processing_time + pickup_delay_hours + total_operational_hours
    ) / 24.0

    # Random noise so the model has genuine uncertainty to learn (not deterministic)
    actual_delivery_days = np.clip(actual_delivery_days_raw + rng.normal(0, 0.4, n), 0.1, None)

    expected_delivery_date = order_date + pd.to_timedelta(promised_delivery_days, unit="D")
    actual_delivery_date = order_date + pd.to_timedelta(actual_delivery_days, unit="D")

    delay_hours_raw = (actual_delivery_date - expected_delivery_date) / pd.Timedelta(hours=1)
    delay_flag = (delay_hours_raw > 0).astype(int)
    delay_hours = np.where(delay_flag == 1, delay_hours_raw, 0.0)

    # ---------------------------------------------------------------
    # Delay category / reason attribution: pick the dominant component
    # among delayed shipments, with some randomness for realism
    # ---------------------------------------------------------------
    component_hours = np.column_stack([
        warehouse_processing_time * (warehouse_capacity_utilization > 80),   # Capacity/Warehouse Processing
        pickup_delay_hours,                                                  # Pickup
        transit_time_hours * (weather_severity < 0.2) * (traffic_severity < 0.2),  # pure Transit
        sorting_time_hours,                                                  # Sorting
        customs_clearance_time_hours,                                        # Customs
        last_mile_time_hours * (traffic_severity < 0.3) * (address_bad < 0.3),  # pure Last Mile
        transit_time_hours * weather_severity,                               # Weather
        (last_mile_time_hours + transit_time_hours) * traffic_severity * 0.5,  # Traffic
        last_mile_time_hours * address_bad,                                  # Address Issue
        warehouse_processing_time * (warehouse_capacity_utilization / 100),  # Capacity
        pickup_delay_hours * ((100 - driver_availability) / 100),            # Driver Availability
    ])
    category_names = [
        "Warehouse Processing", "Pickup", "Transit", "Sorting", "Customs",
        "Last Mile", "Weather", "Traffic", "Address Issue", "Capacity",
        "Driver Availability",
    ]
    dominant_idx = np.argmax(component_hours, axis=1)
    delay_category = np.array(category_names)[dominant_idx]
    # Some fraction gets bucketed as "Other" for realism (unexplained noise)
    other_mask = rng.random(n) < 0.04
    delay_category = np.where(other_mask, "Other", delay_category)
    # "No Delay" (not "None"/"NA") -- those strings are silently read back as
    # NaN by pandas' default CSV parser and would corrupt this column on load.
    delay_category = np.where(delay_flag == 1, delay_category, "No Delay")

    reason_map = {
        "Warehouse Processing": "High warehouse throughput / backlog at origin facility",
        "Pickup": "Delayed pickup due to driver/vehicle scheduling",
        "Transit": "Extended transit time on route",
        "Sorting": "Multiple handoffs increased sorting time",
        "Customs": "Customs clearance delay for cross-region shipment",
        "Last Mile": "Last-mile delivery delay",
        "Weather": "Severe weather impacted transit",
        "Traffic": "Heavy traffic congestion en route",
        "Address Issue": "Incomplete or inaccurate delivery address",
        "Capacity": "Origin warehouse over capacity utilization",
        "Driver Availability": "Low driver/vehicle availability at dispatch",
        "Other": "Unclassified operational delay",
        "No Delay": "On-time delivery",
    }
    delay_reason = np.array([reason_map[c] for c in delay_category])

    # ---------------------------------------------------------------
    # IDs
    # ---------------------------------------------------------------
    shipment_id = np.array([f"SH{100000 + i}" for i in range(n)])
    order_id = np.array([f"OR{200000 + i}" for i in range(n)])
    customer_id = np.array([f"CU{rng.integers(1, 25000)}" for _ in range(n)])

    df = pd.DataFrame({
        "Shipment_ID": shipment_id,
        "Order_ID": order_id,
        "Customer_ID": customer_id,
        "Origin_City": origin_city,
        "Origin_Region": origin_region,
        "Destination_City": dest_city,
        "Destination_Region": dest_region,
        "Origin_Warehouse": origin_warehouse,
        "Destination_Warehouse": dest_warehouse,
        "Shipping_Mode": shipping_mode,
        "Carrier": carrier,
        "Service_Type": service_type,
        "Package_Type": package_type,
        "Package_Weight": np.round(package_weight, 2),
        "Package_Volume": np.round(package_volume, 3),
        "Order_Date": order_date,
        "Pickup_Date": pickup_date,
        "Expected_Delivery_Date": expected_delivery_date,
        "Actual_Delivery_Date": actual_delivery_date,
        "Promised_Delivery_Days": promised_delivery_days,
        "Actual_Delivery_Days": np.round(actual_delivery_days, 2),
        "Distance_KM": np.round(distance_km, 1),
        "Warehouse_Processing_Time_Hours": np.round(warehouse_processing_time, 2),
        "Pickup_Delay_Hours": np.round(pickup_delay_hours, 2),
        "Transit_Time_Hours": np.round(transit_time_hours, 2),
        "Sorting_Time_Hours": np.round(sorting_time_hours, 2),
        "Last_Mile_Time_Hours": np.round(last_mile_time_hours, 2),
        "Number_of_Handoffs": number_of_handoffs,
        "Weather_Condition": weather_condition,
        "Traffic_Level": traffic_level,
        "Holiday_Flag": holiday_flag,
        "Weekend_Flag": weekend_flag,
        "Peak_Season_Flag": peak_season_flag,
        "Vehicle_Availability": np.round(vehicle_availability, 1),
        "Driver_Availability": np.round(driver_availability, 1),
        "Warehouse_Capacity_Utilization": np.round(warehouse_capacity_utilization, 1),
        "Customs_Clearance_Time_Hours": np.round(customs_clearance_time_hours, 2),
        "Payment_Status": payment_status,
        "Address_Quality": address_quality,
        "Customer_Priority": customer_priority,
        "Delay_Flag": delay_flag,
        "Delay_Hours": np.round(delay_hours, 2),
        "Delay_Category": delay_category,
        "Delay_Reason": delay_reason,
    })

    df = _inject_data_quality_issues(df, rng)
    return df


def _inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Introduce a small, controlled amount of missing values, duplicates,
    outliers, and inconsistent categorical spellings, to give the app's
    data-quality pipeline real work to do."""
    df = df.copy()
    n = len(df)

    # Missing values (~1.5% of a handful of columns)
    for col, frac in [
        ("Package_Weight", 0.01),
        ("Driver_Availability", 0.015),
        ("Address_Quality", 0.01),
        ("Weather_Condition", 0.008),
        ("Customs_Clearance_Time_Hours", 0.01),
    ]:
        idx = rng.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # Inconsistent categorical spellings
    inconsistent_map = {
        "Road": ["road", "ROAD", "Road "],
        "Air": ["air", "AIR"],
    }
    idx = rng.choice(n, size=int(n * 0.02), replace=False)
    for i in idx:
        val = df.at[i, "Shipping_Mode"]
        if val in inconsistent_map:
            df.at[i, "Shipping_Mode"] = rng.choice(inconsistent_map[val])

    carrier_variants = {"Carrier A": ["carrier a", "CARRIER A", "Carrier-A"]}
    idx = rng.choice(n, size=int(n * 0.015), replace=False)
    for i in idx:
        val = df.at[i, "Carrier"]
        if val in carrier_variants:
            df.at[i, "Carrier"] = rng.choice(carrier_variants[val])

    # Outliers: implausible package weight / distance / delay hours
    idx = rng.choice(n, size=int(n * 0.003), replace=False)
    df.loc[idx, "Package_Weight"] = df.loc[idx, "Package_Weight"] * rng.uniform(20, 40, len(idx))

    # Only ever inflate Delay_Hours for shipments that were actually delayed --
    # otherwise this would create the impossible state of an on-time shipment
    # (Delay_Flag == 0) with nonzero Delay_Hours.
    delayed_idx = df.index[df["Delay_Flag"] == 1].to_numpy()
    idx = rng.choice(delayed_idx, size=min(int(n * 0.002), len(delayed_idx)), replace=False)
    df.loc[idx, "Delay_Hours"] = df.loc[idx, "Delay_Hours"] + rng.uniform(500, 2000, len(idx))

    # Duplicate records (~0.5%)
    dup_idx = rng.choice(n, size=int(n * 0.005), replace=False)
    dup_rows = df.loc[dup_idx]
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Shuffle row order so duplicates aren't trivially adjacent
    df = df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = generate_dataset(120_000)
    out_path = "data/synthetic_logistics_data.csv"
    data.to_csv(out_path, index=False)
    print(f"Generated {len(data):,} records -> {out_path}")
    print(data.head())
    print("\nDelay rate:", data["Delay_Flag"].mean())
