import simpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# 1. LOAD DATA
# =====================================================
print("\nLoading data...")

FILE_NAME = "gemigedara_sep2025_report.xlsx"
df = pd.read_excel(FILE_NAME)

df["order_placement_timestamp"] = pd.to_datetime(df["order_placement_timestamp"], errors="coerce")

# Aggregate per order (max for prep/courier/delivery to account for multiple items)
order_df = df.groupby("order_id").agg({
    "order_placement_timestamp": "min",
    "preparation_time_min": "max",
    "courier_wait_time_min": "max",
    "delivery_time_min": "max"
}).reset_index()

order_df = order_df.sort_values("order_placement_timestamp")

# Arrival time in minutes since start
start_time = order_df["order_placement_timestamp"].min()
order_df["Arrival_Min"] = (order_df["order_placement_timestamp"] - start_time).dt.total_seconds() / 60

# Day number
order_df["Day"] = (order_df["order_placement_timestamp"].dt.floor('D') - start_time.floor('D')).dt.days + 1

# Hour of day (0-23)
order_df["Hour"] = order_df["order_placement_timestamp"].dt.hour

# Create results folders
Path("results/historical").mkdir(parents=True, exist_ok=True)
Path("results/future").mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. USER INPUTS
# =====================================================
print("\n===== SYSTEM CONFIGURATION =====")
print("Provide the number of kitchen servers and couriers for simulation.")
KITCHENS = int(input("Kitchen servers: "))
COURIERS = int(input("Couriers: "))

# =====================================================
# 3. SIMULATION FUNCTION
# =====================================================
def simulate(data, kitchens, couriers, sim_time=None):
    if data.empty:
        return pd.DataFrame(columns=["Day","Hour","prep_queue","courier_queue","prep_time","delivery_time","total_time"])
    
    env = simpy.Environment()
    kitchen = simpy.Resource(env, kitchens)
    courier = simpy.Resource(env, couriers)
    records = []

    def order(env, arrival, prep, cwait, deliv, day, hour):
        yield env.timeout(max(arrival - env.now, 0))
        # Kitchen process
        with kitchen.request() as r:
            yield r
            kitchen_wait = env.now - arrival
            yield env.timeout(prep)
        # Courier process
        with courier.request() as r:
            yield r
            courier_wait = env.now - (arrival + prep)
            yield env.timeout(cwait + deliv)
        total_time = env.now - arrival
        records.append({
            "Day": day,
            "Hour": hour,
            "prep_queue": kitchen_wait,
            "courier_queue": courier_wait,
            "prep_time": prep,
            "delivery_time": deliv,
            "total_time": total_time
        })

    for _, r in data.iterrows():
        env.process(order(
            env,
            r["Arrival_Min"],
            r["preparation_time_min"] or 20,
            r["courier_wait_time_min"] or 10,
            r["delivery_time_min"] or 15,
            r["Day"],
            r["Hour"]
        ))

    if sim_time is None:
        last_day = data["Day"].max()
        sim_time = int(last_day * 24 * 60 + 60)

    env.run(until=sim_time)
    return pd.DataFrame(records)

# =====================================================
# 4. RUN HISTORICAL SIMULATION
# =====================================================
print("\nRunning historical simulation...")
hist_results = simulate(order_df, KITCHENS, COURIERS)

# Aggregate daily metrics
hist_perf_df = hist_results.groupby("Day").agg(
    Orders=("total_time","count"),
    Avg_Kitchen_Wait=("prep_queue","mean"),
    Avg_Courier_Wait=("courier_queue","mean"),
    Avg_Prep_Time=("prep_time","mean"),
    Avg_Delivery=("delivery_time","mean"),
    Avg_Total=("total_time","mean")
).reset_index()

hist_perf_df.to_csv("results/historical/2_week_historical_performance.csv", index=False)
print("\n===== 2-WEEK HISTORICAL PERFORMANCE =====")
print(hist_perf_df)

# =====================================================
# 5. SAVE HISTORICAL GRAPHS
# =====================================================
metrics = ["prep_queue", "courier_queue", "prep_time", "delivery_time", "total_time"]
metric_titles = {
    "prep_queue": "Kitchen Queue Wait Time",
    "courier_queue": "Courier Queue Wait Time",
    "prep_time": "Preparation Time",
    "delivery_time": "Delivery Time",
    "total_time": "Total Fulfillment Time"
}

# Daily histograms
for day in hist_results["Day"].unique():
    day_data = hist_results[hist_results["Day"] == day]
    for metric in metrics:
        if not day_data.empty:
            plt.figure()
            plt.hist(day_data[metric], bins=30, color="skyblue", edgecolor="black")
            plt.title(f"{metric_titles[metric]} - Day {day}")
            plt.xlabel("Minutes")
            plt.ylabel("Orders")
            plt.savefig(f"results/historical/day_{day}_{metric}.png")
            plt.close()

# Overall 2-week histograms
for metric in metrics:
    if not hist_results.empty:
        plt.figure()
        plt.hist(hist_results[metric], bins=50, color="lightgreen", edgecolor="black")
        plt.title(f"{metric_titles[metric]} - 2 Weeks")
        plt.xlabel("Minutes")
        plt.ylabel("Orders")
        plt.savefig(f"results/historical/total_2weeks_{metric}.png")
        plt.close()

# =====================================================
# 6. HOURLY HISTORICAL BREAKDOWN (NEW)
# =====================================================
hourly_perf_df = hist_results.groupby(["Day","Hour"]).agg(
    Orders=("total_time","count"),
    Avg_Kitchen_Wait=("prep_queue","mean"),
    Avg_Courier_Wait=("courier_queue","mean"),
    Avg_Prep_Time=("prep_time","mean"),
    Avg_Delivery=("delivery_time","mean"),
    Avg_Total=("total_time","mean")
).reset_index()
hourly_perf_df.to_csv("results/historical/hourly_historical_performance.csv", index=False)

# Line plots: average wait times per hour
plt.figure(figsize=(12,6))
for day in sorted(hist_results["Day"].unique()):
    day_data = hourly_perf_df[hourly_perf_df["Day"]==day]
    plt.plot(day_data["Hour"], day_data["Avg_Kitchen_Wait"], label=f"Day {day}")
plt.xlabel("Hour")
plt.ylabel("Avg Kitchen Wait (min)")
plt.title("Hourly Average Kitchen Wait")
plt.legend()
plt.grid(True)
plt.savefig("results/historical/hourly_avg_kitchen_wait.png")
plt.close()

plt.figure(figsize=(12,6))
for day in sorted(hist_results["Day"].unique()):
    day_data = hourly_perf_df[hourly_perf_df["Day"]==day]
    plt.plot(day_data["Hour"], day_data["Avg_Courier_Wait"], label=f"Day {day}")
plt.xlabel("Hour")
plt.ylabel("Avg Courier Wait (min)")
plt.title("Hourly Average Courier Wait")
plt.legend()
plt.grid(True)
plt.savefig("results/historical/hourly_avg_courier_wait.png")
plt.close()

# =====================================================
# 7. FUTURE SCENARIO INPUTS
# =====================================================
print("\n===== FUTURE SCENARIO INPUT =====")
FUTURE_ORDERS = int(input("Enter expected orders per day: "))
FUTURE_KITCHENS = int(input("Enter future kitchen servers: "))
FUTURE_COURIERS = int(input("Enter future couriers: "))

print(f"\nFuture simulation: {FUTURE_ORDERS} orders per day over 14 hours")
RATE_PER_HOUR = FUTURE_ORDERS / 14
print(f"Average expected arrival rate: {RATE_PER_HOUR:.2f} orders per hour")

prep_dist = hist_results["prep_time"].dropna()
courier_dist = hist_results["courier_queue"].dropna()
delivery_dist = hist_results["delivery_time"].dropna()

# =====================================================
# 8. FUTURE SIMULATION FUNCTION (deterministic arrivals)
# =====================================================
def future_sim(future_orders, kitchens, couriers, hours=14):
    env = simpy.Environment()
    kitchen = simpy.Resource(env, kitchens)
    courier = simpy.Resource(env, couriers)
    q_p, q_c, p_t, d_t, tot = [], [], [], [], []

    interval = (hours * 60) / future_orders
    arrival_times = np.arange(interval, hours*60 + interval, interval)[:future_orders]

    def order(env, arrival):
        yield env.timeout(arrival)
        prep = prep_dist.sample(1).iloc[0]
        cwait = courier_dist.sample(1).iloc[0]
        deliv = delivery_dist.sample(1).iloc[0]

        with kitchen.request() as r:
            yield r
            q_p.append(env.now - arrival)
            p_t.append(prep)
            yield env.timeout(prep)
        with courier.request() as r:
            yield r
            q_c.append(env.now - (arrival + prep))
            d_t.append(deliv)
            yield env.timeout(cwait + deliv)
        tot.append(env.now - arrival)

    for a in arrival_times:
        env.process(order(env, a))

    env.run(until=hours*60)

    min_len = min(len(q_p), len(q_c), len(p_t), len(d_t), len(tot))
    df = pd.DataFrame({
        "prep_queue": q_p[:min_len],
        "courier_queue": q_c[:min_len],
        "prep_time": p_t[:min_len],
        "delivery_time": d_t[:min_len],
        "total_time": tot[:min_len]
    })

    for metric in metrics:
        if metric in df.columns and not df.empty:
            plt.figure()
            plt.hist(df[metric], bins=50, color="salmon", edgecolor="black")
            plt.title(f"{metric_titles[metric]} - Future Prediction")
            plt.xlabel("Minutes")
            plt.ylabel("Orders")
            plt.savefig(f"results/future/future_{metric}.png")
            plt.close()

    return df

future_results = future_sim(FUTURE_ORDERS, FUTURE_KITCHENS, FUTURE_COURIERS)

# =====================================================
# 9. FUTURE PERFORMANCE OUTPUT
# =====================================================
future_perf = {
    "Orders": len(future_results),
    "Avg_Kitchen_Wait": future_results["prep_queue"].mean(),
    "Avg_Courier_Wait": future_results["courier_queue"].mean(),
    "Avg_Prep_Time": future_results["prep_time"].mean(),
    "Avg_Delivery": future_results["delivery_time"].mean(),
    "Avg_Total": future_results["total_time"].mean()
}
future_perf_df = pd.DataFrame([future_perf])
future_perf_df.to_csv("results/future/future_performance_prediction.csv", index=False)

print("\n===== FUTURE PERFORMANCE PREDICTION =====")
print(future_perf_df)

# =====================================================
# 10. FUTURE HOURLY PREDICTION (NEW)
# =====================================================
def future_sim_hourly(future_orders, kitchens, couriers, hours=14):
    env = simpy.Environment()
    kitchen = simpy.Resource(env, kitchens)
    courier = simpy.Resource(env, couriers)
    q_p, q_c, p_t, d_t, tot, hours_list = [], [], [], [], [], []

    # Historical hourly distribution
    hourly_counts = hist_results.groupby("Hour")["total_time"].count()
    hourly_pct = hourly_counts / hourly_counts.sum()

    orders_per_hour = (hourly_pct * future_orders).round().astype(int)

    current_min = 0
    def order_process(env, arrival_min, hour):
        yield env.timeout(arrival_min)
        prep = prep_dist.sample(1).iloc[0]
        cwait = courier_dist.sample(1).iloc[0]
        deliv = delivery_dist.sample(1).iloc[0]

        with kitchen.request() as r:
            yield r
            q_p.append(env.now - arrival_min)
            p_t.append(prep)
            yield env.timeout(prep)
        with courier.request() as r:
            yield r
            q_c.append(env.now - (arrival_min + prep))
            d_t.append(deliv)
            yield env.timeout(cwait + deliv)
        tot.append(env.now - arrival_min)
        hours_list.append(hour)

    for hour in range(14):
        n_orders = orders_per_hour.get(hour, 0)
        if n_orders == 0:
            continue
        interval = 60 / n_orders
        for i in range(n_orders):
            current_min += interval
            env.process(order_process(env, current_min, hour))

    env.run(until=hours*60)

    min_len = min(len(q_p), len(q_c), len(p_t), len(d_t), len(tot))
    df = pd.DataFrame({
        "Hour": hours_list[:min_len],
        "prep_queue": q_p[:min_len],
        "courier_queue": q_c[:min_len],
        "prep_time": p_t[:min_len],
        "delivery_time": d_t[:min_len],
        "total_time": tot[:min_len]
    })
    return df

future_hourly_results = future_sim_hourly(FUTURE_ORDERS, FUTURE_KITCHENS, FUTURE_COURIERS)

future_hourly_perf = future_hourly_results.groupby("Hour").agg(
    Orders=("total_time","count"),
    Avg_Kitchen_Wait=("prep_queue","mean"),
    Avg_Courier_Wait=("courier_queue","mean"),
    Avg_Prep_Time=("prep_time","mean"),
    Avg_Delivery=("delivery_time","mean"),
    Avg_Total=("total_time","mean")
).reset_index()
future_hourly_perf.to_csv("results/future/future_hourly_performance.csv", index=False)

# Hourly line plots for future scenario
plt.figure(figsize=(12,6))
plt.plot(future_hourly_perf["Hour"], future_hourly_perf["Avg_Kitchen_Wait"], marker="o")
plt.xlabel("Hour")
plt.ylabel("Avg Kitchen Wait (min)")
plt.title("Future Hourly Average Kitchen Wait")
plt.grid(True)
plt.savefig("results/future/future_hourly_avg_kitchen_wait.png")
plt.close()

plt.figure(figsize=(12,6))
plt.plot(future_hourly_perf["Hour"], future_hourly_perf["Avg_Courier_Wait"], marker="o", color="orange")
plt.xlabel("Hour")
plt.ylabel("Avg Courier Wait (min)")
plt.title("Future Hourly Average Courier Wait")
plt.grid(True)
plt.savefig("results/future/future_hourly_avg_courier_wait.png")
plt.close()
