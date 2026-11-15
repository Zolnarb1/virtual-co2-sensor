"""Charts for the README.  Usage:  python -m vsensor.plots"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from vsensor import RESULTS_DIR


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def trace_chart():
    t = pd.read_csv(RESULTS_DIR / "co2_test_trace.csv", parse_dates=["date"])
    days = pd.read_csv(RESULTS_DIR / "co2_per_day.csv", parse_dates=["day"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(t["date"], t["lower"], t["upper"], color="#9ecae9", alpha=0.6,
                    label="virtual sensor: 90% interval")
    ax.plot(t["date"], t["pred"], color="#4C78A8", lw=0.8, label="virtual sensor estimate")
    ax.plot(t["date"], t["co2"], color="#222222", lw=1, label="physical CO₂ sensor")
    for _, d in days[days["flag"] != "ok"].iterrows():
        ax.axvspan(d["day"], d["day"] + pd.Timedelta(days=1), color="#E45756", alpha=0.12)
        ax.text(d["day"] + pd.Timedelta(hours=1), ax.get_ylim()[1] * 0.97,
                "fault flagged:\nroom empty, sensor\nreads > 1,000 ppm",
                color="#E45756", fontsize=9, va="top")
    ax.set_ylabel("CO₂ (ppm)")
    ax.set_title("Virtual CO₂ sensor (temperature, humidity, light, time) vs the real sensor, "
                 "on 8 unseen days", loc="left", fontsize=11.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "co2_virtual_vs_real.png", dpi=160)


def ablation_chart():
    o = pd.read_csv(RESULTS_DIR / "occupancy_ablation.csv")
    best = (o[o["model"] != "-"].sort_values("f1").groupby("sensors").tail(1)
            .sort_values("f1"))
    colors = ["#E45756" if s == "CO2 only" else "#F2A541" if s == "Light only" else "#4C78A8"
              for s in best["sensors"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(best["sensors"], best["f1"], color=colors)
    for i, (f1, acc) in enumerate(zip(best["f1"], best["accuracy"])):
        ax.text(f1 + 0.01, i, f"F1 {f1:.3f}  (acc {acc:.1%})", va="center", fontsize=9)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("F1 score on unseen days (best of logistic regression / gradient boosting)")
    ax.set_title("Which sensors do you need to detect occupancy?", loc="left", fontsize=11.5)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "occupancy_by_sensor.png", dpi=160)


if __name__ == "__main__":
    trace_chart()
    ablation_chart()
    print("charts saved to", RESULTS_DIR)
