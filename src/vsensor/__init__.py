"""Virtual CO2 sensor: estimate CO2 and occupancy from cheap sensors, with honest uncertainty."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
