"""
Configuration parameters, file paths, and feature definitions for Predictive Maintenance.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Data file paths
DEFAULT_DATASET_PATH = BASE_DIR / "ai4i2020.csv"

# Model artifact paths
DEFAULT_MODEL_PATH = BASE_DIR / "predictive maintenance.pkl"
ALT_MODEL_PATH = MODELS_DIR / "predictive_maintenance.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "le.pkl"

# Feature definitions
CATEGORICAL_FEATURES = ["Type"]

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

TARGET_COLUMN = "Machine failure"

COLUMNS_TO_DROP = ["UDI", "Product ID", "\ufeffUDI"]

# Full ordered feature list expected by the trained model
MODEL_FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES + NUMERICAL_FEATURES + FAILURE_MODE_COLUMNS
)

# Overstrain threshold limits per machine type (Tool Wear * Torque in min*Nm)
OVERSTRAIN_LIMITS = {
    "L": 11000,
    "M": 12000,
    "H": 13000,
}

# Heat dissipation critical limits
HDF_TEMP_DIFF_MIN = 8.6  # Kelvin
HDF_ROTATIONAL_SPEED_MAX = 1380  # rpm

# Power failure limits (Watts)
POWER_MIN = 3500.0  # W
POWER_MAX = 9000.0  # W

# Tool wear failure warning limit
TOOL_WEAR_CRITICAL_MIN = 200  # minutes
