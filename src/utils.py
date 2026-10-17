"""
Utility functions for metrics reporting, visualization, and system helpers.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def get_sample_inputs() -> Dict[str, Dict[str, Any]]:
    """
    Predefined operational presets for interactive testing in Streamlit.
    """
    return {
        "Normal Operation (Safe)": {
            "Type": "L",
            "Air temperature [K]": 298.5,
            "Process temperature [K]": 308.7,
            "Rotational speed [rpm]": 1520,
            "Torque [Nm]": 38.0,
            "Tool wear [min]": 45,
            "description": "Standard operating parameters with low thermal load and moderate tool wear.",
        },
        "High Thermal Strain (HDF Risk)": {
            "Type": "M",
            "Air temperature [K]": 303.5,
            "Process temperature [K]": 308.0,
            "Rotational speed [rpm]": 1250,
            "Torque [Nm]": 52.0,
            "Tool wear [min]": 90,
            "description": "Temperature delta < 8.6 K combined with low rotational speed induces heat dissipation failure.",
        },
        "Extreme Tool Wear (TWF Risk)": {
            "Type": "H",
            "Air temperature [K]": 300.2,
            "Process temperature [K]": 310.4,
            "Rotational speed [rpm]": 1420,
            "Torque [Nm]": 48.0,
            "Tool wear [min]": 235,
            "description": "Tool wear surpasses the 200-240 min endurance threshold.",
        },
        "Overstrain Stress (OSF Risk)": {
            "Type": "L",
            "Air temperature [K]": 299.0,
            "Process temperature [K]": 309.5,
            "Rotational speed [rpm]": 1360,
            "Torque [Nm]": 68.0,
            "Tool wear [min]": 195,
            "description": "Tool wear x Torque product exceeds 11,000 min*Nm for Type L equipment.",
        },
        "Power Anomaly (PWF Risk)": {
            "Type": "M",
            "Air temperature [K]": 301.0,
            "Process temperature [K]": 311.2,
            "Rotational speed [rpm]": 2750,
            "Torque [Nm]": 36.0,
            "Tool wear [min]": 80,
            "description": "Motor mechanical power exceeds 9000 Watts envelope.",
        },
    }


def format_risk_badge(risk_level: str) -> str:
    """Return HTML formatted badge for risk level."""
    colors = {
        "Low": "#10b981",
        "Moderate": "#f59e0b",
        "High": "#f97316",
        "Critical": "#ef4444",
    }
    bg = colors.get(risk_level, "#6b7280")
    return (
        f"<span style='background-color: {bg}; color: white; padding: 4px 12px; "
        f"border-radius: 9999px; font-weight: 600; font-size: 0.875rem;'>{risk_level} Risk</span>"
    )
