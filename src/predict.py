"""
Inference module for predictive maintenance, providing single-instance
and batch predictions with explainable risk assessment and failure mode diagnostics.
"""

import math
import os
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_MODEL_PATH,
    ALT_MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    MODEL_FEATURE_COLUMNS,
    OVERSTRAIN_LIMITS,
    HDF_TEMP_DIFF_MIN,
    HDF_ROTATIONAL_SPEED_MAX,
    POWER_MIN,
    POWER_MAX,
    TOOL_WEAR_CRITICAL_MIN,
)


class PredictiveMaintenanceModel:
    """
    Wrapper for loading trained ML models and running inference
    with physics-based failure mode diagnostics.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        encoder_path: Optional[str] = None,
    ):
        self.model_path = model_path or self._resolve_model_path()
        self.scaler_path = scaler_path or str(SCALER_PATH)
        self.encoder_path = encoder_path or str(ENCODER_PATH)

        self.model = None
        self.scaler = None
        self.encoder = None

        self._load_artifacts()

    def _resolve_model_path(self) -> str:
        """Find the model file in root or models/ directory."""
        if os.path.exists(DEFAULT_MODEL_PATH):
            return str(DEFAULT_MODEL_PATH)
        elif os.path.exists(ALT_MODEL_PATH):
            return str(ALT_MODEL_PATH)
        return str(DEFAULT_MODEL_PATH)

    def _load_artifacts(self):
        """Load pickled model, scaler, and label encoder."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found at: {self.scaler_path}")
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Encoder file not found at: {self.encoder_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.encoder = joblib.load(self.encoder_path)

    @staticmethod
    def calculate_physics_diagnostics(
        machine_type: str,
        air_temp: float,
        process_temp: float,
        rot_speed: float,
        torque: float,
        tool_wear: float,
    ) -> Dict[str, Any]:
        """
        Evaluate physical operating conditions against known failure criteria:
        - HDF (Heat Dissipation): Delta T < 8.6 K and Speed < 1380 rpm
        - PWF (Power Failure): Power < 3500 W or > 9000 W
        - OSF (Overstrain Failure): Tool wear * Torque exceeds limit
        - TWF (Tool Wear Failure): Tool wear >= 200 min
        """
        temp_diff = process_temp - air_temp
        power_watts = torque * (rot_speed * (2 * math.pi / 60))
        overstrain_metric = tool_wear * torque
        overstrain_threshold = OVERSTRAIN_LIMITS.get(machine_type, 11000)

        diagnostics = {
            "temp_diff_k": round(temp_diff, 2),
            "power_watts": round(power_watts, 1),
            "overstrain_metric": round(overstrain_metric, 1),
            "overstrain_threshold": overstrain_threshold,
            "hdf_risk": temp_diff < HDF_TEMP_DIFF_MIN and rot_speed < HDF_ROTATIONAL_SPEED_MAX,
            "pwf_risk": power_watts < POWER_MIN or power_watts > POWER_MAX,
            "osf_risk": overstrain_metric > overstrain_threshold,
            "twf_risk": tool_wear >= TOOL_WEAR_CRITICAL_MIN,
        }

        potential_causes = []
        if diagnostics["hdf_risk"]:
            potential_causes.append("Heat Dissipation Failure (HDF) - Inadequate cooling with low air flow")
        if diagnostics["pwf_risk"]:
            potential_causes.append(
                f"Power Failure (PWF) - Motor operating outside safe power envelope ({power_watts:.0f} W)"
            )
        if diagnostics["osf_risk"]:
            potential_causes.append(
                f"Overstrain Failure (OSF) - High torque & tool wear product ({overstrain_metric:.0f} > {overstrain_threshold})"
            )
        if diagnostics["twf_risk"]:
            potential_causes.append(
                f"Tool Wear Failure (TWF) - Tool wear exceeds critical threshold ({tool_wear:.0f} min >= 200 min)"
            )

        diagnostics["potential_causes"] = potential_causes
        return diagnostics

    def prepare_input_dataframe(
        self,
        machine_type: str,
        air_temp: float,
        process_temp: float,
        rot_speed: float,
        torque: float,
        tool_wear: float,
        twf: int = 0,
        hdf: int = 0,
        pwf: int = 0,
        osf: int = 0,
        rnf: int = 0,
    ) -> pd.DataFrame:
        """
        Encode inputs and construct properly ordered DataFrame matching scaler expectations.
        """
        type_encoded = self.encoder.transform([machine_type])[0]

        df_input = pd.DataFrame(
            {
                "Type": [type_encoded],
                "Air temperature [K]": [air_temp],
                "Process temperature [K]": [process_temp],
                "Rotational speed [rpm]": [rot_speed],
                "Torque [Nm]": [torque],
                "Tool wear [min]": [tool_wear],
                "TWF": [twf],
                "HDF": [hdf],
                "PWF": [pwf],
                "OSF": [osf],
                "RNF": [rnf],
            }
        )

        return df_input[MODEL_FEATURE_COLUMNS]

    def predict_single(
        self,
        machine_type: str,
        air_temp: float,
        process_temp: float,
        rot_speed: float,
        torque: float,
        tool_wear: float,
    ) -> Dict[str, Any]:
        """
        Run complete prediction pipeline on a single machine's parameters.
        Returns predicted class, probability, risk tier, and physics root-cause diagnosis.
        """
        diagnostics = self.calculate_physics_diagnostics(
            machine_type=machine_type,
            air_temp=air_temp,
            process_temp=process_temp,
            rot_speed=rot_speed,
            torque=torque,
            tool_wear=tool_wear,
        )

        input_df = self.prepare_input_dataframe(
            machine_type=machine_type,
            air_temp=air_temp,
            process_temp=process_temp,
            rot_speed=rot_speed,
            torque=torque,
            tool_wear=tool_wear,
            twf=1 if diagnostics["twf_risk"] else 0,
            hdf=1 if diagnostics["hdf_risk"] else 0,
            pwf=1 if diagnostics["pwf_risk"] else 0,
            osf=1 if diagnostics["osf_risk"] else 0,
            rnf=0,
        )

        input_scaled = self.scaler.transform(input_df)

        prediction = int(self.model.predict(input_scaled)[0])

        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(input_scaled)[0][1])
        else:
            probability = 1.0 if prediction == 1 else 0.0

        # Determine risk tier
        if probability < 0.25:
            risk_level = "Low"
            risk_color = "green"
        elif probability < 0.50:
            risk_level = "Moderate"
            risk_color = "yellow"
        elif probability < 0.75:
            risk_level = "High"
            risk_color = "orange"
        else:
            risk_level = "Critical"
            risk_color = "red"

        return {
            "prediction": prediction,
            "failure_predicted": bool(prediction == 1),
            "failure_probability": probability,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "diagnostics": diagnostics,
            "input_features": {
                "Type": machine_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rot_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            },
        }

    def predict_batch(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        """
        Run predictions over a batch DataFrame of machine sensor readings.
        """
        results = []
        df_clean = df_batch.copy()

        # Clean column names
        df_clean.columns = df_clean.columns.str.replace("^\ufeff", "", regex=True).str.strip()

        for idx, row in df_clean.iterrows():
            m_type = str(row.get("Type", "L"))
            air_t = float(row.get("Air temperature [K]", 300.0))
            proc_t = float(row.get("Process temperature [K]", 310.0))
            rot_s = float(row.get("Rotational speed [rpm]", 1500.0))
            torq = float(row.get("Torque [Nm]", 40.0))
            wear = float(row.get("Tool wear [min]", 0.0))

            pred_res = self.predict_single(
                machine_type=m_type,
                air_temp=air_t,
                process_temp=proc_t,
                rot_speed=rot_s,
                torque=torq,
                tool_wear=wear,
            )

            causes = ", ".join(pred_res["diagnostics"]["potential_causes"]) or "None"

            results.append(
                {
                    "Failure Predicted": "YES" if pred_res["failure_predicted"] else "NO",
                    "Failure Probability (%)": round(pred_res["failure_probability"] * 100, 2),
                    "Risk Level": pred_res["risk_level"],
                    "Potential Failure Causes": causes,
                }
            )

        res_df = pd.DataFrame(results)
        return pd.concat([df_clean.reset_index(drop=True), res_df], axis=1)
