"""
Data preprocessing, cleaning, outlier handling, and feature transformation pipeline.
"""

import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.config import (
    DEFAULT_DATASET_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FAILURE_MODE_COLUMNS,
    TARGET_COLUMN,
    COLUMNS_TO_DROP,
    MODEL_FEATURE_COLUMNS,
)

logger = logging.getLogger(__name__)


def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the AI4I 2020 predictive maintenance dataset from disk.
    Handles UTF-8 BOM if present.
    """
    path = file_path or DEFAULT_DATASET_PATH
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    # Clean potential BOM in column names
    df.columns = df.columns.str.replace("^\ufeff", "", regex=True).str.strip()
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop non-predictive metadata identifier columns like UDI and Product ID.
    """
    df_cleaned = df.copy()
    existing_drops = [col for col in COLUMNS_TO_DROP if col in df_cleaned.columns]
    if existing_drops:
        df_cleaned.drop(columns=existing_drops, inplace=True)
    return df_cleaned


def detect_outliers_iqr(
    df: pd.DataFrame, features: List[str]
) -> dict:
    """
    Detect outliers in numerical features using the 1.5 * IQR method.
    Returns a dictionary of outlier counts per feature.
    """
    outlier_counts = {}
    for col in features:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_counts[col] = len(outliers)
    return outlier_counts


def cap_outliers_iqr(
    df: pd.DataFrame, features: List[str]
) -> pd.DataFrame:
    """
    Cap outliers in numerical features using 1.5 * IQR upper and lower bounds.
    """
    df_capped = df.copy()
    for col in features:
        if col in df_capped.columns:
            q1 = df_capped[col].quantile(0.25)
            q3 = df_capped[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_capped[col] = np.clip(df_capped[col], lower_bound, upper_bound)
    return df_capped


def encode_type(
    df: pd.DataFrame, encoder: Optional[LabelEncoder] = None
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode the 'Type' categorical feature ('L', 'M', 'H') using LabelEncoder.
    """
    df_encoded = df.copy()
    if encoder is None:
        encoder = LabelEncoder()
        df_encoded["Type"] = encoder.fit_transform(df_encoded["Type"])
    else:
        df_encoded["Type"] = encoder.transform(df_encoded["Type"])
    return df_encoded, encoder


def prepare_training_data(
    file_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True,
    cap_outliers: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    End-to-end data preparation pipeline:
    1. Loads dataset
    2. Drops identifiers
    3. Encodes 'Type' column
    4. Optionally caps outliers via IQR
    5. Applies SMOTE if requested
    6. Splits into stratified train and test sets
    7. Scales numerical features using StandardScaler
    """
    df = load_dataset(file_path)
    df = clean_dataframe(df)

    # Encode categorical Type column
    df, le = encode_type(df)

    # Cap outliers
    if cap_outliers:
        df = cap_outliers_iqr(df, NUMERICAL_FEATURES)

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Reorder columns to match standard order
    cols_order = [c for c in MODEL_FEATURE_COLUMNS if c in X.columns]
    X = X[cols_order]

    # Handle class imbalance using SMOTE
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    else:
        X_resampled, y_resampled = X, y

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=test_size,
        random_state=random_state,
        stratify=y_resampled,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le
