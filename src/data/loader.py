from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np

from src.core.logger import get_logger
from src.config.settings import (
    RAW_DIR,
    INTERVAL,
)

logger = get_logger("loader")


class DataValidationError(Exception):
    pass


REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def load_csv(
    symbol: str,
    timeframe: str = INTERVAL,
) -> pd.DataFrame:

    path = RAW_DIR / f"{symbol.lower()}_{timeframe}.csv"

    logger.info(f"Load start | {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataValidationError(f"Erro ao ler CSV: {e}")

    df = _validate_schema(df)
    df = _normalize_types(df)
    df = _sort_and_deduplicate(df)
    df = _detect_gaps(df, timeframe)
    _final_validation(df)

    logger.info(
        "Load done | rows=%d | valid=%d | invalid=%d",
        len(df),
        df["is_valid"].sum(),
        (~df["is_valid"]).sum(),
    )

    return df


# ====================================================
# INTERNALS
# ====================================================


def _validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    return df.copy()


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ====================================================
    # TIMESTAMP (COESO COM FETCHER)
    # ====================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    invalid_ts = df["timestamp"].isna()

    if invalid_ts.any():
        logger.warning(
            "Timestamps inválidos removidos: %d",
            invalid_ts.sum(),
        )
        df = df[~invalid_ts].reset_index(drop=True)

    # ====================================================
    # NUMÉRICOS
    # ====================================================

    numeric_cols = ["open", "high", "low", "close", "volume"]

    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # ====================================================
    # QUALIDADE
    # ====================================================

    df["is_valid"] = True

    null_mask = df[REQUIRED_COLUMNS].isnull().any(axis=1)
    df.loc[null_mask, "is_valid"] = False

    # sanity OHLC
    invalid_ohlc = (
        (df["high"] < df["low"]) |
        (df["open"] > df["high"]) |
        (df["open"] < df["low"]) |
        (df["close"] > df["high"]) |
        (df["close"] < df["low"])
    )

    df.loc[invalid_ohlc, "is_valid"] = False

    # ====================================================
    # BASE FEATURES
    # ====================================================

    df["return"] = df["close"].pct_change()

    ratio = df["close"] / df["close"].shift(1)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio[ratio <= 0] = np.nan

    df["log_return"] = np.log(ratio)

    df["volatility"] = df["return"].rolling(20).std()

    return df


def _sort_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp")

    df = df.drop_duplicates(
        subset="timestamp",
        keep="last",
    )

    if not df["timestamp"].is_monotonic_increasing:
        raise DataValidationError("Timestamp não monotônico")

    return df.reset_index(drop=True)


def _detect_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = df.copy()

    expected = pd.Timedelta(_timeframe_to_freq(timeframe))

    df["delta"] = df["timestamp"].diff()

    tolerance = pd.Timedelta(seconds=1)

    df["gap"] = (df["delta"] - expected).abs() > tolerance

    df.loc[0, "gap"] = False

    gap_count = df["gap"].sum()

    if gap_count > 0:
        logger.warning(
            "Gaps detectados: %d | expected=%s",
            gap_count,
            expected,
        )

    df["has_gap"] = df["gap"]

    return df.drop(columns=["delta"])


def _final_validation(df: pd.DataFrame) -> None:
    if df["timestamp"].duplicated().any():
        raise DataValidationError("Duplicatas após processamento")

    if not df["timestamp"].is_monotonic_increasing:
        raise DataValidationError("Timestamp não monotônico final")

    if "is_valid" not in df.columns:
        raise DataValidationError("Coluna is_valid ausente")

    invalid_count = (~df["is_valid"]).sum()

    if invalid_count > 0:
        logger.warning("Linhas inválidas: %d", invalid_count)


def _timeframe_to_freq(tf: str) -> str:
    if tf.endswith("m"):
        return f"{int(tf[:-1])}min"
    if tf.endswith("h"):
        return f"{int(tf[:-1])}h"
    raise ValueError(f"Timeframe inválido: {tf}")