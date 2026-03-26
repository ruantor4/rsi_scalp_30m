from __future__ import annotations

from typing import List
import pandas as pd

from src.core.logger import get_logger


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


def load_csv(path: str, timeframe: str) -> pd.DataFrame:
    logger.info(f"Load start | {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataValidationError(f"Erro ao ler CSV: {e}")

    df = _validate_schema(df)
    df = _normalize_types(df)
    df = _sort_and_deduplicate(df)
    df = _ensure_continuity(df, timeframe)
    _final_validation(df, timeframe)

    logger.info(f"Load done | rows={len(df)}")

    return df


def _validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    return df.copy()


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise DataValidationError("Nulls após coerção")

    # sanity check OHLC
    if (df["high"] < df["low"]).any():
        raise DataValidationError("High < Low detectado")

    if (df["open"] > df["high"]).any() or (df["open"] < df["low"]).any():
        raise DataValidationError("Open fora do range")

    if (df["close"] > df["high"]).any() or (df["close"] < df["low"]).any():
        raise DataValidationError("Close fora do range")

    return df


def _sort_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp", keep="last")

    if not df["timestamp"].is_monotonic_increasing:
        raise DataValidationError("Timestamp não monotônico")

    return df.reset_index(drop=True)


def _ensure_continuity(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    freq = _timeframe_to_freq(timeframe)

    full_index = pd.date_range(
        start=df["timestamp"].iloc[0],
        end=df["timestamp"].iloc[-1],
        freq=freq,
        tz="UTC",
    )

    df = df.set_index("timestamp")
    df["is_synthetic"] = False

    df = df.reindex(full_index)

    # garantir primeiro valor válido
    if df.iloc[0][["open", "high", "low", "close"]].isna().any():
        raise DataValidationError("Primeiro candle inválido após reindex")

    # 🔴 FIX: estava dentro do if (errado)
    synthetic_mask = df["open"].isna()

    df[["open", "high", "low", "close"]] = df[
        ["open", "high", "low", "close"]
    ].ffill()

    df.loc[synthetic_mask, "volume"] = 0.0
    df["volume"] = df["volume"].fillna(0.0)

    df.loc[synthetic_mask, "is_synthetic"] = True

    df = df.reset_index().rename(columns={"index": "timestamp"})

    df["is_synthetic"] = df["is_synthetic"].astype(bool)

    return df


def _final_validation(df: pd.DataFrame, timeframe: str) -> None:
    if df["timestamp"].duplicated().any():
        raise DataValidationError("Duplicatas após processamento")

    if not df["timestamp"].is_monotonic_increasing:
        raise DataValidationError("Timestamp não monotônico final")

    expected = pd.Timedelta(_timeframe_to_freq(timeframe))

    diffs = df["timestamp"].diff().dropna()

    if diffs.nunique() != 1 or diffs.iloc[0] != expected:
        raise DataValidationError("Dataset não contínuo")


def _timeframe_to_freq(tf: str) -> str:
    if tf.endswith("m"):
        return f"{int(tf[:-1])}min"
    if tf.endswith("h"):
        return f"{int(tf[:-1])}h"
    raise ValueError(f"Timeframe inválido: {tf}")