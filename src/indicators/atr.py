from __future__ import annotations

import pandas as pd


class ATRComputationError(Exception):
    pass


def compute_atr(
    df: pd.DataFrame,
    period: int,
) -> pd.Series:
    """
    Calcula ATR (Average True Range)

    Args:
        df: DataFrame com colunas [high, low, close]
        period: período do ATR

    Returns:
        pd.Series com ATR
    """

    required = {"high", "low", "close"}
    missing = required - set(df.columns)

    if missing:
        raise ATRComputationError(f"Colunas ausentes: {missing}")

    if period <= 0:
        raise ATRComputationError("Período inválido")

    if df.empty:
        raise ATRComputationError("DataFrame vazio")

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(
        window=period,
        min_periods=period
    ).mean()

    atr.name = f"atr_{period}"

    return atr