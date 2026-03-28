from __future__ import annotations

import pandas as pd

from src.indicators.ema import compute_ema
from src.config.settings import EMA_FAST, EMA_SLOW


class RegimeComputationError(Exception):
    pass


def compute_regime(
    df: pd.DataFrame,
    fast_period: int = EMA_FAST,
    slow_period: int = EMA_SLOW,
) -> pd.Series:

    if df.empty:
        raise RegimeComputationError("DataFrame vazio")

    if fast_period <= 0 or slow_period <= 0:
        raise RegimeComputationError("Períodos inválidos")

    if fast_period >= slow_period:
        raise RegimeComputationError("fast_period deve ser < slow_period")

    ema_fast = compute_ema(df, period=fast_period)
    ema_slow = compute_ema(df, period=slow_period)

    regime = pd.Series(0, index=df.index, dtype="int8")

    regime[ema_fast > ema_slow] = 1
    regime[ema_fast < ema_slow] = -1

    regime.name = "regime"

    return regime