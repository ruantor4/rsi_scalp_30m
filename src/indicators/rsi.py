from __future__ import annotations

import logging
import pandas as pd


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger("rsi")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ============================================================
# EXCEPTIONS
# ============================================================

class RSIComputationError(Exception):
    pass


# ============================================================
# CORE
# ============================================================

def compute_rsi(
    df: pd.DataFrame,
    period: int,
    price_col: str = "close",
) -> pd.Series:

    if df.empty:
        raise RSIComputationError("DataFrame vazio")

    if price_col not in df.columns:
        raise RSIComputationError(f"Coluna não encontrada: {price_col}")

    if period <= 0:
        raise RSIComputationError("Período inválido")

    try:
        price = df[price_col]

        delta = price.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder smoothing
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        rsi.name = f"rsi_{period}"

        # ====================================================
        # LOG
        # ====================================================

        warmup = rsi.isna().sum()

        logger.info(
            "RSI calculado | period=%d | warmup=%d",
            period,
            warmup,
        )

    except Exception as e:
        raise RSIComputationError(f"Erro ao calcular RSI: {e}")

    return rsi