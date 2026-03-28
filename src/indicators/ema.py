from __future__ import annotations

import logging
import pandas as pd


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger("ema")
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

class EMAComputationError(Exception):
    pass


# ============================================================
# CORE
# ============================================================

def compute_ema(
    df: pd.DataFrame,
    period: int,
    price_col: str = "close",
) -> pd.Series:

    if df.empty:
        raise EMAComputationError("DataFrame vazio")

    if price_col not in df.columns:
        raise EMAComputationError(f"Coluna não encontrada: {price_col}")

    if period <= 0:
        raise EMAComputationError("Período da EMA deve ser > 0")

    try:
        price = df[price_col]

        ema = price.ewm(
            span=period,
            adjust=False,
            min_periods=period
        ).mean()

        ema.name = f"ema_{period}"

        # ====================================================
        # LOG (opcional, leve)
        # ====================================================

        warmup = ema.isna().sum()

        logger.info(
            "EMA calculada | period=%d | warmup=%d",
            period,
            warmup,
        )

    except Exception as e:
        raise EMAComputationError(f"Erro ao calcular EMA: {e}")

    return ema