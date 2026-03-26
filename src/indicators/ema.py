from __future__ import annotations

import logging
import pandas as pd
import pandas_ta as ta


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
    """Erro ao calcular EMA."""


# ============================================================
# CORE
# ============================================================

def compute_ema(
    df: pd.DataFrame,
    period: int,
    price_col: str = "close",
) -> pd.Series:
    """
    Calcula EMA utilizando pandas-ta.

    Parâmetros:
        df: DataFrame com OHLCV
        period: período da EMA
        price_col: coluna de preço (default: close)

    Retorno:
        pd.Series com EMA

    Raises:
        EMAComputationError
    """

    logger.info(f"Calculando EMA | period={period}")

    if price_col not in df.columns:
        raise EMAComputationError(f"Coluna não encontrada: {price_col}")

    if period <= 0:
        raise EMAComputationError("Período da EMA deve ser > 0")

    try:
        ema = ta.ema(df[price_col], length=period)

        if ema is None or ema.isna().all():
            raise EMAComputationError("EMA retornou valores inválidos")

    except Exception as e:
        raise EMAComputationError(f"Erro ao calcular EMA: {e}")

    return ema