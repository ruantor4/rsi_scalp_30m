from __future__ import annotations

import logging
import pandas as pd
import pandas_ta as ta


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
    """Erro ao calcular RSI."""


# ============================================================
# CORE
# ============================================================

def compute_rsi(
    df: pd.DataFrame,
    period: int,
    price_col: str = "close",
) -> pd.Series:
    """
    Calcula RSI utilizando pandas-ta.

    Parâmetros:
        df: DataFrame com OHLCV
        period: período do RSI
        price_col: coluna de preço (default: close)

    Retorno:
        pd.Series com RSI

    Raises:
        RSIComputationError
    """

    logger.info(f"Calculando RSI | period={period}")

    if price_col not in df.columns:
        raise RSIComputationError(f"Coluna não encontrada: {price_col}")

    if period <= 0:
        raise RSIComputationError("Período do RSI deve ser > 0")

    try:
        rsi = ta.rsi(df[price_col], length=period)

        if rsi is None or rsi.isna().all():
            raise RSIComputationError("RSI retornou valores inválidos")

    except Exception as e:
        raise RSIComputationError(f"Erro ao calcular RSI: {e}")

    return rsi