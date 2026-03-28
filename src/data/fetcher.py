from __future__ import annotations

from pathlib import Path
import time
from typing import List, Optional

import requests
import pandas as pd

from src.core.logger import get_logger
from src.config.settings import (
    INTERVAL,
    LIMIT,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    BACKOFF_BASE,
    FETCH_SLEEP,
    BINANCE_BASE_URL,
)

DEFAULT_LIMIT = LIMIT

logger = get_logger("fetcher")


# ============================================================
# EXCEPTIONS
# ============================================================

class BinanceAPIError(Exception):
    pass


class BinanceRateLimitError(BinanceAPIError):
    pass


# ============================================================
# HELPERS
# ============================================================

def _interval_to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    raise ValueError(f"Intervalo não suportado: {interval}")


def align_timestamp(ts: int, interval_minutes: int) -> int:
    interval_ms = interval_minutes * 60 * 1000
    return ts - (ts % interval_ms)


# ============================================================
# API REQUEST
# ============================================================

def _request_klines(
    symbol: str,
    interval: str,
    start_ts: int,
    end_ts: Optional[int],
    limit: int,
) -> List:

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ts,
        "limit": limit,
    }

    if end_ts is not None:
        params["endTime"] = end_ts

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                BINANCE_BASE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()

                if not isinstance(data, list):
                    raise BinanceAPIError("Resposta inválida")

                return data or []

            if response.status_code == 429:
                raise BinanceRateLimitError("Rate limit")

            raise BinanceAPIError(f"{response.status_code} - {response.text}")

        except (BinanceRateLimitError, requests.RequestException) as e:
            sleep = BACKOFF_BASE * (2 ** attempt)
            logger.warning(f"{e} | retry={attempt} sleep={sleep:.2f}s")
            time.sleep(sleep)

    raise BinanceAPIError("Falha após retries")


# ============================================================
# FETCH MAIN
# ============================================================

def fetch_all_klines(
    symbol: str,
    interval: str = INTERVAL,
    start_ts: int = 0,
    end_ts: Optional[int] = None,
    limit: int = DEFAULT_LIMIT,
    sleep: float = FETCH_SLEEP,
) -> pd.DataFrame:

    if start_ts == 0:
        raise ValueError("start_ts não pode ser 0")

    logger.info(f"Fetch start | {symbol} {interval}")

    interval_min = _interval_to_minutes(interval)
    interval_ms = interval_min * 60 * 1000

    start_ts = align_timestamp(start_ts, interval_min)

    if end_ts is not None:
        end_ts = align_timestamp(end_ts, interval_min)

    all_rows: List = []
    current = start_ts

    while True:
        batch = _request_klines(
            symbol=symbol,
            interval=interval,
            start_ts=current,
            end_ts=end_ts,
            limit=limit,
        )

        if not batch:
            logger.warning("Batch vazio - encerrando")
            break

        all_rows.extend(batch)

        last_open = batch[-1][0]

        if last_open <= current:
            logger.warning("Loop detectado - interrompendo")
            break

        current = last_open + interval_ms

        logger.info(
            f"Batch={len(batch)} | Total={len(all_rows)} | next_ts={current}"
        )

        if len(batch) < limit:
            break

        if end_ts is not None and current > end_ts:
            break

        time.sleep(sleep)

    df = _normalize(all_rows)

    if df.empty:
        raise BinanceAPIError("Nenhum dado retornado da API")

    logger.info(f"Fetch done | rows={len(df)}")

    return df


# ============================================================
# NORMALIZE
# ============================================================

def _normalize(data: List) -> pd.DataFrame:

    if not data:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "_1", "_2", "_3", "_4", "_5", "_6",
        ],
    )

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # ====================================================
    # TIMESTAMP PADRONIZADO (CRÍTICO)
    # ====================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # ====================================================
    # NUMÉRICOS
    # ====================================================

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float64")

    # ====================================================
    # CLEAN
    # ====================================================

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# SAVE
# ============================================================

def save_csv(df: pd.DataFrame, symbol: str, interval: str) -> None:
    path = Path(f"data/raw/{symbol.lower()}_{interval}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        raise ValueError("Tentativa de salvar CSV vazio")

    df.to_csv(path, index=False)

    logger.info(f"CSV salvo: {path}")