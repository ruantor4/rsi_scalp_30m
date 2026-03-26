from __future__ import annotations

from pathlib import Path
import time
from typing import List, Optional

import requests
import pandas as pd

from src.core.logger import get_logger


BASE_URL = "https://api.binance.com/api/v3/klines"

DEFAULT_LIMIT = 1000
REQUEST_TIMEOUT = 10

MAX_RETRIES = 5
BACKOFF_BASE = 0.5

logger = get_logger("fetcher")


class BinanceAPIError(Exception):
    pass


class BinanceRateLimitError(BinanceAPIError):
    pass


def _interval_to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    raise ValueError(f"Intervalo não suportado: {interval}")


def align_timestamp(ts: int, interval_minutes: int) -> int:
    interval_ms = interval_minutes * 60 * 1000
    return ts - (ts % interval_ms)


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
                BASE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                data = response.json()

                # 🔴 validação crítica de contrato
                if not isinstance(data, list):
                    raise BinanceAPIError("Resposta inválida da API (não é lista)")

                return data or []

            if response.status_code == 429:
                raise BinanceRateLimitError("Rate limit")

            raise BinanceAPIError(f"{response.status_code} - {response.text}")

        except (BinanceRateLimitError, requests.RequestException) as e:
            sleep = BACKOFF_BASE * (2 ** attempt)
            logger.warning(f"{e} | retry {sleep:.2f}s")
            time.sleep(sleep)

    raise BinanceAPIError("Falha após retries")


def fetch_all_klines(
    symbol: str,
    interval: str,
    start_ts: int,
    end_ts: Optional[int],
    limit: int = DEFAULT_LIMIT,
    sleep: float = 0.1,
) -> pd.DataFrame:

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
            break

        all_rows.extend(batch)

        last_open = batch[-1][0]

        if last_open <= current:
            logger.warning("Loop detectado - interrompendo")
            break

        current = last_open + interval_ms

        logger.info(f"Batch {len(batch)} | Total {len(all_rows)}")

        if len(batch) < limit:
            break

        if end_ts is not None and current > end_ts:
            break

        time.sleep(sleep)

    df = _normalize(all_rows)

    logger.info(f"Fetch done | rows={len(df)}")

    return df


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

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def save_csv(df: pd.DataFrame, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path_obj, index=False)

    logger.info(f"CSV salvo: {path_obj}")