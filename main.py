from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.logger import get_logger
from src.data.fetcher import fetch_all_klines, save_csv
from src.data.loader import load_csv
from src.config.settings import (
    SYMBOLS,
    INTERVAL,
    START_DATE,
    END_DATE,
    RAW_DIR,
    PROCESSED_DIR,
    MAX_INVALID_RATIO,
)

logger = get_logger("main")


# ============================================================
# HELPERS
# ============================================================

def to_milliseconds(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def interval_to_ms(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60 * 1000
    if interval.endswith("h"):
        return int(interval[:-1]) * 60 * 60 * 1000
    raise ValueError(f"Intervalo inválido: {interval}")


def read_raw_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        logger.warning(f"Erro ao ler CSV raw: {e}")
        return None


def get_last_timestamp(df: Optional[pd.DataFrame]) -> Optional[int]:
    if df is None or df.empty:
        return None

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    last_ts = ts.max()

    return int(last_ts.timestamp() * 1000)


def log_dataset_info(df: pd.DataFrame) -> None:
    logger.info(f"Rows: {len(df)}")
    logger.info(f"Range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    invalid_ratio = (~df["is_valid"]).mean()
    logger.info(f"Invalid ratio: {invalid_ratio:.6f}")

    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.info("\n" + buffer.getvalue())


# ============================================================
# PROCESS SYMBOL
# ============================================================

def process_symbol(symbol: str) -> pd.DataFrame:
    logger.info(f"=== PROCESSING {symbol} ===")

    raw_path = RAW_DIR / f"{symbol.lower()}_{INTERVAL}.csv"

    end_ts = to_milliseconds(END_DATE)
    interval_ms = interval_to_ms(INTERVAL)

    df_old = read_raw_csv(raw_path)
    last_ts = get_last_timestamp(df_old)

    if last_ts:
        logger.info("Modo incremental")
        start_ts = last_ts + interval_ms
    else:
        logger.info("Modo full load")
        start_ts = to_milliseconds(START_DATE)

    # ====================================================
    # FETCH
    # ====================================================

    if last_ts and start_ts >= end_ts:
        logger.info("Dataset já atualizado")
        df_new = pd.DataFrame()
    else:
        try:
            df_new = fetch_all_klines(
                symbol=symbol,
                interval=INTERVAL,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        except Exception as e:
            logger.warning(f"Fetch falhou {symbol}: {e}")
            df_new = pd.DataFrame()

    # ====================================================
    # MERGE RAW
    # ====================================================

    if df_old is not None and not df_new.empty:
        df_raw = pd.concat([df_old, df_new], ignore_index=True)
    elif df_old is not None:
        df_raw = df_old
    else:
        df_raw = df_new

    if df_raw is None or df_raw.empty:
        raise RuntimeError(f"{symbol} sem dados")

    df_raw = (
        df_raw
        .sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="last")
        .reset_index(drop=True)
    )

    save_csv(df_raw, symbol, INTERVAL)

    # ====================================================
    # LOAD (NORMALIZAÇÃO)
    # ====================================================

    df = load_csv(symbol=symbol, timeframe=INTERVAL)

    if df.empty:
        raise RuntimeError(f"{symbol} dataset vazio")

    invalid_ratio = (~df["is_valid"]).mean()

    if invalid_ratio > MAX_INVALID_RATIO:
        raise RuntimeError(
            f"{symbol} dataset ruim: {invalid_ratio:.6f}"
        )

    df["symbol"] = symbol

    return df


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    logger.info("=== START PIPELINE ===")

    all_dfs = []

    for symbol in SYMBOLS:
        df_symbol = process_symbol(symbol)
        all_dfs.append(df_symbol)

    if not all_dfs:
        raise RuntimeError("Nenhum dataset gerado")

    df_final = (
        pd.concat(all_dfs)
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    # ====================================================
    # SAVE
    # ====================================================

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # versionado
    versioned_path = PROCESSED_DIR / f"dataset_{version}.parquet"

    df_final.to_parquet(
        versioned_path,
        index=False,
        compression="snappy",
    )

    logger.info(f"Saved parquet: {versioned_path}")

    # 🔥 latest (ESSENCIAL)
    latest_path = PROCESSED_DIR / "dataset.parquet"

    df_final.to_parquet(
        latest_path,
        index=False,
        compression="snappy",
    )

    logger.info(f"Saved latest dataset: {latest_path}")

    # metadata
    metadata_path = versioned_path.with_suffix(".json")

    metadata = {
        "symbols": SYMBOLS,
        "interval": INTERVAL,
        "rows": len(df_final),
        "start": str(df_final["timestamp"].min()),
        "end": str(df_final["timestamp"].max()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    pd.DataFrame([metadata]).to_json(metadata_path, orient="records")

    logger.info(f"Metadata salvo: {metadata_path}")

    log_dataset_info(df_final)

    logger.info("=== PIPELINE OK ===")


if __name__ == "__main__":
    main()