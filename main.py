from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.logger import get_logger
from src.data.fetcher import fetch_all_klines, save_csv
from src.data.loader import load_csv


logger = get_logger("main")


# ============================================================
# CONFIG
# ============================================================

SYMBOL = "BTCUSDT"
INTERVAL = "30m"

START_DATE = "2022-01-01"
END_DATE   = "2025-01-01"

RAW_PATH = Path("data/raw/btcusdt_30m.csv")
PROCESSED_DIR = Path("data/processed")

MAX_SYNTHETIC_RATIO = 0.01


# ============================================================
# HELPERS
# ============================================================

def to_milliseconds(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def get_last_timestamp(path: Path) -> Optional[int]:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)

        if df.empty:
            return None

        last_ts = pd.to_datetime(df["timestamp"]).max()

        return int(last_ts.timestamp() * 1000)

    except Exception as e:
        logger.warning(f"Erro ao ler último timestamp: {e}")
        return None


def log_dataset_info(df: pd.DataFrame) -> None:
    logger.info(f"Rows: {len(df)}")
    logger.info(f"Range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    synthetic_ratio = df["is_synthetic"].mean()
    logger.info(f"Synthetic ratio: {synthetic_ratio:.6f}")

    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.info("\n" + buffer.getvalue())


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    logger.info("=== START PIPELINE ===")

    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Interval: {INTERVAL}")
    logger.info(f"Start: {START_DATE}")
    logger.info(f"End: {END_DATE}")

    try:
        end_ts = to_milliseconds(END_DATE)

        # ====================================================
        # INCREMENTAL LOGIC
        # ====================================================

        last_ts = get_last_timestamp(RAW_PATH)

        if last_ts:
            logger.info("Modo incremental")

            start_ts = last_ts + 1
            df_old = pd.read_csv(RAW_PATH)

        else:
            logger.info("Modo full load")

            start_ts = to_milliseconds(START_DATE)
            df_old = None

        # ====================================================
        # FETCH
        # ====================================================

        df_new = fetch_all_klines(
            symbol=SYMBOL,
            interval=INTERVAL,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        # ====================================================
        # MERGE RAW (CORRIGIDO)
        # ====================================================

        if df_old is not None and not df_new.empty:
            df_raw = pd.concat([df_old, df_new], ignore_index=True)
        elif df_old is not None:
            df_raw = df_old
        else:
            df_raw = df_new

        # 🔴 FIX CRÍTICO: normalização do RAW
        if not df_raw.empty:
            before = len(df_raw)

            df_raw = (
                df_raw
                .sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"], keep="last")
                .reset_index(drop=True)
            )

            after = len(df_raw)

            if before != after:
                logger.warning(f"RAW deduplicado: {before} → {after}")

        save_csv(df_raw, str(RAW_PATH))

        # ====================================================
        # LOAD (VALIDAÇÃO)
        # ====================================================

        df = load_csv(
            path=str(RAW_PATH),
            timeframe=INTERVAL,
        )

        if df.empty:
            raise RuntimeError("Dataset vazio após load")

        synthetic_ratio = df["is_synthetic"].mean()

        if synthetic_ratio > MAX_SYNTHETIC_RATIO:
            raise RuntimeError(
                f"Dataset ruim: synthetic_ratio={synthetic_ratio:.6f}"
            )

        # ====================================================
        # VERSIONAMENTO
        # ====================================================

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_name = f"{SYMBOL.lower()}_{INTERVAL}_{version}"

        parquet_path = PROCESSED_DIR / f"{base_name}.parquet"
        metadata_path = PROCESSED_DIR / f"{base_name}.json"

        # ====================================================
        # SAVE DATASET
        # ====================================================

        try:
            df.to_parquet(
                parquet_path,
                index=False,
                compression="snappy",
            )
            logger.info(f"Saved parquet: {parquet_path}")

        except Exception as e:
            fallback_path = parquet_path.with_suffix(".csv")

            df.to_csv(fallback_path, index=False)

            logger.warning(
                f"Parquet falhou ({e}), fallback CSV usado: {fallback_path}"
            )

        # ====================================================
        # SAVE METADATA
        # ====================================================

        metadata = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "rows": len(df),
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
            "synthetic_ratio": float(synthetic_ratio),
            "created_at": datetime.utcnow().isoformat(),
        }

        pd.DataFrame([metadata]).to_json(
            metadata_path,
            orient="records",
        )

        logger.info(f"Metadata salvo: {metadata_path}")

        # ====================================================
        # DEBUG
        # ====================================================

        log_dataset_info(df)

        logger.info("=== PIPELINE OK ===")

    except Exception as e:
        logger.exception(f"Erro no pipeline: {e}")
        raise


if __name__ == "__main__":
    main()