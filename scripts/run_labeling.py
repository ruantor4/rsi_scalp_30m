from __future__ import annotations

import logging
import pandas as pd

from src.indicators.atr import compute_atr
from src.indicators.rsi import compute_rsi
from src.indicators.ema import compute_ema
from src.indicators.regime import compute_regime
from src.labeling.label_long import label_long
from src.labeling.label_short import label_short
from src.config.settings import (
    BASE_DATASET_PATH,
    LABELED_DATASET_PATH,
    ATR_PERIOD,
    RSI_PERIOD,
    EMA_FAST,
    EMA_SLOW,
)

logger = logging.getLogger("run_labeling")
logging.basicConfig(level=logging.INFO)


def run() -> None:
    logger.info("=== LABELING START ===")

    df = pd.read_parquet(BASE_DATASET_PATH)
    logger.info("Loaded rows: %d", len(df))

    all_dfs = []

    for symbol in df["symbol"].unique():
        logger.info("Processing symbol: %s", symbol)

        df_sym = df[df["symbol"] == symbol].copy()

        # ====================================================
        # FEATURES (COESO + ML READY)
        # ====================================================

        df_sym["atr"] = compute_atr(df_sym, ATR_PERIOD)
        df_sym["rsi"] = compute_rsi(df_sym, RSI_PERIOD)

        df_sym["ema_fast"] = compute_ema(df_sym, EMA_FAST)
        df_sym["ema_slow"] = compute_ema(df_sym, EMA_SLOW)

        df_sym["ema_spread"] = df_sym["ema_fast"] - df_sym["ema_slow"]

        # 🔥 NORMALIZAÇÃO (ESSENCIAL)
        df_sym["ema_ratio"] = df_sym["ema_fast"] / df_sym["ema_slow"]
        df_sym["price_to_ema"] = df_sym["close"] / df_sym["ema_slow"]

        df_sym["regime"] = compute_regime(df_sym)

        # remove warmup mínimo
        df_sym = df_sym[df_sym["atr"].notna()].reset_index(drop=True)

        # ====================================================
        # GAP FILTER (SEM LEAK)
        # ====================================================

        df_sym["is_gap_edge"] = (
            df_sym["has_gap"] |
            df_sym["has_gap"].shift(1).fillna(False)
        )

        df_sym = df_sym[~df_sym["is_gap_edge"]].reset_index(drop=True)

        # ====================================================
        # LABELING
        # ====================================================

        df_sym = label_long(df_sym)
        df_sym = label_short(df_sym)

        all_dfs.append(df_sym)

    df = pd.concat(all_dfs).reset_index(drop=True)

    logger.info("After labeling: %d", len(df))

    # ====================================================
    # METRICS
    # ====================================================

    long_df = df[df["label_long"] != -1]
    short_df = df[df["label_short"] != -1]

    if not long_df.empty:
        logger.info("Long samples: %d", len(long_df))
        logger.info(
            "Long winrate: %.2f%%",
            (long_df["label_long"] == 1).mean() * 100
        )

    if not short_df.empty:
        logger.info("Short samples: %d", len(short_df))
        logger.info(
            "Short winrate: %.2f%%",
            (short_df["label_short"] == 1).mean() * 100
        )

    # ====================================================
    # SAVE
    # ====================================================

    df.to_parquet(LABELED_DATASET_PATH, index=False)

    logger.info("Saved dataset: %s", LABELED_DATASET_PATH)
    logger.info("=== LABELING END ===")


if __name__ == "__main__":
    run()