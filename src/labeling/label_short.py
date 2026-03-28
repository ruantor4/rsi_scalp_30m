from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src.config.settings import LABEL_HORIZON, TP_ATR_MULT, SL_ATR_MULT

logger = logging.getLogger("label_short")


def label_short(df: pd.DataFrame) -> pd.DataFrame:
    required = {"close", "high", "low", "atr", "rsi", "regime", "is_valid", "has_gap"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    df = df.copy()

    # ====================================================
    # PARAMS (CENTRALIZADO)
    # ====================================================

    horizon = LABEL_HORIZON
    tp_mult = TP_ATR_MULT
    sl_mult = SL_ATR_MULT

    # ====================================================
    # ARRAYS
    # ====================================================

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df["atr"].to_numpy()
    rsi = df["rsi"].to_numpy()
    regime = df["regime"].to_numpy()
    is_valid = df["is_valid"].to_numpy()
    has_gap = df["has_gap"].to_numpy()

    n = close.shape[0]

    setup = np.zeros(n, dtype=np.int8)
    outcome = np.full(n, -1, dtype=np.int8)

    # ====================================================
    # VOL FILTER
    # ====================================================

    atr_threshold = df["atr"].rolling(50).median().to_numpy()
    vol_ok = atr > atr_threshold

    # ====================================================
    # SETUP
    # ====================================================

    setup_mask = (
        (atr > 0) &
        vol_ok &
        is_valid &
        (~has_gap) &
        (regime == -1) &
        (rsi > 65)
    )

    setup[setup_mask] = 1
    setup_indices = np.flatnonzero(setup_mask)

    # ====================================================
    # SIMULAÇÃO
    # ====================================================

    cooldown = horizon
    last_entry = -cooldown

    for i in setup_indices:

        if i - last_entry < cooldown:
            continue

        end = i + horizon + 1
        if end > n:
            break

        entry = close[i]
        atr_i = atr[i]

        tp = entry - tp_mult * atr_i
        sl = entry + sl_mult * atr_i

        fh = high[i + 1:end]
        fl = low[i + 1:end]

        tp_hits = np.flatnonzero(fl <= tp)
        sl_hits = np.flatnonzero(fh >= sl)

        if tp_hits.size == 0 and sl_hits.size == 0:
            continue

        if tp_hits.size > 0 and sl_hits.size > 0:
            outcome[i] = 1 if tp_hits[0] < sl_hits[0] else 0
        elif tp_hits.size > 0:
            outcome[i] = 1
        else:
            outcome[i] = 0

        last_entry = i

    # ====================================================
    # LABEL FINAL
    # ====================================================

    valid = (setup == 1) & (outcome != -1)

    label = np.full(n, -1, dtype=np.int8)
    label[valid] = outcome[valid]

    df["setup_short"] = setup
    df["label_short"] = label

    total_setups = int(setup.sum())
    total_valid = int(valid.sum())

    winrate = float((label[valid] == 1).mean()) if total_valid > 0 else 0.0

    logger.info(
        "SHORT | setups=%d | valid=%d | winrate=%.2f%%",
        total_setups,
        total_valid,
        winrate * 100,
    )

    return df