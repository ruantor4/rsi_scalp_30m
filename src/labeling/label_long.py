from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src.config.settings import LABEL_HORIZON, TP_ATR_MULT, SL_ATR_MULT

logger = logging.getLogger("label_long")


def label_long(df: pd.DataFrame) -> pd.DataFrame:
    required = {"close", "high", "low", "atr", "rsi", "regime", "is_valid", "has_gap"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    df = df.copy()

    # ====================================================
    # ARRAYS (ACESSO RÁPIDO)
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
    # SETUP (VETORIZADO)
    # ====================================================

    setup_mask = (
        (atr > 0) &
        is_valid &
        (~has_gap) &
        (regime == 1) &
        (rsi < 37)
    )

    setup[setup_mask] = 1

    # ====================================================
    # ÍNDICES DE INTERESSE (REDUZ LOOP)
    # ====================================================

    setup_indices = np.flatnonzero(setup_mask)

    horizon = LABEL_HORIZON
    tp_mult = TP_ATR_MULT
    sl_mult = SL_ATR_MULT

    cooldown = horizon
    last_entry = -cooldown

    # ====================================================
    # LOOP OTIMIZADO
    # ====================================================

    for i in setup_indices:

        # cooldown
        if i - last_entry < cooldown:
            continue

        end = i + horizon + 1
        if end > n:
            break  # importante: dados ordenados

        entry = close[i]
        atr_i = atr[i]

        tp = entry + tp_mult * atr_i
        sl = entry - sl_mult * atr_i

        fh = high[i + 1:end]
        fl = low[i + 1:end]

        # hits
        tp_hits = np.flatnonzero(fh >= tp)
        sl_hits = np.flatnonzero(fl <= sl)

        if tp_hits.size == 0 and sl_hits.size == 0:
            continue

        if tp_hits.size > 0 and sl_hits.size > 0:
            outcome[i] = 1 if tp_hits[0] < sl_hits[0] else 0
        elif tp_hits.size > 0:
            outcome[i] = 1
        else:
            outcome[i] = 0

        last_entry = i  # só atualiza se houve trade válido

    # ====================================================
    # LABEL FINAL (VETORIZADO)
    # ====================================================

    valid = (setup == 1) & (outcome != -1)

    label = np.full(n, -1, dtype=np.int8)
    label[valid] = outcome[valid]

    df["setup_long"] = setup
    df["label_long"] = label

    # ====================================================
    # LOG (SEM CUSTO EXTRA)
    # ====================================================

    total_setups = int(setup.sum())
    total_valid = int(valid.sum())

    if total_valid > 0:
        winrate = float((label[valid] == 1).mean())
    else:
        winrate = 0.0

    logger.info(
        "LONG | setups=%d | valid=%d | winrate=%.2f%%",
        total_setups,
        total_valid,
        winrate * 100,
    )

    return df