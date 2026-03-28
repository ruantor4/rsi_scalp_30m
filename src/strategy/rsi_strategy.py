import pandas as pd


def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # =========================
    # BREAKOUT BASE
    # =========================
    low_break = df["low"].rolling(20).min().shift(1)

    # =========================
    # SCORE SHORT (ranking)
    # =========================
    # quanto mais negativo dist_to_low_break → melhor
    # quanto maior atr_norm → melhor

    df["score_short"] = (
        (-df["dist_to_low_break"]) +
        (df["atr_norm"])
    )

    # =========================
    # SHORT (com threshold)
    # =========================
    df["signal_short"] = (
        (df["close"] < low_break) &
        (df["score_short"] > 1.5) &
        (df["close"].shift(1) > low_break)  # breakout recente
    ).astype(int)

    # =========================
    # SCORE LONG (ranking)
    # =========================
    df["score_long"] = (
        (30 - df["rsi"]) +
        (-df["return_3"] * 100)
    )

    # =========================
    # LONG (com threshold)
    # =========================
    df["signal_long"] = (
        (df["score_long"] > 5) &
        (df["rsi"] < 35) &
        (df["close"] > df["close"].shift(1))
    ).astype(int)

    return df