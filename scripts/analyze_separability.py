from __future__ import annotations

import pandas as pd


def analyze_separability(df: pd.DataFrame) -> None:
    # ====================================================
    # FILTRO (SÓ LONG)
    # ====================================================

    df = df[df["label_long"] != -1].copy()

    if df.empty:
        print("Sem dados válidos")
        return

    # target
    df["target"] = df["label_long"]

    # features que vamos testar
    features = [
        "rsi",
        "atr",
        "ema_9",
        "ema_21",
        "ema_spread",
        "volatility",
        "return",
    ]

    print("\n=== SEPARABILITY ANALYSIS ===\n")

    for col in features:
        if col not in df.columns:
            continue

        win = df[df["target"] == 1][col]
        loss = df[df["target"] == 0][col]

        if win.empty or loss.empty:
            continue

        win_mean = win.mean()
        loss_mean = loss.mean()

        diff = win_mean - loss_mean

        print(f"{col}")
        print(f"  win_mean : {win_mean:.6f}")
        print(f"  loss_mean: {loss_mean:.6f}")
        print(f"  diff     : {diff:.6f}")

        # sinal fraco vs forte
        if abs(diff) < 0.01:
            print("  → fraco")
        elif abs(diff) < 0.05:
            print("  → médio")
        else:
            print("  → forte")

        print("-" * 40)


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/dataset_labeled.parquet")
    analyze_separability(df)