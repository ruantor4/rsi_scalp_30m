from __future__ import annotations

import pandas as pd


def validate_label_returns(df: pd.DataFrame) -> None:
    df = df.copy()

    # ====================================================
    # LONG
    # ====================================================

    long_df = df[df["label_long"] != -1]

    winrate = (long_df["label_long"] == 1).mean()

    avg_win = long_df[long_df["label_long"] == 1]["label_long"].map(lambda x: 2).mean()
    avg_loss = long_df[long_df["label_long"] == 0]["label_long"].map(lambda x: -1).mean()

    expectancy = (winrate * 2) + ((1 - winrate) * -1)

    print("=== LONG ===")
    print(f"samples: {len(long_df)}")
    print(f"winrate: {winrate:.4f}")
    print(f"expectancy: {expectancy:.4f}")

    # ====================================================
    # SHORT
    # ====================================================

    short_df = df[df["label_short"] != -1]

    winrate = (short_df["label_short"] == 1).mean()

    avg_win = short_df[short_df["label_short"] == 1]["label_short"].map(lambda x: 2).mean()
    avg_loss = short_df[short_df["label_short"] == 0]["label_short"].map(lambda x: -1).mean()

    expectancy = (winrate * 2) + ((1 - winrate) * -1)

    print("\n=== SHORT ===")
    print(f"samples: {len(short_df)}")
    print(f"winrate: {winrate:.4f}")
    print(f"expectancy: {expectancy:.4f}")


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/dataset_labeled.parquet")
    validate_label_returns(df)