from src.data.loader import load_csv
from src.strategy.rsi_strategy import generate_signals
from src.backtest.engine import run_backtest


def main():
    path = "data/raw/btcusdt_30m.csv"

    # =========================
    # LOAD
    # =========================
    df = load_csv(path, timeframe="30m")

    # =========================
    # STRATEGY
    # =========================
    df = generate_signals(
        df,
        rsi_period=14,
        ema_period=50,
    )

    # =========================
    # BACKTEST
    # =========================
    df_bt = run_backtest(
        df,
        initial_balance=1000,
        fee=0.001,
    )

    # =========================
    # OUTPUT
    # =========================
    print(df_bt[["timestamp", "close", "position", "equity"]].tail())


if __name__ == "__main__":
    main()