from __future__ import annotations

import logging
import pandas as pd


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger("engine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ============================================================
# EXCEPTIONS
# ============================================================

class BacktestError(Exception):
    pass


# ============================================================
# CORE
# ============================================================

def run_backtest(
    df: pd.DataFrame,
    initial_balance: float = 1000.0,
    fee: float = 0.001,  # 0.1%
) -> pd.DataFrame:
    """
    Executa backtest baseado em position (já shiftado pela strategy).

    Regras:
        - position = 1 → comprado
        - position = 0 → fora

    Métricas:
        - equity
        - return
        - trades
        - winrate
    """

    logger.info("Iniciando backtest")

    if df.empty:
        raise BacktestError("DataFrame vazio")

    required_cols = ["close", "position", "signal"]
    for col in required_cols:
        if col not in df.columns:
            raise BacktestError(f"Coluna ausente: {col}")

    df = df.copy()

    # ============================================================
    # RETURNS
    # ============================================================

    df["market_return"] = df["close"].pct_change().fillna(0.0)

    # estratégia só ganha quando está posicionado
    df["strategy_return"] = df["position"] * df["market_return"]

    # ============================================================
    # TRADES (entrada + saída)
    # ============================================================

    df["trade"] = df["signal"].abs()

    # custo por trade (entrada OU saída)
    df["fee_cost"] = df["trade"] * fee

    df["strategy_return"] = df["strategy_return"] - df["fee_cost"]

    # ============================================================
    # EQUITY CURVE
    # ============================================================

    df["equity"] = (1 + df["strategy_return"]).cumprod() * initial_balance

    # ============================================================
    # METRICS
    # ============================================================

    total_return = df["equity"].iloc[-1] / initial_balance - 1

    trades = int(df["trade"].sum())

    trade_returns = df.loc[df["trade"] > 0, "strategy_return"]

    wins = (trade_returns > 0).sum()
    losses = (trade_returns < 0).sum()

    winrate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

    max_equity = df["equity"].cummax()
    drawdown = (df["equity"] - max_equity) / max_equity
    max_dd = drawdown.min()

    logger.info(
        f"Backtest | return={total_return:.4f} | trades={trades} | winrate={winrate:.2%} | max_dd={max_dd:.2%}"
    )

    return df