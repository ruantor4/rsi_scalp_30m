from __future__ import annotations

from pathlib import Path


# ============================================================
# BASE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


# ============================================================
# FILE PATHS
# ============================================================

BASE_DATASET_PATH = PROCESSED_DIR / "dataset.parquet"
LABELED_DATASET_PATH = PROCESSED_DIR / "dataset_labeled.parquet"


# ============================================================
# MARKET CONFIG
# ============================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
]

INTERVAL = "30m"

START_DATE = "2020-01-01"
END_DATE   = "2025-01-01"


# ============================================================
# FETCH CONFIG
# ============================================================

BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"

LIMIT = 1000
REQUEST_TIMEOUT = 10

MAX_RETRIES = 5
BACKOFF_BASE = 0.5
FETCH_SLEEP = 0.2


# ============================================================
# DATA QUALITY
# ============================================================

MAX_INVALID_RATIO = 0.02


# ============================================================
# INDICATORS
# ============================================================

ATR_PERIOD = 14
RSI_PERIOD = 14

EMA_FAST = 50
EMA_SLOW = 200


# ============================================================
# LABELING (CORE STRATEGY)
# ============================================================

TP_ATR_MULT = 1.5
SL_ATR_MULT = 1.0
LABEL_HORIZON = 4


# ============================================================
# BACKTEST (PREPARADO)
# ============================================================

FEE_RATE = 0.0006        # 0.04% (Binance spot aproximado)
SLIPPAGE = 0.0002       # conservador

INITIAL_CAPITAL = 10000
RISK_PER_TRADE = 0.01   # 1% por trade