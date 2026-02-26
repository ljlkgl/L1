import asyncio
import pandas as pd
import numpy as np
import time
import glob
import os
from datetime import datetime, timedelta
from binance.client import Client
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException, BinanceWebsocketUnableToConnect

# ==============================================
# Core Configuration (CSV Collection Only)
# ==============================================
SYMBOL_LIST = ["BTCUSDT", "ETHUSDT"]  # Binance official perpetual contract symbols
TIMEFRAME = Client.KLINE_INTERVAL_15MINUTE  # Fixed 15-minute interval
LABEL_WINDOW = 3  # Label based on next 3 K-lines (45 minutes)
LONG_THRESHOLD = 0.005  # Long if return > 0.5%
SHORT_THRESHOLD = -0.005  # Short if return < -0.5%
# Technical indicator periods
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLL_PERIOD = 20
# CSV file configuration (sequential naming: 001.csv, 002.csv...)
CSV_FILE_PREFIX = "./"
CSV_FILE_TPL = f"{CSV_FILE_PREFIX}{{:03d}}.csv"  # Format: 001.csv, 002.csv...

# Global real-time data cache
REAL_TIME_CACHE = {symbol: {
    "kline_buffer": [],
    "depth_buffer": [],
    "force_order_buffer": [],
    "open_interest": 0,
    "long_short_ratio": 0,
    "funding_rate": 0,
    "unlabeled_samples": []
} for symbol in SYMBOL_LIST}
# Global runtime variables
CURRENT_CSV_FILE = ""
async_client = None
client = Client()

# ==============================================
# 1. CSV File Management (Sequential Naming + No Frequent New Files)
# ==============================================
def get_latest_numeric_file():
    """Get the latest sequential CSV file (001.csv, 002.csv...), return None if none"""
    file_list = glob.glob(f"{CSV_FILE_PREFIX}*.csv")
    if not file_list:
        return None
    
    # Extract numeric part from filenames (e.g., 001.csv → 1)
    numeric_files = []
    for file in file_list:
        try:
            file_name = os.path.basename(file)
            num = int(file_name.split('.')[0])
            numeric_files.append((num, file))
        except ValueError:
            continue  # Skip non-sequential files
    
    if not numeric_files:
        return None
    
    # Sort by number, get the largest one
    numeric_files.sort(key=lambda x: x[0], reverse=True)
    return numeric_files[0][1]

def get_next_numeric_file():
    """Get the next sequential filename (e.g., if latest is 003.csv → 004.csv)"""
    latest_file = get_latest_numeric_file()
    if not latest_file:
        return CSV_FILE_TPL.format(1)  # Start with 001.csv
    
    # Extract number from latest file
    file_name = os.path.basename(latest_file)
    current_num = int(file_name.split('.')[0])
    return CSV_FILE_TPL.format(current_num + 1)

def load_existing_data(file_path):
    """Load historical data from existing CSV for continuation"""
    try:
        df = pd.read_csv(file_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        # Keep only last 1000 K-lines for accurate indicator calculation
        df = df.tail(1000)
        print(f"Loaded historical file: {file_path}, total {len(df)} historical records")
        return df.to_dict("records")
    except Exception as e:
        print(f"Failed to load historical file, will use new file: {e}")
        return []

def init_csv_file():
    """Initialize CSV file (continue existing or create new sequential file)"""
    global CURRENT_CSV_FILE
    latest_file = get_latest_numeric_file()
    
    # Continue existing file if available
    if latest_file:
        CURRENT_CSV_FILE = latest_file
        # Load historical data to cache
        for symbol in SYMBOL_LIST:
            history_kline = load_existing_data(latest_file)
            REAL_TIME_CACHE[symbol]["kline_buffer"] = history_kline
        print(f"Continuing with existing CSV file: {CURRENT_CSV_FILE}")
    else:
        # Create first file (001.csv) if no existing files
        CURRENT_CSV_FILE = get_next_numeric_file()
        print(f"No existing files found, created new CSV file: {CURRENT_CSV_FILE}")

# ==============================================
# 2. 20 Fixed Feature Calculation
# ==============================================
def calculate_rsi(prices, period):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period):
    tr = np.zeros_like(high)
    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    atr = np.zeros_like(tr)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period + 1, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def calculate_features(df):
    """Calculate 20 fixed features (same as original scheme)"""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    # 1. Technical factors (5)
    df["rsi_14"] = calculate_rsi(close, RSI_PERIOD)
    df["ema_fast"] = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["dif"] = df["ema_fast"] - df["ema_slow"]
    df["dea"] = df["dif"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_diff"] = df["dif"] - df["dea"]
    df["boll_mid"] = df["close"].rolling(BOLL_PERIOD).mean()
    df["boll_std"] = df["close"].rolling(BOLL_PERIOD).std()
    df["boll_upper"] = df["boll_mid"] + 2 * df["boll_std"]
    df["boll_lower"] = df["boll_mid"] - 2 * df["boll_std"]
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"]
    df["atr_14"] = calculate_atr(high, low, close, ATR_PERIOD)
    df["volume_3d_mean"] = df["volume"].rolling(288).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_3d_mean"] + 1e-10)

    # 2. Perpetual micro factors (6, real-time data)
    df["order_imbalance"] = df["order_imbalance"]
    df["taker_ratio"] = (df["taker_buy_base"] - df["taker_sell_base"]) / (df["volume"] + 1e-10)
    df["force_vol_ratio"] = df["force_vol"] / (df["volume"] + 1e-10)
    df["oi_change"] = df["oi"].pct_change(fill_method=None)
    df["oi_volume_ratio"] = df["oi"] / (df["volume"] + 1e-10)
    df["cancel_ratio"] = df["cancel_ratio"]

    # 3. Funding rate factors (4)
    df["funding_rate_3d_mean"] = df["funding_rate"].rolling(288).mean()
    df["funding_rate_30d_mean"] = df["funding_rate"].rolling(2880).mean()
    df["funding_dev"] = df["funding_rate"] - df["funding_rate_30d_mean"]
    df["funding_sign"] = np.sign(df["funding_rate"])
    df["funding_switch"] = df["funding_sign"].diff().abs()
    df["funding_switch_3d"] = df["funding_switch"].rolling(288).sum()

    # 4. Market sentiment factors (5)
    df["ls_ratio_30d_mean"] = df["long_short_ratio"].rolling(2880).mean()
    df["ls_ratio_dev"] = df["long_short_ratio"] - df["ls_ratio_30d_mean"]
    df["stablecoin_inflow"] = df["quote_volume"].pct_change(fill_method=None).rolling(24).mean()
    df["basis"] = df["close"] - df["spot_close"]
    df["basis_ratio"] = df["basis"] / df["spot_close"]
    df["atr_30d_rank"] = df["atr_14"].rolling(2880).rank(pct=True)
    df["market_sentiment"] = np.where(df["close"].pct_change() > 0, 1, 0).rolling(100).mean()

    # Fixed 20 features in exact order
    feature_cols = [
        "rsi_14", "macd_diff", "boll_width", "atr_14", "volume_ratio",
        "order_imbalance", "taker_ratio", "force_vol_ratio", "oi_change", "oi_volume_ratio", "cancel_ratio",
        "funding_rate", "funding_rate_3d_mean", "funding_dev", "funding_switch_3d",
        "ls_ratio_dev", "stablecoin_inflow", "basis_ratio", "atr_30d_rank", "market_sentiment"
    ]
    # Handle missing values
    df[feature_cols] = df[feature_cols].ffill().bfill()
    return df.dropna(subset=feature_cols).reset_index(drop=True), feature_cols

# ==============================================
# 3. Label Calculation (Same as Original Rules)
# ==============================================
def calculate_label_for_sample(sample, future_close):
    future_return = future_close / sample["close"] - 1
    kline_range = (sample["high"] - sample["low"]) / sample["close"]
    max_allowed_range = 2 * sample["atr_14"] / sample["close"]
    
    # Filter wick spikes
    if kline_range > max_allowed_range:
        return np.nan, np.abs(future_return)
    # Label assignment
    if future_return > LONG_THRESHOLD:
        return 1, np.abs(future_return)
    elif future_return < SHORT_THRESHOLD:
        return 0, np.abs(future_return)
    else:
        return np.nan, np.abs(future_return)

# ==============================================
# 4. Real-time WebSocket Message Handling
# ==============================================
async def handle_kline_socket(symbol, msg):
    """Process closed 15min K-lines only, trigger feature calculation and label backfilling"""
    if msg.get("e") != "kline" or not msg.get("k", {}).get("x"):
        return  # Only process closed K-line events
    kline_data = msg["k"]
    timestamp = pd.to_datetime(kline_data["t"], unit="ms", utc=True)
    cache = REAL_TIME_CACHE[symbol]

    # Aggregate real-time micro data for current 15min period
    kline_row = {
        "timestamp": timestamp,
        "open": float(kline_data["o"]),
        "high": float(kline_data["h"]),
        "low": float(kline_data["l"]),
        "close": float(kline_data["c"]),
        "volume": float(kline_data["v"]),
        "quote_volume": float(kline_data["q"]),
        "trades": int(kline_data["n"]),
        "taker_buy_base": float(kline_data["V"]),
        "taker_sell_base": float(kline_data["v"]) - float(kline_data["V"]),
        "symbol": symbol,
        "oi": cache["open_interest"],
        "long_short_ratio": cache["long_short_ratio"],
        "funding_rate": cache["funding_rate"]
    }

    # Calculate order book and cancel data
    if cache["depth_buffer"]:
        depth_df = pd.DataFrame(cache["depth_buffer"])
        kline_row["order_imbalance"] = depth_df["order_imbalance"].mean()
        total_add = depth_df["add_volume"].sum()
        total_cancel = depth_df["cancel_volume"].sum()
        kline_row["cancel_ratio"] = total_cancel / (total_add + 1e-10)
    else:
        kline_row["order_imbalance"] = 0
        kline_row["cancel_ratio"] = 0

    # Calculate liquidation volume
    kline_row["force_vol"] = sum([x["volume"] for x in cache["force_order_buffer"]])

    # Get spot price in real-time for basis calculation
    try:
        spot_ticker = await async_client.get_symbol_ticker(symbol=symbol)
        kline_row["spot_close"] = float(spot_ticker["price"])
    except:
        kline_row["spot_close"] = kline_row["close"]

    # Add to K-line buffer (control memory usage)
    cache["kline_buffer"].append(kline_row)
    if len(cache["kline_buffer"]) > 1000:
        cache["kline_buffer"] = cache["kline_buffer"][-1000:]

    # Check if enough K-lines for feature calculation (minimum 60)
    if len(cache["kline_buffer"]) < 60:
        print(f"[{symbol}] Accumulating K-lines: {len(cache['kline_buffer'])}/60, cannot calculate features yet")
        # Clear period cache
        cache["depth_buffer"] = []
        cache["force_order_buffer"] = []
        return

    # Calculate 20 features
    kline_df = pd.DataFrame(cache["kline_buffer"])
    kline_df, feature_cols = calculate_features(kline_df)
    current_sample = kline_df.iloc[-1].to_dict()

    # Save sample for label backfilling
    cache["unlabeled_samples"].append({
        "timestamp": current_sample["timestamp"],
        "close": current_sample["close"],
        "atr_14": current_sample["atr_14"],
        "feature_dict": current_sample
    })
    print(f"[{symbol}] K-line closed, features calculated, pending label samples: {len(cache['unlabeled_samples'])}")

    # Backfill labels (wait for next 3 K-lines)
    if len(cache["unlabeled_samples"]) > LABEL_WINDOW:
        target_sample = cache["unlabeled_samples"].pop(0)
        future_close = current_sample["close"]
        label, conf_label = calculate_label_for_sample(target_sample, future_close)
        
        # Write valid sample to CSV
        if not np.isnan(label):
            train_row = target_sample["feature_dict"]
            train_row["label"] = int(label)
            train_row["conf_label"] = conf_label
            # Incremental write to CSV (auto handle header)
            is_header = not os.path.exists(CURRENT_CSV_FILE)
            pd.DataFrame([train_row]).to_csv(
                CURRENT_CSV_FILE,
                mode="a",
                header=is_header,
                index=False,
                encoding="utf-8-sig"
            )
            print(f"[{symbol}] Valid sample written to CSV, label: {'Long' if label ==1 else 'Short'}, file: {CURRENT_CSV_FILE}")

    # Clear current period cache
    cache["depth_buffer"] = []
    cache["force_order_buffer"] = []

async def handle_depth_socket(symbol, msg):
    """Process real-time order book depth stream, calculate order imbalance and cancel ratio"""
    if msg.get("e") != "depthUpdate":
        return
    bids = msg.get("b", [])
    asks = msg.get("a", [])
    if not bids or not asks:
        return
    bid1_vol = float(bids[0][1])
    ask1_vol = float(asks[0][1])
    order_imbalance = (bid1_vol - ask1_vol) / (bid1_vol + ask1_vol + 1e-10)
    
    cache = REAL_TIME_CACHE[symbol]
    last_bid = cache.get("last_bid_vol", 0)
    last_ask = cache.get("last_ask_vol", 0)
    add_volume = max(0, bid1_vol - last_bid) + max(0, ask1_vol - last_ask)
    cancel_volume = max(0, last_bid - bid1_vol) + max(0, last_ask - ask1_vol)
    
    cache["last_bid_vol"] = bid1_vol
    cache["last_ask_vol"] = ask1_vol
    cache["depth_buffer"].append({
        "order_imbalance": order_imbalance,
        "add_volume": add_volume,
        "cancel_volume": cancel_volume
    })

async def handle_force_order_socket(symbol, msg):
    """Process real-time liquidation order stream"""
    if msg.get("e") != "forceOrder" or msg.get("o", {}).get("s") != symbol:
        return
    REAL_TIME_CACHE[symbol]["force_order_buffer"].append({
        "volume": float(msg["o"]["q"]),
        "side": msg["o"]["S"]
    })

async def handle_open_interest_socket(symbol, msg):
    """Update open interest in real-time"""
    if msg.get("e") != "openInterest":
        return
    REAL_TIME_CACHE[symbol]["open_interest"] = float(msg["o"])

async def handle_funding_rate_socket(symbol, msg):
    """Update funding rate in real-time"""
    if msg.get("e") != "markPriceUpdate":
        return
    REAL_TIME_CACHE[symbol]["funding_rate"] = float(msg["r"])

# ==============================================
# 5. WebSocket Subscription (Fixed: async with + await recv())
# ==============================================
async def subscribe_symbol_streams(symbol, bm):
    """Subscribe to all real-time streams for a symbol (fixed callback issue)"""
    # K-line stream task
    async def kline_task():
        async with bm.kline_socket(symbol=symbol, interval=TIMEFRAME) as stream:
            while True:
                msg = await stream.recv()
                await handle_kline_socket(symbol, msg)

    # Depth stream task
    async def depth_task():
        async with bm.depth_socket(symbol=symbol, depth="100ms") as stream:
            while True:
                msg = await stream.recv()
                await handle_depth_socket(symbol, msg)

    # Liquidation order stream task
    async def force_order_task():
        async with bm.force_order_socket(symbol=symbol) as stream:
            while True:
                msg = await stream.recv()
                await handle_force_order_socket(symbol, msg)

    # Open interest stream task
    async def open_interest_task():
        async with bm.open_interest_socket(symbol=symbol) as stream:
            while True:
                msg = await stream.recv()
                await handle_open_interest_socket(symbol, msg)

    # Funding rate stream task
    async def mark_price_task():
        async with bm.mark_price_socket(symbol=symbol, fast=False) as stream:
            while True:
                msg = await stream.recv()
                await handle_funding_rate_socket(symbol, msg)

    # Long/short ratio update (every minute)
    async def update_long_short():
        while True:
            try:
                ls_data = await async_client.futures_global_long_short_account_ratio(
                    symbol=symbol, period="5m", limit=1
                )
                if ls_data:
                    REAL_TIME_CACHE[symbol]["long_short_ratio"] = float(ls_data[0]["longShortRatio"])
            except:
                pass
            await asyncio.sleep(60)

    # Return all tasks
    return [
        asyncio.create_task(kline_task()),
        asyncio.create_task(depth_task()),
        asyncio.create_task(force_order_task()),
        asyncio.create_task(open_interest_task()),
        asyncio.create_task(mark_price_task()),
        asyncio.create_task(update_long_short())
    ]

# ==============================================
# Main Program Entry
# ==============================================
async def main():
    global async_client
    print("="*60)
    print("Binance Perpetual Contract CSV Collection Script (No Torch Dependency)")
    print("DISCLAIMER: This script is for technical research only, not investment advice")
    print("="*60)

    # 1. Initialize CSV file (continue existing or create new)
    init_csv_file()

    # 2. Initialize async client and WebSocket manager
    async_client = await AsyncClient.create()
    bm = BinanceSocketManager(async_client)

    # 3. Subscribe to real-time streams for all symbols
    all_tasks = []
    for symbol in SYMBOL_LIST:
        symbol_tasks = await subscribe_symbol_streams(symbol, bm)
        all_tasks.extend(symbol_tasks)
    print(f"Subscribed to real-time streams for {len(SYMBOL_LIST)} symbols, collecting data...")
    print(f"Note: Need to accumulate 60 K-lines (≈15 hours) to generate first valid sample")

    # 4. Auto-reconnect on exceptions
    try:
        await asyncio.gather(*all_tasks)
    except BinanceWebsocketUnableToConnect:
        print("WebSocket connection disconnected, restarting in 5 seconds...")
        await asyncio.sleep(5)
        await main()
    except Exception as e:
        print(f"Program exception: {e}, restarting in 5 seconds")
        await asyncio.sleep(5)
        await main()
    finally:
        await async_client.close_connection()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram stopped manually, CSV data saved to: ", CURRENT_CSV_FILE)