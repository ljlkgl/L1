import asyncio
import pandas as pd
import numpy as np
import glob
import os
import traceback
from datetime import datetime
from binance.client import Client
from binance import AsyncClient

# ======================== 配置项 ========================
SYMBOL_LIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAME = Client.KLINE_INTERVAL_15MINUTE
LABEL_WINDOW = 3
LONG_THRESHOLD = 0.005
SHORT_THRESHOLD = -0.005
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLL_PERIOD = 20
CSV_FILE_PREFIX = "./"
CSV_FILE_TPL = f"{CSV_FILE_PREFIX}{{:03d}}.csv"

# ======================== 全局缓存 ========================
REAL_TIME_CACHE = {symbol: {
    # 期货永续数据
    "fut_kline_buffer": [],
    "fut_depth_buffer": [],
    "open_interest": 0.0,
    "long_short_ratio": 0.0,
    "funding_rate": 0.0,
    "fut_last_bid_vol": 0.0,
    "fut_last_ask_vol": 0.0,
    # 现货数据
    "spot_kline_buffer": [],
    "spot_depth_buffer": [],
    "spot_last_bid_vol": 0.0,
    "spot_last_ask_vol": 0.0,
    "spot_latest_kline": None,
    # 标签缓存
    "unlabeled_samples": []
} for symbol in SYMBOL_LIST}
CURRENT_CSV_FILE = ""
async_client = None

# ======================== CSV文件管理 ========================
def get_latest_numeric_file():
    file_list = glob.glob(f"{CSV_FILE_PREFIX}*.csv")
    if not file_list:
        return None
    numeric_files = []
    for file in file_list:
        try:
            file_name = os.path.basename(file)
            num = int(file_name.split('.')[0])
            numeric_files.append((num, file))
        except ValueError:
            continue
    if not numeric_files:
        return None
    numeric_files.sort(key=lambda x: x[0], reverse=True)
    return numeric_files[0][1]

def get_next_numeric_file():
    latest_file = get_latest_numeric_file()
    if not latest_file:
        return CSV_FILE_TPL.format(1)
    file_name = os.path.basename(latest_file)
    current_num = int(file_name.split('.')[0])
    return CSV_FILE_TPL.format(current_num + 1)

def load_existing_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        df = df.tail(1000)
        print(f"Loaded historical file: {file_path}, total {len(df)} historical records")
        return df.to_dict("records")
    except Exception as e:
        print(f"Failed to load historical file, will use new file: {e}")
        return []

def init_csv_file():
    global CURRENT_CSV_FILE
    latest_file = get_latest_numeric_file()
    if latest_file:
        CURRENT_CSV_FILE = latest_file
        for symbol in SYMBOL_LIST:
            history_kline = load_existing_data(latest_file)
            REAL_TIME_CACHE[symbol]["fut_kline_buffer"] = history_kline
        print(f"Continuing with existing CSV file: {CURRENT_CSV_FILE}")
    else:
        CURRENT_CSV_FILE = get_next_numeric_file()
        print(f"No existing files found, created new CSV file: {CURRENT_CSV_FILE}")

# ======================== 特征计算（增加类型安全） ========================
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.zeros_like(tr)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period + 1, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def calculate_features(df):
    # 确保 df 是 pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    # 期货数据列
    fut_close = df["fut_close"].values
    fut_high = df["fut_high"].values
    fut_low = df["fut_low"].values
    fut_volume = df["fut_volume"].values
    # 现货数据列
    spot_close = df["spot_close"].values

    # 1. 期货技术指标
    df["rsi_14"] = calculate_rsi(fut_close, RSI_PERIOD)
    df["ema_fast"] = df["fut_close"].ewm(span=MACD_FAST, adjust=False).mean()
    df["ema_slow"] = df["fut_close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["dif"] = df["ema_fast"] - df["ema_slow"]
    df["dea"] = df["dif"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_diff"] = df["dif"] - df["dea"]
    df["boll_mid"] = df["fut_close"].rolling(BOLL_PERIOD).mean()
    df["boll_std"] = df["fut_close"].rolling(BOLL_PERIOD).std()
    df["boll_upper"] = df["boll_mid"] + 2 * df["boll_std"]
    df["boll_lower"] = df["boll_mid"] - 2 * df["boll_std"]
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"]
    df["atr_14"] = calculate_atr(fut_high, fut_low, fut_close, ATR_PERIOD)
    df["volume_ratio"] = df["fut_volume"] / (df["fut_volume"].rolling(288, min_periods=1).mean() + 1e-10)

    # 2. 期货微观指标
    df["fut_order_imbalance"] = df["fut_order_imbalance"]
    df["taker_ratio"] = (df["fut_taker_buy_base"] - df["fut_taker_sell_base"]) / (df["fut_volume"] + 1e-10)
    df["taker_volume_ratio"] = (df["fut_taker_buy_base"] + df["fut_taker_sell_base"]) / (df["fut_volume"] + 1e-10)
    df["oi_change"] = df["oi"].pct_change(fill_method=None).fillna(0)
    df["oi_volume_ratio"] = df["oi"] / (df["fut_volume"] + 1e-10)
    df["fut_cancel_ratio"] = df["fut_cancel_ratio"]

    # 3. 资金费率指标
    df["funding_rate_3d_mean"] = df["funding_rate"].rolling(288, min_periods=1).mean()
    df["funding_rate_30d_mean"] = df["funding_rate"].rolling(2880, min_periods=1).mean()
    df["funding_dev"] = df["funding_rate"] - df["funding_rate_30d_mean"]
    df["funding_sign"] = np.sign(df["funding_rate"])
    df["funding_switch"] = df["funding_sign"].diff().abs().fillna(0)
    df["funding_switch_3d"] = df["funding_switch"].rolling(288, min_periods=1).sum()

    # 4. 市场情绪+基差指标
    df["ls_ratio_dev"] = df["long_short_ratio"] - df["long_short_ratio"].rolling(2880, min_periods=1).mean()
    df["stablecoin_inflow"] = df["fut_quote_volume"].pct_change(fill_method=None).rolling(24, min_periods=1).mean().fillna(0)
    df["basis"] = df["fut_close"] - df["spot_close"]
    df["basis_ratio"] = df["basis"] / df["spot_close"]
    df["atr_30d_rank"] = df["atr_14"].rolling(2880, min_periods=1).rank(pct=True).fillna(0.5)
    df["market_sentiment"] = np.where(df["fut_close"].pct_change() > 0, 1, 0).rolling(100, min_periods=1).mean().fillna(0.5)

    # 20个核心特征列
    feature_cols = [
        "rsi_14", "macd_diff", "boll_width", "atr_14", "volume_ratio",
        "fut_order_imbalance", "taker_ratio", "taker_volume_ratio", "oi_change", "oi_volume_ratio", "fut_cancel_ratio",
        "funding_rate", "funding_rate_3d_mean", "funding_dev", "funding_switch_3d",
        "ls_ratio_dev", "stablecoin_inflow", "basis_ratio", "atr_30d_rank", "market_sentiment"
    ]
    df[feature_cols] = df[feature_cols].ffill().bfill()
    return df.dropna(subset=feature_cols).reset_index(drop=True), feature_cols

def calculate_label_for_sample(sample, future_close):
    future_return = future_close / sample["fut_close"] - 1
    kline_range = (sample["fut_high"] - sample["fut_low"]) / sample["fut_close"]
    max_allowed_range = 2 * sample["atr_14"] / sample["fut_close"]
    if kline_range > max_allowed_range:
        return np.nan, np.abs(future_return)
    if future_return > LONG_THRESHOLD:
        return 1, np.abs(future_return)
    elif future_return < SHORT_THRESHOLD:
        return 0, np.abs(future_return)
    else:
        return np.nan, np.abs(future_return)

# ======================== 数据处理函数（被轮询任务调用） ========================
async def handle_spot_kline_socket(symbol, msg):
    """处理现货K线数据（由轮询任务调用）"""
    if msg.get("e") != "kline" or not msg.get("k", {}).get("x"):
        return
    kline_data = msg["k"]
    cache = REAL_TIME_CACHE[symbol]
    spot_kline = {
        "spot_timestamp": pd.to_datetime(kline_data["t"], unit="ms", utc=True),
        "spot_open": float(kline_data["o"]),
        "spot_high": float(kline_data["h"]),
        "spot_low": float(kline_data["l"]),
        "spot_close": float(kline_data["c"]),
        "spot_volume": float(kline_data["v"]),
        "spot_quote_volume": float(kline_data["q"]),
        "spot_trades": int(kline_data["n"])
    }
    cache["spot_kline_buffer"].append(spot_kline)
    if len(cache["spot_kline_buffer"]) > 1000:
        cache["spot_kline_buffer"] = cache["spot_kline_buffer"][-1000:]
    cache["spot_latest_kline"] = spot_kline

async def handle_fut_kline_socket(symbol, msg):
    """处理期货K线数据（由轮询任务调用）"""
    if msg.get("e") != "continuous_kline" or not msg.get("k", {}).get("x"):
        return
    kline_data = msg["k"]
    timestamp = pd.to_datetime(kline_data["t"], unit="ms", utc=True)
    cache = REAL_TIME_CACHE[symbol]

    # 1. 获取最新现货数据（从缓存）
    spot_latest = cache.get("spot_latest_kline", {})
    spot_close = spot_latest.get("spot_close", float(kline_data["c"]))
    spot_open = spot_latest.get("spot_open", float(kline_data["o"]))
    spot_high = spot_latest.get("spot_high", float(kline_data["h"]))
    spot_low = spot_latest.get("spot_low", float(kline_data["l"]))
    spot_volume = spot_latest.get("spot_volume", 0.0)

    # 2. 构造期货K线行
    fut_kline_row = {
        "timestamp": timestamp,
        "fut_open": float(kline_data["o"]),
        "fut_high": float(kline_data["h"]),
        "fut_low": float(kline_data["l"]),
        "fut_close": float(kline_data["c"]),
        "fut_volume": float(kline_data["v"]),
        "fut_quote_volume": float(kline_data["q"]),
        "fut_trades": int(kline_data["n"]),
        "fut_taker_buy_base": float(kline_data["V"]),
        "fut_taker_sell_base": float(kline_data["v"]) - float(kline_data["V"]),
        "symbol": symbol,
        "oi": cache["open_interest"],
        "long_short_ratio": cache["long_short_ratio"],
        "funding_rate": cache["funding_rate"],
        "fut_order_imbalance": 0.0,
        "fut_cancel_ratio": 0.0,
        "spot_open": spot_open,
        "spot_high": spot_high,
        "spot_low": spot_low,
        "spot_close": spot_close,
        "spot_volume": spot_volume
    }

    # 3. 补充期货深度指标
    if cache["fut_depth_buffer"]:
        fut_depth_df = pd.DataFrame(cache["fut_depth_buffer"])
        fut_kline_row["fut_order_imbalance"] = fut_depth_df["fut_order_imbalance"].mean()
        total_add = fut_depth_df["fut_add_volume"].sum()
        total_cancel = fut_depth_df["fut_cancel_volume"].sum()
        fut_kline_row["fut_cancel_ratio"] = total_cancel / (total_add + 1e-10)

    # 4. 更新期货K线缓存
    cache["fut_kline_buffer"].append(fut_kline_row)
    if len(cache["fut_kline_buffer"]) > 1000:
        cache["fut_kline_buffer"] = cache["fut_kline_buffer"][-1000:]

    # 5. 数据量校验
    if len(cache["fut_kline_buffer"]) < 60:
        print(f"[{symbol}] Accumulating K-lines: {len(cache['fut_kline_buffer'])}/60 (futures)")
        cache["fut_depth_buffer"] = []
        return

    # 6. 计算特征并写入CSV
    try:
        fut_df = pd.DataFrame(cache["fut_kline_buffer"])
        fut_df, feature_cols = calculate_features(fut_df)
        current_sample = fut_df.iloc[-1].to_dict()

        is_header = not os.path.exists(CURRENT_CSV_FILE)
        pd.DataFrame([current_sample]).to_csv(
            CURRENT_CSV_FILE,
            mode="a",
            header=is_header,
            index=False,
            encoding="utf-8-sig"
        )
        print(f"[{symbol}] Saved sample | Fut Close: {current_sample['fut_close']:.2f} | Spot Close: {current_sample['spot_close']:.2f} | Basis: {current_sample['basis']:.2f}")

        # 7. 标签回填
        cache["unlabeled_samples"].append({
            "timestamp": current_sample["timestamp"],
            "fut_close": current_sample["fut_close"],
            "atr_14": current_sample["atr_14"],
            "feature_dict": current_sample
        })

        if len(cache["unlabeled_samples"]) > LABEL_WINDOW:
            target_sample = cache["unlabeled_samples"].pop(0)
            future_close = current_sample["fut_close"]
            label, conf_label = calculate_label_for_sample(target_sample, future_close)
            if not np.isnan(label):
                print(f"[{symbol}] Label backfilled: {'Long' if label ==1 else 'Short'}, Conf: {conf_label:.4f}")

        # 清空当期期货深度缓存
        cache["fut_depth_buffer"] = []
    except Exception as e:
        print(f"[{symbol}] Error in handle_fut_kline_socket: {e}")
        traceback.print_exc()

# ======================== 轮询任务 ========================
async def poll_spot_klines(symbol):
    """轮询现货K线"""
    global async_client
    while True:
        try:
            klines = await async_client.get_klines(symbol=symbol, interval=TIMEFRAME, limit=1)
            if klines:
                k = klines[0]
                msg = {
                    "e": "kline",
                    "k": {
                        "t": k[0],
                        "o": k[1],
                        "h": k[2],
                        "l": k[3],
                        "c": k[4],
                        "v": k[5],
                        "x": True,
                        "q": k[7],
                        "n": k[8],
                        "V": k[9],
                    }
                }
                await handle_spot_kline_socket(symbol, msg)
        except Exception as e:
            print(f"[{symbol}] Spot Kline poll error: {e}")
        await asyncio.sleep(15)

async def poll_spot_depth(symbol):
    """轮询现货深度（每秒一次）"""
    global async_client
    last_bid1_vol = 0.0
    last_ask1_vol = 0.0
    while True:
        try:
            depth = await async_client.get_order_book(symbol=symbol, limit=10)
            bid1_vol = float(depth['bids'][0][1]) if depth['bids'] else 0.0
            ask1_vol = float(depth['asks'][0][1]) if depth['asks'] else 0.0

            add_volume = max(0.0, bid1_vol - last_bid1_vol) + max(0.0, ask1_vol - last_ask1_vol)
            cancel_volume = max(0.0, last_bid1_vol - bid1_vol) + max(0.0, last_ask1_vol - ask1_vol)
            order_imbalance = (bid1_vol - ask1_vol) / (bid1_vol + ask1_vol + 1e-10)

            cache = REAL_TIME_CACHE[symbol]
            cache["spot_last_bid_vol"] = bid1_vol
            cache["spot_last_ask_vol"] = ask1_vol
            cache["spot_depth_buffer"].append({
                "spot_order_imbalance": order_imbalance,
                "spot_add_volume": add_volume,
                "spot_cancel_volume": cancel_volume
            })
            if len(cache["spot_depth_buffer"]) > 1000:
                cache["spot_depth_buffer"] = cache["spot_depth_buffer"][-1000:]

            last_bid1_vol = bid1_vol
            last_ask1_vol = ask1_vol

        except Exception as e:
            print(f"[{symbol}] Spot Depth poll error: {e}")
        await asyncio.sleep(1)

async def poll_fut_klines(symbol):
    """轮询期货永续K线"""
    global async_client
    while True:
        try:
            klines = await async_client.futures_continuous_klines(
                pair=symbol,
                contractType='PERPETUAL',
                interval=TIMEFRAME,
                limit=1
            )
            if klines:
                k = klines[0]
                msg = {
                    "e": "continuous_kline",
                    "k": {
                        "t": k[0],
                        "o": k[1],
                        "h": k[2],
                        "l": k[3],
                        "c": k[4],
                        "v": k[5],
                        "x": True,
                        "q": k[7],
                        "n": k[8],
                        "V": k[9],
                    }
                }
                await handle_fut_kline_socket(symbol, msg)
        except Exception as e:
            print(f"[{symbol}] Futures Kline poll error: {e}")
            traceback.print_exc()
        await asyncio.sleep(15)

async def poll_fut_depth(symbol):
    """轮询期货深度（每秒一次）"""
    global async_client
    last_bid1_vol = 0.0
    last_ask1_vol = 0.0
    while True:
        try:
            depth = await async_client.futures_order_book(symbol=symbol, limit=10)
            bid1_vol = float(depth['bids'][0][1]) if depth['bids'] else 0.0
            ask1_vol = float(depth['asks'][0][1]) if depth['asks'] else 0.0

            add_volume = max(0.0, bid1_vol - last_bid1_vol) + max(0.0, ask1_vol - last_ask1_vol)
            cancel_volume = max(0.0, last_bid1_vol - bid1_vol) + max(0.0, last_ask1_vol - ask1_vol)
            order_imbalance = (bid1_vol - ask1_vol) / (bid1_vol + ask1_vol + 1e-10)

            cache = REAL_TIME_CACHE[symbol]
            cache["fut_last_bid_vol"] = bid1_vol
            cache["fut_last_ask_vol"] = ask1_vol
            cache["fut_depth_buffer"].append({
                "fut_order_imbalance": order_imbalance,
                "fut_add_volume": add_volume,
                "fut_cancel_volume": cancel_volume
            })
            if len(cache["fut_depth_buffer"]) > 1000:
                cache["fut_depth_buffer"] = cache["fut_depth_buffer"][-1000:]

            last_bid1_vol = bid1_vol
            last_ask1_vol = ask1_vol

        except Exception as e:
            print(f"[{symbol}] Futures Depth poll error: {e}")
        await asyncio.sleep(1)

async def poll_fut_funding_rate(symbol):
    """轮询期货资金费率（官方专用接口）"""
    global async_client
    while True:
        try:
            funding_data = await async_client.futures_funding_rate(
                symbol=symbol,
                limit=1
            )
            if funding_data and len(funding_data) > 0:
                REAL_TIME_CACHE[symbol]["funding_rate"] = float(funding_data[0]['fundingRate'])
        except Exception as e:
            print(f"[{symbol}] Funding rate poll error: {e}")
        await asyncio.sleep(60)  # 资金费率每小时更新，60秒轮询足够

# ======================== 原有轮询任务（不变） ========================
async def update_open_interest(symbol):
    while True:
        try:
            oi_data = await async_client.futures_open_interest(symbol=symbol)
            if oi_data:
                REAL_TIME_CACHE[symbol]["open_interest"] = float(oi_data["openInterest"])
        except Exception as e:
            print(f"[{symbol}] OI update failed: {e}")
        await asyncio.sleep(60)

async def update_long_short(symbol):
    while True:
        try:
            ls_data = await async_client.futures_top_longshort_account_ratio(
                symbol=symbol, period="5m", limit=1
            )
            if ls_data:
                REAL_TIME_CACHE[symbol]["long_short_ratio"] = float(ls_data[0]["longShortRatio"])
        except Exception as e:
            print(f"[{symbol}] Long/Short ratio update failed: {e}")
        await asyncio.sleep(60)

# ======================== 安全退出 ========================
async def safe_shutdown(all_tasks):
    print("\nInitiating safe shutdown...")
    for task in all_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    global async_client
    if async_client is not None:
        await async_client.close_connection()
        print("Async client closed")
    print(f"Data saved to: {CURRENT_CSV_FILE}")

# ======================== 主程序 ========================
async def main():
    global async_client
    print("="*60)
    print("Binance Futures + Spot CSV Collector (Polling Mode)")
    print("="*60)

    init_csv_file()
    async_client = await AsyncClient.create()

    all_tasks = []
    for symbol in SYMBOL_LIST:
        all_tasks.append(asyncio.create_task(poll_spot_klines(symbol)))
        all_tasks.append(asyncio.create_task(poll_spot_depth(symbol)))
        all_tasks.append(asyncio.create_task(poll_fut_klines(symbol)))
        all_tasks.append(asyncio.create_task(poll_fut_depth(symbol)))
        all_tasks.append(asyncio.create_task(poll_fut_funding_rate(symbol)))
        all_tasks.append(asyncio.create_task(update_open_interest(symbol)))
        all_tasks.append(asyncio.create_task(update_long_short(symbol)))

    print(f"Started polling for {len(SYMBOL_LIST)} symbols (Futures + Spot). Collecting data...")
    print(f"Note: Need 60 K-lines (≈15 hours) for first valid sample.")

    try:
        await asyncio.gather(*all_tasks)
    except KeyboardInterrupt:
        await safe_shutdown(all_tasks)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        await safe_shutdown(all_tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram stopped manually.")