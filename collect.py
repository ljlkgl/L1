import asyncio
import pandas as pd
import numpy as np
import torch
import time
import glob
import os
from datetime import datetime, timedelta
from binance.client import Client
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException, BinanceWebsocketUnableToConnect

# ==============================================
# 【固定参数，100%匹配原方案，仅中断阈值可按需调整】
# ==============================================
# 核心交易配置
SYMBOL_LIST = ["BTCUSDT", "ETHUSDT"]  # 币安官方永续合约标的
TIMEFRAME = Client.KLINE_INTERVAL_15MINUTE  # 固定15min周期
# 模型输入配置
SEQ_LENGTH = 60  # 60根15minK线=15小时序列长度
FEATURE_DIM = 20  # 固定20个输入特征
# 标签规则（完全匹配原方案）
LABEL_WINDOW = 3  # 未来3根K线（45分钟）打标签
LONG_THRESHOLD = 0.005  # 收益>0.5%=做多
SHORT_THRESHOLD = -0.005  # 收益<-0.5%=做空
# 技术指标固定周期
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLL_PERIOD = 20
# 中断恢复配置（核心需求）
MAX_INTERRUPT_MINUTES = 30  # 中断超过30分钟，自动新开文件
TRAIN_FILE_PREFIX = "./perpetual_train_set"
TRAIN_FILE_TPL = f"{TRAIN_FILE_PREFIX}_{{utc_start_time}}.csv"
TENSOR_SAVE_PATH = "./perpetual_train_tensor.pt"

# 全局实时数据缓存
REAL_TIME_CACHE = {symbol: {
    "kline_buffer": [],
    "depth_buffer": [],
    "trade_buffer": [],
    "force_order_buffer": [],
    "cancel_buffer": [],
    "open_interest": 0,
    "long_short_ratio": 0,
    "funding_rate": 0,
    "unlabeled_samples": []
} for symbol in SYMBOL_LIST}
# 全局运行时变量
CURRENT_TRAIN_FILE = ""
async_client = None
client = Client()

# ==============================================
# 1. 中断恢复与文件管理工具（核心需求实现）
# ==============================================
def get_latest_train_file():
    """获取目录下最新的训练集文件，无则返回None"""
    file_list = glob.glob(f"{TRAIN_FILE_PREFIX}_*.csv")
    if not file_list:
        return None
    # 按文件修改时间倒序，取最新的
    file_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return file_list[0]

def load_existing_data(file_path):
    """加载已有文件的历史数据，用于短中断接续，保证特征计算连续性"""
    try:
        df = pd.read_csv(file_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        # 仅保留最近1000根K线，保证指标计算准确且不占用过多内存
        df = df.tail(1000)
        print(f"已加载历史文件：{file_path}，共{len(df)}条历史数据")
        # 转换为kline_buffer格式
        kline_buffer = df.to_dict("records")
        return kline_buffer
    except Exception as e:
        print(f"历史文件加载失败，将新开文件：{e}")
        return []

def init_train_file():
    """初始化训练文件，判断是否新开/接续"""
    global CURRENT_TRAIN_FILE
    latest_file = get_latest_train_file()
    now_utc = datetime.utcnow()

    # 无历史文件，直接新开
    if not latest_file:
        start_time_str = now_utc.strftime("%Y%m%d_%H%M%S")
        CURRENT_TRAIN_FILE = TRAIN_FILE_TPL.format(utc_start_time=start_time_str)
        print(f"无历史文件，新开训练文件：{CURRENT_TRAIN_FILE}")
        return

    # 有历史文件，判断中断时长
    try:
        # 读取历史文件最后一行的时间
        last_df = pd.read_csv(latest_file, nrows=0)
        if len(last_df) == 0:
            file_end_time = datetime.fromtimestamp(os.path.getmtime(latest_file), tz=datetime.utcnow().tzinfo)
        else:
            last_line = pd.read_csv(latest_file, skiprows=range(1, len(last_df)-1), nrows=1)
            file_end_time = pd.to_datetime(last_line["timestamp"].iloc[0], utc=True)
        # 计算中断时长
        interrupt_duration = (now_utc - file_end_time).total_seconds() / 60
        print(f"检测到历史文件：{latest_file}，中断时长：{interrupt_duration:.1f}分钟")

        # 短中断，接续原有文件
        if interrupt_duration <= MAX_INTERRUPT_MINUTES:
            CURRENT_TRAIN_FILE = latest_file
            # 加载历史数据到缓存
            for symbol in SYMBOL_LIST:
                history_kline = load_existing_data(latest_file)
                REAL_TIME_CACHE[symbol]["kline_buffer"] = history_kline
            print(f"短中断，接续原有文件：{CURRENT_TRAIN_FILE}")
        # 长中断，新开文件
        else:
            start_time_str = now_utc.strftime("%Y%m%d_%H%M%S")
            CURRENT_TRAIN_FILE = TRAIN_FILE_TPL.format(utc_start_time=start_time_str)
            print(f"长中断，新开训练文件：{CURRENT_TRAIN_FILE}")
    except Exception as e:
        # 任何异常，直接新开文件，保证程序正常启动
        start_time_str = now_utc.strftime("%Y%m%d_%H%M%S")
        CURRENT_TRAIN_FILE = TRAIN_FILE_TPL.format(utc_start_time=start_time_str)
        print(f"历史文件校验失败，新开训练文件：{CURRENT_TRAIN_FILE}，错误：{e}")

# ==============================================
# 2. 20个固定特征计算（100%匹配原方案）
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
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    """严格匹配原方案20个特征，顺序完全一致"""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    # 1. 技术因子（5个）
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

    # 2. 永续微观因子（6个，纯实时真实数据）
    df["order_imbalance"] = df["order_imbalance"]
    df["taker_ratio"] = (df["taker_buy_base"] - df["taker_sell_base"]) / (df["volume"] + 1e-10)
    df["force_vol_ratio"] = df["force_vol"] / (df["volume"] + 1e-10)
    df["oi_change"] = df["oi"].pct_change(fill_method=None)
    df["oi_volume_ratio"] = df["oi"] / (df["volume"] + 1e-10)
    df["cancel_ratio"] = df["cancel_ratio"]

    # 3. 资金费率因子（4个）
    df["funding_rate_3d_mean"] = df["funding_rate"].rolling(288).mean()
    df["funding_rate_30d_mean"] = df["funding_rate"].rolling(2880).mean()
    df["funding_dev"] = df["funding_rate"] - df["funding_rate_30d_mean"]
    df["funding_sign"] = np.sign(df["funding_rate"])
    df["funding_switch"] = df["funding_sign"].diff().abs()
    df["funding_switch_3d"] = df["funding_switch"].rolling(288).sum()

    # 4. 市场情绪因子（5个）
    df["ls_ratio_30d_mean"] = df["long_short_ratio"].rolling(2880).mean()
    df["ls_ratio_dev"] = df["long_short_ratio"] - df["ls_ratio_30d_mean"]
    df["stablecoin_inflow"] = df["quote_volume"].pct_change(fill_method=None).rolling(24).mean()
    df["basis"] = df["close"] - df["spot_close"]
    df["basis_ratio"] = df["basis"] / df["spot_close"]
    df["atr_30d_rank"] = df["atr_14"].rolling(2880).rank(pct=True)
    df["market_sentiment"] = np.where(df["close"].pct_change() > 0, 1, 0).rolling(100).mean()

    # 严格锁定20个特征，顺序与原方案完全一致
    feature_cols = [
        "rsi_14", "macd_diff", "boll_width", "atr_14", "volume_ratio",
        "order_imbalance", "taker_ratio", "force_vol_ratio", "oi_change", "oi_volume_ratio", "cancel_ratio",
        "funding_rate", "funding_rate_3d_mean", "funding_dev", "funding_switch_3d",
        "ls_ratio_dev", "stablecoin_inflow", "basis_ratio", "atr_30d_rank", "market_sentiment"
    ]
    # 缺失值处理
    df[feature_cols] = df[feature_cols].ffill().bfill()
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols

# ==============================================
# 3. 标签计算（完全匹配原方案规则）
# ==============================================
def calculate_label_for_sample(sample, future_close):
    future_return = future_close / sample["close"] - 1
    kline_range = (sample["high"] - sample["low"]) / sample["close"]
    max_allowed_range = 2 * sample["atr_14"] / sample["close"]
    
    # 插针过滤
    if kline_range > max_allowed_range:
        return np.nan, np.abs(future_return)
    # 标签标注
    if future_return > LONG_THRESHOLD:
        return 1, np.abs(future_return)
    elif future_return < SHORT_THRESHOLD:
        return 0, np.abs(future_return)
    else:
        return np.nan, np.abs(future_return)

# ==============================================
# 4. 币安官方Websocket实时流处理（纯实时，无历史调用）
# ==============================================
async def handle_kline_socket(symbol, msg):
    """仅处理闭合的15min K线，触发特征计算与标签回填"""
    if msg["e"] != "kline" or not msg["k"]["x"]:
        return  # 仅处理K线闭合事件，不处理盘中数据
    kline_data = msg["k"]
    timestamp = pd.to_datetime(kline_data["t"], unit="ms", utc=True)
    cache = REAL_TIME_CACHE[symbol]

    # 聚合当前15min周期内的实时微观数据
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

    # 计算盘口与撤单数据
    if cache["depth_buffer"]:
        depth_df = pd.DataFrame(cache["depth_buffer"])
        kline_row["order_imbalance"] = depth_df["order_imbalance"].mean()
        total_add = depth_df["add_volume"].sum()
        total_cancel = depth_df["cancel_volume"].sum()
        kline_row["cancel_ratio"] = total_cancel / (total_add + 1e-10)
    else:
        kline_row["order_imbalance"] = 0
        kline_row["cancel_ratio"] = 0

    # 计算爆仓量
    kline_row["force_vol"] = sum([x["volume"] for x in cache["force_order_buffer"]])

    # 实时获取现货价格计算基差
    try:
        spot_ticker = await async_client.get_symbol_ticker(symbol=symbol)
        kline_row["spot_close"] = float(spot_ticker["price"])
    except:
        kline_row["spot_close"] = kline_row["close"]

    # 加入K线缓存，控制内存占用
    cache["kline_buffer"].append(kline_row)
    if len(cache["kline_buffer"]) > 1000:
        cache["kline_buffer"] = cache["kline_buffer"][-1000:]

    # 检查K线数量是否足够计算特征（至少需要SEQ_LENGTH根）
    if len(cache["kline_buffer"]) < SEQ_LENGTH:
        print(f"[{symbol}] 积累K线中：{len(cache['kline_buffer'])}/{SEQ_LENGTH}，暂无法计算特征")
        # 清空周期缓存
        cache["depth_buffer"] = []
        cache["force_order_buffer"] = []
        return

    # 计算20个特征
    kline_df = pd.DataFrame(cache["kline_buffer"])
    kline_df, feature_cols = calculate_features(kline_df)
    current_sample = kline_df.iloc[-1].to_dict()

    # 保存待回填标签的样本
    cache["unlabeled_samples"].append({
        "timestamp": current_sample["timestamp"],
        "close": current_sample["close"],
        "atr_14": current_sample["atr_14"],
        "feature_dict": current_sample
    })
    print(f"[{symbol}] K线闭合，特征计算完成，待回填样本数：{len(cache['unlabeled_samples'])}")

    # 标签回填（等待未来3根K线）
    if len(cache["unlabeled_samples"]) > LABEL_WINDOW:
        target_sample = cache["unlabeled_samples"].pop(0)
        future_close = current_sample["close"]
        label, conf_label = calculate_label_for_sample(target_sample, future_close)
        
        # 有效样本写入文件
        if not np.isnan(label):
            train_row = target_sample["feature_dict"]
            train_row["label"] = int(label)
            train_row["conf_label"] = conf_label
            # 增量写入CSV，自动处理表头
            is_header = not os.path.exists(CURRENT_TRAIN_FILE)
            pd.DataFrame([train_row]).to_csv(
                CURRENT_TRAIN_FILE,
                mode="a",
                header=is_header,
                index=False,
                encoding="utf-8-sig"
            )
            print(f"[{symbol}] 有效样本已写入，标签：{'做多' if label ==1 else '做空'}，文件：{CURRENT_TRAIN_FILE}")

    # 清空当前周期缓存
    cache["depth_buffer"] = []
    cache["force_order_buffer"] = []

async def handle_depth_socket(symbol, msg):
    """实时处理盘口深度流，计算盘口失衡率与撤单量"""
    if msg["e"] != "depthUpdate":
        return
    bids = msg["b"]
    asks = msg["a"]
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
    """实时处理爆仓单流"""
    if msg["e"] != "forceOrder" or msg["o"]["s"] != symbol:
        return
    REAL_TIME_CACHE[symbol]["force_order_buffer"].append({
        "volume": float(msg["o"]["q"]),
        "side": msg["o"]["S"]
    })

async def handle_open_interest_socket(symbol, msg):
    """实时更新未平仓量"""
    if msg["e"] != "openInterest":
        return
    REAL_TIME_CACHE[symbol]["open_interest"] = float(msg["o"])

async def handle_funding_rate_socket(symbol, msg):
    """实时更新资金费率"""
    if msg["e"] != "markPriceUpdate":
        return
    REAL_TIME_CACHE[symbol]["funding_rate"] = float(msg["r"])

async def subscribe_symbol_streams(symbol, bm):
    """订阅标的所有实时流，完全基于币安官方Websocket"""
    # 15min K线流
    kline_task = asyncio.create_task(bm.start_kline_socket(
        callback=lambda msg: handle_kline_socket(symbol, msg),
        symbol=symbol,
        interval=TIMEFRAME
    ))
    # 100ms深度流
    depth_task = asyncio.create_task(bm.start_depth_socket(
        callback=lambda msg: handle_depth_socket(symbol, msg),
        symbol=symbol,
        depth="100ms"
    ))
    # 爆仓单流
    force_task = asyncio.create_task(bm.start_force_order_socket(
        callback=lambda msg: handle_force_order_socket(symbol, msg),
        symbol=symbol
    ))
    # 未平仓量流
    oi_task = asyncio.create_task(bm.start_open_interest_socket(
        callback=lambda msg: handle_open_interest_socket(symbol, msg),
        symbol=symbol
    ))
    # 资金费率流
    funding_task = asyncio.create_task(bm.start_mark_price_socket(
        callback=lambda msg: handle_funding_rate_socket(symbol, msg),
        symbol=symbol,
        fast=False
    ))
    # 多空比定时更新（每分钟）
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
    ls_task = asyncio.create_task(update_long_short())

    return [kline_task, depth_task, force_task, oi_task, funding_task, ls_task]

# ==============================================
# 5. 训练张量生成（适配原方案LSTM模型）
# ==============================================
def build_train_tensor(merge_all_files=True):
    """生成适配原方案模型的训练张量，支持合并所有历史文件"""
    if merge_all_files:
        # 合并所有符合规范的训练文件
        file_list = glob.glob(f"{TRAIN_FILE_PREFIX}_*.csv")
        if not file_list:
            print("暂无训练集文件，请先运行采集脚本")
            return
        df_list = []
        for file in file_list:
            try:
                df = pd.read_csv(file)
                df_list.append(df)
            except:
                continue
        if not df_list:
            print("无有效训练数据")
            return
        full_df = pd.concat(df_list, axis=0)
    else:
        # 仅使用当前正在写入的文件
        if not os.path.exists(CURRENT_TRAIN_FILE):
            print("当前训练文件不存在")
            return
        full_df = pd.read_csv(CURRENT_TRAIN_FILE)

    # 数据预处理
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], utc=True)
    full_df = full_df.sort_values("timestamp").drop_duplicates(subset=["timestamp", "symbol"]).reset_index(drop=True)
    feature_cols = [
        "rsi_14", "macd_diff", "boll_width", "atr_14", "volume_ratio",
        "order_imbalance", "taker_ratio", "force_vol_ratio", "oi_change", "oi_volume_ratio", "cancel_ratio",
        "funding_rate", "funding_rate_3d_mean", "funding_dev", "funding_switch_3d",
        "ls_ratio_dev", "stablecoin_inflow", "basis_ratio", "atr_30d_rank", "market_sentiment"
    ]
    full_df = full_df.dropna(subset=feature_cols + ["label", "conf_label"]).reset_index(drop=True)

    # 滑窗构建LSTM输入
    features = full_df[feature_cols].values
    labels = full_df["label"].values
    conf_labels = full_df["conf_label"].values

    X, y, conf = [], [], []
    for i in range(SEQ_LENGTH, len(features)):
        X.append(features[i-SEQ_LENGTH:i, :])
        y.append(labels[i])
        conf.append(conf_labels[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    conf = np.array(conf, dtype=np.float32)

    # 标准化（仅用训练集，避免数据泄露）
    train_size = int(len(X) * 0.85)
    X_train = X[:train_size]
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    X = (X - mean) / (std + 1e-10)

    # 划分数据集
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    conf_train, conf_test = conf[:train_size], conf[train_size:]

    # 保存张量
    torch.save({
        "train": {"X": torch.tensor(X_train), "y": torch.tensor(y_train), "conf": torch.tensor(conf_train), "mean": mean, "std": std},
        "test": {"X": torch.tensor(X_test), "y": torch.tensor(y_test), "conf": torch.tensor(conf_test)},
        "feature_cols": feature_cols,
        "seq_length": SEQ_LENGTH,
        "feature_dim": FEATURE_DIM
    }, TENSOR_SAVE_PATH)
    print(f"训练张量生成完成，总样本数：{len(X)}，训练集：{len(X_train)}，测试集：{len(X_test)}")
    print(f"输入维度：{X_train.shape}，完全适配原方案PerpetualNN模型")

# ==============================================
# 【主程序入口，一键启动】
# ==============================================
async def main():
    global async_client
    print("="*60)
    print("币安永续合约纯实时训练集采集脚本")
    print("【风险提示】本脚本仅为技术研究，不构成任何投资建议")
    print("="*60)

    # 1. 初始化训练文件（判断是否新开/接续）
    init_train_file()

    # 2. 初始化异步客户端与Websocket管理器
    async_client = await AsyncClient.create()
    bm = BinanceSocketManager(async_client)

    # 3. 订阅所有标的的实时流
    all_tasks = []
    for symbol in SYMBOL_LIST:
        symbol_tasks = await subscribe_symbol_streams(symbol, bm)
        all_tasks.extend(symbol_tasks)
    print(f"已启动{len(SYMBOL_LIST)}个标的的实时流订阅，正在采集数据...")
    print(f"提示：启动后需积累{SEQ_LENGTH}根K线（约15小时）才会开始生成有效样本")

    # 4. 每日自动生成训练张量
    async def auto_build_tensor():
        while True:
            await asyncio.sleep(86400)
            build_train_tensor()
    all_tasks.append(asyncio.create_task(auto_build_tensor()))

    # 5. 异常自动重连
    try:
        await asyncio.gather(*all_tasks)
    except BinanceWebsocketUnableToConnect:
        print("Websocket连接断开，5秒后自动重启...")
        await asyncio.sleep(5)
        await main()
    except Exception as e:
        print(f"程序异常：{e}，5秒后自动重启")
        await asyncio.sleep(5)
        await main()
    finally:
        await async_client.close_connection()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序手动停止，正在生成最新训练张量...")
        build_train_tensor()
        print("程序已退出")