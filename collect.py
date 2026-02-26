import asyncio
import pandas as pd
import numpy as np
import glob
import os
import logging
from datetime import datetime
from binance.client import Client
from binance import AsyncClient
from binance.enums import FuturesType, ContractType
from binance.exceptions import BinanceWebsocketUnableToConnect

# ======================== 导入你提供的官方SocketManager源码 ========================
from enum import Enum
from typing import Optional, List, Dict, Callable, Any
from binance.ws.constants import KEEPALIVE_TIMEOUT
from binance.ws.keepalive_websocket import KeepAliveWebsocket
from binance.ws.reconnecting_websocket import ReconnectingWebsocket
from binance.ws.threaded_stream import ThreadedApiManager
from binance.helpers import get_loop

class BinanceSocketType(str, Enum):
    SPOT = "Spot"
    USD_M_FUTURES = "USD_M_Futures"
    COIN_M_FUTURES = "Coin_M_Futures"
    OPTIONS = "Vanilla_Options"
    ACCOUNT = "Account"

class BinanceSocketManager:
    STREAM_URL = "wss://stream.binance.{}:9443/"
    STREAM_TESTNET_URL = "wss://stream.testnet.binance.vision/"
    STREAM_DEMO_URL = "wss://demo-stream.binance.com/"
    FSTREAM_URL = "wss://fstream.binance.{}/"
    FSTREAM_TESTNET_URL = "wss://stream.binancefuture.com/"
    FSTREAM_DEMO_URL = "wss://fstream.binancefuture.com/"
    DSTREAM_URL = "wss://dstream.binance.{}/"
    DSTREAM_TESTNET_URL = "wss://dstream.binancefuture.com/"
    DSTREAM_DEMO_URL = "wss://dstream.binancefuture.com/"
    OPTIONS_URL = "wss://nbstream.binance.{}/eoptions/"

    WEBSOCKET_DEPTH_5 = "5"
    WEBSOCKET_DEPTH_10 = "10"
    WEBSOCKET_DEPTH_20 = "20"

    def __init__(
        self,
        client: AsyncClient,
        user_timeout=KEEPALIVE_TIMEOUT,
        max_queue_size: int = 100,
        verbose: bool = False,
    ):
        self.STREAM_URL = self.STREAM_URL.format(client.tld)
        self.FSTREAM_URL = self.FSTREAM_URL.format(client.tld)
        self.DSTREAM_URL = self.DSTREAM_URL.format(client.tld)
        self.OPTIONS_URL = self.OPTIONS_URL.format(client.tld)

        self._conns = {}
        self._loop = get_loop()
        self._client = client
        self._user_timeout = user_timeout
        self.testnet = self._client.testnet
        self.demo = self._client.demo
        self._max_queue_size = max_queue_size
        self.verbose = verbose
        self.ws_kwargs = {}

        if verbose:
            logging.getLogger('binance.ws').setLevel(logging.DEBUG)

    def _get_stream_url(self, stream_url: Optional[str] = None):
        if stream_url:
            return stream_url
        stream_url = self.STREAM_URL
        if self.testnet:
            stream_url = self.STREAM_TESTNET_URL
        elif self.demo:
            stream_url = self.STREAM_DEMO_URL
        return stream_url

    def _get_socket(
        self,
        path: str,
        stream_url: Optional[str] = None,
        prefix: str = "ws/",
        is_binary: bool = False,
        socket_type: BinanceSocketType = BinanceSocketType.SPOT,
    ) -> ReconnectingWebsocket:
        conn_id = f"{socket_type}_{path}"
        time_unit = getattr(self._client, "TIME_UNIT", None)
        if time_unit:
            path = f"{path}?timeUnit={time_unit}"
        if conn_id not in self._conns:
            self._conns[conn_id] = ReconnectingWebsocket(
                path=path,
                url=self._get_stream_url(stream_url),
                prefix=prefix,
                exit_coro=lambda p: self._exit_socket(f"{socket_type}_{p}"),
                is_binary=is_binary,
                https_proxy=self._client.https_proxy,
                max_queue_size=self._max_queue_size,** self.ws_kwargs,
            )

        return self._conns[conn_id]

    def _get_account_socket(
        self,
        path: str,
        stream_url: Optional[str] = None,
        prefix: str = "ws/",
        is_binary: bool = False,
    ) -> KeepAliveWebsocket:
        conn_id = f"{BinanceSocketType.ACCOUNT}_{path}"
        if conn_id not in self._conns:
            self._conns[conn_id] = KeepAliveWebsocket(
                client=self._client,
                url=self._get_stream_url(stream_url),
                keepalive_type=path,
                prefix=prefix,
                exit_coro=lambda p: self._exit_socket(conn_id),
                is_binary=is_binary,
                user_timeout=self._user_timeout,
                https_proxy=self._client.https_proxy,
                max_queue_size=self._max_queue_size,
                **self.ws_kwargs,
            )

        return self._conns[conn_id]

    def _get_futures_socket(
        self, path: str, futures_type: FuturesType, prefix: str = "stream?streams="
    ):
        socket_type: BinanceSocketType = BinanceSocketType.USD_M_FUTURES
        if futures_type == FuturesType.USD_M:
            stream_url = self.FSTREAM_URL
            if self.testnet:
                stream_url = self.FSTREAM_TESTNET_URL
            elif self.demo:
                stream_url = self.FSTREAM_DEMO_URL
        else:
            stream_url = self.DSTREAM_URL
            if self.testnet:
                stream_url = self.DSTREAM_TESTNET_URL
            elif self.demo:
                stream_url = self.DSTREAM_DEMO_URL
        return self._get_socket(path, stream_url, prefix, socket_type=socket_type)

    def _get_options_socket(self, path: str, prefix: str = "ws/"):
        stream_url = self.OPTIONS_URL
        return self._get_socket(
            path,
            stream_url,
            prefix,
            is_binary=False,
            socket_type=BinanceSocketType.OPTIONS,
        )

    async def _exit_socket(self, path: str):
        await self._stop_socket(path)

    def depth_socket(
        self, symbol: str, depth: Optional[str] = None, interval: Optional[int] = None
    ):
        socket_name = symbol.lower() + "@depth"
        if depth and depth != "1":
            socket_name = f"{socket_name}{depth}"
        if interval:
            if interval in [0, 100]:
                socket_name = f"{socket_name}@{interval}ms"
            else:
                raise ValueError(
                    "Websocket interval value not allowed. Allowed values are [0, 100]"
                )
        return self._get_socket(socket_name)

    def kline_socket(self, symbol: str, interval=AsyncClient.KLINE_INTERVAL_1MINUTE):
        path = f"{symbol.lower()}@kline_{interval}"
        return self._get_socket(path)

    def kline_futures_socket(
        self,
        symbol: str,
        interval=AsyncClient.KLINE_INTERVAL_1MINUTE,
        futures_type: FuturesType = FuturesType.USD_M,
        contract_type: ContractType = ContractType.PERPETUAL,
    ):
        path = f"{symbol.lower()}_{contract_type.value}@continuousKline_{interval}"
        return self._get_futures_socket(path, prefix="ws/", futures_type=futures_type)

    def miniticker_socket(self, update_time: int = 1000):
        return self._get_socket(f"!miniTicker@arr@{update_time}ms")

    def trade_socket(self, symbol: str):
        return self._get_socket(symbol.lower() + "@trade")

    def aggtrade_socket(self, symbol: str):
        return self._get_socket(symbol.lower() + "@aggTrade")

    def aggtrade_futures_socket(
        self, symbol: str, futures_type: FuturesType = FuturesType.USD_M
    ):
        return self._get_futures_socket(
            symbol.lower() + "@aggTrade", futures_type=futures_type
        )

    def symbol_miniticker_socket(self, symbol: str):
        return self._get_socket(symbol.lower() + "@miniTicker")

    def symbol_ticker_socket(self, symbol: str):
        return self._get_socket(symbol.lower() + "@ticker")

    def ticker_socket(self):
        return self._get_socket("!ticker@arr")

    def futures_ticker_socket(self):
        return self._get_futures_socket("!ticker@arr", FuturesType.USD_M)

    def futures_coin_ticker_socket(self):
        return self._get_futures_socket("!ticker@arr", FuturesType.COIN_M)

    def index_price_socket(self, symbol: str, fast: bool = True):
        stream_name = "@indexPrice@1s" if fast else "@indexPrice"
        return self._get_futures_socket(
            symbol.lower() + stream_name, futures_type=FuturesType.COIN_M
        )

    def symbol_mark_price_socket(
        self,
        symbol: str,
        fast: bool = True,
        futures_type: FuturesType = FuturesType.USD_M,
    ):
        stream_name = "@markPrice@1s" if fast else "@markPrice"
        return self._get_futures_socket(
            symbol.lower() + stream_name, futures_type=futures_type
        )

    def all_mark_price_socket(
        self, fast: bool = True, futures_type: FuturesType = FuturesType.USD_M
    ):
        stream_name = "!markPrice@arr@1s" if fast else "!markPrice@arr"
        return self._get_futures_socket(stream_name, futures_type=futures_type)

    def symbol_ticker_futures_socket(
        self, symbol: str, futures_type: FuturesType = FuturesType.USD_M
    ):
        return self._get_futures_socket(
            symbol.lower() + "@bookTicker", futures_type=futures_type
        )

    def individual_symbol_ticker_futures_socket(
        self, symbol: str, futures_type: FuturesType = FuturesType.USD_M
    ):
        return self._get_futures_socket(
            symbol.lower() + "@ticker", futures_type=futures_type
        )

    def all_ticker_futures_socket(
        self,
        channel: str = "!bookTicker",
        futures_type: FuturesType = FuturesType.USD_M,
    ):
        return self._get_futures_socket(channel, futures_type=futures_type)

    def symbol_book_ticker_socket(self, symbol: str):
        return self._get_socket(symbol.lower() + "@bookTicker")

    def book_ticker_socket(self):
        return self._get_socket("!bookTicker")

    def multiplex_socket(self, streams: List[str]):
        path = f"streams={'/'.join(streams)}"
        return self._get_socket(path, prefix="stream?")

    def options_multiplex_socket(self, streams: List[str]):
        stream_name = "/".join([s for s in streams])
        stream_path = f"streams={stream_name}"
        return self._get_options_socket(stream_path, prefix="stream?")

    def futures_multiplex_socket(
        self, streams: List[str], futures_type: FuturesType = FuturesType.USD_M
    ):
        path = f"streams={'/'.join(streams)}"
        return self._get_futures_socket(
            path, prefix="stream?", futures_type=futures_type
        )

    def user_socket(self):
        stream_url = self.STREAM_URL
        if self.testnet:
            stream_url = self.STREAM_TESTNET_URL
        elif self.demo:
            stream_url = self.STREAM_DEMO_URL
        return self._get_account_socket("user", stream_url=stream_url)

    def futures_user_socket(self):
        stream_url = self.FSTREAM_URL
        if self.testnet:
            stream_url = self.FSTREAM_TESTNET_URL
        elif self.demo:
            stream_url = self.FSTREAM_DEMO_URL
        return self._get_account_socket("futures", stream_url=stream_url)

    def coin_futures_user_socket(self):
        return self._get_account_socket("coin_futures", stream_url=self.DSTREAM_URL)

    def margin_socket(self):
        stream_url = self.STREAM_URL
        if self.testnet:
            stream_url = self.STREAM_TESTNET_URL
        elif self.demo:
            stream_url = self.STREAM_DEMO_URL
        return self._get_account_socket("margin", stream_url=stream_url)

    def futures_socket(self):
        stream_url = self.FSTREAM_URL
        if self.testnet:
            stream_url = self.FSTREAM_TESTNET_URL
        elif self.demo:
            stream_url = self.FSTREAM_DEMO_URL
        return self._get_account_socket("futures", stream_url=stream_url)

    def coin_futures_socket(self):
        stream_url = self.DSTREAM_URL
        if self.testnet:
            stream_url = self.DSTREAM_TESTNET_URL
        elif self.demo:
            stream_url = self.DSTREAM_DEMO_URL
        return self._get_account_socket("coin_futures", stream_url=stream_url)

    def portfolio_margin_socket(self):
        stream_url = self.FSTREAM_URL
        if self.testnet:
            stream_url = self.FSTREAM_TESTNET_URL
        elif self.demo:
            stream_url = self.FSTREAM_DEMO_URL
        stream_url += "pm/"
        return self._get_account_socket("portfolio_margin", stream_url=stream_url)

    def isolated_margin_socket(self, symbol: str):
        stream_url = self.STREAM_URL
        if self.testnet:
            stream_url = self.STREAM_TESTNET_URL
        elif self.demo:
            stream_url = self.STREAM_DEMO_URL
        return self._get_account_socket(symbol, stream_url=stream_url)

    def options_ticker_socket(self, symbol: str):
        return self._get_options_socket(symbol.upper() + "@ticker")

    def options_ticker_by_expiration_socket(self, symbol: str, expiration_date: str):
        return self._get_options_socket(symbol.upper() + "@ticker@" + expiration_date)

    def options_recent_trades_socket(self, symbol: str):
        return self._get_options_socket(symbol.upper() + "@trade")

    def options_kline_socket(
        self, symbol: str, interval=AsyncClient.KLINE_INTERVAL_1MINUTE
    ):
        return self._get_options_socket(symbol.upper() + "@kline_" + interval)

    def options_depth_socket(self, symbol: str, depth: str = "10"):
        return self._get_options_socket(symbol.upper() + "@depth" + str(depth))

    def futures_depth_socket(self, symbol: str, depth: str = "10", futures_type=FuturesType.USD_M):
        return self._get_futures_socket(
            symbol.lower() + "@depth" + str(depth), futures_type=futures_type
        )

    def futures_rpi_depth_socket(self, symbol: str, futures_type=FuturesType.USD_M):
        return self._get_futures_socket(
            symbol.lower() + "@rpiDepth@500ms", futures_type=futures_type
        )

    def options_new_symbol_socket(self):
        return self._get_options_socket("option_pair")

    def options_open_interest_socket(self, symbol: str, expiration_date: str):
        return self._get_options_socket(symbol.upper() + "@openInterest@" + expiration_date)

    def options_mark_price_socket(self, symbol: str):
        return self._get_options_socket(symbol.upper() + "@markPrice")

    def options_index_price_socket(self, symbol: str):
        return self._get_options_socket(symbol.upper() + "@index")

    async def _stop_socket(self, conn_key):
        if conn_key not in self._conns:
            return
        del self._conns[conn_key]

# ======================== 核心业务逻辑（期货+现货双整合） ========================
# 配置项
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
MAX_QUEUE_SIZE = 2000  # 增大队列，适配双轨流数据

# 全局缓存（新增现货专属子缓存）
REAL_TIME_CACHE = {symbol: {
    # 期货永续数据
    "fut_kline_buffer": [],
    "fut_depth_buffer": [],
    "open_interest": 0.0,
    "long_short_ratio": 0.0,
    "funding_rate": 0.0,
    "fut_last_bid_vol": 0.0,
    "fut_last_ask_vol": 0.0,
    # 现货数据（新增）
    "spot_kline_buffer": [],
    "spot_depth_buffer": [],
    "spot_last_bid_vol": 0.0,
    "spot_last_ask_vol": 0.0,
    "spot_latest_kline": None,  # 存储最新现货闭合K线
    # 标签缓存
    "unlabeled_samples": []
} for symbol in SYMBOL_LIST}
CURRENT_CSV_FILE = ""
async_client = None

# ======================== CSV文件管理（无变化） ========================
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

# ======================== 特征计算（适配现货实时数据） ========================
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
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    # 期货数据列
    fut_close = df["fut_close"].values
    fut_high = df["fut_high"].values
    fut_low = df["fut_low"].values
    fut_volume = df["fut_volume"].values
    # 现货数据列（新增）
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

    # 4. 市场情绪+基差指标（新增现货完整数据）
    df["ls_ratio_dev"] = df["long_short_ratio"] - df["long_short_ratio"].rolling(2880, min_periods=1).mean()
    df["stablecoin_inflow"] = df["fut_quote_volume"].pct_change(fill_method=None).rolling(24, min_periods=1).mean().fillna(0)
    df["basis"] = df["fut_close"] - df["spot_close"]  # 用现货WS实时收盘价
    df["basis_ratio"] = df["basis"] / df["spot_close"]
    df["atr_30d_rank"] = df["atr_14"].rolling(2880, min_periods=1).rank(pct=True).fillna(0.5)
    df["market_sentiment"] = np.where(df["fut_close"].pct_change() > 0, 1, 0).rolling(100, min_periods=1).mean().fillna(0.5)

    # 20个核心特征列（适配字段名）
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

# ======================== 消息处理（期货+现货分离处理） ========================
# --- 现货消息处理（新增）---
async def handle_spot_kline_socket(symbol, msg):
    """处理现货K线流，更新最新现货K线缓存"""
    if msg.get("e") != "kline" or not msg.get("k", {}).get("x"):
        return
    kline_data = msg["k"]
    cache = REAL_TIME_CACHE[symbol]
    # 构造现货K线数据
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
    # 更新缓存
    cache["spot_kline_buffer"].append(spot_kline)
    if len(cache["spot_kline_buffer"]) > 1000:
        cache["spot_kline_buffer"] = cache["spot_kline_buffer"][-1000:]
    cache["spot_latest_kline"] = spot_kline  # 存储最新闭合现货K线

async def handle_spot_depth_socket(symbol, msg):
    """处理现货深度流，更新现货深度缓存"""
    if msg.get("e") != "depthUpdate":
        return
    bids = msg.get("b", [])
    asks = msg.get("a", [])
    if not bids or not asks:
        return
    cache = REAL_TIME_CACHE[symbol]
    bid1_vol = float(bids[0][1])
    ask1_vol = float(asks[0][1])
    order_imbalance = (bid1_vol - ask1_vol) / (bid1_vol + ask1_vol + 1e-10)
    add_volume = max(0.0, bid1_vol - cache["spot_last_bid_vol"]) + max(0.0, ask1_vol - cache["spot_last_ask_vol"])
    cancel_volume = max(0.0, cache["spot_last_bid_vol"] - bid1_vol) + max(0.0, cache["spot_last_ask_vol"] - ask1_vol)
    # 更新缓存
    cache["spot_last_bid_vol"] = bid1_vol
    cache["spot_last_ask_vol"] = ask1_vol
    cache["spot_depth_buffer"].append({
        "spot_order_imbalance": order_imbalance,
        "spot_add_volume": add_volume,
        "spot_cancel_volume": cancel_volume
    })

# --- 期货消息处理（适配现货缓存）---
async def handle_fut_kline_socket(symbol, msg):
    """处理期货永续K线流，整合现货数据写入CSV"""
    if msg.get("e") != "continuous_kline" or not msg.get("k", {}).get("x"):
        return
    kline_data = msg["k"]
    timestamp = pd.to_datetime(kline_data["t"], unit="ms", utc=True)
    cache = REAL_TIME_CACHE[symbol]

    # 1. 获取最新现货数据（核心：从WS缓存取，不再轮询）
    spot_latest = cache.get("spot_latest_kline", {})
    spot_close = spot_latest.get("spot_close", float(kline_data["c"]))
    spot_open = spot_latest.get("spot_open", float(kline_data["o"]))
    spot_high = spot_latest.get("spot_high", float(kline_data["h"]))
    spot_low = spot_latest.get("spot_low", float(kline_data["l"]))
    spot_volume = spot_latest.get("spot_volume", 0.0)

    # 2. 构造期货K线行（字段名加fut_前缀，避免与现货冲突）
    fut_kline_row = {
        "timestamp": timestamp,
        # 期货基础列
        "fut_open": float(kline_data["o"]),
        "fut_high": float(kline_data["h"]),
        "fut_low": float(kline_data["l"]),
        "fut_close": float(kline_data["c"]),
        "fut_volume": float(kline_data["v"]),
        "fut_quote_volume": float(kline_data["q"]),
        "fut_trades": int(kline_data["n"]),
        "fut_taker_buy_base": float(kline_data["V"]),
        "fut_taker_sell_base": float(kline_data["v"]) - float(kline_data["V"]),
        # 期货合约列
        "symbol": symbol,
        "oi": cache["open_interest"],
        "long_short_ratio": cache["long_short_ratio"],
        "funding_rate": cache["funding_rate"],
        # 期货深度列
        "fut_order_imbalance": 0.0,
        "fut_cancel_ratio": 0.0,
        # 现货完整列（新增）
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
    fut_df = pd.DataFrame(cache["fut_kline_buffer"])
    fut_df, feature_cols = calculate_features(fut_df)
    current_sample = fut_df.iloc[-1].to_dict()

    # 写入CSV
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

async def handle_fut_depth_socket(symbol, msg):
    """处理期货深度流"""
    if msg.get("e") != "depthUpdate":
        return
    bids = msg.get("b", [])
    asks = msg.get("a", [])
    if not bids or not asks:
        return
    cache = REAL_TIME_CACHE[symbol]
    bid1_vol = float(bids[0][1])
    ask1_vol = float(asks[0][1])
    order_imbalance = (bid1_vol - ask1_vol) / (bid1_vol + ask1_vol + 1e-10)
    add_volume = max(0.0, bid1_vol - cache["fut_last_bid_vol"]) + max(0.0, ask1_vol - cache["fut_last_ask_vol"])
    cancel_volume = max(0.0, cache["fut_last_bid_vol"] - bid1_vol) + max(0.0, cache["fut_last_ask_vol"] - ask1_vol)
    # 更新期货深度缓存
    cache["fut_last_bid_vol"] = bid1_vol
    cache["fut_last_ask_vol"] = ask1_vol
    cache["fut_depth_buffer"].append({
        "fut_order_imbalance": order_imbalance,
        "fut_add_volume": add_volume,
        "fut_cancel_volume": cancel_volume
    })

async def handle_fut_mark_price_socket(symbol, msg):
    """处理期货资金费率流"""
    if msg.get("e") != "markPriceUpdate":
        return
    REAL_TIME_CACHE[symbol]["funding_rate"] = float(msg["r"])

# ======================== 轮询任务（添加异常保护） ========================
async def update_open_interest(symbol):
    while True:
        try:
            oi_data = await async_client.futures_open_interest(symbol=symbol)
            if oi_data:
                REAL_TIME_CACHE[symbol]["open_interest"] = float(oi_data["openInterest"])
        except asyncio.CancelledError:
            break  # 正常退出
        except Exception as e:
            print(f"[{symbol}] OI update failed: {e}, retry in 60s...")
        await asyncio.sleep(60)

async def update_long_short(symbol):
    while True:
        try:
            ls_data = await async_client.futures_top_longshort_account_ratio(
                symbol=symbol, period="5m", limit=1
            )
            if ls_data:
                REAL_TIME_CACHE[symbol]["long_short_ratio"] = float(ls_data[0]["longShortRatio"])
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[{symbol}] Long/Short ratio update failed: {e}, retry in 60s...")
        await asyncio.sleep(60)

# ======================== WebSocket订阅（添加自动重连） ========================
async def subscribe_symbol_streams(symbol, bm):
    """订阅单个币种的【期货流+现货流】，每个流自带重连逻辑"""
    # --- 现货流任务（新增重连）---
    async def spot_kline_task():
        conn_id = f"{BinanceSocketType.SPOT}_{symbol.lower()}@kline_{TIMEFRAME}"
        while True:
            try:
                socket = bm.kline_socket(symbol=symbol, interval=TIMEFRAME)
                while True:
                    msg = await socket.recv()
                    await handle_spot_kline_socket(symbol, msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[{symbol} Spot Kline] Error: {e}, reconnecting in 5s...")
                await bm._stop_socket(conn_id)  # 清理旧连接
                await asyncio.sleep(5)

    async def spot_depth_task():
        conn_id = f"{BinanceSocketType.SPOT}_{symbol.lower()}@depth10@100ms"
        while True:
            try:
                socket = bm.depth_socket(symbol=symbol, depth="10", interval=100)
                while True:
                    msg = await socket.recv()
                    await handle_spot_depth_socket(symbol, msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[{symbol} Spot Depth] Error: {e}, reconnecting in 5s...")
                await bm._stop_socket(conn_id)
                await asyncio.sleep(5)

    # --- 期货流任务（保留并修正）---
    async def fut_kline_task():
        path = f"{symbol.lower()}_PERPETUAL@continuousKline_{TIMEFRAME}"
        conn_id = f"{BinanceSocketType.USD_M_FUTURES}_{path}"
        while True:
            try:
                socket = bm.kline_futures_socket(
                    symbol=symbol,
                    interval=TIMEFRAME,
                    futures_type=FuturesType.USD_M,
                    contract_type=ContractType.PERPETUAL
                )
                while True:
                    msg = await socket.recv()
                    await handle_fut_kline_socket(symbol, msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[{symbol} Futures Kline] Error: {e}, reconnecting in 5s...")
                await bm._stop_socket(conn_id)
                await asyncio.sleep(5)

    async def fut_depth_task():
        path = f"{symbol.lower()}@depth10"
        conn_id = f"{BinanceSocketType.USD_M_FUTURES}_{path}"
        while True:
            try:
                socket = bm.futures_depth_socket(
                    symbol=symbol,
                    depth="10",
                    futures_type=FuturesType.USD_M
                )
                while True:
                    msg = await socket.recv()
                    await handle_fut_depth_socket(symbol, msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[{symbol} Futures Depth] Error: {e}, reconnecting in 5s...")
                await bm._stop_socket(conn_id)
                await asyncio.sleep(5)

    async def fut_mark_price_task():
        path = f"{symbol.lower()}@markPrice"
        conn_id = f"{BinanceSocketType.USD_M_FUTURES}_{path}"
        while True:
            try:
                socket = bm.symbol_mark_price_socket(
                    symbol=symbol,
                    fast=False,
                    futures_type=FuturesType.USD_M
                )
                while True:
                    msg = await socket.recv()
                    await handle_fut_mark_price_socket(symbol, msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[{symbol} Futures Mark Price] Error: {e}, reconnecting in 5s...")
                await bm._stop_socket(conn_id)
                await asyncio.sleep(5)

    # --- 轮询任务（已在上方函数内添加异常保护）---
    oi_task = update_open_interest(symbol)
    ls_task = update_long_short(symbol)

    # 返回所有任务
    return [
        asyncio.create_task(spot_kline_task()),
        asyncio.create_task(spot_depth_task()),
        asyncio.create_task(fut_kline_task()),
        asyncio.create_task(fut_depth_task()),
        asyncio.create_task(fut_mark_price_task()),
        asyncio.create_task(oi_task),
        asyncio.create_task(ls_task)
    ]

# ======================== 安全退出（无变化） ========================
async def safe_shutdown(all_tasks, bm):
    print("\nInitiating safe shutdown...")
    for task in all_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    if bm is not None:
        for conn_key in list(bm._conns.keys()):
            await bm._stop_socket(conn_key)
        print("WebSocket connections closed")
    global async_client
    if async_client is not None:
        await async_client.close_connection()
        print("Async client closed")
    print(f"Data saved to: {CURRENT_CSV_FILE}")

# ======================== 主程序 ========================
async def main():
    global async_client
    print("="*60)
    print("Binance Futures + Spot CSV Collector (Official SocketManager)")
    print("="*60)

    init_csv_file()
    async_client = await AsyncClient.create()
    bm = BinanceSocketManager(
        client=async_client,
        max_queue_size=MAX_QUEUE_SIZE,
        verbose=False
    )

    all_tasks = []
    for symbol in SYMBOL_LIST:
        symbol_tasks = await subscribe_symbol_streams(symbol, bm)
        all_tasks.extend(symbol_tasks)

    print(f"Subscribed to {len(SYMBOL_LIST)} symbols (Futures + Spot). Collecting data...")
    print(f"Note: Need 60 K-lines (≈15 hours) for first valid sample.")  # 修正为60

    try:
        await asyncio.gather(*all_tasks)
    except KeyboardInterrupt:
        await safe_shutdown(all_tasks, bm)
    except BinanceWebsocketUnableToConnect as e:
        print(f"WS connection failed: {e}")
        await safe_shutdown(all_tasks, bm)
    except Exception as e:
        print(f"Unexpected error: {e}")
        await safe_shutdown(all_tasks, bm)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram stopped manually.")