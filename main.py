import os
import sys
import time
import logging
import threading
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from dotenv import load_dotenv
from datetime import datetime
from colorama import init, Fore, Style
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

sys.stdout.reconfigure(encoding='utf-8')
init(autoreset=True)

# ==================== 日志系统 ====================
def setup_logger():
    main_logger = logging.getLogger('VWT_Main')
    main_logger.setLevel(logging.DEBUG)
    main_logger.propagate = False
    signal_logger = logging.getLogger('VWT_Signal')
    signal_logger.setLevel(logging.INFO)
    signal_logger.propagate = False

    if not main_logger.handlers:
        file_handler = logging.FileHandler(f'vwt_full_log_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        main_logger.addHandler(file_handler)
        main_logger.addHandler(console_handler)

    if not signal_logger.handlers:
        signal_file_handler = logging.FileHandler(f'vwt_signal_log_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
        signal_file_handler.setLevel(logging.INFO)
        signal_formatter = logging.Formatter('%(asctime)s | %(message)s')
        signal_file_handler.setFormatter(signal_formatter)
        signal_logger.addHandler(signal_file_handler)

    return main_logger, signal_logger

main_logger, signal_logger = setup_logger()

# ==================== 配置参数 ====================
load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

SPOT_SYMBOL = "ETHUSDC"          # 现货K线源
FUTURES_SYMBOL = "ETHUSDC"       # 合约交易对
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
LOOKBACK = 600

# VWT 参数
VWMA_LENGTH = 34
VWT_ATR_MULT = 1.5

# L1 滤波器参数
L1_ATR_MULT = 1.5
L1_MU = 0.6

# 趋势过滤参数（必须与 TradingView 指标完全一致）
TREND_CONFIRM_BARS = 2
USE_CONSENSUS_FILTER = True
USE_ATR_FILTER = True
MIN_ATR_PCT = 0.001               # 0.1%

# 杠杆与风险管理
LEVERAGE = 20
MARGIN_TYPE = "ISOLATED"
RISK_PERCENTAGE = 50
ADD_RISK_PCT = 20

# 止损（独立于移动止损）
STOP_LOSS_PCT = 1.5
ENABLE_STOP_LOSS = False

# 流动性扫盘参数
LIQ_SWEEP_LENGTH = 8
LIQ_PARTIAL_PROFIT_RATIO = 0.5
BREAKOUT_CONFIRM_BARS = 2
BREAKOUT_THRESHOLD_PCT = 0.1

# 状态管理
MAX_ADD_TIMES = 1
NEW_ZONE_THRESHOLD_PCT = 0.5
STATE_RESET_DELAY = 1

# 移动止损
ENABLE_TRAILING_STOP = True
TRAILING_ATR_MULT = 1.2

# 止损单类型
STOP_ORDER_TYPE = "STOP_MARKET"
STOP_WORKING_TYPE = "MARK_PRICE"
STOP_TIME_IN_FORCE = "GTC"

# 初始化客户端
client = Client(API_KEY, API_SECRET, testnet=False, requests_params={'timeout': 30})
main_logger.info(Fore.CYAN + "✅ Binance live trading client initialized")
main_logger.info(Fore.CYAN + f"📊 Signal: {SPOT_SYMBOL} spot → Trade: {FUTURES_SYMBOL} futures")
main_logger.info(Fore.CYAN + f"📈 Filter params: confirm={TREND_CONFIRM_BARS}, consensus={USE_CONSENSUS_FILTER}, atr_filter={USE_ATR_FILTER}")

# 线程锁
trade_lock = threading.Lock()

# ==================== 状态类 ====================
@dataclass
class SideState:
    position_size: float = 0.0
    entry_price: float = 0.0
    initial_entry_price: float = 0.0
    trend_at_open: int = 0          # 开仓时的过滤趋势：1=看涨，-1=看跌
    is_trend_valid: bool = False
    last_operated_zone_price: float = 0.0
    has_partial_tp_in_zone: bool = False
    has_added_in_zone: bool = False
    total_add_times: int = 0
    last_add_price: float = 0.0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 0.0
    stop_order_id: Optional[int] = None
    # 等待价格触及L1线标志（仅当无持仓且趋势刚变时设置）
    awaiting_entry: bool = False
    awaiting_trend: int = 0

    def reset(self):
        self.position_size = 0.0
        self.entry_price = 0.0
        self.initial_entry_price = 0.0
        self.trend_at_open = 0
        self.is_trend_valid = False
        self.last_operated_zone_price = 0.0
        self.has_partial_tp_in_zone = False
        self.has_added_in_zone = False
        self.total_add_times = 0
        self.last_add_price = 0.0
        self.highest_since_entry = 0.0
        self.lowest_since_entry = 0.0
        self.stop_order_id = None
        self.awaiting_entry = False
        self.awaiting_trend = 0

    def init_new_position(self, pos_size: float, entry_price: float, trend: int):
        self.reset()
        self.position_size = pos_size
        self.entry_price = entry_price
        self.initial_entry_price = entry_price
        self.trend_at_open = trend
        self.is_trend_valid = True
        self.highest_since_entry = entry_price
        self.lowest_since_entry = entry_price

    def update_position(self, pos_size: float, entry_price: float):
        self.position_size = pos_size
        self.entry_price = entry_price

    def is_new_liquidity_zone(self, current_zone_price: float, pos_dir: str) -> bool:
        if self.last_operated_zone_price == 0:
            return True
        diff_pct = abs(current_zone_price - self.last_operated_zone_price) / self.last_operated_zone_price * 100
        if pos_dir == "long":
            is_new = (current_zone_price > self.last_operated_zone_price) and (diff_pct >= NEW_ZONE_THRESHOLD_PCT)
        elif pos_dir == "short":
            is_new = (current_zone_price < self.last_operated_zone_price) and (diff_pct >= NEW_ZONE_THRESHOLD_PCT)
        else:
            is_new = False
        if is_new:
            self.has_partial_tp_in_zone = False
            self.has_added_in_zone = False
            main_logger.info(Fore.CYAN + f"🎯 New liquidity zone | Price:{current_zone_price:.2f} | Diff:{diff_pct:.2f}%")
        return is_new

@dataclass
class TradeState:
    long_state: SideState = field(default_factory=SideState)
    short_state: SideState = field(default_factory=SideState)

    def reset_side(self, side: str):
        if side == "long":
            self.long_state.reset()
            main_logger.info(Fore.YELLOW + "🔄 Long state reset")
        elif side == "short":
            self.short_state.reset()
            main_logger.info(Fore.YELLOW + "🔄 Short state reset")

    def reset_all(self):
        self.long_state.reset()
        self.short_state.reset()
        main_logger.info(Fore.YELLOW + "🔄 All state reset")

trade_state = TradeState()

# ==================== 核心指标计算 ====================
def calculate_atr_rma(data: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = data['high'], data['low'], data['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calculate_vwt(data: pd.DataFrame, vwma_len: int, atr_mult: float) -> tuple:
    close, volume = data['close'], data['volume']
    vwma = (close * volume).rolling(vwma_len).sum() / volume.rolling(vwma_len).sum()
    atr = calculate_atr_rma(data, vwma_len)
    upper = vwma + atr * atr_mult
    lower = vwma - atr * atr_mult
    trend = pd.Series(0, index=data.index)
    trend[close > upper] = 1
    trend[close < lower] = -1
    return vwma, upper, lower, trend, atr

def calculate_l1_filter(data: pd.DataFrame, atr_mult: float, mu: float, src='close') -> pd.Series:
    src_arr = data[src].values
    length = len(src_arr)
    z = np.zeros(length)
    v = np.zeros(length)
    atr_200 = calculate_atr_rma(data, 200).values
    thresh = atr_200 * atr_mult
    for i in range(length):
        if i == 0:
            z[i] = src_arr[i]
            v[i] = 0.0
        else:
            z_prev, v_prev = z[i-1], v[i-1]
            z_pred = z_prev + v_prev
            z_temp = z_pred + mu * (src_arr[i] - z_pred)
            diff = z_temp - z_prev
            if abs(diff) > thresh[i]:
                v[i] = np.sign(diff) * (abs(diff) - thresh[i])
            else:
                v[i] = 0.0
            z[i] = z_prev + v[i]
    return pd.Series(z, index=data.index)

def calculate_l1_trend(l1_series: pd.Series) -> pd.Series:
    trend = pd.Series(0, index=l1_series.index)
    trend[l1_series > l1_series.shift(1)] = 1
    trend[l1_series < l1_series.shift(1)] = -1
    return trend

def calculate_filtered_trend_series(data: pd.DataFrame,
                                     vwma_len: int, vwt_mult: float,
                                     l1_mult: float, l1_mu: float,
                                     confirm: int, use_consensus: bool,
                                     use_atr_filter: bool, min_atr_pct: float) -> pd.Series:
    _, _, _, vwt_trend, atr_series = calculate_vwt(data, vwma_len, vwt_mult)
    if use_consensus:
        l1_series = calculate_l1_filter(data, l1_mult, l1_mu)
        l1_trend = calculate_l1_trend(l1_series)
    else:
        l1_trend = pd.Series(0, index=data.index)

    filtered = pd.Series(0, index=data.index)
    close = data['close']
    for i in range(confirm - 1, len(data)):
        recent_vwt = vwt_trend.iloc[i-confirm+1 : i+1].values
        all_bull = all(v == 1 for v in recent_vwt)
        all_bear = all(v == -1 for v in recent_vwt)
        if use_consensus:
            recent_l1 = l1_trend.iloc[i-confirm+1 : i+1].values
            all_bull = all_bull and all(l == 1 for l in recent_l1)
            all_bear = all_bear and all(l == -1 for l in recent_l1)
        atr_ok = (not use_atr_filter) or (atr_series.iloc[i] / close.iloc[i] >= min_atr_pct)
        if all_bull and atr_ok:
            filtered.iloc[i] = 1
        elif all_bear and atr_ok:
            filtered.iloc[i] = -1
        else:
            filtered.iloc[i] = 0
    return filtered

# ==================== 全局变量（供价格线程） ====================
latest_z_value = 0.0
latest_filtered_trend = 0
latest_price = 0.0
latest_atr = 0.0

# ==================== 流动性区域检测 ====================
def detect_liquidity_zones(data: pd.DataFrame, lookback: int = 8) -> dict:
    df = data.copy()
    closed = df.iloc[:-1].copy()
    nearest_res, nearest_sup = np.nan, np.nan
    if len(closed) < lookback * 2 + 1:
        return {'resistance': nearest_res, 'support': nearest_sup}
    closed['is_pivot_high'] = closed['high'] == closed['high'].rolling(lookback*2+1, center=True).max()
    closed['is_pivot_low'] = closed['low'] == closed['low'].rolling(lookback*2+1, center=True).min()
    pivot_highs = closed[closed['is_pivot_high']]['high']
    pivot_lows = closed[closed['is_pivot_low']]['low']
    cur_price = df['close'].iloc[-1]
    if not pivot_highs.empty:
        valid = pivot_highs[pivot_highs > cur_price]
        if not valid.empty:
            nearest_res = valid.iloc[-1]
    if not pivot_lows.empty:
        valid = pivot_lows[pivot_lows < cur_price]
        if not valid.empty:
            nearest_sup = valid.iloc[-1]
    return {'resistance': nearest_res, 'support': nearest_sup}

def confirm_breakout(data: pd.DataFrame, zone_price: float, pos_dir: str) -> bool:
    if len(data) < BREAKOUT_CONFIRM_BARS:
        return False
    recent = data.iloc[-(BREAKOUT_CONFIRM_BARS+1):-1]
    if pos_dir == "long":
        level = zone_price * (1 + BREAKOUT_THRESHOLD_PCT/100)
        return all(recent['close'] > level)
    elif pos_dir == "short":
        level = zone_price * (1 - BREAKOUT_THRESHOLD_PCT/100)
        return all(recent['close'] < level)
    return False

# ==================== 交易辅助函数 ====================
def setup_hedge_mode(symbol: str):
    try:
        client.futures_change_position_mode(dualSidePosition=True)
        main_logger.info(Fore.GREEN + "✅ Hedge mode enabled")
    except BinanceAPIException as e:
        if "No need to change" not in str(e):
            main_logger.error(Fore.RED + f"❌ Hedge mode failed: {e}")
            raise

def setup_leverage_margin(symbol: str, lev: int, margin: str):
    try:
        client.futures_change_margin_type(symbol=symbol, marginType=margin)
    except BinanceAPIException as e:
        if "No need to change" not in str(e):
            main_logger.warning(Fore.YELLOW + f"⚠️ Margin note: {e}")
    try:
        client.futures_change_leverage(symbol=symbol, leverage=lev)
        main_logger.info(Fore.CYAN + f"🔧 Leverage: {lev}x")
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Leverage setup failed: {e}")
        raise

def get_position(symbol: str) -> Tuple[Dict, Dict]:
    long_info = {'size': 0.0, 'entry_price': 0.0}
    short_info = {'size': 0.0, 'entry_price': 0.0}
    try:
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            if pos['symbol'] != symbol:
                continue
            side = pos['positionSide']
            amt = float(pos['positionAmt'])
            price = float(pos['entryPrice'])
            if side == 'LONG' and amt > 0:
                long_info['size'] = amt
                long_info['entry_price'] = price
                main_logger.info(Fore.CYAN + f"📈 Long: {amt} @ {price}")
            elif side == 'SHORT' and amt > 0:
                short_info['size'] = amt
                short_info['entry_price'] = price
                main_logger.info(Fore.CYAN + f"📉 Short: {amt} @ {price}")
        return long_info, short_info
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Get position error: {e}")
        return long_info, short_info

def get_usdc_balance() -> float:
    try:
        balance = client.futures_account_balance()
        for asset in balance:
            if asset['asset'] == 'USDC':
                return float(asset['availableBalance'])
        return 0.0
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Balance error: {e}")
        return 0.0

def get_precision(symbol: str) -> tuple:
    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                return int(s['pricePrecision']), int(s['quantityPrecision'])
        return 2, 3
    except:
        return 2, 3

def calculate_position_size(symbol: str, usdc: float, risk_pct: float, lev: int, price: float) -> float:
    try:
        info = client.futures_exchange_info()
        sym_info = None
        for s in info['symbols']:
            if s['symbol'] == symbol:
                sym_info = s
                break
        if not sym_info:
            return 0.0
        qty_prec = int(sym_info['quantityPrecision'])
        min_qty = float(sym_info['filters'][1]['minQty'])
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Precision error: {e}")
        return 0.0
    risk_amount = usdc * (risk_pct / 100)
    notional = risk_amount * lev
    size = notional / price
    adjusted = round(size, qty_prec)
    if adjusted < min_qty:
        adjusted = min_qty
    return adjusted

def place_market_order(symbol: str, side: str, qty: float, pos_side: str) -> dict:
    try:
        _, qty_prec = get_precision(symbol)
        qty = round(qty, qty_prec)
        order = client.futures_create_order(
            symbol=symbol, side=side, type=Client.ORDER_TYPE_MARKET,
            quantity=qty, positionSide=pos_side
        )
        action = f"{pos_side} Open" if (pos_side=='LONG' and side==Client.SIDE_BUY) or (pos_side=='SHORT' and side==Client.SIDE_SELL) else f"{pos_side} Close"
        main_logger.info(Fore.GREEN + f"✅ {action} success | ID: {order['orderId']} | Qty: {qty}")
        return order
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Order failed: {e}")
        return None

def place_stop_order(symbol: str, side: str, qty: float, stop_price: float, pos_side: str) -> Optional[dict]:
    try:
        price_prec, qty_prec = get_precision(symbol)
        qty = round(qty, qty_prec)
        stop_price = round(stop_price, price_prec)
        order = client.futures_create_order(
            symbol=symbol, side=side, type=STOP_ORDER_TYPE,
            quantity=qty, stopPrice=stop_price,
            workingType=STOP_WORKING_TYPE, timeInForce=STOP_TIME_IN_FORCE,
            positionSide=pos_side
        )
        main_logger.info(Fore.GREEN + f"✅ Stop order placed | ID: {order['orderId']} | {pos_side} stop@{stop_price}")
        return order
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Stop order failed: {e}")
        return None

def cancel_stop_order(symbol: str, order_id: int) -> bool:
    try:
        client.futures_cancel_order(symbol=symbol, orderId=order_id)
        main_logger.info(Fore.YELLOW + f"✅ Stop cancelled | ID: {order_id}")
        return True
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Cancel stop failed: {e}")
        return False

def cancel_all_stop_orders(symbol: str, pos_side: str):
    try:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        for order in open_orders:
            if order['type'] == STOP_ORDER_TYPE and order['positionSide'] == pos_side:
                cancel_stop_order(symbol, int(order['orderId']))
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Cancel all stops error: {e}")

def update_trailing_stop(symbol: str, pos_side: str, new_stop: float, qty: float) -> bool:
    side = Client.SIDE_SELL if pos_side == 'LONG' else Client.SIDE_BUY
    if pos_side == 'LONG' and trade_state.long_state.stop_order_id:
        cancel_stop_order(symbol, trade_state.long_state.stop_order_id)
        trade_state.long_state.stop_order_id = None
    elif pos_side == 'SHORT' and trade_state.short_state.stop_order_id:
        cancel_stop_order(symbol, trade_state.short_state.stop_order_id)
        trade_state.short_state.stop_order_id = None
    else:
        cancel_all_stop_orders(symbol, pos_side)
    order = place_stop_order(symbol, side, qty, new_stop, pos_side)
    if order:
        if pos_side == 'LONG':
            trade_state.long_state.stop_order_id = int(order['orderId'])
        else:
            trade_state.short_state.stop_order_id = int(order['orderId'])
        return True
    return False

def restore_stop_orders(symbol: str):
    try:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        for order in open_orders:
            if order['type'] != STOP_ORDER_TYPE:
                continue
            pos_side = order['positionSide']
            if pos_side == 'LONG' and trade_state.long_state.position_size > 0:
                trade_state.long_state.stop_order_id = int(order['orderId'])
                main_logger.info(Fore.GREEN + f"🔄 Restored LONG stop {order['orderId']}")
            elif pos_side == 'SHORT' and trade_state.short_state.position_size > 0:
                trade_state.short_state.stop_order_id = int(order['orderId'])
                main_logger.info(Fore.GREEN + f"🔄 Restored SHORT stop {order['orderId']}")
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Restore stops error: {e}")

def check_stop_loss(symbol: str, price: float) -> Tuple[bool, str]:
    if not ENABLE_STOP_LOSS:
        return False, "none"
    long_info, short_info = get_position(symbol)
    if long_info['size'] > 0:
        loss = (long_info['entry_price'] - price) / long_info['entry_price'] * 100
        if loss >= STOP_LOSS_PCT:
            return True, "long"
    if short_info['size'] > 0:
        loss = (price - short_info['entry_price']) / short_info['entry_price'] * 100
        if loss >= STOP_LOSS_PCT:
            return True, "short"
    return False, "none"

def restore_trade_state():
    long_info, short_info = get_position(FUTURES_SYMBOL)
    if long_info['size'] > 0:
        trade_state.long_state.init_new_position(long_info['size'], long_info['entry_price'], 1)
        trade_state.long_state.highest_since_entry = long_info['entry_price']
        trade_state.long_state.lowest_since_entry = long_info['entry_price']
        main_logger.info(Fore.GREEN + f"🔄 Restored long state")
    if short_info['size'] > 0:
        trade_state.short_state.init_new_position(short_info['size'], short_info['entry_price'], -1)
        trade_state.short_state.highest_since_entry = short_info['entry_price']
        trade_state.short_state.lowest_since_entry = short_info['entry_price']
        main_logger.info(Fore.GREEN + f"🔄 Restored short state")
    if long_info['size'] == 0 and short_info['size'] == 0:
        trade_state.reset_all()
    restore_stop_orders(FUTURES_SYMBOL)

def check_partial_take_profit(symbol: str, price: float, zones: dict):
    long_info, short_info = get_position(symbol)
    qty_prec = get_precision(symbol)[1]
    min_qty = float(client.futures_exchange_info()['symbols'][0]['filters'][1]['minQty'])

    if long_info['size'] > 0 and trade_state.long_state.is_trend_valid and not np.isnan(zones['resistance']):
        zone = zones['resistance']
        if trade_state.long_state.is_new_liquidity_zone(zone, "long"):
            if price >= zone and not trade_state.long_state.has_partial_tp_in_zone and long_info['size'] > min_qty:
                qty = round(long_info['size'] * LIQ_PARTIAL_PROFIT_RATIO, qty_prec)
                qty = max(qty, min_qty)
                main_logger.info(Fore.MAGENTA + f"🎯 Long partial TP at {zone:.2f} | Qty: {qty}")
                order = place_market_order(symbol, Client.SIDE_SELL, qty, 'LONG')
                if order:
                    trade_state.long_state.has_partial_tp_in_zone = True
                    trade_state.long_state.last_operated_zone_price = zone
                    new_long, _ = get_position(symbol)
                    trade_state.long_state.update_position(new_long['size'], new_long['entry_price'])
                    signal_logger.info(f"[Long Partial TP] {qty} @ {price} | Remain: {new_long['size']}")

    if short_info['size'] > 0 and trade_state.short_state.is_trend_valid and not np.isnan(zones['support']):
        zone = zones['support']
        if trade_state.short_state.is_new_liquidity_zone(zone, "short"):
            if price <= zone and not trade_state.short_state.has_partial_tp_in_zone and short_info['size'] > min_qty:
                qty = round(short_info['size'] * LIQ_PARTIAL_PROFIT_RATIO, qty_prec)
                qty = max(qty, min_qty)
                main_logger.info(Fore.MAGENTA + f"🎯 Short partial TP at {zone:.2f} | Qty: {qty}")
                order = place_market_order(symbol, Client.SIDE_BUY, qty, 'SHORT')
                if order:
                    trade_state.short_state.has_partial_tp_in_zone = True
                    trade_state.short_state.last_operated_zone_price = zone
                    _, new_short = get_position(symbol)
                    trade_state.short_state.update_position(new_short['size'], new_short['entry_price'])
                    signal_logger.info(f"[Short Partial TP] {qty} @ {price} | Remain: {new_short['size']}")

def check_breakout_and_add(symbol: str, price: float, zones: dict, filtered: int):
    long_info, short_info = get_position(symbol)
    usdc = get_usdc_balance()
    qty_prec = get_precision(symbol)[1]

    if (long_info['size'] > 0 and trade_state.long_state.is_trend_valid and filtered == 1
            and trade_state.long_state.total_add_times < MAX_ADD_TIMES and not np.isnan(zones['resistance'])):
        zone = zones['resistance']
        if trade_state.long_state.is_new_liquidity_zone(zone, "long"):
            klines = client.get_klines(symbol=SPOT_SYMBOL, interval=INTERVAL, limit=LOOKBACK)
            df_k = pd.DataFrame(klines, columns=['t','o','h','l','c','v','ct','qv','trades','tb','tq','ig'])
            for col in ['o','h','l','c','v']:
                df_k[col] = pd.to_numeric(df_k[col], errors='coerce')
            if (confirm_breakout(df_k, zone, "long") and not trade_state.long_state.has_added_in_zone
                    and trade_state.long_state.has_partial_tp_in_zone):
                add_qty = calculate_position_size(symbol, usdc, ADD_RISK_PCT, LEVERAGE, price)
                if add_qty <= 0:
                    return
                main_logger.info(Fore.BLUE + f"🚀 Long add at {zone:.2f} | Qty: {add_qty}")
                order = place_market_order(symbol, Client.SIDE_BUY, add_qty, 'LONG')
                if order:
                    trade_state.long_state.has_added_in_zone = True
                    trade_state.long_state.total_add_times += 1
                    trade_state.long_state.last_add_price = price
                    trade_state.long_state.last_operated_zone_price = zone
                    new_long, _ = get_position(symbol)
                    trade_state.long_state.update_position(new_long['size'], new_long['entry_price'])
                    signal_logger.info(f"[Long Add] {add_qty} @ {price} | Adds: {trade_state.long_state.total_add_times}")

    if (short_info['size'] > 0 and trade_state.short_state.is_trend_valid and filtered == -1
            and trade_state.short_state.total_add_times < MAX_ADD_TIMES and not np.isnan(zones['support'])):
        zone = zones['support']
        if trade_state.short_state.is_new_liquidity_zone(zone, "short"):
            klines = client.get_klines(symbol=SPOT_SYMBOL, interval=INTERVAL, limit=LOOKBACK)
            df_k = pd.DataFrame(klines, columns=['t','o','h','l','c','v','ct','qv','trades','tb','tq','ig'])
            for col in ['o','h','l','c','v']:
                df_k[col] = pd.to_numeric(df_k[col], errors='coerce')
            if (confirm_breakout(df_k, zone, "short") and not trade_state.short_state.has_added_in_zone
                    and trade_state.short_state.has_partial_tp_in_zone):
                add_qty = calculate_position_size(symbol, usdc, ADD_RISK_PCT, LEVERAGE, price)
                if add_qty <= 0:
                    return
                main_logger.info(Fore.BLUE + f"🚀 Short add at {zone:.2f} | Qty: {add_qty}")
                order = place_market_order(symbol, Client.SIDE_SELL, add_qty, 'SHORT')
                if order:
                    trade_state.short_state.has_added_in_zone = True
                    trade_state.short_state.total_add_times += 1
                    trade_state.short_state.last_add_price = price
                    trade_state.short_state.last_operated_zone_price = zone
                    _, new_short = get_position(symbol)
                    trade_state.short_state.update_position(new_short['size'], new_short['entry_price'])
                    signal_logger.info(f"[Short Add] {add_qty} @ {price} | Adds: {trade_state.short_state.total_add_times}")

def force_close_invalid_trend(filtered: int, price: float):
    long_info, short_info = get_position(FUTURES_SYMBOL)
    if long_info['size'] > 0:
        valid = not (filtered == -1 and trade_state.long_state.trend_at_open == 1)
        trade_state.long_state.is_trend_valid = valid
        if not valid:
            main_logger.warning(Fore.YELLOW + "⚠️ Long invalid, force close")
            close_order = place_market_order(FUTURES_SYMBOL, Client.SIDE_SELL, long_info['size'], 'LONG')
            if close_order:
                if trade_state.long_state.stop_order_id:
                    cancel_stop_order(FUTURES_SYMBOL, trade_state.long_state.stop_order_id)
                signal_logger.info(f"[Force Close Long] @ {price}")
            trade_state.reset_side("long")
    if short_info['size'] > 0:
        valid = not (filtered == 1 and trade_state.short_state.trend_at_open == -1)
        trade_state.short_state.is_trend_valid = valid
        if not valid:
            main_logger.warning(Fore.YELLOW + "⚠️ Short invalid, force close")
            close_order = place_market_order(FUTURES_SYMBOL, Client.SIDE_BUY, short_info['size'], 'SHORT')
            if close_order:
                if trade_state.short_state.stop_order_id:
                    cancel_stop_order(FUTURES_SYMBOL, trade_state.short_state.stop_order_id)
                signal_logger.info(f"[Force Close Short] @ {price}")
            trade_state.reset_side("short")

def close_all_if_neutral(filtered: int, price: float):
    if filtered != 0:
        return
    long_info, short_info = get_position(FUTURES_SYMBOL)
    if long_info['size'] > 0:
        main_logger.warning(Fore.YELLOW + "⚠️ Neutral, close long")
        close_order = place_market_order(FUTURES_SYMBOL, Client.SIDE_SELL, long_info['size'], 'LONG')
        if close_order:
            if trade_state.long_state.stop_order_id:
                cancel_stop_order(FUTURES_SYMBOL, trade_state.long_state.stop_order_id)
            signal_logger.info(f"[Neutral Close Long] @ {price}")
        trade_state.reset_side("long")
    if short_info['size'] > 0:
        main_logger.warning(Fore.YELLOW + "⚠️ Neutral, close short")
        close_order = place_market_order(FUTURES_SYMBOL, Client.SIDE_BUY, short_info['size'], 'SHORT')
        if close_order:
            if trade_state.short_state.stop_order_id:
                cancel_stop_order(FUTURES_SYMBOL, trade_state.short_state.stop_order_id)
            signal_logger.info(f"[Neutral Close Short] @ {price}")
        trade_state.reset_side("short")

def update_trailing_stops(symbol: str, price: float, atr_val: float):
    if not ENABLE_TRAILING_STOP:
        return
    long_info, _ = get_position(symbol)
    if long_info['size'] > 0:
        if price > trade_state.long_state.highest_since_entry:
            trade_state.long_state.highest_since_entry = price
        new_stop = trade_state.long_state.highest_since_entry - atr_val * TRAILING_ATR_MULT
        cur_stop = None
        if trade_state.long_state.stop_order_id:
            try:
                order = client.futures_get_order(symbol=symbol, orderId=trade_state.long_state.stop_order_id)
                cur_stop = float(order['stopPrice'])
            except:
                pass
        if cur_stop is None or new_stop > cur_stop:
            main_logger.info(Fore.CYAN + f"🔄 Update LONG trailing stop: {cur_stop} -> {new_stop:.2f}")
            update_trailing_stop(symbol, 'LONG', new_stop, long_info['size'])
    _, short_info = get_position(symbol)
    if short_info['size'] > 0:
        if price < trade_state.short_state.lowest_since_entry:
            trade_state.short_state.lowest_since_entry = price
        new_stop = trade_state.short_state.lowest_since_entry + atr_val * TRAILING_ATR_MULT
        cur_stop = None
        if trade_state.short_state.stop_order_id:
            try:
                order = client.futures_get_order(symbol=symbol, orderId=trade_state.short_state.stop_order_id)
                cur_stop = float(order['stopPrice'])
            except:
                pass
        if cur_stop is None or new_stop < cur_stop:
            main_logger.info(Fore.CYAN + f"🔄 Update SHORT trailing stop: {cur_stop} -> {new_stop:.2f}")
            update_trailing_stop(symbol, 'SHORT', new_stop, short_info['size'])

# ==================== 价格线程（REST 轮询） ====================
def price_polling():
    """独立线程，每秒轮询一次合约最新价格，并检查开仓条件"""
    global latest_price
    main_logger.info(Fore.CYAN + f"📡 启动价格轮询线程，交易对: {FUTURES_SYMBOL}，间隔 1 秒")
    while True:
        try:
            ticker = client.futures_symbol_ticker(symbol=FUTURES_SYMBOL)
            price = float(ticker['price'])
            with trade_lock:
                latest_price = price
                check_price_trigger(price)
        except Exception as e:
            main_logger.error(Fore.RED + f"❌ 价格轮询错误: {e}")
        time.sleep(1)

def check_price_trigger(price: float):
    """检查当前价格是否满足等待开仓条件，若满足则执行开仓"""
    # 多头等待
    if trade_state.long_state.awaiting_entry and trade_state.long_state.awaiting_trend == 1:
        if latest_filtered_trend == 1:
            if price <= latest_z_value:
                main_logger.info(Fore.GREEN + f"🎯 Price {price:.2f} <= z={latest_z_value:.2f}, trigger LONG")
                usdc = get_usdc_balance()
                qty = calculate_position_size(FUTURES_SYMBOL, usdc, RISK_PERCENTAGE, LEVERAGE, price)
                if qty > 0:
                    order = place_market_order(FUTURES_SYMBOL, Client.SIDE_BUY, qty, 'LONG')
                    if order:
                        new_long, _ = get_position(FUTURES_SYMBOL)
                        trade_state.long_state.init_new_position(new_long['size'], new_long['entry_price'], 1)
                        if ENABLE_TRAILING_STOP:
                            initial_stop = new_long['entry_price'] - (latest_atr * TRAILING_ATR_MULT)
                            stop_order = place_stop_order(FUTURES_SYMBOL, Client.SIDE_SELL, new_long['size'], initial_stop, 'LONG')
                            if stop_order:
                                trade_state.long_state.stop_order_id = int(stop_order['orderId'])
                        signal_logger.info(f"[Long Entry via Price] {qty} @ {price}")
                        trade_state.long_state.awaiting_entry = False
                else:
                    trade_state.long_state.awaiting_entry = False
        else:
            trade_state.long_state.awaiting_entry = False

    # 空头等待
    if trade_state.short_state.awaiting_entry and trade_state.short_state.awaiting_trend == -1:
        if latest_filtered_trend == -1:
            if price >= latest_z_value:
                main_logger.info(Fore.RED + f"🎯 Price {price:.2f} >= z={latest_z_value:.2f}, trigger SHORT")
                usdc = get_usdc_balance()
                qty = calculate_position_size(FUTURES_SYMBOL, usdc, RISK_PERCENTAGE, LEVERAGE, price)
                if qty > 0:
                    order = place_market_order(FUTURES_SYMBOL, Client.SIDE_SELL, qty, 'SHORT')
                    if order:
                        _, new_short = get_position(FUTURES_SYMBOL)
                        trade_state.short_state.init_new_position(new_short['size'], new_short['entry_price'], -1)
                        if ENABLE_TRAILING_STOP:
                            initial_stop = new_short['entry_price'] + (latest_atr * TRAILING_ATR_MULT)
                            stop_order = place_stop_order(FUTURES_SYMBOL, Client.SIDE_BUY, new_short['size'], initial_stop, 'SHORT')
                            if stop_order:
                                trade_state.short_state.stop_order_id = int(stop_order['orderId'])
                        signal_logger.info(f"[Short Entry via Price] {qty} @ {price}")
                        trade_state.short_state.awaiting_entry = False
                else:
                    trade_state.short_state.awaiting_entry = False
        else:
            trade_state.short_state.awaiting_entry = False

# ==================== 主策略循环 ====================
def run_strategy():
    global latest_z_value, latest_filtered_trend, latest_atr
    main_logger.info(Fore.CYAN + "="*80)
    main_logger.info(Fore.CYAN + "🚀 VWT + L1 Filter Strategy Started")
    main_logger.info(Fore.CYAN + f"📊 Signal: {SPOT_SYMBOL} → Trade: {FUTURES_SYMBOL}")
    main_logger.info(Fore.CYAN + f"⚙️  Filter: confirm={TREND_CONFIRM_BARS}, consensus={USE_CONSENSUS_FILTER}, atr_filter={USE_ATR_FILTER}")
    main_logger.info(Fore.CYAN + f"💰  Risk: Leverage={LEVERAGE}x | EntryRisk={RISK_PERCENTAGE}%")
    main_logger.info(Fore.CYAN + "="*80)

    try:
        setup_hedge_mode(FUTURES_SYMBOL)
        setup_leverage_margin(FUTURES_SYMBOL, LEVERAGE, MARGIN_TYPE)
        restore_trade_state()
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Initialization error: {e}")
        return

    last_kline_time = 0
    retries = 0
    MAX_RETRIES = 3
    RETRY_INTERVAL = 5

    # 启动价格轮询线程
    price_thread = threading.Thread(target=price_polling, daemon=True)
    price_thread.start()
    time.sleep(2)

    while True:
        try:
            # 获取K线
            klines = None
            for r in range(MAX_RETRIES):
                try:
                    klines = client.get_klines(symbol=SPOT_SYMBOL, interval=INTERVAL, limit=LOOKBACK)
                    if klines and len(klines) > 0:
                        break
                    time.sleep(RETRY_INTERVAL)
                except Exception as e:
                    main_logger.error(Fore.RED + f"❌ Kline fetch error: {e}")
                    time.sleep(RETRY_INTERVAL)
            if not klines:
                time.sleep(30)
                continue

            df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','ct','qv','trades','tb','tq','ig'])
            for col in ['open','high','low','close','volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 检查新K线
            cur_time = int(df['timestamp'].iloc[-1])
            if cur_time == last_kline_time:
                retries += 1
                if retries >= MAX_RETRIES:
                    last_kline_time = 0
                    retries = 0
                else:
                    time.sleep(10)
                    continue
            else:
                retries = 0
            last_kline_time = cur_time
            cur_price = df['close'].iloc[-1]
            if pd.isna(cur_price):
                time.sleep(10)
                continue

            if len(df) < VWMA_LENGTH + 1:
                time.sleep(10)
                continue

            # 计算过滤趋势
            filtered_series = calculate_filtered_trend_series(
                df, VWMA_LENGTH, VWT_ATR_MULT, L1_ATR_MULT, L1_MU,
                TREND_CONFIRM_BARS, USE_CONSENSUS_FILTER, USE_ATR_FILTER, MIN_ATR_PCT
            )
            filtered = filtered_series.iloc[-1]
            prev_filtered = filtered_series.iloc[-2] if len(filtered_series) > 1 else 0

            # 计算ATR和L1 z值
            _, _, _, _, atr_series = calculate_vwt(df, VWMA_LENGTH, VWT_ATR_MULT)
            atr_val = atr_series.iloc[-1]
            atr_pct = atr_val / cur_price
            l1_series = calculate_l1_filter(df, L1_ATR_MULT, L1_MU)
            z_val = l1_series.iloc[-1]

            # 更新全局变量
            with trade_lock:
                latest_z_value = z_val
                latest_filtered_trend = filtered
                latest_atr = atr_val

            # 检测流动性区域
            zones = detect_liquidity_zones(df, LIQ_SWEEP_LENGTH)

            # 获取当前仓位
            long_info, short_info = get_position(FUTURES_SYMBOL)

            # 日志
            color_map = {1: Fore.GREEN+'LONG', -1: Fore.RED+'SHORT', 0: Fore.WHITE+'NEUTRAL'}
            main_logger.info(Fore.CYAN + "="*60)
            main_logger.info(Fore.CYAN + f"🕐 Time: {pd.to_datetime(cur_time, unit='ms')} | Price: {cur_price:.2f}")
            main_logger.info(Fore.CYAN + f"🧭 Filtered: {filtered} ({color_map[filtered]}) | Prev: {prev_filtered} | ATR%: {atr_pct:.4f}")
            main_logger.info(Fore.CYAN + f"📈 L1 z: {z_val:.2f}")
            main_logger.info(Fore.CYAN + f"📈 Long: size={long_info['size']} entry={long_info['entry_price']:.2f} valid={trade_state.long_state.is_trend_valid}")
            main_logger.info(Fore.CYAN + f"📉 Short: size={short_info['size']} entry={short_info['entry_price']:.2f} valid={trade_state.short_state.is_trend_valid}")

            # 止损检查
            sl_trigger, sl_side = check_stop_loss(FUTURES_SYMBOL, cur_price)
            if sl_trigger:
                if sl_side == "long" and long_info['size'] > 0:
                    place_market_order(FUTURES_SYMBOL, Client.SIDE_SELL, long_info['size'], 'LONG')
                    if trade_state.long_state.stop_order_id:
                        cancel_stop_order(FUTURES_SYMBOL, trade_state.long_state.stop_order_id)
                    trade_state.reset_side("long")
                elif sl_side == "short" and short_info['size'] > 0:
                    place_market_order(FUTURES_SYMBOL, Client.SIDE_BUY, short_info['size'], 'SHORT')
                    if trade_state.short_state.stop_order_id:
                        cancel_stop_order(FUTURES_SYMBOL, trade_state.short_state.stop_order_id)
                    trade_state.reset_side("short")
                time.sleep(60)
                continue

            # 移动止损
            if ENABLE_TRAILING_STOP:
                update_trailing_stops(FUTURES_SYMBOL, cur_price, atr_val)

            # 中性平仓
            close_all_if_neutral(filtered, cur_price)

            # 强制平仓
            force_close_invalid_trend(filtered, cur_price)

            # 部分止盈和加仓
            check_partial_take_profit(FUTURES_SYMBOL, cur_price, zones)
            check_breakout_and_add(FUTURES_SYMBOL, cur_price, zones, filtered)

            # 设置等待开仓标志（仅当无持仓且趋势刚变时）
            with trade_lock:
                if filtered == 1 and prev_filtered != 1 and trade_state.long_state.position_size == 0:
                    trade_state.long_state.awaiting_entry = True
                    trade_state.long_state.awaiting_trend = 1
                    main_logger.info(Fore.GREEN + f"⏳ Long awaiting: price <= {z_val:.2f}")
                if filtered == -1 and prev_filtered != -1 and trade_state.short_state.position_size == 0:
                    trade_state.short_state.awaiting_entry = True
                    trade_state.short_state.awaiting_trend = -1
                    main_logger.info(Fore.RED + f"⏳ Short awaiting: price >= {z_val:.2f}")

                # 如果趋势不再支持等待，清除标志
                if filtered != 1:
                    trade_state.long_state.awaiting_entry = False
                if filtered != -1:
                    trade_state.short_state.awaiting_entry = False

            main_logger.info(Fore.CYAN + "="*60 + "\n")
            time.sleep(30)

        except Exception as e:
            main_logger.error(Fore.RED + f"❌ Main loop error: {e}", exc_info=True)
            time.sleep(30)

if __name__ == "__main__":
    try:
        run_strategy()
    except KeyboardInterrupt:
        main_logger.info(Fore.CYAN + "👋 Strategy stopped")