import os
import sys
import time
import logging
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

# Initialize color output
init(autoreset=True)

# ———————————————— Logging System ————————————————
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

# ———————————————— Full Configuration ————————————————
load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# ========== 分离现货数据源和合约交易标的 ==========
SPOT_SYMBOL = "ETHUSDT"          # 用于获取K线计算信号的现货交易对
FUTURES_SYMBOL = "ETHUSDC"        # 实际执行交易的永续合约交易对

INTERVAL = Client.KLINE_INTERVAL_15MINUTE
LOOKBACK = 600

# ========== VWT 趋势参数（与 TradingView 指标默认值一致） ==========
VWMA_LENGTH = 34                  # VWMA 周期
VWT_ATR_MULT = 1.5                # ATR 乘数

# Leverage & Risk Management
LEVERAGE = 20
MARGIN_TYPE = "ISOLATED"
RISK_PERCENTAGE = 50
ADD_RISK_PCT = 20

# Stop Loss Config
STOP_LOSS_PCT = 1.5
ENABLE_STOP_LOSS = False

# Liquidity Sweep Core Parameters
LIQ_SWEEP_LENGTH = 8
LIQ_PARTIAL_PROFIT_RATIO = 0.5
BREAKOUT_CONFIRM_BARS = 2
BREAKOUT_THRESHOLD_PCT = 0.1

# State Management Core Config
MAX_ADD_TIMES = 1
NEW_ZONE_THRESHOLD_PCT = 0.5
STATE_RESET_DELAY = 1

# Trailing Stop Configuration
ENABLE_TRAILING_STOP = True
TRAILING_ATR_MULT = 1.1

# Binance Client Initialization
client = Client(API_KEY, API_SECRET, testnet=False, requests_params={'timeout': 30})
main_logger.info(Fore.CYAN + "✅ Binance live trading client initialized (timeout=30s)")
main_logger.info(Fore.CYAN + f"📊 Signal source: {SPOT_SYMBOL} spot klines | Trading on: {FUTURES_SYMBOL} futures")
main_logger.info(Fore.CYAN + f"📈 Trend indicator: VWT (VWMA + ATR channel) with VWMA length={VWMA_LENGTH}, ATR mult={VWT_ATR_MULT}")

# ———————————————— SideState and TradeState (unchanged) ————————————————
@dataclass
class SideState:
    position_size: float = 0.0
    entry_price: float = 0.0
    initial_entry_price: float = 0.0
    trend_at_open: int = 0          # 开仓时的 VWT 趋势：1=看涨，-1=看跌
    is_trend_valid: bool = False
    last_operated_zone_price: float = 0.0
    has_partial_tp_in_zone: bool = False
    has_added_in_zone: bool = False
    total_add_times: int = 0
    last_add_price: float = 0.0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 0.0

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
        price_diff_pct = abs(current_zone_price - self.last_operated_zone_price) / self.last_operated_zone_price * 100
        if pos_dir == "long":
            is_new = (current_zone_price > self.last_operated_zone_price) and (price_diff_pct >= NEW_ZONE_THRESHOLD_PCT)
        elif pos_dir == "short":
            is_new = (current_zone_price < self.last_operated_zone_price) and (price_diff_pct >= NEW_ZONE_THRESHOLD_PCT)
        else:
            is_new = False
        if is_new:
            self.has_partial_tp_in_zone = False
            self.has_added_in_zone = False
            main_logger.info(Fore.CYAN + f"🎯 New liquidity zone detected | Price:{current_zone_price:.2f} | Diff:{price_diff_pct:.2f}%")
        return is_new

@dataclass
class TradeState:
    long_state: SideState = field(default_factory=SideState)
    short_state: SideState = field(default_factory=SideState)

    def reset_side(self, side: str):
        if side == "long":
            self.long_state.reset()
            main_logger.info(Fore.YELLOW + "🔄 Long side state reset")
        elif side == "short":
            self.short_state.reset()
            main_logger.info(Fore.YELLOW + "🔄 Short side state reset")

    def reset_all(self):
        self.long_state.reset()
        self.short_state.reset()
        main_logger.info(Fore.YELLOW + "🔄 All trading state reset")

trade_state = TradeState()

# ———————————————— 辅助函数：RMA-ATR（与 TradingView 一致） ————————————————
def calculate_atr_rma(data: pd.DataFrame, period: int) -> pd.Series:
    high = data['high']
    low = data['low']
    close = data['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

# ———————————————— VWT 趋势计算（核心信号） ————————————————
def calculate_vwt_trend(data: pd.DataFrame, vwma_length: int, atr_mult: float) -> tuple[int, int, float, float, float]:
    """
    计算 VWT 趋势及通道
    返回: (current_trend, prev_trend, vwma_basis, upper_band, lower_band)
    trend: 1=看涨（收盘价>上轨），-1=看跌（收盘价<下轨），0=中性（通道内）
    """
    close = data['close']
    volume = data['volume']
    
    # VWMA = (price * volume) 的滚动和 / volume 的滚动和
    vwma = (close * volume).rolling(window=vwma_length).sum() / volume.rolling(window=vwma_length).sum()
    
    # 用 RMA 计算 ATR，周期与 VWMA 相同
    atr_vwt = calculate_atr_rma(data, vwma_length)
    
    upper_band = vwma + atr_vwt * atr_mult
    lower_band = vwma - atr_vwt * atr_mult
    
    # 确定当前趋势
    current_trend = 0
    if close.iloc[-1] > upper_band.iloc[-1]:
        current_trend = 1
    elif close.iloc[-1] < lower_band.iloc[-1]:
        current_trend = -1
    
    # 前一趋势（使用上根K线的收盘价比较）
    prev_trend = 0
    if len(close) > 1:
        if close.iloc[-2] > upper_band.iloc[-2]:
            prev_trend = 1
        elif close.iloc[-2] < lower_band.iloc[-2]:
            prev_trend = -1
    
    return current_trend, prev_trend, vwma.iloc[-1], upper_band.iloc[-1], lower_band.iloc[-1]

# ———————————————— 流动性区域检测（不变） ————————————————
def detect_liquidity_zones(data: pd.DataFrame, lookback_len: int = 8) -> dict:
    df = data.copy()
    closed_df = df.iloc[:-1].copy()
    nearest_resistance = np.nan
    nearest_support = np.nan

    if len(closed_df) < lookback_len * 2 + 1:
        return {'resistance': nearest_resistance, 'support': nearest_support}

    closed_df['is_pivot_high'] = closed_df['high'] == closed_df['high'].rolling(window=lookback_len*2+1, center=True).max()
    closed_df['is_pivot_low'] = closed_df['low'] == closed_df['low'].rolling(window=lookback_len*2+1, center=True).min()

    pivot_highs = closed_df[closed_df['is_pivot_high']]['high']
    pivot_lows = closed_df[closed_df['is_pivot_low']]['low']

    current_price = df['close'].iloc[-1]
    if not pivot_highs.empty:
        valid_resistances = pivot_highs[pivot_highs > current_price]
        if not valid_resistances.empty:
            nearest_resistance = valid_resistances.iloc[-1]
    if not pivot_lows.empty:
        valid_supports = pivot_lows[pivot_lows < current_price]
        if not valid_supports.empty:
            nearest_support = valid_supports.iloc[-1]

    return {'resistance': nearest_resistance, 'support': nearest_support}

def confirm_breakout(data: pd.DataFrame, zone_price: float, pos_dir: str) -> bool:
    if len(data) < BREAKOUT_CONFIRM_BARS:
        return False
    recent_bars = data.iloc[-(BREAKOUT_CONFIRM_BARS+1):-1]
    if pos_dir == "long":
        breakout_level = zone_price * (1 + BREAKOUT_THRESHOLD_PCT / 100)
        all_breakout = all(recent_bars['close'] > breakout_level)
    elif pos_dir == "short":
        breakout_level = zone_price * (1 - BREAKOUT_THRESHOLD_PCT / 100)
        all_breakout = all(recent_bars['close'] < breakout_level)
    else:
        all_breakout = False
    if all_breakout:
        main_logger.info(Fore.BLUE + f"✅ Valid breakout confirmed | Zone Price:{zone_price:.2f} | Breakout Level:{breakout_level:.2f} | Confirm Bars:{BREAKOUT_CONFIRM_BARS}")
    return all_breakout

# ———————————————— 交易辅助函数（不变，但使用 FUTURES_SYMBOL） ————————————————
def setup_hedge_mode(symbol: str):
    try:
        client.futures_change_position_mode(dualSidePosition=True)
        main_logger.info(Fore.GREEN + "✅ Successfully switched to HEDGE MODE (dual side position)")
        position_mode = client.futures_get_position_mode()
        main_logger.info(Fore.CYAN + f"🔍 Current position mode: {position_mode}")
    except BinanceAPIException as e:
        if "No need to change position mode" in str(e):
            main_logger.info(Fore.CYAN + "ℹ️ Already in HEDGE MODE")
        else:
            main_logger.error(Fore.RED + f"❌ Failed to set hedge mode: {e}")
            raise

def setup_leverage_and_margin(symbol: str, leverage: int, margin_type: str):
    try:
        client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
        main_logger.info(Fore.CYAN + f"🔧 Margin mode set: {'Isolated' if margin_type == 'ISOLATED' else 'Cross'}")
    except BinanceAPIException as e:
        if "No need to change margin type" not in str(e):
            main_logger.warning(Fore.YELLOW + f"⚠️ Margin mode note: {e}")
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        main_logger.info(Fore.CYAN + f"🔧 Leverage set: {leverage}x")
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Leverage setup failed: {e}")
        raise

def get_position(symbol: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    long_info = {'size': 0.0, 'entry_price': 0.0}
    short_info = {'size': 0.0, 'entry_price': 0.0}
    try:
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            if pos['symbol'] == symbol:
                position_side = pos['positionSide']
                amt = float(pos['positionAmt'])
                entry_price = float(pos['entryPrice'])
                if position_side == 'LONG' and amt > 0:
                    long_info['size'] = amt
                    long_info['entry_price'] = entry_price
                    main_logger.info(Fore.CYAN + f"📈 Long position: {amt} | Entry price: {entry_price}")
                elif position_side == 'SHORT' and amt > 0:
                    short_info['size'] = amt
                    short_info['entry_price'] = entry_price
                    main_logger.info(Fore.CYAN + f"📉 Short position: {amt} | Entry price: {entry_price}")
        if long_info['size'] == 0 and short_info['size'] == 0:
            main_logger.info(Fore.CYAN + "📊 No current positions (both sides)")
        return long_info, short_info
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to get position: {e}")
        return long_info, short_info

def get_usdc_balance() -> float:
    try:
        balance = client.futures_account_balance()
        for asset in balance:
            if asset['asset'] == 'USDC':
                available_balance = float(asset['availableBalance'])
                main_logger.info(Fore.CYAN + f"💰 USDC available balance: {available_balance}")
                return available_balance
        main_logger.error(Fore.RED + "❌ USDC balance not found")
        return 0.0
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to get USDC balance: {e}")
        return 0.0

def get_symbol_precision(symbol: str) -> tuple[int, int]:
    try:
        info = client.futures_exchange_info()
        for symbol_info in info['symbols']:
            if symbol_info['symbol'] == symbol:
                return int(symbol_info['pricePrecision']), int(symbol_info['quantityPrecision'])
        return 2, 3
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to get precision: {e}")
        return 2, 3

def calculate_position_size(symbol: str, usdc_balance: float, risk_pct: float, leverage: int, current_price: float) -> float:
    try:
        info = client.futures_exchange_info()
        symbol_info = None
        for item in info['symbols']:
            if item['symbol'] == symbol:
                symbol_info = item
                break
        if not symbol_info:
            main_logger.error(Fore.RED + f"❌ Trading pair information for {symbol} not found")
            return 0.0
        qty_precision = int(symbol_info['quantityPrecision'])
        min_qty = float(symbol_info['filters'][1]['minQty'])
        main_logger.info(Fore.YELLOW + f"📏 Trading pair parameters | Minimum quantity:{min_qty} | Quantity precision:{qty_precision}")
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to get symbol precision: {e}")
        return 0.0

    risk_amount = usdc_balance * (risk_pct / 100)
    notional_value = risk_amount * leverage
    position_size = notional_value / current_price
    adjusted_size = round(position_size, qty_precision)
    main_logger.info(Fore.YELLOW + f"📏 Position calculation details | Risk amount:{risk_amount} | Notional value:{notional_value} | Raw position:{position_size} | Adjusted:{adjusted_size}")
    if adjusted_size < min_qty:
        main_logger.warning(Fore.YELLOW + f"⚠️ Adjusted position {adjusted_size} is less than minimum quantity {min_qty}, force set to {min_qty}")
        adjusted_size = min_qty
    if adjusted_size <= 0:
        main_logger.error(Fore.RED + f"❌ Calculated position {adjusted_size} is invalid (<=0)")
        return min_qty
    return adjusted_size

def place_market_order(symbol: str, side: str, quantity: float, position_side: str) -> dict:
    try:
        _, qty_precision = get_symbol_precision(symbol)
        quantity = round(quantity, qty_precision)
        order = client.futures_create_order(
            symbol=symbol, 
            side=side, 
            type=Client.ORDER_TYPE_MARKET, 
            quantity=quantity,
            positionSide=position_side
        )
        action = f"{position_side} Open" if (position_side == 'LONG' and side == Client.SIDE_BUY) or (position_side == 'SHORT' and side == Client.SIDE_SELL) else f"{position_side} Close"
        main_logger.info(Fore.GREEN + f"✅ [{action} Success] Order ID: {order['orderId']}, Quantity: {quantity}")
        return order
    except (BinanceAPIException, BinanceOrderException) as e:
        main_logger.error(Fore.RED + f"❌ [Order Failed] {e} | Side: {side} | PositionSide: {position_side} | Quantity: {quantity}")
        return None

def check_stop_loss(symbol: str, current_price: float) -> Tuple[bool, str]:
    long_info, short_info = get_position(symbol)
    if not ENABLE_STOP_LOSS:
        return False, "none"
    if long_info['size'] > 0:
        loss_pct = (long_info['entry_price'] - current_price) / long_info['entry_price'] * 100
        if loss_pct >= STOP_LOSS_PCT:
            warn_msg = f"⚠️ [Long Stop Loss Triggered] Entry: {long_info['entry_price']:.2f}, Current: {current_price:.2f}, Loss: {loss_pct:.2f}%"
            main_logger.warning(Fore.YELLOW + warn_msg)
            signal_logger.warning(warn_msg)
            return True, "long"
    if short_info['size'] > 0:
        loss_pct = (current_price - short_info['entry_price']) / short_info['entry_price'] * 100
        if loss_pct >= STOP_LOSS_PCT:
            warn_msg = f"⚠️ [Short Stop Loss Triggered] Entry: {short_info['entry_price']:.2f}, Current: {current_price:.2f}, Loss: {loss_pct:.2f}%"
            main_logger.warning(Fore.YELLOW + warn_msg)
            signal_logger.warning(warn_msg)
            return True, "short"
    return False, "none"

def restore_trade_state():
    long_info, short_info = get_position(FUTURES_SYMBOL)
    if long_info['size'] > 0:
        trade_state.long_state.init_new_position(
            pos_size=long_info['size'],
            entry_price=long_info['entry_price'],
            trend=1  # 恢复时默认趋势为1（需根据实际情况调整，但无法获取历史趋势，只能假设有效）
        )
        trade_state.long_state.highest_since_entry = long_info['entry_price']
        trade_state.long_state.lowest_since_entry = long_info['entry_price']
        main_logger.info(Fore.GREEN + f"🔄 Restored long state | Size:{long_info['size']} | Entry:{long_info['entry_price']:.2f} | Trend at open:1")
    if short_info['size'] > 0:
        trade_state.short_state.init_new_position(
            pos_size=short_info['size'],
            entry_price=short_info['entry_price'],
            trend=-1
        )
        trade_state.short_state.highest_since_entry = short_info['entry_price']
        trade_state.short_state.lowest_since_entry = short_info['entry_price']
        main_logger.info(Fore.GREEN + f"🔄 Restored short state | Size:{short_info['size']} | Entry:{short_info['entry_price']:.2f} | Trend at open:-1")
    if long_info['size'] == 0 and short_info['size'] == 0:
        trade_state.reset_all()
        main_logger.info(Fore.CYAN + "🔄 No positions, state initialized")

def check_partial_take_profit(symbol: str, current_price: float, liq_zones: dict):
    long_info, short_info = get_position(symbol)
    qty_precision = get_symbol_precision(symbol)[1]
    min_qty = float(client.futures_exchange_info()['symbols'][0]['filters'][1]['minQty'])

    if long_info['size'] > 0 and trade_state.long_state.is_trend_valid and not np.isnan(liq_zones['resistance']):
        zone_price = liq_zones['resistance']
        if trade_state.long_state.is_new_liquidity_zone(zone_price, "long"):
            if (current_price >= zone_price 
                and not trade_state.long_state.has_partial_tp_in_zone 
                and long_info['size'] > min_qty):
                sell_qty = round(long_info['size'] * LIQ_PARTIAL_PROFIT_RATIO, qty_precision)
                sell_qty = max(sell_qty, min_qty)
                main_logger.info(Fore.MAGENTA + "\n" + "="*80)
                main_logger.info(Fore.MAGENTA + f"🎯 [Long Partial TP] Hit resistance: {zone_price:.2f}")
                main_logger.info(Fore.MAGENTA + f"Action: Close {LIQ_PARTIAL_PROFIT_RATIO*100}% long position | Qty: {sell_qty}")
                main_logger.info(Fore.MAGENTA + "="*80 + "\n")
                order = place_market_order(symbol, Client.SIDE_SELL, sell_qty, 'LONG')
                if order:
                    trade_state.long_state.has_partial_tp_in_zone = True
                    trade_state.long_state.last_operated_zone_price = zone_price
                    new_long, _ = get_position(symbol)
                    trade_state.long_state.update_position(new_long['size'], new_long['entry_price'])
                    signal_logger.info(f"[Long Partial TP Done] Close {sell_qty} @ {current_price} | Resistance: {zone_price} | Remaining: {new_long['size']}")

    if short_info['size'] > 0 and trade_state.short_state.is_trend_valid and not np.isnan(liq_zones['support']):
        zone_price = liq_zones['support']
        if trade_state.short_state.is_new_liquidity_zone(zone_price, "short"):
            if (current_price <= zone_price 
                and not trade_state.short_state.has_partial_tp_in_zone 
                and short_info['size'] > min_qty):
                buy_qty = round(short_info['size'] * LIQ_PARTIAL_PROFIT_RATIO, qty_precision)
                buy_qty = max(buy_qty, min_qty)
                main_logger.info(Fore.MAGENTA + "\n" + "="*80)
                main_logger.info(Fore.MAGENTA + f"🎯 [Short Partial TP] Hit support: {zone_price:.2f}")
                main_logger.info(Fore.MAGENTA + f"Action: Close {LIQ_PARTIAL_PROFIT_RATIO*100}% short position | Qty: {buy_qty}")
                main_logger.info(Fore.MAGENTA + "="*80 + "\n")
                order = place_market_order(symbol, Client.SIDE_BUY, buy_qty, 'SHORT')
                if order:
                    trade_state.short_state.has_partial_tp_in_zone = True
                    trade_state.short_state.last_operated_zone_price = zone_price
                    _, new_short = get_position(symbol)
                    trade_state.short_state.update_position(new_short['size'], new_short['entry_price'])
                    signal_logger.info(f"[Short Partial TP Done] Close {buy_qty} @ {current_price} | Support: {zone_price} | Remaining: {new_short['size']}")

def check_breakout_and_add(symbol: str, current_price: float, liq_zones: dict, current_trend: int):
    long_info, short_info = get_position(symbol)
    usdc_balance = get_usdc_balance()
    qty_precision = get_symbol_precision(symbol)[1]

    if (long_info['size'] > 0 
        and trade_state.long_state.is_trend_valid 
        and current_trend == 1 
        and trade_state.long_state.total_add_times < MAX_ADD_TIMES
        and not np.isnan(liq_zones['resistance'])):
        zone_price = liq_zones['resistance']
        if trade_state.long_state.is_new_liquidity_zone(zone_price, "long"):
            klines_data = client.get_klines(symbol=SPOT_SYMBOL, interval=INTERVAL, limit=LOOKBACK)
            df_kline = pd.DataFrame(klines_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume','close_time', 'quote_vol', 'trades', 'taker_buy_base','taker_buy_quote', 'ignore'])
            for col in ['open', 'high', 'low', 'close']:
                df_kline[col] = pd.to_numeric(df_kline[col], errors='coerce')
            if (confirm_breakout(df_kline, zone_price, "long")
                and not trade_state.long_state.has_added_in_zone
                and trade_state.long_state.has_partial_tp_in_zone):
                add_qty = calculate_position_size(symbol, usdc_balance, ADD_RISK_PCT, LEVERAGE, current_price)
                if add_qty <= 0:
                    main_logger.warning(Fore.YELLOW + "⚠️ Long add qty insufficient, skip add")
                    return
                main_logger.info(Fore.BLUE + "\n" + "="*80)
                main_logger.info(Fore.BLUE + f"🚀 [Long Breakout Add] Valid breakout of resistance: {zone_price:.2f}")
                main_logger.info(Fore.BLUE + f"Trend confirmed: VWT remains long | Add count: {trade_state.long_state.total_add_times+1}/{MAX_ADD_TIMES}")
                main_logger.info(Fore.BLUE + f"Action: Add long | Qty: {add_qty}")
                main_logger.info(Fore.BLUE + "="*80 + "\n")
                order = place_market_order(symbol, Client.SIDE_BUY, add_qty, 'LONG')
                if order:
                    trade_state.long_state.has_added_in_zone = True
                    trade_state.long_state.total_add_times += 1
                    trade_state.long_state.last_add_price = current_price
                    trade_state.long_state.last_operated_zone_price = zone_price
                    new_long, _ = get_position(symbol)
                    trade_state.long_state.update_position(new_long['size'], new_long['entry_price'])
                    signal_logger.info(f"[Long Add Done] Add {add_qty} @ {current_price} | Breakout: {zone_price} | Total adds: {trade_state.long_state.total_add_times} | Total pos: {new_long['size']}")

    if (short_info['size'] > 0 
        and trade_state.short_state.is_trend_valid 
        and current_trend == -1 
        and trade_state.short_state.total_add_times < MAX_ADD_TIMES
        and not np.isnan(liq_zones['support'])):
        zone_price = liq_zones['support']
        if trade_state.short_state.is_new_liquidity_zone(zone_price, "short"):
            klines_data = client.get_klines(symbol=SPOT_SYMBOL, interval=INTERVAL, limit=LOOKBACK)
            df_kline = pd.DataFrame(klines_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume','close_time', 'quote_vol', 'trades', 'taker_buy_base','taker_buy_quote', 'ignore'])
            for col in ['open', 'high', 'low', 'close']:
                df_kline[col] = pd.to_numeric(df_kline[col], errors='coerce')
            if (confirm_breakout(df_kline, zone_price, "short")
                and not trade_state.short_state.has_added_in_zone
                and trade_state.short_state.has_partial_tp_in_zone):
                add_qty = calculate_position_size(symbol, usdc_balance, ADD_RISK_PCT, LEVERAGE, current_price)
                if add_qty <= 0:
                    main_logger.warning(Fore.YELLOW + "⚠️ Short add qty insufficient, skip add")
                    return
                main_logger.info(Fore.BLUE + "\n" + "="*80)
                main_logger.info(Fore.BLUE + f"🚀 [Short Breakdown Add] Valid breakdown of support: {zone_price:.2f}")
                main_logger.info(Fore.BLUE + f"Trend confirmed: VWT remains short | Add count: {trade_state.short_state.total_add_times+1}/{MAX_ADD_TIMES}")
                main_logger.info(Fore.BLUE + f"Action: Add short | Qty: {add_qty}")
                main_logger.info(Fore.BLUE + "="*80 + "\n")
                order = place_market_order(symbol, Client.SIDE_SELL, add_qty, 'SHORT')
                if order:
                    trade_state.short_state.has_added_in_zone = True
                    trade_state.short_state.total_add_times += 1
                    trade_state.short_state.last_add_price = current_price
                    trade_state.short_state.last_operated_zone_price = zone_price
                    _, new_short = get_position(symbol)
                    trade_state.short_state.update_position(new_short['size'], new_short['entry_price'])
                    signal_logger.info(f"[Short Add Done] Add {add_qty} @ {current_price} | Breakdown: {zone_price} | Total adds: {trade_state.short_state.total_add_times} | Total pos: {new_short['size']}")

def force_close_invalid_trend_positions(current_trend: int, current_price: float):
    """
    强制平仓与当前 VWT 趋势反向的仓位
    规则：如果开仓趋势为1，当前趋势为-1，则平多；如果开仓趋势为-1，当前趋势为1，则平空。
    趋势为0时不平仓（视为中性，不强制）
    """
    long_info, short_info = get_position(FUTURES_SYMBOL)
    if long_info['size'] > 0:
        # 检查多头是否应平仓：开仓时趋势应为1，当前趋势为-1则无效
        trade_state.long_state.is_trend_valid = not (current_trend == -1 and trade_state.long_state.trend_at_open == 1)
        main_logger.info(Fore.CYAN + f"🧮 Long trend validity | Current VWT trend:{current_trend} | Open trend:{trade_state.long_state.trend_at_open} | Valid:{trade_state.long_state.is_trend_valid}")
        if not trade_state.long_state.is_trend_valid:
            main_logger.warning(Fore.YELLOW + "⚠️ Long trend invalid (VWT turned bearish), force close long position!")
            main_logger.info(Fore.RED + f"\n{'='*80}")
            main_logger.info(Fore.RED + "🔴 [Force Close Long] VWT trend reversed")
            main_logger.info(Fore.RED + f"Reason: Current VWT trend ({current_trend}) vs open trend ({trade_state.long_state.trend_at_open})")
            main_logger.info(Fore.RED + f"Close Quantity: {long_info['size']} | Current Price: {current_price:.2f}")
            main_logger.info(Fore.RED + f"{'='*80}\n")
            close_order = place_market_order(FUTURES_SYMBOL, Client.SIDE_SELL, long_info['size'], 'LONG')
            if close_order:
                signal_logger.info(f"[Force Close Long] Qty: {long_info['size']} @ {current_price:.2f}")
            else:
                main_logger.error(Fore.RED + "❌ Force close long failed! Manual intervention required!")
            trade_state.reset_side("long")
            main_logger.info(Fore.YELLOW + "⏸️ Long force close done")
    if short_info['size'] > 0:
        trade_state.short_state.is_trend_valid = not (current_trend == 1 and trade_state.short_state.trend_at_open == -1)
        main_logger.info(Fore.CYAN + f"🧮 Short trend validity | Current VWT trend:{current_trend} | Open trend:{trade_state.short_state.trend_at_open} | Valid:{trade_state.short_state.is_trend_valid}")
        if not trade_state.short_state.is_trend_valid:
            main_logger.warning(Fore.YELLOW + "⚠️ Short trend invalid (VWT turned bullish), force close short position!")
            main_logger.info(Fore.GREEN + f"\n{'='*80}")
            main_logger.info(Fore.GREEN + "🟢 [Force Close Short] VWT trend reversed")
            main_logger.info(Fore.GREEN + f"Reason: Current VWT trend ({current_trend}) vs open trend ({trade_state.short_state.trend_at_open})")
            main_logger.info(Fore.GREEN + f"Close Quantity: {short_info['size']} | Current Price: {current_price:.2f}")
            main_logger.info(Fore.GREEN + f"{'='*80}\n")
            close_order = place_market_order(FUTURES_SYMBOL, Client.SIDE_BUY, short_info['size'], 'SHORT')
            if close_order:
                signal_logger.info(f"[Force Close Short] Qty: {short_info['size']} @ {current_price:.2f}")
            else:
                main_logger.error(Fore.RED + "❌ Force close short failed! Manual intervention required!")
            trade_state.reset_side("short")
            main_logger.info(Fore.YELLOW + "⏸️ Short force close done")

# ———————————————— 主策略循环（VWT 信号） ————————————————
def run_strategy():
    main_logger.info(Fore.CYAN + "="*80)
    main_logger.info(Fore.CYAN + "🚀 VWT Trend Strategy (霓虹发光指标) Started")
    main_logger.info(Fore.CYAN + f"📊 Signal source: {SPOT_SYMBOL} spot klines | Trading on: {FUTURES_SYMBOL} futures")
    main_logger.info(Fore.CYAN + f"⚙️  VWT Params: VWMA Length={VWMA_LENGTH} | ATR Multiplier={VWT_ATR_MULT}")
    main_logger.info(Fore.CYAN + f"💰  Risk Mgmt: Leverage={LEVERAGE}x | Initial Entry={RISK_PERCENTAGE}% | Add Ratio={ADD_RISK_PCT}%")
    main_logger.info(Fore.CYAN + f"🛡️  Trailing Stop: Enabled={ENABLE_TRAILING_STOP} | ATR Mult={TRAILING_ATR_MULT}")
    main_logger.info(Fore.CYAN + "="*80)

    try:
        setup_hedge_mode(FUTURES_SYMBOL)
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to initialize strategy: {e}")
    setup_leverage_and_margin(FUTURES_SYMBOL, LEVERAGE, MARGIN_TYPE)
    restore_trade_state()
    last_kline_time = 0
    kline_update_retries = 0
    MAX_KLINE_RETRIES = 3
    RETRY_INTERVAL = 5

    while True:
        try:
            # 1. 获取现货K线数据
            klines = None
            for retry in range(MAX_KLINE_RETRIES):
                try:
                    klines = client.get_klines(
                        symbol=SPOT_SYMBOL,
                        interval=INTERVAL,
                        limit=LOOKBACK
                    )
                    if klines and len(klines) > 0:
                        main_logger.info(Fore.CYAN + f"✅ Successfully fetched {len(klines)} spot klines (retry {retry+1})")
                        break
                    main_logger.warning(Fore.YELLOW + f"⚠️ Spot kline fetch retry {retry+1}/{MAX_KLINE_RETRIES}: Empty response")
                    time.sleep(RETRY_INTERVAL)
                except BinanceAPIException as e:
                    main_logger.error(Fore.RED + f"❌ Spot kline fetch failed (retry {retry+1}): Binance API error: {e}")
                    time.sleep(RETRY_INTERVAL)
                except Exception as e:
                    main_logger.error(Fore.RED + f"❌ Spot kline fetch failed (retry {retry+1}): {e}")
                    time.sleep(RETRY_INTERVAL)
            
            if not klines or len(klines) == 0:
                main_logger.error(Fore.RED + "❌ Failed to fetch spot kline data after all retries, skipping this round")
                time.sleep(30)
                continue

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_vol', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 2. 检查新K线
            current_kline_time = int(df['timestamp'].iloc[-1])
            current_kline_dt = pd.to_datetime(current_kline_time, unit='ms')
            last_kline_dt = pd.to_datetime(last_kline_time, unit='ms') if last_kline_time !=0 else "None"
            
            main_logger.info(Fore.CYAN + f"🕒 Spot kline time check | Current: {current_kline_dt} | Previous: {last_kline_dt}")
            
            if current_kline_time == last_kline_time:
                kline_update_retries += 1
                if kline_update_retries >= MAX_KLINE_RETRIES:
                    main_logger.warning(Fore.YELLOW + f"⚠️ Spot kline not updated, resetting last_kline_time")
                    last_kline_time = 0
                    kline_update_retries = 0
                else:
                    main_logger.warning(Fore.YELLOW + f"⚠️ Spot kline not updated, waiting 30s (retry {kline_update_retries}/{MAX_KLINE_RETRIES})")
                    time.sleep(30)
                    continue
            else:
                kline_update_retries = 0
            
            last_kline_time = current_kline_time
            kline_time = pd.to_datetime(current_kline_time, unit='ms')
            current_price_spot = df['close'].iloc[-1]

            if pd.isna(current_price_spot):
                main_logger.error(Fore.RED + "❌ Current spot price is NaN, skip this round")
                time.sleep(30)
                continue

            # 3. 计算 VWT 趋势和通道
            if len(df) < VWMA_LENGTH + 1:
                main_logger.error(Fore.RED + f"❌ Insufficient spot kline data: {len(df)} < {VWMA_LENGTH + 1}")
                time.sleep(30)
                continue

            current_trend, prev_trend, vwma_val, upper_band, lower_band = calculate_vwt_trend(df, VWMA_LENGTH, VWT_ATR_MULT)

            # 4. 检测流动性区域
            liq_zones = detect_liquidity_zones(df, lookback_len=LIQ_SWEEP_LENGTH)
            res_text = f"{liq_zones['resistance']:.2f}" if not np.isnan(liq_zones['resistance']) else "None"
            sup_text = f"{liq_zones['support']:.2f}" if not np.isnan(liq_zones['support']) else "None"

            # 5. 趋势不一致强制平仓（基于 VWT）
            force_close_invalid_trend_positions(current_trend, current_price_spot)

            # 获取合约当前仓位
            long_info, short_info = get_position(FUTURES_SYMBOL)

            # 日志输出
            main_logger.info(Fore.CYAN + "="*60)
            main_logger.info(Fore.CYAN + f"🕐 Spot kline close time: {kline_time} | Spot close: {current_price_spot:.2f}")
            main_logger.info(Fore.CYAN + f"📊 VWT: basis={vwma_val:.2f} | upper={upper_band:.2f} | lower={lower_band:.2f}")
            main_logger.info(Fore.CYAN + f"🧭 VWT Trend: Current={current_trend} | Previous={prev_trend}")
            main_logger.info(Fore.CYAN + f"📈 Long position (futures): Size={long_info['size']} | Avg Price={long_info['entry_price']:.2f} | Trend valid={trade_state.long_state.is_trend_valid}")
            main_logger.info(Fore.CYAN + f"📉 Short position (futures): Size={short_info['size']} | Avg Price={short_info['entry_price']:.2f} | Trend valid={trade_state.short_state.is_trend_valid}")

            # 6. 固定止损检查
            sl_triggered, sl_side = check_stop_loss(FUTURES_SYMBOL, current_price_spot)
            if sl_triggered:
                if sl_side == "long" and long_info['size'] > 0:
                    place_market_order(FUTURES_SYMBOL, Client.SIDE_SELL, long_info['size'], 'LONG')
                    trade_state.reset_side("long")
                    signal_logger.info(f"[SL Close Long] Qty: {long_info['size']} @ {current_price_spot:.2f}")
                elif sl_side == "short" and short_info['size'] > 0:
                    place_market_order(FUTURES_SYMBOL, Client.SIDE_BUY, short_info['size'], 'SHORT')
                    trade_state.reset_side("short")
                    signal_logger.info(f"[SL Close Short] Qty: {short_info['size']} @ {current_price_spot:.2f}")
                main_logger.info(Fore.YELLOW + "⏸️ Stop loss executed, pause 60s")
                time.sleep(60)
                continue

            # 7. 移动止损检查
            if ENABLE_TRAILING_STOP:
                current_atr = calculate_atr_rma(df, VWMA_LENGTH).iloc[-1]  # 复用 VWMA 周期的 ATR
                if long_info['size'] > 0:
                    if current_price_spot > trade_state.long_state.highest_since_entry:
                        trade_state.long_state.highest_since_entry = current_price_spot
                if short_info['size'] > 0:
                    if current_price_spot < trade_state.short_state.lowest_since_entry:
                        trade_state.short_state.lowest_since_entry = current_price_spot

                if long_info['size'] > 0:
                    stop_price = trade_state.long_state.highest_since_entry - current_atr * TRAILING_ATR_MULT
                    if current_price_spot < stop_price:
                        main_logger.warning(Fore.YELLOW + f"🚨 Trailing stop hit for LONG: current={current_price_spot:.2f} < stop={stop_price:.2f} (from high={trade_state.long_state.highest_since_entry:.2f})")
                        close_order = place_market_order(FUTURES_SYMBOL, Client.SIDE_SELL, long_info['size'], 'LONG')
                        if close_order:
                            signal_logger.info(f"[Trailing Stop Close Long] Qty: {long_info['size']} @ {current_price_spot:.2f}")
                            trade_state.reset_side("long")
                            long_info['size'] = 0
                        else:
                            main_logger.error(Fore.RED + "❌ Trailing stop close long failed!")

                if short_info['size'] > 0:
                    stop_price = trade_state.short_state.lowest_since_entry + current_atr * TRAILING_ATR_MULT
                    if current_price_spot > stop_price:
                        main_logger.warning(Fore.YELLOW + f"🚨 Trailing stop hit for SHORT: current={current_price_spot:.2f} > stop={stop_price:.2f} (from low={trade_state.short_state.lowest_since_entry:.2f})")
                        close_order = place_market_order(FUTURES_SYMBOL, Client.SIDE_BUY, short_info['size'], 'SHORT')
                        if close_order:
                            signal_logger.info(f"[Trailing Stop Close Short] Qty: {short_info['size']} @ {current_price_spot:.2f}")
                            trade_state.reset_side("short")
                            short_info['size'] = 0
                        else:
                            main_logger.error(Fore.RED + "❌ Trailing stop close short failed!")

            # 8. 部分止盈和加仓
            long_info, short_info = get_position(FUTURES_SYMBOL)
            check_partial_take_profit(FUTURES_SYMBOL, current_price_spot, liq_zones)
            check_breakout_and_add(FUTURES_SYMBOL, current_price_spot, liq_zones, current_trend)

            # 9. VWT 趋势反转开仓信号
            signal_open_long = (current_trend == 1) and (prev_trend != 1)  # 趋势转为看涨
            signal_open_short = (current_trend == -1) and (prev_trend != -1)  # 趋势转为看跌
            
            main_logger.info(Fore.YELLOW + f"🚨 Entry signals | Long: {signal_open_long} | Short: {signal_open_short}")
            
            usdc_balance = get_usdc_balance()
            adjusted_qty = calculate_position_size(FUTURES_SYMBOL, usdc_balance, RISK_PERCENTAGE, LEVERAGE, current_price_spot)

            # 开多头仓
            if signal_open_long and adjusted_qty > 0:
                main_logger.info(Fore.GREEN + "\n" + "="*80)
                main_logger.info(Fore.GREEN + f"🟢 [Long Signal Triggered] VWT turned bullish: {prev_trend}→{current_trend}")
                main_logger.info(Fore.GREEN + f"Planned entry: {adjusted_qty} @ {current_price_spot:.2f} (approx)")
                main_logger.info(Fore.GREEN + "="*80 + "\n")
                open_order = place_market_order(FUTURES_SYMBOL, Client.SIDE_BUY, adjusted_qty, 'LONG')
                if open_order:
                    new_long, _ = get_position(FUTURES_SYMBOL)
                    trade_state.long_state.init_new_position(new_long['size'], new_long['entry_price'], current_trend)
                    signal_logger.info(f"[Long Entry Done] Qty: {adjusted_qty} @ {current_price_spot:.2f}")
                else:
                    main_logger.error(Fore.RED + "❌ Long entry failed")

            # 开空头仓
            elif signal_open_short and adjusted_qty > 0:
                main_logger.info(Fore.RED + "\n" + "="*80)
                main_logger.info(Fore.RED + f"🔴 [Short Signal Triggered] VWT turned bearish: {prev_trend}→{current_trend}")
                main_logger.info(Fore.RED + f"Planned entry: {adjusted_qty} @ {current_price_spot:.2f} (approx)")
                main_logger.info(Fore.RED + "="*80 + "\n")
                open_order = place_market_order(FUTURES_SYMBOL, Client.SIDE_SELL, adjusted_qty, 'SHORT')
                if open_order:
                    _, new_short = get_position(FUTURES_SYMBOL)
                    trade_state.short_state.init_new_position(new_short['size'], new_short['entry_price'], current_trend)
                    signal_logger.info(f"[Short Entry Done] Qty: {adjusted_qty} @ {current_price_spot:.2f}")
                else:
                    main_logger.error(Fore.RED + "❌ Short entry failed")

            else:
                main_logger.info(Fore.CYAN + f"💤 No new entry signals")

            main_logger.info(Fore.CYAN + "="*60 + "\n")
            time.sleep(60)

        except Exception as e:
            main_logger.error(Fore.RED + f"❌ Main loop error: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    try:
        run_strategy()
    except KeyboardInterrupt:
        main_logger.info(Fore.CYAN + "👋 Strategy manually stopped")