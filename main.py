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

# ———————————————— Logging System (Enhanced Debug Log) ————————————————
def setup_logger():
    main_logger = logging.getLogger('L1_Main')
    main_logger.setLevel(logging.DEBUG)
    main_logger.propagate = False
    signal_logger = logging.getLogger('L1_Signal')
    signal_logger.setLevel(logging.INFO)
    signal_logger.propagate = False

    if not main_logger.handlers:
        file_handler = logging.FileHandler(f'l1_full_log_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
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
        signal_file_handler = logging.FileHandler(f'l1_signal_log_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
        signal_file_handler.setLevel(logging.INFO)
        signal_formatter = logging.Formatter('%(asctime)s | %(message)s')
        signal_file_handler.setFormatter(signal_formatter)
        signal_logger.addHandler(signal_file_handler)

    return main_logger, signal_logger

main_logger, signal_logger = setup_logger()

# ———————————————— Full Configuration (New State Management Parameters) ————————————————
load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Trading Base Config
SYMBOL = "ETHUSDC"
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
LOOKBACK = 600

# L1 Core Filter Parameters
ATR_PERIOD = 200
ATR_MULT = 1.5
MU = 0.6

# Leverage & Risk Management
LEVERAGE = 20
MARGIN_TYPE = "ISOLATED"
RISK_PERCENTAGE = 50          # Initial entry capital percentage per side
ADD_RISK_PCT = 20              # Add position capital percentage per side

# Stop Loss Config
STOP_LOSS_PCT = 1.5
ENABLE_STOP_LOSS = False

# Liquidity Sweep Core Parameters (1:1 TradingView Alignment)
LIQ_SWEEP_LENGTH = 8           # Pivot high/low lookback period
LIQ_PARTIAL_PROFIT_RATIO = 0.5 # Partial TP ratio per liquidity zone hit (50% of current position)
BREAKOUT_CONFIRM_BARS = 2      # Breakout confirmation bars (consecutive N closes outside zone to prevent fakeouts)
BREAKOUT_THRESHOLD_PCT = 0.1   # Breakout threshold (0.1% to filter noise)

# State Management Core Config
MAX_ADD_TIMES = 1               # Max add times per trend per side
NEW_ZONE_THRESHOLD_PCT = 0.5    # New zone threshold (price difference ≥0.5% from last operated zone = new opportunity)
STATE_RESET_DELAY = 1           # State reset delay (reset after bar confirmation to prevent misjudgment)

# Binance Client Initialization (with increased timeout)
client = Client(API_KEY, API_SECRET, testnet=False, requests_params={'timeout': 30})
main_logger.info(Fore.CYAN + "✅ Binance live trading client initialized (timeout=30s)")

# ———————————————— [核心修改] 双向持仓状态管理数据类 ————————————————
@dataclass
class SideState:
    """单方向（多/空）的持仓状态"""
    position_size: float = 0.0              # 仓位大小
    entry_price: float = 0.0                # 平均开仓价格
    initial_entry_price: float = 0.0        # 初始开仓价格
    trend_at_open: int = 0                  # 开仓时的趋势（1/-1）
    is_trend_valid: bool = False            # 趋势是否有效
    last_operated_zone_price: float = 0.0   # 最后操作的流动性区域价格
    has_partial_tp_in_zone: bool = False    # 当前区域是否已部分止盈
    has_added_in_zone: bool = False         # 当前区域是否已加仓
    total_add_times: int = 0                # 总加仓次数
    last_add_price: float = 0.0             # 最后加仓价格

    def reset(self):
        """重置单方向状态"""
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

    def init_new_position(self, pos_size: float, entry_price: float, trend: int):
        """初始化新仓位"""
        self.reset()
        self.position_size = pos_size
        self.entry_price = entry_price
        self.initial_entry_price = entry_price
        self.trend_at_open = trend
        self.is_trend_valid = True

    def update_position(self, pos_size: float, entry_price: float):
        """更新仓位状态"""
        self.position_size = pos_size
        self.entry_price = entry_price

    def is_new_liquidity_zone(self, current_zone_price: float, pos_dir: str) -> bool:
        """判断是否为新的流动性区域"""
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
    """双向持仓总状态"""
    long_state: SideState = field(default_factory=SideState)
    short_state: SideState = field(default_factory=SideState)

    def reset_side(self, side: str):
        """重置指定方向的状态"""
        if side == "long":
            self.long_state.reset()
            main_logger.info(Fore.YELLOW + "🔄 Long side state reset")
        elif side == "short":
            self.short_state.reset()
            main_logger.info(Fore.YELLOW + "🔄 Short side state reset")

    def reset_all(self):
        """重置所有状态"""
        self.long_state.reset()
        self.short_state.reset()
        main_logger.info(Fore.YELLOW + "🔄 All trading state reset")

# 全局状态实例
trade_state = TradeState()

# ———————————————— Core Indicator Calculation Functions (Unchanged) ————————————————
def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high = data['high']
    low = data['low']
    close = data['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr

def l1_proximal_filter(close: pd.Series, atr_200: pd.Series,
                        atr_mult: float = 1.5, mu: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    z = np.zeros(n)
    v = np.zeros(n)
    l1_trend = np.zeros(n, dtype=int)
    z[0] = close.iloc[0]
    l1_trend[0] = 0

    for i in range(1, n):
        z_prev = z[i-1]
        v_prev = v[i-1]
        z_pred = z_prev + v_prev
        z_temp = z_pred + mu * (close.iloc[i] - z_pred)
        diff = z_temp - z_prev
        threshold = atr_200.iloc[i] * atr_mult if not pd.isna(atr_200.iloc[i]) else 0
        
        if abs(diff) > threshold:
            v[i] = np.sign(diff) * (abs(diff) - threshold)
        else:
            v[i] = 0.0
        z[i] = z_prev + v[i]
        
        if z[i] > z[i-1]:
            l1_trend[i] = 1
        elif z[i] < z[i-1]:
            l1_trend[i] = -1
        else:
            l1_trend[i] = l1_trend[i-1]
    return z, l1_trend

# ———————————————— Liquidity Zone Detection (Unchanged) ————————————————
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

    return {
        'resistance': nearest_resistance,
        'support': nearest_support
    }

# ———————————————— Breakout Validity Confirmation (Unchanged) ————————————————
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

# ———————————————— [核心修改] 双向持仓相关工具函数 ————————————————
def setup_hedge_mode(symbol: str):
    """设置为双向持仓（对冲）模式"""
    try:
        # 切换到对冲模式
        client.futures_change_position_mode(dualSidePosition=True)
        main_logger.info(Fore.GREEN + "✅ Successfully switched to HEDGE MODE (dual side position)")
        
        # 确认模式切换
        position_mode = client.futures_get_position_mode()
        main_logger.info(Fore.CYAN + f"🔍 Current position mode: {position_mode}")
        
    except BinanceAPIException as e:
        if "No need to change position mode" in str(e):
            main_logger.info(Fore.CYAN + "ℹ️ Already in HEDGE MODE")
        else:
            main_logger.error(Fore.RED + f"❌ Failed to set hedge mode: {e}")
            raise

def setup_leverage_and_margin(symbol: str, leverage: int, margin_type: str):
    """设置杠杆和保证金模式（适配双向持仓）"""
    try:
        # 分别设置多空方向的保证金模式
        client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
        main_logger.info(Fore.CYAN + f"🔧 Margin mode set: {'Isolated' if margin_type == 'ISOLATED' else 'Cross'}")
    except BinanceAPIException as e:
        if "No need to change margin type" not in str(e):
            main_logger.warning(Fore.YELLOW + f"⚠️ Margin mode note: {e}")
    
    try:
        # 设置杠杆（双向持仓下多空杠杆相同）
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        main_logger.info(Fore.CYAN + f"🔧 Leverage set: {leverage}x")
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Leverage setup failed: {e}")
        raise

def get_position(symbol: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    查询双向持仓的仓位信息
    返回: (long_pos_info, short_pos_info)
    long_pos_info: {'size': 仓位大小, 'entry_price': 开仓均价}
    short_pos_info: {'size': 仓位大小, 'entry_price': 开仓均价}
    """
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
                elif position_side == 'SHORT' and amt > 0:  # 双向持仓下amt始终为正
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
    """获取USDC可用余额"""
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
    """获取交易对精度"""
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
    """计算仓位大小（单向）"""
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
    """
    双向持仓下的市价单
    side: Client.SIDE_BUY/Client.SIDE_SELL
    position_side: 'LONG'/'SHORT'
    """
    try:
        _, qty_precision = get_symbol_precision(symbol)
        quantity = round(quantity, qty_precision)
        
        order = client.futures_create_order(
            symbol=symbol, 
            side=side, 
            type=Client.ORDER_TYPE_MARKET, 
            quantity=quantity,
            positionSide=position_side  # 双向持仓必须指定positionSide
        )
        
        action = f"{position_side} Open" if (position_side == 'LONG' and side == Client.SIDE_BUY) or (position_side == 'SHORT' and side == Client.SIDE_SELL) else f"{position_side} Close"
        main_logger.info(Fore.GREEN + f"✅ [{action} Success] Order ID: {order['orderId']}, Quantity: {quantity}")
        return order
    except (BinanceAPIException, BinanceOrderException) as e:
        main_logger.error(Fore.RED + f"❌ [Order Failed] {e} | Side: {side} | PositionSide: {position_side} | Quantity: {quantity}")
        return None

def check_stop_loss(symbol: str, current_price: float) -> Tuple[bool, str]:
    """
    检查止损（双向持仓）
    返回: (是否触发止损, 触发的方向 long/short/none)
    """
    long_info, short_info = get_position(symbol)
    if not ENABLE_STOP_LOSS:
        return False, "none"

    # 检查多头止损
    if long_info['size'] > 0:
        loss_pct = (long_info['entry_price'] - current_price) / long_info['entry_price'] * 100
        if loss_pct >= STOP_LOSS_PCT:
            warn_msg = f"⚠️ [Long Stop Loss Triggered] Entry: {long_info['entry_price']:.2f}, Current: {current_price:.2f}, Loss: {loss_pct:.2f}%"
            main_logger.warning(Fore.YELLOW + warn_msg)
            signal_logger.warning(warn_msg)
            return True, "long"
    
    # 检查空头止损
    if short_info['size'] > 0:
        loss_pct = (current_price - short_info['entry_price']) / short_info['entry_price'] * 100
        if loss_pct >= STOP_LOSS_PCT:
            warn_msg = f"⚠️ [Short Stop Loss Triggered] Entry: {short_info['entry_price']:.2f}, Current: {current_price:.2f}, Loss: {loss_pct:.2f}%"
            main_logger.warning(Fore.YELLOW + warn_msg)
            signal_logger.warning(warn_msg)
            return True, "short"
    
    return False, "none"

# ———————————————— [核心修改] 状态恢复（适配双向持仓） ————————————————
def restore_trade_state():
    """恢复双向持仓状态"""
    long_info, short_info = get_position(SYMBOL)
    
    # 恢复多头状态
    if long_info['size'] > 0:
        trade_state.long_state.init_new_position(
            pos_size=long_info['size'],
            entry_price=long_info['entry_price'],
            trend=1  # 多头开仓趋势为1
        )
        main_logger.info(Fore.GREEN + f"🔄 Restored long state | Size:{long_info['size']} | Entry:{long_info['entry_price']:.2f} | Trend at open:1")
    
    # 恢复空头状态
    if short_info['size'] > 0:
        trade_state.short_state.init_new_position(
            pos_size=short_info['size'],
            entry_price=short_info['entry_price'],
            trend=-1  # 空头开仓趋势为-1
        )
        main_logger.info(Fore.GREEN + f"🔄 Restored short state | Size:{short_info['size']} | Entry:{short_info['entry_price']:.2f} | Trend at open:-1")
    
    if long_info['size'] == 0 and short_info['size'] == 0:
        trade_state.reset_all()
        main_logger.info(Fore.CYAN + "🔄 No positions, state initialized")

# ———————————————— [核心修改] 止盈/加仓逻辑（适配双向持仓） ————————————————
def check_partial_take_profit(symbol: str, current_price: float, liq_zones: dict):
    """双向持仓的部分止盈"""
    long_info, short_info = get_position(symbol)
    qty_precision = get_symbol_precision(symbol)[1]
    min_qty = float(client.futures_exchange_info()['symbols'][0]['filters'][1]['minQty'])

    # 多头止盈（阻力位）
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
                    # 更新状态
                    new_long, _ = get_position(symbol)
                    trade_state.long_state.update_position(new_long['size'], new_long['entry_price'])
                    signal_logger.info(f"[Long Partial TP Done] Close {sell_qty} @ {current_price} | Resistance: {zone_price} | Remaining: {new_long['size']}")

    # 空头止盈（支撑位）
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
                    # 更新状态
                    _, new_short = get_position(symbol)
                    trade_state.short_state.update_position(new_short['size'], new_short['entry_price'])
                    signal_logger.info(f"[Short Partial TP Done] Close {buy_qty} @ {current_price} | Support: {zone_price} | Remaining: {new_short['size']}")

def check_breakout_and_add(symbol: str, current_price: float, liq_zones: dict, current_trend: int):
    """双向持仓的突破加仓"""
    long_info, short_info = get_position(symbol)
    usdc_balance = get_usdc_balance()
    qty_precision = get_symbol_precision(symbol)[1]

    # 多头加仓
    if (long_info['size'] > 0 
        and trade_state.long_state.is_trend_valid 
        and current_trend == 1 
        and trade_state.long_state.total_add_times < MAX_ADD_TIMES
        and not np.isnan(liq_zones['resistance'])):
        
        zone_price = liq_zones['resistance']
        if trade_state.long_state.is_new_liquidity_zone(zone_price, "long"):
            # 获取K线数据验证突破
            klines_data = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=LOOKBACK)
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
                main_logger.info(Fore.BLUE + f"Trend confirmed: L1 remains long | Add count: {trade_state.long_state.total_add_times+1}/{MAX_ADD_TIMES}")
                main_logger.info(Fore.BLUE + f"Action: Add long | Qty: {add_qty}")
                main_logger.info(Fore.BLUE + "="*80 + "\n")
                
                order = place_market_order(symbol, Client.SIDE_BUY, add_qty, 'LONG')
                if order:
                    trade_state.long_state.has_added_in_zone = True
                    trade_state.long_state.total_add_times += 1
                    trade_state.long_state.last_add_price = current_price
                    trade_state.long_state.last_operated_zone_price = zone_price
                    # 更新状态
                    new_long, _ = get_position(symbol)
                    trade_state.long_state.update_position(new_long['size'], new_long['entry_price'])
                    signal_logger.info(f"[Long Add Done] Add {add_qty} @ {current_price} | Breakout: {zone_price} | Total adds: {trade_state.long_state.total_add_times} | Total pos: {new_long['size']}")

    # 空头加仓
    if (short_info['size'] > 0 
        and trade_state.short_state.is_trend_valid 
        and current_trend == -1 
        and trade_state.short_state.total_add_times < MAX_ADD_TIMES
        and not np.isnan(liq_zones['support'])):
        
        zone_price = liq_zones['support']
        if trade_state.short_state.is_new_liquidity_zone(zone_price, "short"):
            # 获取K线数据验证突破
            klines_data = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=LOOKBACK)
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
                main_logger.info(Fore.BLUE + f"Trend confirmed: L1 remains short | Add count: {trade_state.short_state.total_add_times+1}/{MAX_ADD_TIMES}")
                main_logger.info(Fore.BLUE + f"Action: Add short | Qty: {add_qty}")
                main_logger.info(Fore.BLUE + "="*80 + "\n")
                
                order = place_market_order(symbol, Client.SIDE_SELL, add_qty, 'SHORT')
                if order:
                    trade_state.short_state.has_added_in_zone = True
                    trade_state.short_state.total_add_times += 1
                    trade_state.short_state.last_add_price = current_price
                    trade_state.short_state.last_operated_zone_price = zone_price
                    # 更新状态
                    _, new_short = get_position(symbol)
                    trade_state.short_state.update_position(new_short['size'], new_short['entry_price'])
                    signal_logger.info(f"[Short Add Done] Add {add_qty} @ {current_price} | Breakdown: {zone_price} | Total adds: {trade_state.short_state.total_add_times} | Total pos: {new_short['size']}")

# ———————————————— [核心修改] 趋势不一致强制平仓（双向持仓） ————————————————
def force_close_invalid_trend_positions(current_trend: int, current_price: float):
    """强制平仓趋势不一致的仓位（双向持仓）"""
    long_info, short_info = get_position(SYMBOL)
    
    # 检查多头仓位趋势有效性
    if long_info['size'] > 0:
        trade_state.long_state.is_trend_valid = (current_trend == trade_state.long_state.trend_at_open)
        main_logger.info(Fore.CYAN + f"🧮 Long trend validity | Current:{current_trend} | Open:{trade_state.long_state.trend_at_open} | Valid:{trade_state.long_state.is_trend_valid}")
        
        if not trade_state.long_state.is_trend_valid:
            main_logger.warning(Fore.YELLOW + "⚠️ Long trend invalid, force close long position!")
            main_logger.info(Fore.RED + f"\n{'='*80}")
            main_logger.info(Fore.RED + "🔴 [Force Close Long] Trend reversed")
            main_logger.info(Fore.RED + f"Reason: Current trend ({current_trend}) != Entry trend ({trade_state.long_state.trend_at_open})")
            main_logger.info(Fore.RED + f"Close Quantity: {long_info['size']} | Current Price: {current_price:.2f}")
            main_logger.info(Fore.RED + f"{'='*80}\n")
            
            close_order = place_market_order(SYMBOL, Client.SIDE_SELL, long_info['size'], 'LONG')
            if close_order:
                signal_logger.info(f"[Force Close Long] Qty: {long_info['size']} @ {current_price:.2f}")
            else:
                main_logger.error(Fore.RED + "❌ Force close long failed! Manual intervention required!")
            
            # 重置多头状态
            trade_state.reset_side("long")
            main_logger.info(Fore.YELLOW + "⏸️ Long force close done")
    
    # 检查空头仓位趋势有效性
    if short_info['size'] > 0:
        trade_state.short_state.is_trend_valid = (current_trend == trade_state.short_state.trend_at_open)
        main_logger.info(Fore.CYAN + f"🧮 Short trend validity | Current:{current_trend} | Open:{trade_state.short_state.trend_at_open} | Valid:{trade_state.short_state.is_trend_valid}")
        
        if not trade_state.short_state.is_trend_valid:
            main_logger.warning(Fore.YELLOW + "⚠️ Short trend invalid, force close short position!")
            main_logger.info(Fore.GREEN + f"\n{'='*80}")
            main_logger.info(Fore.GREEN + "🟢 [Force Close Short] Trend reversed")
            main_logger.info(Fore.GREEN + f"Reason: Current trend ({current_trend}) != Entry trend ({trade_state.short_state.trend_at_open})")
            main_logger.info(Fore.GREEN + f"Close Quantity: {short_info['size']} | Current Price: {current_price:.2f}")
            main_logger.info(Fore.GREEN + f"{'='*80}\n")
            
            close_order = place_market_order(SYMBOL, Client.SIDE_BUY, short_info['size'], 'SHORT')
            if close_order:
                signal_logger.info(f"[Force Close Short] Qty: {short_info['size']} @ {current_price:.2f}")
            else:
                main_logger.error(Fore.RED + "❌ Force close short failed! Manual intervention required!")
            
            # 重置空头状态
            trade_state.reset_side("short")
            main_logger.info(Fore.YELLOW + "⏸️ Short force close done")

# ———————————————— [核心修改] 主策略循环（适配双向持仓） ————————————————
def run_strategy():
    main_logger.info(Fore.CYAN + "="*80)
    main_logger.info(Fore.CYAN + "🚀 L1 Proximal Filter + Liquidity Sweep (HEDGE MODE) Started")
    main_logger.info(Fore.CYAN + f"📊 Symbol: {SYMBOL} | Kline Interval: {INTERVAL} | Mode: HEDGE (Dual Side)")
    main_logger.info(Fore.CYAN + f"⚙️  Core Params: ATR Period={ATR_PERIOD} | Pivot Lookback={LIQ_SWEEP_LENGTH} | Max Adds={MAX_ADD_TIMES}")
    main_logger.info(Fore.CYAN + f"💰  Risk Mgmt: Leverage={LEVERAGE}x | Initial Entry={RISK_PERCENTAGE}% | Add Ratio={ADD_RISK_PCT}%")
    main_logger.info(Fore.CYAN + "="*80)

    # 初始化：启用对冲模式、设置杠杆
    try:
        setup_hedge_mode(SYMBOL)
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to initialize strategy: {e}")
    setup_leverage_and_margin(SYMBOL, LEVERAGE, MARGIN_TYPE)
    restore_trade_state()
    last_kline_time = 0
    kline_update_retries = 0
    MAX_KLINE_RETRIES = 3
    RETRY_INTERVAL = 5

    while True:
        try:
            # 1. 获取K线数据
            klines = None
            for retry in range(MAX_KLINE_RETRIES):
                try:
                    klines = client.futures_klines(
                        symbol=SYMBOL,
                        interval=INTERVAL,
                        limit=LOOKBACK
                    )
                    if klines and len(klines) > 0:
                        main_logger.info(Fore.CYAN + f"✅ Successfully fetched {len(klines)} klines (retry {retry+1})")
                        break
                    main_logger.warning(Fore.YELLOW + f"⚠️ Kline fetch retry {retry+1}/{MAX_KLINE_RETRIES}: Empty response")
                    time.sleep(RETRY_INTERVAL)
                except BinanceAPIException as e:
                    main_logger.error(Fore.RED + f"❌ Kline fetch failed (retry {retry+1}): Binance API error: {e}")
                    time.sleep(RETRY_INTERVAL)
                except Exception as e:
                    main_logger.error(Fore.RED + f"❌ Kline fetch failed (retry {retry+1}): {e}")
                    time.sleep(RETRY_INTERVAL)
            
            if not klines or len(klines) == 0:
                main_logger.error(Fore.RED + "❌ Failed to fetch kline data after all retries, skipping this round")
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
            
            main_logger.info(Fore.CYAN + f"🕒 Kline time check | Current: {current_kline_dt} | Previous: {last_kline_dt}")
            
            if current_kline_time == last_kline_time:
                kline_update_retries += 1
                if kline_update_retries >= MAX_KLINE_RETRIES:
                    main_logger.warning(Fore.YELLOW + f"⚠️ Kline not updated, resetting last_kline_time")
                    last_kline_time = 0
                    kline_update_retries = 0
                else:
                    main_logger.warning(Fore.YELLOW + f"⚠️ Kline not updated, waiting 30s (retry {kline_update_retries}/{MAX_KLINE_RETRIES})")
                    time.sleep(30)
                    continue
            else:
                kline_update_retries = 0
            
            last_kline_time = current_kline_time
            kline_time = pd.to_datetime(current_kline_time, unit='ms')
            current_price = df['close'].iloc[-1]

            if pd.isna(current_price):
                main_logger.error(Fore.RED + "❌ Current price is NaN, skip this round")
                time.sleep(30)
                continue

            # 3. 计算核心指标
            if len(df) < ATR_PERIOD + 1:
                main_logger.error(Fore.RED + f"❌ Insufficient kline data: {len(df)} < {ATR_PERIOD + 1}")
                time.sleep(30)
                continue

            df['atr_200'] = calculate_atr(df, period=ATR_PERIOD)
            
            if pd.isna(df['atr_200'].iloc[-1]):
                main_logger.error(Fore.RED + "❌ ATR value is NaN, skip this round")
                time.sleep(30)
                continue
                
            z, l1_trend = l1_proximal_filter(df['close'], df['atr_200'], ATR_MULT, MU)
            current_trend = int(l1_trend[-1])
            prev_trend = int(l1_trend[-2])

            # 4. 检测流动性区域
            liq_zones = detect_liquidity_zones(df, lookback_len=LIQ_SWEEP_LENGTH)
            res_text = f"{liq_zones['resistance']:.2f}" if not np.isnan(liq_zones['resistance']) else "None"
            sup_text = f"{liq_zones['support']:.2f}" if not np.isnan(liq_zones['support']) else "None"

            # 5. 趋势不一致强制平仓（最高优先级）
            force_close_invalid_trend_positions(current_trend, current_price)

            # 日志输出
            long_info, short_info = get_position(SYMBOL)
            main_logger.info(Fore.CYAN + "="*60)
            main_logger.info(Fore.CYAN + f"🕐 Kline close time: {kline_time} | Close price: {current_price:.2f}")
            main_logger.info(Fore.CYAN + f"📊 Liquidity zones: Resistance=[{res_text}] | Support=[{sup_text}]")
            main_logger.info(Fore.CYAN + f"🧭 L1 Trend: Current={current_trend} | Previous={prev_trend}")
            main_logger.info(Fore.CYAN + f"📈 Long position: Size={long_info['size']} | Avg Price={long_info['entry_price']:.2f} | Trend valid={trade_state.long_state.is_trend_valid}")
            main_logger.info(Fore.CYAN + f"📉 Short position: Size={short_info['size']} | Avg Price={short_info['entry_price']:.2f} | Trend valid={trade_state.short_state.is_trend_valid}")

            # 6. 止损检查
            sl_triggered, sl_side = check_stop_loss(SYMBOL, current_price)
            if sl_triggered:
                if sl_side == "long" and long_info['size'] > 0:
                    place_market_order(SYMBOL, Client.SIDE_SELL, long_info['size'], 'LONG')
                    trade_state.reset_side("long")
                    signal_logger.info(f"[SL Close Long] Qty: {long_info['size']} @ {current_price:.2f}")
                elif sl_side == "short" and short_info['size'] > 0:
                    place_market_order(SYMBOL, Client.SIDE_BUY, short_info['size'], 'SHORT')
                    trade_state.reset_side("short")
                    signal_logger.info(f"[SL Close Short] Qty: {short_info['size']} @ {current_price:.2f}")
                
                main_logger.info(Fore.YELLOW + "⏸️ Stop loss executed, pause 60s")
                time.sleep(60)
                continue

            # 7. 止盈和加仓
            check_partial_take_profit(SYMBOL, current_price, liq_zones)
            check_breakout_and_add(SYMBOL, current_price, liq_zones, current_trend)

            # 8. 趋势反转开仓信号
            signal_open_long = (current_trend == 1) and (prev_trend == -1)
            signal_open_short = (current_trend == -1) and (prev_trend == 1)
            
            main_logger.info(Fore.YELLOW + f"🚨 Entry signals | Long: {signal_open_long} | Short: {signal_open_short}")
            
            usdc_balance = get_usdc_balance()
            adjusted_qty = calculate_position_size(SYMBOL, usdc_balance, RISK_PERCENTAGE, LEVERAGE, current_price)

            # 开多头仓
            if signal_open_long and adjusted_qty > 0:
                main_logger.info(Fore.GREEN + "\n" + "="*80)
                main_logger.info(Fore.GREEN + "🟢 [Long Signal Triggered] Trend reversal: {prev_trend}→{current_trend}")
                main_logger.info(Fore.GREEN + f"Planned entry: {adjusted_qty} @ {current_price:.2f}")
                main_logger.info(Fore.GREEN + "="*80 + "\n")

                # 开多头仓（双向持仓无需平仓空头）
                open_order = place_market_order(SYMBOL, Client.SIDE_BUY, adjusted_qty, 'LONG')
                if open_order:
                    new_long, _ = get_position(SYMBOL)
                    trade_state.long_state.init_new_position(new_long['size'], new_long['entry_price'], current_trend)
                    signal_logger.info(f"[Long Entry Done] Qty: {adjusted_qty} @ {current_price:.2f}")
                else:
                    main_logger.error(Fore.RED + "❌ Long entry failed")

            # 开空头仓
            elif signal_open_short and adjusted_qty > 0:
                main_logger.info(Fore.RED + "\n" + "="*80)
                main_logger.info(Fore.RED + "🔴 [Short Signal Triggered] Trend reversal: {prev_trend}→{current_trend}")
                main_logger.info(Fore.RED + f"Planned entry: {adjusted_qty} @ {current_price:.2f}")
                main_logger.info(Fore.RED + "="*80 + "\n")

                # 开空头仓（双向持仓无需平仓多头）
                open_order = place_market_order(SYMBOL, Client.SIDE_SELL, adjusted_qty, 'SHORT')
                if open_order:
                    _, new_short = get_position(SYMBOL)
                    trade_state.short_state.init_new_position(new_short['size'], new_short['entry_price'], current_trend)
                    signal_logger.info(f"[Short Entry Done] Qty: {adjusted_qty} @ {current_price:.2f}")
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