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
from typing import Optional
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
RISK_PERCENTAGE = 50          # Initial entry capital percentage
ADD_RISK_PCT = 20              # Add position capital percentage (conservative half of initial)

# Stop Loss Config
STOP_LOSS_PCT = 1.5
ENABLE_STOP_LOSS = False

# Liquidity Sweep Core Parameters (1:1 TradingView Alignment)
LIQ_SWEEP_LENGTH = 8           # Pivot high/low lookback period
LIQ_PARTIAL_PROFIT_RATIO = 0.5 # Partial TP ratio per liquidity zone hit (50% of current position)
BREAKOUT_CONFIRM_BARS = 2      # Breakout confirmation bars (consecutive N closes outside zone to prevent fakeouts)
BREAKOUT_THRESHOLD_PCT = 0.1   # Breakout threshold (0.1% to filter noise)

# State Management Core Config
MAX_ADD_TIMES = 1               # Max add times per trend (prevent heavy position blowup)
NEW_ZONE_THRESHOLD_PCT = 0.5    # New zone threshold (price difference ≥0.5% from last operated zone = new opportunity)
STATE_RESET_DELAY = 1           # State reset delay (reset after bar confirmation to prevent misjudgment)

# Binance Client Initialization
client = Client(API_KEY, API_SECRET, testnet=False)
main_logger.info(Fore.CYAN + "✅ Binance live trading client initialized")

# ———————————————— [Core Addition] Full Lifecycle State Management Dataclass ————————————————
@dataclass
class TradeState:
    # Position Base State
    position_dir: str = "none"             # Current position direction: long/short/none
    position_size: float = 0.0              # Current position size
    entry_price: float = 0.0                # Weighted average entry price
    initial_entry_price: float = 0.0        # Initial entry price (first entry of trend, unchanged)
    
    # Liquidity Operation State (Core Anti-Duplication)
    last_operated_zone_price: float = 0.0   # Last operated liquidity zone price (TP/Add)
    has_partial_tp_in_zone: bool = False    # Has partial TP been executed in current zone
    has_added_in_zone: bool = False          # Has add been executed in current zone
    
    # Add Position Control State
    total_add_times: int = 0                 # Total add times in current trend
    last_add_price: float = 0.0              # Last add price
    
    # Trend Lock State
    trend_at_open: int = 0                    # L1 trend at entry (1 long/-1 short, prevent mid-trend reversal misoperation)
    is_trend_valid: bool = False              # Is current trend valid (matches entry trend)

    # Reset State (Call on close/stop loss/trend reversal)
    def reset(self):
        self.position_dir = "none"
        self.position_size = 0.0
        self.entry_price = 0.0
        self.initial_entry_price = 0.0
        self.last_operated_zone_price = 0.0
        self.has_partial_tp_in_zone = False
        self.has_added_in_zone = False
        self.total_add_times = 0
        self.last_add_price = 0.0
        self.trend_at_open = 0
        self.is_trend_valid = False
        main_logger.info(Fore.YELLOW + "🔄 Trading state fully reset")
        signal_logger.info("[State Reset] Position cleared, all trading flags reset")

    # Initialize New Trend Position State
    def init_new_position(self, pos_dir: str, pos_size: float, entry_price: float, trend: int):
        self.reset()  # Clear previous trend residual state before new position
        self.position_dir = pos_dir
        self.position_size = pos_size
        self.entry_price = entry_price
        self.initial_entry_price = entry_price
        self.trend_at_open = trend
        self.is_trend_valid = True
        main_logger.info(Fore.GREEN + f"📝 New position state initialized | Dir:{pos_dir} | Size:{pos_size} | Entry:{entry_price:.2f}")
        signal_logger.info(f"[State Init] {pos_dir} position | Size:{pos_size} | Entry:{entry_price:.2f} | Trend:{trend}")

    # Update Position State (Call after TP/Add/Close to sync latest position data)
    def update_position(self, pos_dir: str, pos_size: float, entry_price: float):
        self.position_dir = pos_dir
        self.position_size = pos_size
        self.entry_price = entry_price
        main_logger.debug(Fore.CYAN + f"📊 Position state updated | Dir:{pos_dir} | Size:{pos_size} | Avg Price:{entry_price:.2f}")

    # Check if New Liquidity Zone (Core Anti-Duplication)
    def is_new_liquidity_zone(self, current_zone_price: float, pos_dir: str) -> bool:
        # First operation, no history, directly judge as new zone
        if self.last_operated_zone_price == 0:
            return True
        
        # Calculate price difference percentage between current zone and last operated zone
        price_diff_pct = abs(current_zone_price - self.last_operated_zone_price) / self.last_operated_zone_price * 100
        
        # Long: New resistance must be above last operated zone and price difference meets threshold
        if pos_dir == "long":
            is_new = (current_zone_price > self.last_operated_zone_price) and (price_diff_pct >= NEW_ZONE_THRESHOLD_PCT)
        # Short: New support must be below last operated zone and price difference meets threshold
        elif pos_dir == "short":
            is_new = (current_zone_price < self.last_operated_zone_price) and (price_diff_pct >= NEW_ZONE_THRESHOLD_PCT)
        else:
            is_new = False

        if is_new:
            # New zone resets current zone operation flags
            self.has_partial_tp_in_zone = False
            self.has_added_in_zone = False
            main_logger.info(Fore.CYAN + f"🎯 New liquidity zone detected | Price:{current_zone_price:.2f} | Diff:{price_diff_pct:.2f}%")
        return is_new

# Global Unique State Instance (Single-threaded loop, thread-safe)
trade_state = TradeState()

# ———————————————— Core Indicator Calculation Functions (Unchanged, 1:1 TradingView Alignment) ————————————————
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

# ———————————————— Liquidity Zone Detection (Optimized, TradingView Pivot Logic Alignment) ————————————————
def detect_liquidity_zones(data: pd.DataFrame, lookback_len: int = 8) -> dict:
    """
    1:1 aligned with TradingView Pivot High/Low detection, output valid support/resistance levels
    Logic: Find confirmed Pivot High/Low (verified by lookback_len bars on each side, no lookahead bias)
    """
    df = data.copy()
    # Only use closed bars, exclude current incomplete bar to avoid lookahead bias
    closed_df = df.iloc[:-1].copy()
    nearest_resistance = np.nan
    nearest_support = np.nan

    if len(closed_df) < lookback_len * 2 + 1:
        return {'resistance': nearest_resistance, 'support': nearest_support}

    # Calculate Pivot High: Current high is the highest of lookback_len*2+1 bars (confirmed, no lookahead)
    closed_df['is_pivot_high'] = closed_df['high'] == closed_df['high'].rolling(window=lookback_len*2+1, center=True).max()
    # Calculate Pivot Low: Current low is the lowest of lookback_len*2+1 bars
    closed_df['is_pivot_low'] = closed_df['low'] == closed_df['low'].rolling(window=lookback_len*2+1, center=True).min()

    # Extract valid pivot points
    pivot_highs = closed_df[closed_df['is_pivot_high']]['high']
    pivot_lows = closed_df[closed_df['is_pivot_low']]['low']

    # Take the nearest valid pivot point outside current price (avoid already broken zones)
    current_price = df['close'].iloc[-1]
    if not pivot_highs.empty:
        # Resistance: Nearest pivot high above current price
        valid_resistances = pivot_highs[pivot_highs > current_price]
        if not valid_resistances.empty:
            nearest_resistance = valid_resistances.iloc[-1]
    if not pivot_lows.empty:
        # Support: Nearest pivot low below current price
        valid_supports = pivot_lows[pivot_lows < current_price]
        if not valid_supports.empty:
            nearest_support = valid_supports.iloc[-1]

    return {
        'resistance': nearest_resistance,
        'support': nearest_support
    }

# ———————————————— [New] Breakout Validity Confirmation (Core Anti-Fakeout) ————————————————
def confirm_breakout(data: pd.DataFrame, zone_price: float, pos_dir: str) -> bool:
    """
    Confirm breakout validity: Consecutive N bars close outside zone with threshold
    :param data: Kline data
    :param zone_price: Liquidity zone price (resistance/support)
    :param pos_dir: Position direction
    :return: Is valid breakout
    """
    if len(data) < BREAKOUT_CONFIRM_BARS:
        return False
    
    # Take last N closed bars
    recent_bars = data.iloc[-(BREAKOUT_CONFIRM_BARS+1):-1]
    
    if pos_dir == "long":
        # Long breakout: Consecutive N closes > resistance * (1+threshold)
        breakout_level = zone_price * (1 + BREAKOUT_THRESHOLD_PCT / 100)
        all_breakout = all(recent_bars['close'] > breakout_level)
    elif pos_dir == "short":
        # Short breakout: Consecutive N closes < support * (1-threshold)
        breakout_level = zone_price * (1 - BREAKOUT_THRESHOLD_PCT / 100)
        all_breakout = all(recent_bars['close'] < breakout_level)
    else:
        all_breakout = False

    if all_breakout:
        main_logger.info(Fore.BLUE + f"✅ Valid breakout confirmed | Zone Price:{zone_price:.2f} | Breakout Level:{breakout_level:.2f} | Confirm Bars:{BREAKOUT_CONFIRM_BARS}")
    return all_breakout

# ———————————————— [New] Script Restart State Auto-Restore ————————————————
def restore_trade_state():
    """Automatically restore trading state from Binance on script start/restart to avoid state loss"""
    pos_dir, pos_size, entry_price = get_position(SYMBOL)
    if pos_dir != "none" and pos_size > 0:
        # Has position, restore state
        klines = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=LOOKBACK)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume','close_time', 'quote_vol', 'trades', 'taker_buy_base','taker_buy_quote', 'ignore'])
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate current trend
        df['atr_200'] = calculate_atr(df, period=ATR_PERIOD)
        _, l1_trend = l1_proximal_filter(df['close'], df['atr_200'], ATR_MULT, MU)
        current_trend = int(l1_trend[-1])

        # Restore state
        trade_state.init_new_position(pos_dir, pos_size, entry_price, current_trend)
        main_logger.info(Fore.GREEN + f"🔄 Restart state restored | Position:{pos_dir} {pos_size} | Avg Price:{entry_price:.2f}")
        signal_logger.info(f"[Restart Restore] {pos_dir} position | Size:{pos_size} | Avg Price:{entry_price:.2f}")
    else:
        # No position, reset state
        trade_state.reset()
        main_logger.info(Fore.CYAN + "🔄 No position on start, state initialized")

# ———————————————— Trading Helper Functions (Enhanced with Debug Log) ————————————————
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

def calculate_position_size(symbol: str, usdc_balance: float, risk_pct: float, leverage: int, current_price: float) -> float:
    """Enhanced position size calculation with detailed debug log"""
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

    # Calculate position size
    risk_amount = usdc_balance * (risk_pct / 100)
    notional_value = risk_amount * leverage
    position_size = notional_value / current_price
    adjusted_size = round(position_size, qty_precision)
    
    # Debug log
    main_logger.info(Fore.YELLOW + f"📏 Position calculation details | Risk amount:{risk_amount} | Notional value:{notional_value} | Raw position:{position_size} | Adjusted:{adjusted_size}")
    
    # Ensure position size is not less than minimum quantity
    if adjusted_size < min_qty:
        main_logger.warning(Fore.YELLOW + f"⚠️ Adjusted position {adjusted_size} is less than minimum quantity {min_qty}, force set to {min_qty}")
        adjusted_size = min_qty
    
    # Ensure position size is greater than 0
    if adjusted_size <= 0:
        main_logger.error(Fore.RED + f"❌ Calculated position {adjusted_size} is invalid (<=0)")
        return min_qty  # Return minimum quantity at least
    
    return adjusted_size

def get_position(symbol: str) -> tuple[str, float, float]:
    """Enhanced position query with debug log"""
    try:
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            if pos['symbol'] == symbol:
                amt = float(pos['positionAmt'])
                entry_price = float(pos['entryPrice'])
                if amt > 0:
                    main_logger.info(Fore.CYAN + f"📈 Current position: Long {amt} | Entry price: {entry_price}")
                    return 'long', amt, entry_price
                elif amt < 0:
                    main_logger.info(Fore.CYAN + f"📉 Current position: Short {abs(amt)} | Entry price: {entry_price}")
                    return 'short', abs(amt), entry_price
        main_logger.info(Fore.CYAN + "📊 No current positions")
        return 'none', 0, 0
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to get position: {e}")
        return 'none', 0, 0

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

def place_market_order(symbol: str, side: str, quantity: float) -> dict:
    """Enhanced order placement with error handling"""
    try:
        # Ensure quantity meets precision requirements
        _, qty_precision = get_symbol_precision(symbol)
        quantity = round(quantity, qty_precision)
        
        order = client.futures_create_order(
            symbol=symbol, 
            side=side, 
            type=Client.ORDER_TYPE_MARKET, 
            quantity=quantity
        )
        action = "Long Open" if side == Client.SIDE_BUY else "Short Open" if side == Client.SIDE_SELL else "Close Position"
        main_logger.info(Fore.GREEN + f"✅ [{action} Success] Order ID: {order['orderId']}, Quantity: {quantity}")
        return order
    except (BinanceAPIException, BinanceOrderException) as e:
        main_logger.error(Fore.RED + f"❌ [Order Failed] {e} | Side: {side} | Quantity: {quantity}")
        return None

def check_stop_loss(symbol: str, current_price: float) -> bool:
    pos, pos_amt, entry_price = get_position(symbol)
    if pos == 'none' or not ENABLE_STOP_LOSS:
        return False

    is_stop_triggered = False
    if pos == 'long':
        loss_pct = (entry_price - current_price) / entry_price * 100
        if loss_pct >= STOP_LOSS_PCT:
            warn_msg = f"⚠️ [Long Stop Loss Triggered] Entry: {entry_price:.2f}, Current: {current_price:.2f}, Loss: {loss_pct:.2f}%"
            main_logger.warning(Fore.YELLOW + warn_msg)
            signal_logger.warning(warn_msg)
            is_stop_triggered = True
    elif pos == 'short':
        loss_pct = (current_price - entry_price) / entry_price * 100
        if loss_pct >= STOP_LOSS_PCT:
            warn_msg = f"⚠️ [Short Stop Loss Triggered] Entry: {entry_price:.2f}, Current: {current_price:.2f}, Loss: {loss_pct:.2f}%"
            main_logger.warning(Fore.YELLOW + warn_msg)
            signal_logger.warning(warn_msg)
            is_stop_triggered = True
    return is_stop_triggered

# ———————————————— [Improved] Liquidity Strategy Core Logic (Full State Control) ————————————————
def check_partial_take_profit(symbol: str, current_price: float, liq_zones: dict) -> None:
    """
    Partial take profit logic with state control:
    1. Only execute when current trend is valid
    2. Only one TP per liquidity zone
    3. New zone auto resets TP flag
    """
    # No position / invalid trend, skip
    if trade_state.position_dir == "none" or not trade_state.is_trend_valid:
        return
    
    pos_dir = trade_state.position_dir
    pos_size = trade_state.position_size
    qty_precision = get_symbol_precision(symbol)[1]

    # Long TP: Hit resistance
    if pos_dir == "long" and not np.isnan(liq_zones['resistance']):
        zone_price = liq_zones['resistance']
        # Check if new zone, update state flags
        trade_state.is_new_liquidity_zone(zone_price, pos_dir)
        
        # Trigger conditions: Price hits resistance, no TP in current zone, position > min qty
        min_qty = float(client.futures_exchange_info()['symbols'][0]['filters'][1]['minQty'])
        if (current_price >= zone_price 
            and not trade_state.has_partial_tp_in_zone 
            and pos_size > min_qty):
            
            # Calculate TP qty (50% of current position)
            sell_qty = round(pos_size * LIQ_PARTIAL_PROFIT_RATIO, qty_precision)
            sell_qty = max(sell_qty, min_qty) # Ensure not less than min qty

            # Execute TP
            main_logger.info(Fore.MAGENTA + "\n" + "="*80)
            main_logger.info(Fore.MAGENTA + f"🎯 [Liquidity Partial TP] Hit resistance: {zone_price:.2f}")
            main_logger.info(Fore.MAGENTA + f"Action: Close {LIQ_PARTIAL_PROFIT_RATIO*100}% position | Qty: {sell_qty}")
            main_logger.info(Fore.MAGENTA + "="*80 + "\n")
            
            order = place_market_order(symbol, Client.SIDE_SELL, sell_qty)
            if order:
                # Order success, update state
                trade_state.has_partial_tp_in_zone = True
                trade_state.last_operated_zone_price = zone_price
                # Sync latest position state
                new_pos_dir, new_pos_size, new_entry_price = get_position(symbol)
                trade_state.update_position(new_pos_dir, new_pos_size, new_entry_price)
                # Log
                signal_logger.info(f"[Partial TP Done] Close long {sell_qty} @ {current_price} | Resistance: {zone_price} | Remaining: {new_pos_size}")

    # Short TP: Hit support
    elif pos_dir == "short" and not np.isnan(liq_zones['support']):
        zone_price = liq_zones['support']
        # Check if new zone, update state flags
        trade_state.is_new_liquidity_zone(zone_price, pos_dir)
        
        # Trigger conditions
        min_qty = float(client.futures_exchange_info()['symbols'][0]['filters'][1]['minQty'])
        if (current_price <= zone_price 
            and not trade_state.has_partial_tp_in_zone 
            and pos_size > min_qty):
            
            # Calculate TP qty
            buy_qty = round(pos_size * LIQ_PARTIAL_PROFIT_RATIO, qty_precision)
            buy_qty = max(buy_qty, min_qty)

            # Execute TP
            main_logger.info(Fore.MAGENTA + "\n" + "="*80)
            main_logger.info(Fore.MAGENTA + f"🎯 [Liquidity Partial TP] Hit support: {zone_price:.2f}")
            main_logger.info(Fore.MAGENTA + f"Action: Close {LIQ_PARTIAL_PROFIT_RATIO*100}% position | Qty: {buy_qty}")
            main_logger.info(Fore.MAGENTA + "="*80 + "\n")
            
            order = place_market_order(symbol, Client.SIDE_BUY, buy_qty)
            if order:
                # Update state
                trade_state.has_partial_tp_in_zone = True
                trade_state.last_operated_zone_price = zone_price
                # Sync position
                new_pos_dir, new_pos_size, new_entry_price = get_position(symbol)
                trade_state.update_position(new_pos_dir, new_pos_size, new_entry_price)
                # Log
                signal_logger.info(f"[Partial TP Done] Close short {buy_qty} @ {current_price} | Support: {zone_price} | Remaining: {new_pos_size}")

def check_breakout_and_add(symbol: str, current_price: float, liq_zones: dict, current_trend: int) -> None:
    """
    Breakout add position logic with state control:
    1. Only execute when trend matches entry trend
    2. Only one add per zone
    3. Strict max add limit
    4. Must confirm valid breakout first
    """
    # No position / invalid trend / max adds reached, skip
    if (trade_state.position_dir == "none" 
        or not trade_state.is_trend_valid 
        or trade_state.total_add_times >= MAX_ADD_TIMES):
        return
    
    pos_dir = trade_state.position_dir
    usdc_balance = get_usdc_balance()
    qty_precision = get_symbol_precision(symbol)[1]

    # Long add: Valid breakout of resistance, trend remains long
    if pos_dir == "long" and current_trend == 1 and not np.isnan(liq_zones['resistance']):
        zone_price = liq_zones['resistance']
        # Check if new zone
        trade_state.is_new_liquidity_zone(zone_price, pos_dir)
        
        # Trigger conditions: Valid breakout, no add in current zone, partial TP done in this zone (per your logic)
        klines_data = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=LOOKBACK)
        df_kline = pd.DataFrame(klines_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume','close_time', 'quote_vol', 'trades', 'taker_buy_base','taker_buy_quote', 'ignore'])
        for col in ['open', 'high', 'low', 'close']:
            df_kline[col] = pd.to_numeric(df_kline[col], errors='coerce')
            
        if (confirm_breakout(df_kline, zone_price, pos_dir)
            and not trade_state.has_added_in_zone
            and trade_state.has_partial_tp_in_zone):
            
            # Calculate add qty
            add_qty = calculate_position_size(symbol, usdc_balance, ADD_RISK_PCT, LEVERAGE, current_price)
            if add_qty <= 0:
                main_logger.warning(Fore.YELLOW + "⚠️ Add qty insufficient, skip add")
                return

            # Execute add
            main_logger.info(Fore.BLUE + "\n" + "="*80)
            main_logger.info(Fore.BLUE + f"🚀 [Breakout Add] Valid breakout of resistance: {zone_price:.2f}")
            main_logger.info(Fore.BLUE + f"Trend confirmed: L1 remains long | Add count: {trade_state.total_add_times+1}/{MAX_ADD_TIMES}")
            main_logger.info(Fore.BLUE + f"Action: Add long | Qty: {add_qty}")
            main_logger.info(Fore.BLUE + "="*80 + "\n")
            
            order = place_market_order(symbol, Client.SIDE_BUY, add_qty)
            if order:
                # Update state
                trade_state.has_added_in_zone = True
                trade_state.total_add_times += 1
                trade_state.last_add_price = current_price
                trade_state.last_operated_zone_price = zone_price
                # Sync position
                new_pos_dir, new_pos_size, new_entry_price = get_position(symbol)
                trade_state.update_position(new_pos_dir, new_pos_size, new_entry_price)
                # Log
                signal_logger.info(f"[Breakout Add Done] Add long {add_qty} @ {current_price} | Breakout: {zone_price} | Total adds: {trade_state.total_add_times} | Total pos: {new_pos_size}")

    # Short add: Valid breakdown of support, trend remains short
    elif pos_dir == "short" and current_trend == -1 and not np.isnan(liq_zones['support']):
        zone_price = liq_zones['support']
        # Check if new zone
        trade_state.is_new_liquidity_zone(zone_price, pos_dir)
        
        # Trigger conditions
        klines_data = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=LOOKBACK)
        df_kline = pd.DataFrame(klines_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume','close_time', 'quote_vol', 'trades', 'taker_buy_base','taker_buy_quote', 'ignore'])
        for col in ['open', 'high', 'low', 'close']:
            df_kline[col] = pd.to_numeric(df_kline[col], errors='coerce')
            
        if (confirm_breakout(df_kline, zone_price, pos_dir)
            and not trade_state.has_added_in_zone
            and trade_state.has_partial_tp_in_zone):
            
            # Calculate add qty
            add_qty = calculate_position_size(symbol, usdc_balance, ADD_RISK_PCT, LEVERAGE, current_price)
            if add_qty <= 0:
                main_logger.warning(Fore.YELLOW + "⚠️ Add qty insufficient, skip add")
                return

            # Execute add
            main_logger.info(Fore.BLUE + "\n" + "="*80)
            main_logger.info(Fore.BLUE + f"🚀 [Breakdown Add] Valid breakdown of support: {zone_price:.2f}")
            main_logger.info(Fore.BLUE + f"Trend confirmed: L1 remains short | Add count: {trade_state.total_add_times+1}/{MAX_ADD_TIMES}")
            main_logger.info(Fore.BLUE + f"Action: Add short | Qty: {add_qty}")
            main_logger.info(Fore.BLUE + "="*80 + "\n")
            
            order = place_market_order(symbol, Client.SIDE_SELL, add_qty)
            if order:
                # Update state
                trade_state.has_added_in_zone = True
                trade_state.total_add_times += 1
                trade_state.last_add_price = current_price
                trade_state.last_operated_zone_price = zone_price
                # Sync position
                new_pos_dir, new_pos_size, new_entry_price = get_position(symbol)
                trade_state.update_position(new_pos_dir, new_pos_size, new_entry_price)
                # Log
                signal_logger.info(f"[Breakdown Add Done] Add short {add_qty} @ {current_price} | Breakdown: {zone_price} | Total adds: {trade_state.total_add_times} | Total pos: {new_pos_size}")

# ———————————————— [Refactored] Main Strategy Loop (Full Process State Control + Debug) ————————————————
def run_strategy():
    main_logger.info(Fore.CYAN + "="*80)
    main_logger.info(Fore.CYAN + "🚀 L1 Proximal Filter + Liquidity Sweep Enhanced Strategy (with Closed-Loop State Control) Started")
    main_logger.info(Fore.CYAN + f"📊 Symbol: {SYMBOL} | Kline Interval: {INTERVAL}")
    main_logger.info(Fore.CYAN + f"⚙️  Core Params: ATR Period={ATR_PERIOD} | Pivot Lookback={LIQ_SWEEP_LENGTH} | Max Adds={MAX_ADD_TIMES}")
    main_logger.info(Fore.CYAN + f"💰  Risk Mgmt: Leverage={LEVERAGE}x | Initial Entry={RISK_PERCENTAGE}% | Add Ratio={ADD_RISK_PCT}%")
    main_logger.info(Fore.CYAN + "="*80)

    # Startup initialization
    setup_leverage_and_margin(SYMBOL, LEVERAGE, MARGIN_TYPE)
    price_precision, qty_precision = get_symbol_precision(SYMBOL)
    restore_trade_state() # Auto restore state on restart
    last_kline_time = 0
    kline_update_retries = 0
    MAX_KLINE_RETRIES = 3

    while True:
        try:
            # 1. Get kline data with retry logic
            klines = None
            for retry in range(MAX_KLINE_RETRIES):
                try:
                    # If we have a previous kline time, fetch data after it to avoid duplicates
                    if last_kline_time > 0:
                        klines = client.futures_klines(
                            symbol=SYMBOL,
                            interval=INTERVAL,
                            limit=LOOKBACK,
                            startTime=last_kline_time + 1  # Fetch data after last kline time
                        )
                    else:
                        klines = client.futures_klines(
                            symbol=SYMBOL,
                            interval=INTERVAL,
                            limit=LOOKBACK
                        )
                    if klines:
                        break
                    main_logger.warning(Fore.YELLOW + f"⚠️ Kline fetch retry {retry+1}/{MAX_KLINE_RETRIES}")
                    time.sleep(2)
                except Exception as e:
                    main_logger.error(Fore.RED + f"❌ Kline fetch failed (retry {retry+1}): {e}")
                    time.sleep(2)
            
            if not klines:
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

            # 2. New kline check (Enhanced with retry and debug log)
            current_kline_time = int(df['timestamp'].iloc[-1])
            current_kline_dt = pd.to_datetime(current_kline_time, unit='ms')
            last_kline_dt = pd.to_datetime(last_kline_time, unit='ms') if last_kline_time !=0 else "None"
            
            main_logger.info(Fore.CYAN + f"🕒 Kline time check | Current kline time: {current_kline_dt} | Previous kline time: {last_kline_dt}")
            
            if current_kline_time == last_kline_time:
                kline_update_retries += 1
                if kline_update_retries >= MAX_KLINE_RETRIES:
                    main_logger.warning(Fore.YELLOW + f"⚠️ Kline not updated after {MAX_KLINE_RETRIES} retries, resetting last_kline_time to force refresh")
                    last_kline_time = 0  # Reset to force full data fetch next time
                    kline_update_retries = 0
                else:
                    main_logger.warning(Fore.YELLOW + f"⚠️ Kline not updated, waiting 30 seconds (retry {kline_update_retries}/{MAX_KLINE_RETRIES})")
                    time.sleep(30)
                    continue
            else:
                kline_update_retries = 0  # Reset retries on successful update
            
            last_kline_time = current_kline_time
            kline_time = pd.to_datetime(current_kline_time, unit='ms')
            current_price = df['close'].iloc[-1]

            # Check price validity
            if pd.isna(current_price):
                main_logger.error(Fore.RED + "❌ Current price is NaN, skip this round")
                time.sleep(30)
                continue

            # 3. Core indicator calculation with data validation
            if len(df) < ATR_PERIOD + 1:
                main_logger.error(Fore.RED + f"❌ Insufficient kline data: {len(df)} bars, need at least {ATR_PERIOD + 1} for ATR calculation")
                time.sleep(30)
                continue

            df['atr_200'] = calculate_atr(df, period=ATR_PERIOD)
            
            # Check ATR validity
            if pd.isna(df['atr_200'].iloc[-1]):
                main_logger.error(Fore.RED + "❌ ATR value is NaN (insufficient kline data: <200), skip this round")
                time.sleep(30)
                continue
                
            z, l1_trend = l1_proximal_filter(df['close'], df['atr_200'], ATR_MULT, MU)
            
            # Unify trend variable type to int (solve numpy type comparison issue)
            current_trend = int(l1_trend[-1])
            prev_trend = int(l1_trend[-2])

            # 4. Liquidity zone detection
            liq_zones = detect_liquidity_zones(df, lookback_len=LIQ_SWEEP_LENGTH)
            res_text = f"{liq_zones['resistance']:.2f}" if not np.isnan(liq_zones['resistance']) else "None"
            sup_text = f"{liq_zones['support']:.2f}" if not np.isnan(liq_zones['support']) else "None"

            # 5. Trend validity check (Core: Trend reversed after entry, forbid TP/add)
            if trade_state.position_dir != "none":
                trade_state.is_trend_valid = (current_trend == trade_state.trend_at_open)
                if not trade_state.is_trend_valid:
                    main_logger.warning(Fore.YELLOW + "⚠️ Trend reversed, lock current zone operations, wait for close signal")

            # Log output
            main_logger.info(Fore.CYAN + "="*60)
            main_logger.info(Fore.CYAN + f"🕐 Kline close time: {kline_time} | Close price: {current_price:.2f}")
            main_logger.info(Fore.CYAN + f"📊 Liquidity zones: Nearest resistance=[{res_text}] | Nearest support=[{sup_text}]")
            main_logger.info(Fore.CYAN + f"🧭 L1 Trend: Current={current_trend} | Previous={prev_trend} | Entry trend={trade_state.trend_at_open}")
            main_logger.info(Fore.CYAN + f"📈 Position state: Direction={trade_state.position_dir} | Size={trade_state.position_size} | Avg Price={trade_state.entry_price:.2f}")
            main_logger.info(Fore.CYAN + f"🔢 Operation log: Total adds={trade_state.total_add_times} | Last operated zone={trade_state.last_operated_zone_price:.2f}")

            # 6. Stop loss logic (Highest priority, reset all states after SL)
            if check_stop_loss(SYMBOL, current_price):
                pos, pos_amt, _ = get_position(SYMBOL)
                if pos == 'long':
                    place_market_order(SYMBOL, Client.SIDE_SELL, pos_amt)
                    signal_logger.info(f"[SL Close] Close long {pos_amt} @ {current_price:.2f}")
                elif pos == 'short':
                    place_market_order(SYMBOL, Client.SIDE_BUY, pos_amt)
                    signal_logger.info(f"[SL Close] Close short {pos_amt} @ {current_price:.2f}")
                # Reset state after SL
                trade_state.reset()
                main_logger.info(Fore.YELLOW + "⏸️ Stop loss executed, pause remaining operations this round")
                main_logger.info(Fore.CYAN + "="*60 + "\n")
                time.sleep(60)
                continue

            # 7. Liquidity strategy execution (TP → Add, order cannot be changed)
            check_partial_take_profit(SYMBOL, current_price, liq_zones)
            check_breakout_and_add(SYMBOL, current_price, liq_zones, current_trend)

            # 8. Trend reversal open/close signals (Core entry logic with enhanced debug)
            # Force convert to int to ensure accurate comparison
            signal_open_long = (current_trend == 1) and (prev_trend == -1)
            signal_open_short = (current_trend == -1) and (prev_trend == 1)
            
            # Key debug log: Print signal status
            main_logger.info(Fore.YELLOW + f"🚨 Entry signal detection | Long signal: {signal_open_long} | Short signal: {signal_open_short}")
            
            usdc_balance = get_usdc_balance()
            adjusted_qty = calculate_position_size(SYMBOL, usdc_balance, RISK_PERCENTAGE, LEVERAGE, current_price)
            current_pos, current_pos_amt, _ = get_position(SYMBOL)

            # Long entry execution
            if signal_open_long:
                main_logger.info(Fore.GREEN + "\n" + "="*80)
                main_logger.info(Fore.GREEN + "🟢 🟢 🟢 [High Probability Long Signal Triggered] 🟢 🟢 🟢")
                main_logger.info(Fore.GREEN + f"Trigger time: {kline_time} | Close price: {current_price:.2f}")
                main_logger.info(Fore.GREEN + f"Trend reversal: {prev_trend} → {current_trend}")
                main_logger.info(Fore.GREEN + f"Planned entry quantity: {adjusted_qty} | Current position: {current_pos}")
                main_logger.info(Fore.GREEN + "="*80 + "\n")

                signal_logger.info(f"[Long Signal Triggered] Trend Reversal: {prev_trend}→{current_trend} Close: {current_price:.2f} Planned Qty: {adjusted_qty}")

                # Close opposite short position
                if current_pos == 'short':
                    main_logger.info(Fore.GREEN + f"🔄 [Close Short] Current short position {current_pos_amt}")
                    close_order = place_market_order(SYMBOL, Client.SIDE_BUY, current_pos_amt)
                    if close_order:
                        signal_logger.info(f"[Close Short Done] Qty: {current_pos_amt} Close Price: {current_price:.2f}")
                        # Re-get position status after closing
                        current_pos, current_pos_amt, _ = get_position(SYMBOL)

                # Open new long position (force entry if signal triggered and quantity valid)
                if adjusted_qty > 0:
                    main_logger.info(Fore.GREEN + f"🚀 [Open Long] Buy {adjusted_qty} {SYMBOL}")
                    open_order = place_market_order(SYMBOL, Client.SIDE_BUY, adjusted_qty)
                    if open_order:
                        # Entry success, initialize trading state
                        new_pos_dir, new_pos_size, new_entry_price = get_position(SYMBOL)
                        trade_state.init_new_position(new_pos_dir, new_pos_size, new_entry_price, current_trend)
                        signal_logger.info(f"[Long Entry Done] Qty: {adjusted_qty} Entry Price: {current_price:.2f}")
                    else:
                        main_logger.error(Fore.RED + "❌ Long entry failed, check API permissions/balance")
                else:
                    main_logger.error(Fore.RED + f"❌ Calculated entry quantity {adjusted_qty} is invalid, cannot open position")

            # Short entry execution
            elif signal_open_short:
                main_logger.info(Fore.RED + "\n" + "="*80)
                main_logger.info(Fore.RED + "🔴 🔴 🔴 [High Probability Short Signal Triggered] 🔴 🔴 🔴")
                main_logger.info(Fore.RED + f"Trigger time: {kline_time} | Close price: {current_price:.2f}")
                main_logger.info(Fore.RED + f"Trend reversal: {prev_trend} → {current_trend}")
                main_logger.info(Fore.RED + f"Planned entry quantity: {adjusted_qty} | Current position: {current_pos}")
                main_logger.info(Fore.RED + "="*80 + "\n")

                signal_logger.info(f"[Short Signal Triggered] Trend Reversal: {prev_trend}→{current_trend} Close: {current_price:.2f} Planned Qty: {adjusted_qty}")

                # Close opposite long position
                if current_pos == 'long':
                    main_logger.info(Fore.RED + f"🔄 [Close Long] Current long position {current_pos_amt}")
                    close_order = place_market_order(SYMBOL, Client.SIDE_SELL, current_pos_amt)
                    if close_order:
                        signal_logger.info(f"[Close Long Done] Qty: {current_pos_amt} Close Price: {current_price:.2f}")
                        # Re-get position status after closing
                        current_pos, current_pos_amt, _ = get_position(SYMBOL)

                # Open new short position (force entry if signal triggered and quantity valid)
                if adjusted_qty > 0:
                    main_logger.info(Fore.RED + f"🚀 [Open Short] Sell {adjusted_qty} {SYMBOL}")
                    open_order = place_market_order(SYMBOL, Client.SIDE_SELL, adjusted_qty)
                    if open_order:
                        # Entry success, initialize trading state
                        new_pos_dir, new_pos_size, new_entry_price = get_position(SYMBOL)
                        trade_state.init_new_position(new_pos_dir, new_pos_size, new_entry_price, current_trend)
                        signal_logger.info(f"[Short Entry Done] Qty: {adjusted_qty} Entry Price: {current_price:.2f}")
                    else:
                        main_logger.error(Fore.RED + "❌ Short entry failed, check API permissions/balance")
                else:
                    main_logger.error(Fore.RED + f"❌ Calculated entry quantity {adjusted_qty} is invalid, cannot open position")

            # No signal log
            else:
                main_logger.info(Fore.CYAN + f"💤 [No Open/Close Signal] Current Position: {current_pos} {current_pos_amt if current_pos != 'none' else ''}")

            main_logger.info(Fore.CYAN + "="*60 + "\n")
            time.sleep(60)

        except Exception as e:
            main_logger.error(Fore.RED + f"❌ Strategy main loop error: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    try:
        run_strategy()
    except KeyboardInterrupt:
        main_logger.info(Fore.CYAN + "👋 Strategy manually stopped")