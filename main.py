import os
import sys
import time
import json
import logging
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from dotenv import load_dotenv
from datetime import datetime
from colorama import init, Fore, Style
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Tuple, List, Any

sys.stdout.reconfigure(encoding='utf-8')

# Initialize color output
init(autoreset=True)

# ------------------------------ Logging System ------------------------------
def setup_logger():
    main_logger = logging.getLogger('Gainz_Main')
    main_logger.setLevel(logging.DEBUG)
    main_logger.propagate = False
    signal_logger = logging.getLogger('Gainz_Signal')
    signal_logger.setLevel(logging.INFO)
    signal_logger.propagate = False

    if not main_logger.handlers:
        file_handler = logging.FileHandler(f'gainz_full_log_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
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
        signal_file_handler = logging.FileHandler(f'gainz_signal_log_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
        signal_file_handler.setLevel(logging.INFO)
        signal_formatter = logging.Formatter('%(asctime)s | %(message)s')
        signal_file_handler.setFormatter(signal_formatter)
        signal_logger.addHandler(signal_file_handler)

    return main_logger, signal_logger

main_logger, signal_logger = setup_logger()

# ------------------------------ Configuration ------------------------------
load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Trading pairs
SPOT_SYMBOL = "ETHUSDT"          # Symbol for fetching klines (signal source)
FUTURES_SYMBOL = "ETHUSDC"       # Actual trading symbol on futures

# Timeframe for main signal calculation
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
LOOKBACK = 600                    # Number of klines to fetch

# ========== GainzAlgo Indicator Parameters ==========
# Pivot and momentum
GA_PIVOT_LENGTH = 5               # length for pivot highs/lows
GA_MOMENTUM_THRESHOLD_BASE = 0.01 # base momentum threshold (%)
GA_PRE_MOMENTUM_FACTOR_BASE = 0.5 # factor for get-ready signals (not used)

# TP/SL in points
GA_TP_POINTS = 10
GA_SL_POINTS = 10
GA_MIN_SIGNAL_DISTANCE = 5        # minimum bars between signals

# Filters (enable/disable)
GA_USE_MOMENTUM_FILTER = True
GA_USE_TREND_FILTER = False       # set to True to enable multi-timeframe trend filter (requires extra kline requests)
GA_HIGHER_TF_CHOICE = "5M"        # only used if trend filter enabled
GA_USE_LOWER_TF_FILTER = False
GA_LOWER_TF_CHOICE = "5M"
GA_USE_VOLUME_FILTER = False
GA_USE_BREAKOUT_FILTER = False
GA_SHOW_GET_READY = False         # ignored

# Repeated signal restriction
GA_RESTRICT_REPEATED_SIGNALS = True
GA_RESTRICT_TREND_TF_CHOICE = "5M"  # timeframe used to check trend for repeated signals

# Volume filter parameters
GA_VOLUME_LONG_PERIOD = 50
GA_VOLUME_SHORT_PERIOD = 5

# Breakout filter parameters
GA_BREAKOUT_PERIOD = 5

# ========== Risk Management ==========
LEVERAGE = 20
MARGIN_TYPE = "ISOLATED"
RISK_PERCENTAGE = 50               # % of USDC balance to risk per entry

# State persistence file
STATE_FILE = "gainz_state.json"

# Binance Client Initialization
client = Client(API_KEY, API_SECRET, testnet=False, requests_params={'timeout': 30})
main_logger.info(Fore.CYAN + "✅ Binance live trading client initialized (timeout=30s)")
main_logger.info(Fore.CYAN + f"📊 Signal source: {SPOT_SYMBOL} {INTERVAL} | Trading on: {FUTURES_SYMBOL}")
main_logger.info(Fore.CYAN + f"🎯 Strategy: GainzAlgo Smart Money Structure (Full Implementation with Exchange TP/SL)")

# ------------------------------ State Management ------------------------------
@dataclass
class SideState:
    position_size: float = 0.0
    entry_price: float = 0.0
    # The following fields are not critical for operation after TP/SL are placed on exchange,
    # but kept for logging and potential future use.
    signal_bar_low: float = 0.0
    signal_bar_high: float = 0.0
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 0.0

    def reset(self):
        self.position_size = 0.0
        self.entry_price = 0.0
        self.signal_bar_low = 0.0
        self.signal_bar_high = 0.0
        self.highest_since_entry = 0.0
        self.lowest_since_entry = 0.0

    def init_new_position(self, pos_size: float, entry_price: float, bar_low: float, bar_high: float):
        self.reset()
        self.position_size = pos_size
        self.entry_price = entry_price
        self.signal_bar_low = bar_low
        self.signal_bar_high = bar_high
        self.highest_since_entry = entry_price
        self.lowest_since_entry = entry_price

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

# Global signal restriction state (persisted)
@dataclass
class SignalHistory:
    last_signal_bar: int = -GA_MIN_SIGNAL_DISTANCE - 1
    last_signal_direction: str = "Neutral"   # "Buy", "Sell", "Neutral"
    last_signal_trend: int = 0                # trend value from restrict TF at last signal

signal_history = SignalHistory()

# ------------------------------ State Persistence ------------------------------
def save_state():
    """Save signal history and trade state to file."""
    state = {
        'signal_history': {
            'last_signal_bar': signal_history.last_signal_bar,
            'last_signal_direction': signal_history.last_signal_direction,
            'last_signal_trend': signal_history.last_signal_trend
        },
        'trade_state': {
            'long': asdict(trade_state.long_state),
            'short': asdict(trade_state.short_state)
        }
    }
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        main_logger.debug("✅ State saved to file")
    except Exception as e:
        main_logger.error(f"❌ Failed to save state: {e}")

def load_state():
    """Load signal history and trade state from file."""
    global signal_history, trade_state
    if not os.path.exists(STATE_FILE):
        main_logger.info("ℹ️ No state file found, starting fresh")
        return
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        # Load signal history
        sh = state.get('signal_history', {})
        signal_history.last_signal_bar = sh.get('last_signal_bar', -GA_MIN_SIGNAL_DISTANCE-1)
        signal_history.last_signal_direction = sh.get('last_signal_direction', 'Neutral')
        signal_history.last_signal_trend = sh.get('last_signal_trend', 0)

        # Load trade state (only for logging, will be overridden by exchange later)
        ts = state.get('trade_state', {})
        long_dict = ts.get('long', {})
        trade_state.long_state = SideState(**long_dict)
        short_dict = ts.get('short', {})
        trade_state.short_state = SideState(**short_dict)

        main_logger.info("✅ State loaded from file")
    except Exception as e:
        main_logger.error(f"❌ Failed to load state: {e}")

# ------------------------------ Helper Functions ------------------------------
def get_tick_size(symbol: str) -> float:
    """Get minimum price tick size for the symbol."""
    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                for f in s['filters']:
                    if f['filterType'] == 'PRICE_FILTER':
                        return float(f['tickSize'])
        return 0.01
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to get tick size: {e}")
        return 0.01

def get_position(symbol: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Get current long and short positions."""
    long_info = {'size': 0.0, 'entry_price': 0.0}
    short_info = {'size': 0.0, 'entry_price': 0.0}
    try:
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            if pos['symbol'] == symbol:
                side = pos['positionSide']
                amt = float(pos['positionAmt'])
                entry = float(pos['entryPrice'])
                if side == 'LONG' and amt > 0:
                    long_info['size'] = amt
                    long_info['entry_price'] = entry
                elif side == 'SHORT' and amt > 0:
                    short_info['size'] = amt
                    short_info['entry_price'] = entry
        return long_info, short_info
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to get position: {e}")
        return long_info, short_info

def get_usdc_balance() -> float:
    """Get available USDC balance in futures account."""
    try:
        balance = client.futures_account_balance()
        for asset in balance:
            if asset['asset'] == 'USDC':
                return float(asset['availableBalance'])
        return 0.0
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to get USDC balance: {e}")
        return 0.0

def get_symbol_precision(symbol: str) -> tuple[int, int]:
    """Get price and quantity precision for the symbol."""
    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                return int(s['pricePrecision']), int(s['quantityPrecision'])
        return 2, 3
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to get precision: {e}")
        return 2, 3

def calculate_position_size(symbol: str, usdc_balance: float, risk_pct: float, leverage: int, current_price: float) -> float:
    """Calculate position size based on risk % and leverage."""
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
        # Find minQty filter
        min_qty = 0.0
        for f in symbol_info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                min_qty = float(f['minQty'])
                break
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to get symbol precision: {e}")
        return 0.0

    risk_amount = usdc_balance * (risk_pct / 100)
    notional_value = risk_amount * leverage
    position_size = notional_value / current_price
    adjusted_size = round(position_size, qty_precision)
    if adjusted_size < min_qty:
        adjusted_size = min_qty
    if adjusted_size <= 0:
        return min_qty
    return adjusted_size

def place_market_order(symbol: str, side: str, quantity: float, position_side: str) -> Optional[dict]:
    """Place a market order."""
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

def cancel_all_orders(symbol: str):
    """Cancel all open orders for the symbol."""
    try:
        result = client.futures_cancel_all_open_orders(symbol=symbol)
        main_logger.info(Fore.YELLOW + f"✅ Cancelled all open orders for {symbol}")
        return result
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to cancel orders: {e}")
        return None

def place_tp_sl_orders(symbol: str, position_side: str, quantity: float,
                       tp_price: float, sl_price: float) -> Tuple[bool, Optional[dict], Optional[dict]]:
    """
    Place take-profit and stop-loss orders on the exchange.
    Returns (success, tp_order, sl_order)
    """
    price_precision, qty_precision = get_symbol_precision(symbol)
    quantity_rounded = round(quantity, qty_precision)

    # Determine order side
    if position_side == 'LONG':
        order_side = Client.SIDE_SELL
    else:  # SHORT
        order_side = Client.SIDE_BUY

    tp_order = None
    sl_order = None
    success = True

    # Place take-profit market order
    try:
        tp_order = client.futures_create_order(
            symbol=symbol,
            side=order_side,
            type='TAKE_PROFIT_MARKET',
            quantity=quantity_rounded,
            stopPrice=round(tp_price, price_precision),
            positionSide=position_side,
            workingType='MARK_PRICE'   # Use mark price to avoid manipulation
        )
        main_logger.info(Fore.GREEN + f"✅ TP order placed for {position_side} at {tp_price:.2f}")
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to place TP order: {e}")
        success = False

    # Place stop-loss market order
    try:
        sl_order = client.futures_create_order(
            symbol=symbol,
            side=order_side,
            type='STOP_MARKET',
            quantity=quantity_rounded,
            stopPrice=round(sl_price, price_precision),
            positionSide=position_side,
            workingType='MARK_PRICE'
        )
        main_logger.info(Fore.GREEN + f"✅ SL order placed for {position_side} at {sl_price:.2f}")
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to place SL order: {e}")
        success = False

    return success, tp_order, sl_order

def restore_trade_state_from_exchange():
    """Restore trade state from actual positions on exchange."""
    long_info, short_info = get_position(FUTURES_SYMBOL)
    # Update long state
    if long_info['size'] > 0:
        trade_state.long_state.position_size = long_info['size']
        trade_state.long_state.entry_price = long_info['entry_price']
        trade_state.long_state.highest_since_entry = long_info['entry_price']
        trade_state.long_state.lowest_since_entry = long_info['entry_price']
        main_logger.info(Fore.GREEN + f"🔄 Restored long state | Size:{long_info['size']} | Entry:{long_info['entry_price']:.2f}")
    else:
        trade_state.long_state.reset()
    # Update short state
    if short_info['size'] > 0:
        trade_state.short_state.position_size = short_info['size']
        trade_state.short_state.entry_price = short_info['entry_price']
        trade_state.short_state.highest_since_entry = short_info['entry_price']
        trade_state.short_state.lowest_since_entry = short_info['entry_price']
        main_logger.info(Fore.GREEN + f"🔄 Restored short state | Size:{short_info['size']} | Entry:{short_info['entry_price']:.2f}")
    else:
        trade_state.short_state.reset()
    # Log any existing orders (optional)
    try:
        open_orders = client.futures_get_open_orders(symbol=FUTURES_SYMBOL)
        main_logger.info(Fore.CYAN + f"📋 Found {len(open_orders)} open orders")
    except Exception as e:
        main_logger.error(f"Failed to get open orders: {e}")

# ------------------------------ Technical Indicator Functions ------------------------------
def calculate_atr_rma(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR using RMA (exponential moving average with alpha=1/period)."""
    high = data['high']
    low = data['low']
    close = data['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def pivot_high(high: pd.Series, length: int) -> pd.Series:
    """Return boolean series where high is a pivot high (highest in left+right length bars)."""
    pivots = pd.Series(False, index=high.index)
    for i in range(length, len(high)-length):
        if high.iloc[i] == high.iloc[i-length:i+length+1].max():
            pivots.iloc[i] = True
    return pivots

def pivot_low(low: pd.Series, length: int) -> pd.Series:
    """Return boolean series where low is a pivot low."""
    pivots = pd.Series(False, index=low.index)
    for i in range(length, len(low)-length):
        if low.iloc[i] == low.iloc[i-length:i+length+1].min():
            pivots.iloc[i] = True
    return pivots

def compute_multi_tf_trend(symbol: str, tf_choice: str, current_close: float) -> int:
    """
    Compute trend for a given timeframe based on EMA20 and VWAP.
    Returns: 1 (bullish), -1 (bearish), 0 (neutral/undefined)
    """
    tf_map = {
        "1M": Client.KLINE_INTERVAL_1MINUTE,
        "5M": Client.KLINE_INTERVAL_5MINUTE,
        "15M": Client.KLINE_INTERVAL_15MINUTE,
        "30M": Client.KLINE_INTERVAL_30MINUTE,
        "1H": Client.KLINE_INTERVAL_1HOUR,
        "4H": Client.KLINE_INTERVAL_4HOUR,
        "D": Client.KLINE_INTERVAL_1DAY
    }
    interval = tf_map.get(tf_choice)
    if not interval:
        return 0

    try:
        # Fetch enough klines to compute EMA20 (at least 20)
        limit = 50
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if len(klines) < 20:
            return 0

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_vol', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        close_series = df['close']
        # EMA20
        ema20 = close_series.ewm(span=20, adjust=False).mean().iloc[-1]
        # VWAP (using typical price = (high+low+close)/3)
        typical_price = (pd.to_numeric(df['high']) + pd.to_numeric(df['low']) + pd.to_numeric(df['close'])) / 3
        volume = pd.to_numeric(df['volume'])
        vwap = (typical_price * volume).sum() / volume.sum() if volume.sum() != 0 else close_series.iloc[-1]

        if current_close > ema20 and current_close > vwap:
            return 1
        elif current_close < ema20 and current_close < vwap:
            return -1
        else:
            return 0
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Failed to compute multi-tf trend for {tf_choice}: {e}")
        return 0

# ------------------------------ GainzAlgo Signal Detection ------------------------------
def check_gainz_signal(data: pd.DataFrame,
                       pivot_len: int,
                       momo_base: float,
                       tp_points: int,
                       sl_points: int,
                       use_momo: bool,
                       use_volume: bool,
                       use_breakout: bool,
                       vol_long: int,
                       vol_short: int,
                       breakout_period: int,
                       tick_size: float,
                       use_trend_filter: bool,
                       higher_tf: str,
                       use_lower_tf_filter: bool,
                       lower_tf: str,
                       restrict_repeated: bool,
                       restrict_tf: str,
                       current_tick_close: float,
                       last_signal_dir: str,
                       last_signal_trend_val: int) -> Tuple[bool, bool, float, float, float, float, bool]:
    """
    Main signal detection function.
    Returns: (buy_signal, sell_signal, buy_sl, buy_tp, sell_sl, sell_tp, signal_allowed_by_restrict)
    """
    if len(data) < pivot_len * 2 + 2:
        return False, False, np.nan, np.nan, np.nan, np.nan, False

    df = data.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    volume = df['volume']

    # --- Dynamic momentum threshold based on ATR ---
    atr = calculate_atr_rma(df, 14)
    volatility_factor = atr / close
    momentum_threshold = momo_base * (1 + volatility_factor * 2)
    price_change = ((close - close.shift(1)) / close.shift(1)) * 100

    # --- Volume condition ---
    vol_avg_long = volume.rolling(window=vol_long).mean()
    vol_avg_short = volume.rolling(window=vol_short).mean()
    vol_condition = (volume > vol_avg_long) & (vol_avg_short.diff() > 0)

    # --- Breakout condition ---
    highest_breakout = high.rolling(window=breakout_period).max().shift(1)
    lowest_breakout = low.rolling(window=breakout_period).min().shift(1)

    # --- Pivot points and structure ---
    pivot_highs = pivot_high(high, pivot_len)
    pivot_lows = pivot_low(low, pivot_len)

    # Create series for last pivot high/low values (forward filled)
    df['pivot_high_val'] = np.where(pivot_highs, high, np.nan)
    df['pivot_low_val'] = np.where(pivot_lows, low, np.nan)
    df['last_high'] = df['pivot_high_val'].ffill()
    df['last_low'] = df['pivot_low_val'].ffill()
    df['last_high_prev'] = df['last_high'].shift(1)
    df['last_low_prev'] = df['last_low'].shift(1)

    # CHoCH and BOS conditions
    choch_sell = (low < df['last_high']) & (low.shift(1) >= df['last_high'].shift(1)) & (close < open_)
    choch_buy = (high > df['last_low']) & (high.shift(1) <= df['last_low'].shift(1)) & (close > open_)
    bos_sell = (low < df['last_low_prev']) & (close < open_)
    bos_buy = (high > df['last_high_prev']) & (close > open_)

    struct_sell = choch_sell | bos_sell
    struct_buy = choch_buy | bos_buy

    # --- Momentum early signal ---
    momo_sell = price_change < -momentum_threshold if use_momo else True
    momo_buy = price_change > momentum_threshold if use_momo else True

    # --- Multi-timeframe trend filters (if enabled) ---
    if use_trend_filter or use_lower_tf_filter or restrict_repeated:
        # For simplicity, compute only restrict TF trend for repeated signal restriction.
        restrict_trend = compute_multi_tf_trend(SPOT_SYMBOL, restrict_tf, current_tick_close) if restrict_repeated else 0
    else:
        restrict_trend = 0

    # Higher TF trend filter
    if use_trend_filter:
        higher_trend = compute_multi_tf_trend(SPOT_SYMBOL, higher_tf, current_tick_close)
        trend_ok_buy = (higher_trend == 1)
        trend_ok_sell = (higher_trend == -1)
    else:
        trend_ok_buy = True
        trend_ok_sell = True

    # Lower TF filter
    if use_lower_tf_filter:
        lower_trend = compute_multi_tf_trend(SPOT_SYMBOL, lower_tf, current_tick_close)
        lower_ok_buy = (lower_trend != -1) and (lower_trend != 0)
        lower_ok_sell = (lower_trend != 1) and (lower_trend != 0)
    else:
        lower_ok_buy = True
        lower_ok_sell = True

    # --- Repeated signal restriction ---
    signal_allowed = True
    if restrict_repeated:
        if last_signal_dir == "Buy":
            signal_allowed = (restrict_trend != 1)
        elif last_signal_dir == "Sell":
            signal_allowed = (restrict_trend != -1)
        # else last_signal_dir == "Neutral" -> allowed

    # --- Combine all filters for the latest bar ---
    sell_condition = struct_sell & momo_sell & trend_ok_sell & lower_ok_sell
    buy_condition = struct_buy & momo_buy & trend_ok_buy & lower_ok_buy

    if use_volume:
        sell_condition &= vol_condition
        buy_condition &= vol_condition
    if use_breakout:
        sell_condition &= (close < lowest_breakout)
        buy_condition &= (close > highest_breakout)

    # Take last bar's signal
    sell_signal = sell_condition.iloc[-1] if not sell_condition.empty else False
    buy_signal = buy_condition.iloc[-1] if not buy_condition.empty else False

    # Avoid conflicting signals
    if buy_signal and sell_signal:
        buy_signal = sell_signal = False

    # Calculate SL/TP prices based on signal bar's low/high
    current_bar = df.iloc[-1]
    bar_low = current_bar['low']
    bar_high = current_bar['high']

    buy_sl = bar_low - sl_points * tick_size if buy_signal else np.nan
    buy_tp = bar_high + tp_points * tick_size if buy_signal else np.nan
    sell_sl = bar_high + sl_points * tick_size if sell_signal else np.nan
    sell_tp = bar_low - tp_points * tick_size if sell_signal else np.nan

    return buy_signal, sell_signal, buy_sl, buy_tp, sell_sl, sell_tp, signal_allowed

# ------------------------------ Main Strategy Loop ------------------------------
def run_strategy():
    global signal_history

    main_logger.info(Fore.CYAN + "="*80)
    main_logger.info(Fore.CYAN + "🚀 GainzAlgo Smart Money Structure Strategy (Full Implementation with Exchange TP/SL)")
    main_logger.info(Fore.CYAN + f"📊 Signal source: {SPOT_SYMBOL} {INTERVAL} | Trading on: {FUTURES_SYMBOL}")
    main_logger.info(Fore.CYAN + f"⚙️  Pivot Len={GA_PIVOT_LENGTH}, Momo Base={GA_MOMENTUM_THRESHOLD_BASE}%, TP={GA_TP_POINTS}, SL={GA_SL_POINTS}")
    main_logger.info(Fore.CYAN + f"🔧 Filters: Momo={GA_USE_MOMENTUM_FILTER}, Trend={GA_USE_TREND_FILTER}, LowerTF={GA_USE_LOWER_TF_FILTER}, Vol={GA_USE_VOLUME_FILTER}, Breakout={GA_USE_BREAKOUT_FILTER}")
    main_logger.info(Fore.CYAN + f"💰 Risk: Leverage={LEVERAGE}x, EntryRisk={RISK_PERCENTAGE}%")
    main_logger.info(Fore.CYAN + "="*80)

    # Setup hedge mode and leverage
    try:
        client.futures_change_position_mode(dualSidePosition=True)
        main_logger.info(Fore.GREEN + "✅ Hedge mode enabled")
    except BinanceAPIException as e:
        if "No need to change position mode" in str(e):
            main_logger.info(Fore.CYAN + "ℹ️ Already in hedge mode")
        else:
            main_logger.error(Fore.RED + f"❌ Failed to set hedge mode: {e}")
            return

    try:
        client.futures_change_margin_type(symbol=FUTURES_SYMBOL, marginType=MARGIN_TYPE)
    except BinanceAPIException as e:
        if "No need to change margin type" not in str(e):
            main_logger.warning(Fore.YELLOW + f"⚠️ Margin type change: {e}")

    try:
        client.futures_change_leverage(symbol=FUTURES_SYMBOL, leverage=LEVERAGE)
        main_logger.info(Fore.GREEN + f"✅ Leverage set to {LEVERAGE}x")
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ Leverage setup failed: {e}")
        return

    # Load persisted state and sync with exchange
    load_state()
    restore_trade_state_from_exchange()

    tick_size = get_tick_size(FUTURES_SYMBOL)

    last_kline_time = 0
    kline_update_retries = 0
    MAX_KLINE_RETRIES = 3
    RETRY_INTERVAL = 5

    while True:
        try:
            # 1. Fetch spot klines
            klines = None
            for retry in range(MAX_KLINE_RETRIES):
                try:
                    klines = client.get_klines(symbol=SPOT_SYMBOL, interval=INTERVAL, limit=LOOKBACK)
                    if klines and len(klines) > 0:
                        break
                    time.sleep(RETRY_INTERVAL)
                except Exception as e:
                    main_logger.error(Fore.RED + f"❌ Kline fetch error (retry {retry+1}): {e}")
                    time.sleep(RETRY_INTERVAL)

            if not klines:
                main_logger.error("❌ No kline data, skipping cycle")
                time.sleep(30)
                continue

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_vol', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 2. Check if new kline closed
            current_kline_time = int(df['timestamp'].iloc[-1])
            if current_kline_time == last_kline_time:
                kline_update_retries += 1
                if kline_update_retries >= MAX_KLINE_RETRIES:
                    last_kline_time = 0
                    kline_update_retries = 0
                else:
                    time.sleep(30)
                    continue
            else:
                kline_update_retries = 0

            last_kline_time = current_kline_time
            current_price = df['close'].iloc[-1]

            # 3. Detect GainzAlgo signals
            buy_signal, sell_signal, buy_sl, buy_tp, sell_sl, sell_tp, signal_allowed = check_gainz_signal(
                df,
                GA_PIVOT_LENGTH,
                GA_MOMENTUM_THRESHOLD_BASE,
                GA_TP_POINTS,
                GA_SL_POINTS,
                GA_USE_MOMENTUM_FILTER,
                GA_USE_VOLUME_FILTER,
                GA_USE_BREAKOUT_FILTER,
                GA_VOLUME_LONG_PERIOD,
                GA_VOLUME_SHORT_PERIOD,
                GA_BREAKOUT_PERIOD,
                tick_size,
                GA_USE_TREND_FILTER,
                GA_HIGHER_TF_CHOICE,
                GA_USE_LOWER_TF_FILTER,
                GA_LOWER_TF_CHOICE,
                GA_RESTRICT_REPEATED_SIGNALS,
                GA_RESTRICT_TREND_TF_CHOICE,
                current_price,
                signal_history.last_signal_direction,
                signal_history.last_signal_trend
            )

            # 4. Get current positions (sync with exchange)
            long_info, short_info = get_position(FUTURES_SYMBOL)
            # Update local state if positions changed due to external TP/SL hits
            if long_info['size'] == 0 and trade_state.long_state.position_size > 0:
                main_logger.info(Fore.YELLOW + "📉 Long position closed (likely by TP/SL)")
                trade_state.long_state.reset()
            if short_info['size'] == 0 and trade_state.short_state.position_size > 0:
                main_logger.info(Fore.YELLOW + "📈 Short position closed (likely by TP/SL)")
                trade_state.short_state.reset()

            # 5. Logging
            main_logger.info(Fore.CYAN + "="*60)
            main_logger.info(Fore.CYAN + f"🕐 Time: {pd.to_datetime(current_kline_time, unit='ms')} | Price: {current_price:.2f}")
            main_logger.info(Fore.CYAN + f"📊 Signals: Buy={buy_signal} | Sell={sell_signal} | Allowed={signal_allowed}")
            main_logger.info(Fore.CYAN + f"📈 Long: {long_info['size']} @ {long_info['entry_price']:.2f} | Local: {trade_state.long_state.position_size}")
            main_logger.info(Fore.CYAN + f"📉 Short: {short_info['size']} @ {short_info['entry_price']:.2f} | Local: {trade_state.short_state.position_size}")

            # 6. Reverse signal: close opposite position if new signal appears (and cancel its TP/SL orders)
            if sell_signal and long_info['size'] > 0:
                main_logger.info(Fore.YELLOW + "🔄 Sell signal detected, closing long and cancelling orders")
                cancel_all_orders(FUTURES_SYMBOL)
                order = place_market_order(FUTURES_SYMBOL, Client.SIDE_SELL, long_info['size'], 'LONG')
                if order:
                    signal_logger.info(f"[Reverse Close Long] Qty: {long_info['size']} @ {current_price:.2f}")
                    trade_state.long_state.reset()
            if buy_signal and short_info['size'] > 0:
                main_logger.info(Fore.YELLOW + "🔄 Buy signal detected, closing short and cancelling orders")
                cancel_all_orders(FUTURES_SYMBOL)
                order = place_market_order(FUTURES_SYMBOL, Client.SIDE_BUY, short_info['size'], 'SHORT')
                if order:
                    signal_logger.info(f"[Reverse Close Short] Qty: {short_info['size']} @ {current_price:.2f}")
                    trade_state.short_state.reset()

            # 7. Check min distance and open new position
            bars_since_last = len(df) - 1 - signal_history.last_signal_bar if signal_history.last_signal_bar >= 0 else GA_MIN_SIGNAL_DISTANCE + 1
            if bars_since_last >= GA_MIN_SIGNAL_DISTANCE:
                usdc_balance = get_usdc_balance()
                adjusted_qty = calculate_position_size(FUTURES_SYMBOL, usdc_balance, RISK_PERCENTAGE, LEVERAGE, current_price)

                if buy_signal and adjusted_qty > 0 and long_info['size'] == 0 and signal_allowed:
                    # Cancel any lingering orders before opening new position
                    cancel_all_orders(FUTURES_SYMBOL)
                    # Open long
                    main_logger.info(Fore.GREEN + f"\n{'='*80}\n🟢 [Long Entry] Buy signal\n{'='*80}")
                    order = place_market_order(FUTURES_SYMBOL, Client.SIDE_BUY, adjusted_qty, 'LONG')
                    if order:
                        # Get actual filled quantity and price
                        new_long, _ = get_position(FUTURES_SYMBOL)
                        trade_state.long_state.init_new_position(
                            new_long['size'], new_long['entry_price'],
                            df['low'].iloc[-1], df['high'].iloc[-1]
                        )
                        # Place TP/SL orders on exchange
                        tp_sl_ok, _, _ = place_tp_sl_orders(FUTURES_SYMBOL, 'LONG', new_long['size'], buy_tp, buy_sl)
                        if tp_sl_ok:
                            signal_logger.info(f"[Long Entry] Qty: {adjusted_qty} @ {current_price:.2f} | TP: {buy_tp:.2f} SL: {buy_sl:.2f}")
                        else:
                            main_logger.error(Fore.RED + "❌ Failed to place TP/SL for long, consider manual check")
                        # Update signal history
                        signal_history.last_signal_bar = len(df) - 1
                        signal_history.last_signal_direction = "Buy"
                        if GA_RESTRICT_REPEATED_SIGNALS:
                            signal_history.last_signal_trend = compute_multi_tf_trend(SPOT_SYMBOL, GA_RESTRICT_TREND_TF_CHOICE, current_price)
                        save_state()

                elif sell_signal and adjusted_qty > 0 and short_info['size'] == 0 and signal_allowed:
                    cancel_all_orders(FUTURES_SYMBOL)
                    # Open short
                    main_logger.info(Fore.RED + f"\n{'='*80}\n🔴 [Short Entry] Sell signal\n{'='*80}")
                    order = place_market_order(FUTURES_SYMBOL, Client.SIDE_SELL, adjusted_qty, 'SHORT')
                    if order:
                        _, new_short = get_position(FUTURES_SYMBOL)
                        trade_state.short_state.init_new_position(
                            new_short['size'], new_short['entry_price'],
                            df['low'].iloc[-1], df['high'].iloc[-1]
                        )
                        tp_sl_ok, _, _ = place_tp_sl_orders(FUTURES_SYMBOL, 'SHORT', new_short['size'], sell_tp, sell_sl)
                        if tp_sl_ok:
                            signal_logger.info(f"[Short Entry] Qty: {adjusted_qty} @ {current_price:.2f} | TP: {sell_tp:.2f} SL: {sell_sl:.2f}")
                        else:
                            main_logger.error(Fore.RED + "❌ Failed to place TP/SL for short")
                        signal_history.last_signal_bar = len(df) - 1
                        signal_history.last_signal_direction = "Sell"
                        if GA_RESTRICT_REPEATED_SIGNALS:
                            signal_history.last_signal_trend = compute_multi_tf_trend(SPOT_SYMBOL, GA_RESTRICT_TREND_TF_CHOICE, current_price)
                        save_state()

            # 8. No signal
            if not buy_signal and not sell_signal:
                main_logger.info(Fore.CYAN + "💤 No signal")

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
        save_state()