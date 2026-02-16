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

# 初始化颜色输出
init(autoreset=True)

# ———————————————— 日志系统（无修改，兼容原有复盘逻辑） ————————————————
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

# ———————————————— 全量配置项（新增状态管理专属参数） ————————————————
load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# 交易基础配置
SYMBOL = "ETHUSDC"
INTERVAL = Client.KLINE_INTERVAL_15MINUTE
LOOKBACK = 600

# L1核心滤波器参数
ATR_PERIOD = 200
ATR_MULT = 1.5
MU = 0.6

# 杠杆与资金管理
LEVERAGE = 20
MARGIN_TYPE = "ISOLATED"
RISK_PERCENTAGE = 50          # 初始开仓使用可用资金比例
ADD_RISK_PCT = 20              # 加仓使用可用资金比例（保守设置为初始的一半）

# 止损配置
STOP_LOSS_PCT = 1.5
ENABLE_STOP_LOSS = False

# 流动性扫盘核心参数（1:1对齐TradingView）
LIQ_SWEEP_LENGTH = 8           # Pivot高低点回溯周期
LIQ_PARTIAL_PROFIT_RATIO = 0.5 # 单次触及流动性区域止盈比例（当前持仓的50%）
BREAKOUT_CONFIRM_BARS = 2      # 突破确认K线数（连续N根收盘在区域外，防假突破）
BREAKOUT_THRESHOLD_PCT = 0.1   # 突破幅度阈值（0.1%，过滤毛刺）

# 状态管理核心配置
MAX_ADD_TIMES = 1               # 单趋势最大加仓次数（防重仓爆仓）
NEW_ZONE_THRESHOLD_PCT = 0.5    # 新区域判定阈值（与上一次操作区域价差≥0.5%，才算新的交易机会）
STATE_RESET_DELAY = 1           # 状态重置延迟（K线确认后再重置，防误判）

# 币安客户端初始化
client = Client(API_KEY, API_SECRET, testnet=False)
main_logger.info(Fore.CYAN + "✅ Binance实盘客户端初始化完成")

# ———————————————— 【核心新增】全生命周期状态管理数据类 ————————————————
@dataclass
class TradeState:
    # 持仓基础状态
    position_dir: str = "none"             # 当前持仓方向：long/short/none
    position_size: float = 0.0              # 当前持仓数量
    entry_price: float = 0.0                # 加权平均开仓价
    initial_entry_price: float = 0.0        # 初始开仓价（趋势首次开仓的价格，不变）
    
    # 流动性操作状态（核心防重复）
    last_operated_zone_price: float = 0.0   # 上一次操作的流动性区域价格（止盈/加仓）
    has_partial_tp_in_zone: bool = False    # 当前区域是否已执行部分止盈
    has_added_in_zone: bool = False          # 当前区域是否已执行加仓
    
    # 加仓管控状态
    total_add_times: int = 0                 # 当前趋势累计加仓次数
    last_add_price: float = 0.0              # 上一次加仓价格
    
    # 趋势锁定状态
    trend_at_open: int = 0                    # 开仓时的L1趋势（1多/-1空，防趋势中途反转误操作）
    is_trend_valid: bool = False              # 当前趋势是否有效（与开仓时一致）

    # 重置状态（平仓/止损/趋势反转时调用）
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
        main_logger.info(Fore.YELLOW + "🔄 交易状态已全量重置")
        signal_logger.info("【状态重置】持仓已清空，所有交易标记重置")

    # 初始化新趋势开仓状态
    def init_new_position(self, pos_dir: str, pos_size: float, entry_price: float, trend: int):
        self.reset()  # 开新仓前先清空上一个趋势的残留状态
        self.position_dir = pos_dir
        self.position_size = pos_size
        self.entry_price = entry_price
        self.initial_entry_price = entry_price
        self.trend_at_open = trend
        self.is_trend_valid = True
        main_logger.info(Fore.GREEN + f"📝 新仓位状态初始化 | 方向:{pos_dir} | 数量:{pos_size} | 开仓价:{entry_price:.2f}")
        signal_logger.info(f"【状态初始化】{pos_dir}仓 | 数量:{pos_size} | 开仓价:{entry_price:.2f} | 趋势:{trend}")

    # 更新持仓状态（止盈/加仓/平仓后调用，同步最新持仓数据）
    def update_position(self, pos_dir: str, pos_size: float, entry_price: float):
        self.position_dir = pos_dir
        self.position_size = pos_size
        self.entry_price = entry_price
        main_logger.debug(Fore.CYAN + f"📊 持仓状态更新 | 方向:{pos_dir} | 数量:{pos_size} | 均价:{entry_price:.2f}")

    # 检查是否为新的流动性区域（核心防重复操作）
    def is_new_liquidity_zone(self, current_zone_price: float, pos_dir: str) -> bool:
        # 首次操作，无历史记录，直接判定为新区域
        if self.last_operated_zone_price == 0:
            return True
        
        # 计算当前区域与上一次操作区域的价差比例
        price_diff_pct = abs(current_zone_price - self.last_operated_zone_price) / self.last_operated_zone_price * 100
        
        # 多头：新阻力位必须高于上一次操作区域，且价差达标
        if pos_dir == "long":
            is_new = (current_zone_price > self.last_operated_zone_price) and (price_diff_pct >= NEW_ZONE_THRESHOLD_PCT)
        # 空头：新支撑位必须低于上一次操作区域，且价差达标
        elif pos_dir == "short":
            is_new = (current_zone_price < self.last_operated_zone_price) and (price_diff_pct >= NEW_ZONE_THRESHOLD_PCT)
        else:
            is_new = False

        if is_new:
            # 新区域重置当前区域的操作标记
            self.has_partial_tp_in_zone = False
            self.has_added_in_zone = False
            main_logger.info(Fore.CYAN + f"🎯 检测到新流动性区域 | 价格:{current_zone_price:.2f} | 价差:{price_diff_pct:.2f}%")
        return is_new

# 全局唯一状态实例（单线程循环，线程安全）
trade_state = TradeState()

# ———————————————— 核心指标计算函数（无修改，1:1对齐TradingView） ————————————————
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

# ———————————————— 流动性区域检测（优化，对齐TradingView枢轴点逻辑） ————————————————
def detect_liquidity_zones(data: pd.DataFrame, lookback_len: int = 8) -> dict:
    """
    1:1对齐TradingView的Pivot高低点识别，输出有效支撑/阻力位
    逻辑：寻找已确认的Pivot High/Low（左右各lookback_len根K线验证，无未来函数）
    """
    df = data.copy()
    # 只使用已收盘的K线，排除当前未完成K线，避免未来函数
    closed_df = df.iloc[:-1].copy()
    nearest_resistance = np.nan
    nearest_support = np.nan

    if len(closed_df) < lookback_len * 2 + 1:
        return {'resistance': nearest_resistance, 'support': nearest_support}

    # 计算Pivot High：当前高点是左右lookback_len根K线的最高点（已确认，无未来函数）
    closed_df['is_pivot_high'] = closed_df['high'] == closed_df['high'].rolling(window=lookback_len*2+1, center=True).max()
    # 计算Pivot Low：当前低点是左右lookback_len根K线的最低点
    closed_df['is_pivot_low'] = closed_df['low'] == closed_df['low'].rolling(window=lookback_len*2+1, center=True).min()

    # 提取有效枢轴点
    pivot_highs = closed_df[closed_df['is_pivot_high']]['high']
    pivot_lows = closed_df[closed_df['is_pivot_low']]['low']

    # 取最近的、且在当前价格之外的有效枢轴点（避免取已经突破的区域）
    current_price = df['close'].iloc[-1]
    if not pivot_highs.empty:
        # 阻力位：取最近的、高于当前价格的枢轴高点
        valid_resistances = pivot_highs[pivot_highs > current_price]
        if not valid_resistances.empty:
            nearest_resistance = valid_resistances.iloc[-1]
    if not pivot_lows.empty:
        # 支撑位：取最近的、低于当前价格的枢轴低点
        valid_supports = pivot_lows[pivot_lows < current_price]
        if not valid_supports.empty:
            nearest_support = valid_supports.iloc[-1]

    return {
        'resistance': nearest_resistance,
        'support': nearest_support
    }

# ———————————————— 【新增】突破有效性确认（防假突破核心） ————————————————
def confirm_breakout(data: pd.DataFrame, zone_price: float, pos_dir: str) -> bool:
    """
    确认突破有效性：连续N根K线收盘在区域外，且达到突破幅度阈值
    :param data: K线数据
    :param zone_price: 流动性区域价格（阻力/支撑）
    :param pos_dir: 持仓方向
    :return: 是否有效突破
    """
    if len(data) < BREAKOUT_CONFIRM_BARS:
        return False
    
    # 取最近N根已收盘的K线
    recent_bars = data.iloc[-(BREAKOUT_CONFIRM_BARS+1):-1]
    
    if pos_dir == "long":
        # 多头突破：连续N根收盘价 > 阻力位 * (1+阈值)
        breakout_level = zone_price * (1 + BREAKOUT_THRESHOLD_PCT / 100)
        all_breakout = all(recent_bars['close'] > breakout_level)
    elif pos_dir == "short":
        # 空头突破：连续N根收盘价 < 支撑位 * (1-阈值)
        breakout_level = zone_price * (1 - BREAKOUT_THRESHOLD_PCT / 100)
        all_breakout = all(recent_bars['close'] < breakout_level)
    else:
        all_breakout = False

    if all_breakout:
        main_logger.info(Fore.BLUE + f"✅ 突破有效确认 | 区域价:{zone_price:.2f} | 突破位:{breakout_level:.2f} | 确认K线数:{BREAKOUT_CONFIRM_BARS}")
    return all_breakout

# ———————————————— 【新增】脚本重启状态自动恢复 ————————————————
def restore_trade_state():
    """脚本启动/重启时，自动从币安获取当前持仓，恢复交易状态，避免重启后状态丢失"""
    pos_dir, pos_size, entry_price = get_position(SYMBOL)
    if pos_dir != "none" and pos_size > 0:
        # 有持仓，恢复状态
        klines = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=LOOKBACK)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume','close_time', 'quote_vol', 'trades', 'taker_buy_base','taker_buy_quote', 'ignore'])
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 计算当前趋势
        df['atr_200'] = calculate_atr(df, period=ATR_PERIOD)
        _, l1_trend = l1_proximal_filter(df['close'], df['atr_200'], ATR_MULT, MU)
        current_trend = l1_trend[-1]

        # 恢复状态
        trade_state.init_new_position(pos_dir, pos_size, entry_price, current_trend)
        main_logger.info(Fore.GREEN + f"🔄 重启状态恢复成功 | 持仓:{pos_dir} {pos_size} | 均价:{entry_price:.2f}")
        signal_logger.info(f"【重启恢复】{pos_dir}仓 | 数量:{pos_size} | 均价:{entry_price:.2f}")
    else:
        # 无持仓，重置状态
        trade_state.reset()
        main_logger.info(Fore.CYAN + "🔄 启动无持仓，状态已初始化")

# ———————————————— 交易辅助函数（优化，适配状态管理） ————————————————
def setup_leverage_and_margin(symbol: str, leverage: int, margin_type: str):
    try:
        client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
        main_logger.info(Fore.CYAN + f"🔧 保证金模式已设置: {'逐仓' if margin_type == 'ISOLATED' else '全仓'}")
    except BinanceAPIException as e:
        if "No need to change margin type" not in str(e):
            main_logger.warning(Fore.YELLOW + f"⚠️ 保证金模式提示: {e}")
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        main_logger.info(Fore.CYAN + f"🔧 杠杆倍数已设置: {leverage}x")
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ 杠杆设置失败: {e}")

def get_usdc_balance() -> float:
    try:
        balance = client.futures_account_balance()
        for asset in balance:
            if asset['asset'] == 'USDC':
                available_balance = float(asset['availableBalance'])
                return available_balance
        main_logger.error(Fore.RED + "❌ 未找到USDC余额信息")
        return 0.0
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ 获取USDC余额失败: {e}")
        return 0.0

def calculate_position_size(symbol: str, usdc_balance: float, risk_pct: float, leverage: int, current_price: float) -> float:
    try:
        info = client.futures_exchange_info()
        for symbol_info in info['symbols']:
            if symbol_info['symbol'] == symbol:
                qty_precision = int(symbol_info['quantityPrecision'])
                min_qty = float(symbol_info['filters'][1]['minQty'])
                break
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ 获取交易对精度失败: {e}")
        return 0.0

    risk_amount = usdc_balance * (risk_pct / 100)
    notional_value = risk_amount * leverage
    position_size = notional_value / current_price
    adjusted_size = round(position_size, qty_precision)
    
    if adjusted_size < min_qty:
        adjusted_size = min_qty
    return adjusted_size

def get_position(symbol: str) -> tuple[str, float, float]:
    try:
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            if pos['symbol'] == symbol:
                amt = float(pos['positionAmt'])
                entry_price = float(pos['entryPrice'])
                if amt > 0:
                    return 'long', amt, entry_price
                elif amt < 0:
                    return 'short', abs(amt), entry_price
        return 'none', 0, 0
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ 获取持仓失败: {e}")
        return 'none', 0, 0

def get_symbol_precision(symbol: str) -> tuple[int, int]:
    try:
        info = client.futures_exchange_info()
        for symbol_info in info['symbols']:
            if symbol_info['symbol'] == symbol:
                return int(symbol_info['pricePrecision']), int(symbol_info['quantityPrecision'])
        return 2, 3
    except Exception as e:
        main_logger.error(Fore.RED + f"❌ 获取精度失败: {e}")
        return 2, 3

def place_market_order(symbol: str, side: str, quantity: float) -> dict:
    try:
        order = client.futures_create_order(
            symbol=symbol, side=side, type=Client.ORDER_TYPE_MARKET, quantity=quantity
        )
        action = "开多" if side == Client.SIDE_BUY else "开空" if side == Client.SIDE_SELL else "平仓"
        main_logger.info(Fore.GREEN + f"✅ 【{action}成功】订单ID: {order['orderId']}, 数量: {quantity}")
        return order
    except (BinanceAPIException, BinanceOrderException) as e:
        main_logger.error(Fore.RED + f"❌ 【下单失败】{e}")
        return None

def check_stop_loss(symbol: str, current_price: float) -> bool:
    pos, pos_amt, entry_price = get_position(symbol)
    if pos == 'none' or not ENABLE_STOP_LOSS:
        return False

    is_stop_triggered = False
    if pos == 'long':
        loss_pct = (entry_price - current_price) / entry_price * 100
        if loss_pct >= STOP_LOSS_PCT:
            warn_msg = f"⚠️ 【多头止损触发】入场价: {entry_price:.2f}, 当前价: {current_price:.2f}, 亏损: {loss_pct:.2f}%"
            main_logger.warning(Fore.YELLOW + warn_msg)
            signal_logger.warning(warn_msg)
            is_stop_triggered = True
    elif pos == 'short':
        loss_pct = (current_price - entry_price) / entry_price * 100
        if loss_pct >= STOP_LOSS_PCT:
            warn_msg = f"⚠️ 【空头止损触发】入场价: {entry_price:.2f}, 当前价: {current_price:.2f}, 亏损: {loss_pct:.2f}%"
            main_logger.warning(Fore.YELLOW + warn_msg)
            signal_logger.warning(warn_msg)
            is_stop_triggered = True
    return is_stop_triggered

# ———————————————— 【完善】流动性策略核心逻辑（全状态管控） ————————————————
def check_partial_take_profit(symbol: str, current_price: float, liq_zones: dict) -> None:
    """
    带状态管控的部分止盈逻辑：
    1. 仅在当前趋势有效时执行
    2. 同一个流动性区域仅执行一次止盈
    3. 新区域自动重置止盈标记
    """
    # 无持仓/趋势无效，直接跳过
    if trade_state.position_dir == "none" or not trade_state.is_trend_valid:
        return
    
    pos_dir = trade_state.position_dir
    pos_size = trade_state.position_size
    qty_precision = get_symbol_precision(symbol)[1]

    # 多头止盈：触及阻力位
    if pos_dir == "long" and not np.isnan(liq_zones['resistance']):
        zone_price = liq_zones['resistance']
        # 检查是否为新区域，更新状态标记
        trade_state.is_new_liquidity_zone(zone_price, pos_dir)
        
        # 触发条件：价格触及阻力位，且当前区域未止盈，持仓量>最小下单量
        min_qty = float(client.futures_exchange_info()['symbols'][0]['filters'][1]['minQty'])
        if (current_price >= zone_price 
            and not trade_state.has_partial_tp_in_zone 
            and pos_size > min_qty):
            
            # 计算止盈数量（当前持仓的50%）
            sell_qty = round(pos_size * LIQ_PARTIAL_PROFIT_RATIO, qty_precision)
            sell_qty = max(sell_qty, min_qty) # 确保不小于最小下单量

            # 执行止盈
            main_logger.info(Fore.MAGENTA + "\n" + "="*80)
            main_logger.info(Fore.MAGENTA + f"🎯 【流动性部分止盈】触及阻力位: {zone_price:.2f}")
            main_logger.info(Fore.MAGENTA + f"执行动作: 平掉 {LIQ_PARTIAL_PROFIT_RATIO*100}% 仓位 | 数量: {sell_qty}")
            main_logger.info(Fore.MAGENTA + "="*80 + "\n")
            
            order = place_market_order(symbol, Client.SIDE_SELL, sell_qty)
            if order:
                # 下单成功，更新状态
                trade_state.has_partial_tp_in_zone = True
                trade_state.last_operated_zone_price = zone_price
                # 同步最新持仓状态
                new_pos_dir, new_pos_size, new_entry_price = get_position(symbol)
                trade_state.update_position(new_pos_dir, new_pos_size, new_entry_price)
                # 记录日志
                signal_logger.info(f"【部分止盈完成】平多 {sell_qty} @ {current_price} | 阻力位: {zone_price} | 剩余持仓: {new_pos_size}")

    # 空头止盈：触及支撑位
    elif pos_dir == "short" and not np.isnan(liq_zones['support']):
        zone_price = liq_zones['support']
        # 检查是否为新区域，更新状态标记
        trade_state.is_new_liquidity_zone(zone_price, pos_dir)
        
        # 触发条件
        min_qty = float(client.futures_exchange_info()['symbols'][0]['filters'][1]['minQty'])
        if (current_price <= zone_price 
            and not trade_state.has_partial_tp_in_zone 
            and pos_size > min_qty):
            
            # 计算止盈数量
            buy_qty = round(pos_size * LIQ_PARTIAL_PROFIT_RATIO, qty_precision)
            buy_qty = max(buy_qty, min_qty)

            # 执行止盈
            main_logger.info(Fore.MAGENTA + "\n" + "="*80)
            main_logger.info(Fore.MAGENTA + f"🎯 【流动性部分止盈】触及支撑位: {zone_price:.2f}")
            main_logger.info(Fore.MAGENTA + f"执行动作: 平掉 {LIQ_PARTIAL_PROFIT_RATIO*100}% 仓位 | 数量: {buy_qty}")
            main_logger.info(Fore.MAGENTA + "="*80 + "\n")
            
            order = place_market_order(symbol, Client.SIDE_BUY, buy_qty)
            if order:
                # 更新状态
                trade_state.has_partial_tp_in_zone = True
                trade_state.last_operated_zone_price = zone_price
                # 同步持仓
                new_pos_dir, new_pos_size, new_entry_price = get_position(symbol)
                trade_state.update_position(new_pos_dir, new_pos_size, new_entry_price)
                # 日志
                signal_logger.info(f"【部分止盈完成】平空 {buy_qty} @ {current_price} | 支撑位: {zone_price} | 剩余持仓: {new_pos_size}")

def check_breakout_and_add(symbol: str, current_price: float, liq_zones: dict, current_trend: int) -> None:
    """
    带状态管控的突破加仓逻辑：
    1. 仅在趋势与开仓时一致时执行
    2. 同一个区域仅加仓一次
    3. 严格限制最大加仓次数
    4. 必须有效突破确认后才执行
    """
    # 无持仓/趋势无效/已达最大加仓次数，直接跳过
    if (trade_state.position_dir == "none" 
        or not trade_state.is_trend_valid 
        or trade_state.total_add_times >= MAX_ADD_TIMES):
        return
    
    pos_dir = trade_state.position_dir
    usdc_balance = get_usdc_balance()
    qty_precision = get_symbol_precision(symbol)[1]

    # 多头加仓：有效突破阻力位，且趋势保持多头
    if pos_dir == "long" and current_trend == 1 and not np.isnan(liq_zones['resistance']):
        zone_price = liq_zones['resistance']
        # 检查是否为新区域
        trade_state.is_new_liquidity_zone(zone_price, pos_dir)
        
        # 触发条件：有效突破、当前区域未加仓、已在该区域完成部分止盈（符合你的逻辑）
        if (confirm_breakout(pd.DataFrame(client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=LOOKBACK), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume','close_time', 'quote_vol', 'trades', 'taker_buy_base','taker_buy_quote', 'ignore']).apply(pd.to_numeric, errors='coerce'), 
                              zone_price, pos_dir)
            and not trade_state.has_added_in_zone
            and trade_state.has_partial_tp_in_zone):
            
            # 计算加仓数量
            add_qty = calculate_position_size(symbol, usdc_balance, ADD_RISK_PCT, LEVERAGE, current_price)
            if add_qty <= 0:
                main_logger.warning(Fore.YELLOW + "⚠️ 加仓数量不足，跳过加仓")
                return

            # 执行加仓
            main_logger.info(Fore.BLUE + "\n" + "="*80)
            main_logger.info(Fore.BLUE + f"🚀 【突破加仓】有效突破阻力位: {zone_price:.2f}")
            main_logger.info(Fore.BLUE + f"趋势确认: L1保持多头 | 加仓次数: {trade_state.total_add_times+1}/{MAX_ADD_TIMES}")
            main_logger.info(Fore.BLUE + f"执行动作: 加多 | 数量: {add_qty}")
            main_logger.info(Fore.BLUE + "="*80 + "\n")
            
            order = place_market_order(symbol, Client.SIDE_BUY, add_qty)
            if order:
                # 更新状态
                trade_state.has_added_in_zone = True
                trade_state.total_add_times += 1
                trade_state.last_add_price = current_price
                trade_state.last_operated_zone_price = zone_price
                # 同步持仓
                new_pos_dir, new_pos_size, new_entry_price = get_position(symbol)
                trade_state.update_position(new_pos_dir, new_pos_size, new_entry_price)
                # 日志
                signal_logger.info(f"【突破加仓完成】加多 {add_qty} @ {current_price} | 突破位: {zone_price} | 累计加仓: {trade_state.total_add_times}次 | 总持仓: {new_pos_size}")

    # 空头加仓：有效跌破支撑位，且趋势保持空头
    elif pos_dir == "short" and current_trend == -1 and not np.isnan(liq_zones['support']):
        zone_price = liq_zones['support']
        # 检查是否为新区域
        trade_state.is_new_liquidity_zone(zone_price, pos_dir)
        
        # 触发条件
        if (confirm_breakout(pd.DataFrame(client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=LOOKBACK), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume','close_time', 'quote_vol', 'trades', 'taker_buy_base','taker_buy_quote', 'ignore']).apply(pd.to_numeric, errors='coerce'), 
                              zone_price, pos_dir)
            and not trade_state.has_added_in_zone
            and trade_state.has_partial_tp_in_zone):
            
            # 计算加仓数量
            add_qty = calculate_position_size(symbol, usdc_balance, ADD_RISK_PCT, LEVERAGE, current_price)
            if add_qty <= 0:
                main_logger.warning(Fore.YELLOW + "⚠️ 加仓数量不足，跳过加仓")
                return

            # 执行加仓
            main_logger.info(Fore.BLUE + "\n" + "="*80)
            main_logger.info(Fore.BLUE + f"🚀 【跌破加仓】有效跌破支撑位: {zone_price:.2f}")
            main_logger.info(Fore.BLUE + f"趋势确认: L1保持空头 | 加仓次数: {trade_state.total_add_times+1}/{MAX_ADD_TIMES}")
            main_logger.info(Fore.BLUE + f"执行动作: 加空 | 数量: {add_qty}")
            main_logger.info(Fore.BLUE + "="*80 + "\n")
            
            order = place_market_order(symbol, Client.SIDE_SELL, add_qty)
            if order:
                # 更新状态
                trade_state.has_added_in_zone = True
                trade_state.total_add_times += 1
                trade_state.last_add_price = current_price
                trade_state.last_operated_zone_price = zone_price
                # 同步持仓
                new_pos_dir, new_pos_size, new_entry_price = get_position(symbol)
                trade_state.update_position(new_pos_dir, new_pos_size, new_entry_price)
                # 日志
                signal_logger.info(f"【跌破加仓完成】加空 {add_qty} @ {current_price} | 跌破位: {zone_price} | 累计加仓: {trade_state.total_add_times}次 | 总持仓: {new_pos_size}")

# ———————————————— 【重构】主策略循环（全流程状态管控） ————————————————
def run_strategy():
    main_logger.info(Fore.CYAN + "="*80)
    main_logger.info(Fore.CYAN + "🚀 L1近端滤波器 + 流动性扫盘 增强策略（带闭环状态管理）启动")
    main_logger.info(Fore.CYAN + f"📊 交易对: {SYMBOL} | K线周期: {INTERVAL}")
    main_logger.info(Fore.CYAN + f"⚙️  核心参数: ATR周期={ATR_PERIOD} | Pivot回溯={LIQ_SWEEP_LENGTH} | 最大加仓={MAX_ADD_TIMES}次")
    main_logger.info(Fore.CYAN + f"💰 资金管理: 杠杆={LEVERAGE}x | 初始开仓比例={RISK_PERCENTAGE}% | 加仓比例={ADD_RISK_PCT}%")
    main_logger.info(Fore.CYAN + "="*80)

    # 启动初始化
    setup_leverage_and_margin(SYMBOL, LEVERAGE, MARGIN_TYPE)
    price_precision, qty_precision = get_symbol_precision(SYMBOL)
    restore_trade_state() # 重启自动恢复状态
    last_kline_time = 0

    while True:
        try:
            # 1. 获取K线数据
            klines = client.futures_klines(
                symbol=SYMBOL,
                interval=INTERVAL,
                limit=LOOKBACK
            )
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_vol', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 2. 新K线校验（仅在K线收盘后执行一次策略逻辑，避免盘中反复触发）
            current_kline_time = df['timestamp'].iloc[-1]
            if current_kline_time == last_kline_time:
                time.sleep(30)
                continue
            last_kline_time = current_kline_time
            kline_time = pd.to_datetime(current_kline_time, unit='ms')
            current_price = df['close'].iloc[-1]

            # 3. 核心指标计算
            df['atr_200'] = calculate_atr(df, period=ATR_PERIOD)
            z, l1_trend = l1_proximal_filter(df['close'], df['atr_200'], ATR_MULT, MU)
            current_trend = l1_trend[-1]
            prev_trend = l1_trend[-2]

            # 4. 流动性区域检测
            liq_zones = detect_liquidity_zones(df, lookback_len=LIQ_SWEEP_LENGTH)
            res_text = f"{liq_zones['resistance']:.2f}" if not np.isnan(liq_zones['resistance']) else "无"
            sup_text = f"{liq_zones['support']:.2f}" if not np.isnan(liq_zones['support']) else "无"

            # 5. 趋势有效性校验（核心：开仓后趋势反转，禁止止盈/加仓）
            if trade_state.position_dir != "none":
                trade_state.is_trend_valid = (current_trend == trade_state.trend_at_open)
                if not trade_state.is_trend_valid:
                    main_logger.warning(Fore.YELLOW + "⚠️ 趋势已反转，锁定当前区域操作，等待平仓信号")

            # 日志输出
            main_logger.info(Fore.CYAN + "="*60)
            main_logger.info(Fore.CYAN + f"🕐 K线收盘时间: {kline_time} | 收盘价: {current_price:.2f}")
            main_logger.info(Fore.CYAN + f"📊 流动性区域: 最近阻力=[{res_text}] | 最近支撑=[{sup_text}]")
            main_logger.info(Fore.CYAN + f"🧭 L1趋势: 当前={current_trend} | 前值={prev_trend} | 开仓时趋势={trade_state.trend_at_open}")
            main_logger.info(Fore.CYAN + f"📈 持仓状态: 方向={trade_state.position_dir} | 数量={trade_state.position_size} | 均价={trade_state.entry_price:.2f}")
            main_logger.info(Fore.CYAN + f"🔢 操作记录: 累计加仓={trade_state.total_add_times}次 | 上一次操作区域={trade_state.last_operated_zone_price:.2f}")

            # 6. 止损逻辑（最高优先级，止损后重置所有状态）
            if check_stop_loss(SYMBOL, current_price):
                pos, pos_amt, _ = get_position(SYMBOL)
                if pos == 'long':
                    place_market_order(SYMBOL, Client.SIDE_SELL, pos_amt)
                    signal_logger.info(f"【止损平仓】平多仓 数量: {pos_amt} 价格: {current_price:.2f}")
                elif pos == 'short':
                    place_market_order(SYMBOL, Client.SIDE_BUY, pos_amt)
                    signal_logger.info(f"【止损平仓】平空仓 数量: {pos_amt} 价格: {current_price:.2f}")
                # 止损后重置状态
                trade_state.reset()
                main_logger.info(Fore.YELLOW + "⏸️ 止损执行完成，暂停本轮后续操作")
                main_logger.info(Fore.CYAN + "="*60 + "\n")
                time.sleep(60)
                continue

            # 7. 流动性策略执行（止盈→加仓，顺序不可变）
            check_partial_take_profit(SYMBOL, current_price, liq_zones)
            check_breakout_and_add(SYMBOL, current_price, liq_zones, current_trend)

            # 8. 趋势反转开平仓信号（核心开仓逻辑）
            signal_open_long = (current_trend == 1) and (prev_trend == -1)
            signal_open_short = (current_trend == -1) and (prev_trend == 1)
            usdc_balance = get_usdc_balance()
            adjusted_qty = calculate_position_size(SYMBOL, usdc_balance, RISK_PERCENTAGE, LEVERAGE, current_price)
            current_pos, current_pos_amt, _ = get_position(SYMBOL)

            # 开多执行
            if signal_open_long:
                main_logger.info(Fore.GREEN + "\n" + "="*80)
                main_logger.info(Fore.GREEN + "🟢 🟢 🟢 【高概率开多信号触发】 🟢 🟢 🟢")
                main_logger.info(Fore.GREEN + f"触发时间: {kline_time} | 收盘价格: {current_price:.2f}")
                main_logger.info(Fore.GREEN + f"趋势反转: {prev_trend} → {current_trend}")
                main_logger.info(Fore.GREEN + "="*80 + "\n")

                signal_logger.info(f"【开多信号触发】趋势反转: {prev_trend}→{current_trend} 收盘价: {current_price:.2f} 计划数量: {adjusted_qty}")

                # 平掉反向空仓
                if current_pos == 'short':
                    main_logger.info(Fore.GREEN + f"🔄 【平空执行】当前持有空仓 {current_pos_amt}")
                    close_order = place_market_order(SYMBOL, Client.SIDE_BUY, current_pos_amt)
                    if close_order:
                        signal_logger.info(f"【平空完成】数量: {current_pos_amt} 平仓价格: {current_price:.2f}")

                # 开新多仓
                if current_pos != 'long' and adjusted_qty > 0:
                    main_logger.info(Fore.GREEN + f"🚀 【开多执行】买入 {adjusted_qty} {SYMBOL}")
                    open_order = place_market_order(SYMBOL, Client.SIDE_BUY, adjusted_qty)
                    if open_order:
                        # 开仓成功，初始化交易状态
                        new_pos_dir, new_pos_size, new_entry_price = get_position(SYMBOL)
                        trade_state.init_new_position(new_pos_dir, new_pos_size, new_entry_price, current_trend)
                        signal_logger.info(f"【开多完成】数量: {adjusted_qty} 开仓价格: {current_price:.2f}")

            # 开空执行
            elif signal_open_short:
                main_logger.info(Fore.RED + "\n" + "="*80)
                main_logger.info(Fore.RED + "🔴 🔴 🔴 【高概率开空信号触发】 🔴 🔴 🔴")
                main_logger.info(Fore.RED + f"触发时间: {kline_time} | 收盘价格: {current_price:.2f}")
                main_logger.info(Fore.RED + f"趋势反转: {prev_trend} → {current_trend}")
                main_logger.info(Fore.RED + "="*80 + "\n")

                signal_logger.info(f"【开空信号触发】趋势反转: {prev_trend}→{current_trend} 收盘价: {current_price:.2f} 计划数量: {adjusted_qty}")

                # 平掉反向多仓
                if current_pos == 'long':
                    main_logger.info(Fore.RED + f"🔄 【平多执行】当前持有多仓 {current_pos_amt}")
                    close_order = place_market_order(SYMBOL, Client.SIDE_SELL, current_pos_amt)
                    if close_order:
                        signal_logger.info(f"【平多完成】数量: {current_pos_amt} 平仓价格: {current_price:.2f}")

                # 开新空仓
                if current_pos != 'short' and adjusted_qty > 0:
                    main_logger.info(Fore.RED + f"🚀 【开空执行】卖出 {adjusted_qty} {SYMBOL}")
                    open_order = place_market_order(SYMBOL, Client.SIDE_SELL, adjusted_qty)
                    if open_order:
                        # 开仓成功，初始化交易状态
                        new_pos_dir, new_pos_size, new_entry_price = get_position(SYMBOL)
                        trade_state.init_new_position(new_pos_dir, new_pos_size, new_entry_price, current_trend)
                        signal_logger.info(f"【开空完成】数量: {adjusted_qty} 开仓价格: {current_price:.2f}")

            # 无信号日志
            else:
                main_logger.info(Fore.CYAN + f"💤 【无开平仓信号】当前持仓: {current_pos} {current_pos_amt if current_pos != 'none' else ''}")

            main_logger.info(Fore.CYAN + "="*60 + "\n")
            time.sleep(60)

        except Exception as e:
            main_logger.error(Fore.RED + f"❌ 策略主循环异常: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    try:
        run_strategy()
    except KeyboardInterrupt:
        main_logger.info(Fore.CYAN + "👋 策略手动停止运行")