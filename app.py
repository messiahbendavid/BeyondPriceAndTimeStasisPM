# -*- coding: utf-8 -*-
"""
STASIS PM — Prediction Markets Server (Jackpot Edition)
Deploy to: stasisPM.beyondpriceandtime.com
Copyright © 2026 Truth Communications LLC. All Rights Reserved.

Requirements:
    pip install dash dash-bootstrap-components pandas numpy websocket-client requests gunicorn
"""

import time
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import copy
import json
import os
import uuid

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, no_update, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import websocket
import ssl
import requests

# ============================================================================
# CONFIG
# ============================================================================

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "PnzhJOXEJO7tSpHr0ct2zjFKi6XO0yGi")
BET_SIZES = [25, 50, 100, 250, 500, 1000]
STARTING_BALANCE = 10000.00

JACKPOT_TIERS = {
    'GRAND_JACKPOT': {'min_levels': 8, 'min_alignment': 100, 'emoji': '💎', 'color': '#ff00ff'},
    'MEGA_JACKPOT': {'min_levels': 6, 'min_alignment': 100, 'emoji': '🎰', 'color': '#ffff00'},
    'SUPER_JACKPOT': {'min_levels': 5, 'min_alignment': 100, 'emoji': '💰', 'color': '#00ffff'},
    'JACKPOT': {'min_levels': 4, 'min_alignment': 100, 'emoji': '🍀', 'color': '#00ff88'},
    'BIG_WIN': {'min_levels': 3, 'min_alignment': 100, 'emoji': '⭐', 'color': '#88ff88'},
}


@dataclass
class Config:
    symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLU", "XLK",
        "XLP", "XLB", "XLV", "XLI", "XLY", "XLC", "XLRE", "KRE",
        "SMH", "XBI", "GDX",
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META',
        'TSLA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'INTC', 'CSCO',
        'QCOM', 'IBM', 'NOW', 'INTU', 'AMAT', 'MU', 'LRCX', 'ADI',
        'KLAC', 'SNPS', 'CDNS', 'MRVL', 'FTNT', 'PANW', 'CRWD',
        'ZS', 'DDOG', 'SNOW', 'PLTR', 'NET', 'MDB', 'TEAM', 'WDAY',
        'OKTA', 'HUBS', 'ZM', 'DOCU', 'SQ', 'PYPL', 'SHOP', 'MELI',
        'SE', 'UBER', 'LYFT', 'DASH', 'TXN', 'NXPI', 'MPWR', 'ON',
        'SWKS', 'QRVO', 'MCHP', 'ANET', 'CIEN', 'SMCI',
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC',
        'SCHW', 'BK', 'STT', 'NTRS', 'FITB', 'MTB', 'CFG', 'RF',
        'HBAN', 'KEY', 'ZION', 'V', 'MA', 'AXP', 'DFS', 'COF', 'SYF',
        'ALLY', 'SOFI', 'AFRM', 'UPST', 'LC', 'HOOD', 'COIN',
        'BRK.B', 'CB', 'PGR', 'ALL', 'MET', 'PRU', 'AIG', 'AFL',
        'TRV', 'HIG', 'BLK', 'SPGI', 'MCO', 'CME', 'ICE', 'NDAQ',
        'BX', 'KKR', 'APO', 'CG', 'ARES',
        'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'BMY', 'AMGN', 'GILD',
        'BIIB', 'REGN', 'VRTX', 'MRNA', 'BNTX', 'ZTS', 'ABT', 'MDT',
        'DHR', 'TMO', 'SYK', 'BSX', 'EW', 'ISRG', 'BDX', 'ZBH',
        'ALGN', 'IDXX', 'RMD', 'DXCM', 'PODD', 'UNH', 'ELV', 'CVS',
        'CI', 'HUM', 'CNC', 'MOH', 'HCA', 'MCK', 'CAH',
        'HD', 'LOW', 'TJX', 'ROST', 'NKE', 'LULU', 'CROX', 'DECK',
        'WMT', 'TGT', 'COST', 'BJ', 'DG', 'DLTR', 'FIVE', 'OLLI',
        'BBY', 'MCD', 'SBUX', 'YUM', 'CMG', 'DPZ', 'DRI', 'MAR',
        'HLT', 'ABNB', 'BKNG', 'EXPE', 'RCL', 'CCL', 'NCLH', 'LVS',
        'MGM', 'WYNN', 'CZR', 'DKNG', 'F', 'GM', 'RIVN', 'LCID',
        'NIO', 'DIS', 'NFLX', 'WBD', 'PARA', 'TTWO', 'EA', 'RBLX',
        'PEP', 'KO', 'MNST', 'KDP', 'STZ', 'GIS', 'K', 'CPB', 'SJM',
        'CAG', 'HRL', 'TSN', 'MDLZ', 'HSY', 'PG', 'CL', 'KMB', 'CHD',
        'CLX', 'PM', 'MO', 'BTI', 'KR', 'WBA', 'SYY',
        'BA', 'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'TDG', 'AXON',
        'CAT', 'DE', 'PCAR', 'CMI', 'IR', 'ITW', 'ETN', 'PH', 'ROK',
        'AME', 'ROP', 'DOV', 'UNP', 'CSX', 'NSC', 'JBHT', 'ODFL',
        'FDX', 'UPS', 'DAL', 'UAL', 'AAL', 'LUV', 'SHW', 'PPG',
        'GE', 'HON', 'MMM', 'WM', 'RSG', 'PWR', 'EME',
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO',
        'OXY', 'HES', 'PXD', 'DVN', 'HAL', 'BKR', 'FANG', 'KMI',
        'WMB', 'OKE', 'NEE', 'ENPH', 'SEDG', 'FSLR', 'RUN',
        'LIN', 'APD', 'ECL', 'DD', 'DOW', 'LYB', 'NUE', 'STLD',
        'FCX', 'NEM', 'GOLD', 'ALB', 'FMC', 'CF', 'MOS',
        'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'WEC', 'ED', 'AWK',
        'PLD', 'EQR', 'AVB', 'SPG', 'O', 'AMT', 'CCI', 'EQIX', 'DLR',
        'PSA', 'EXR', 'WELL', 'VTR',
        'T', 'VZ', 'TMUS', 'CHTR', 'CMCSA',
        'BABA', 'JD', 'PDD', 'BIDU', 'ASML', 'NVO', 'SAP', 'TSM',
    ])
    etf_symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLU", "XLK",
        "XLP", "XLB", "XLV", "XLI", "XLY", "XLC", "XLRE", "KRE",
        "SMH", "XBI", "GDX",
    ])
    thresholds: List[float] = field(default_factory=lambda: [
        0.000625, 0.00125, 0.0025, 0.005, 0.0075, 0.01, 0.0125,
        0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.10
    ])
    pm_thresholds: List[float] = field(default_factory=lambda: [
        0.000625, 0.00125, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10
    ])
    update_interval_ms: int = 1000
    cache_refresh_interval: float = 0.5
    history_days: int = 5
    polygon_api_key: str = POLYGON_API_KEY
    polygon_ws_url: str = "wss://delayed.polygon.io/stocks"
    polygon_rest_url: str = "https://api.polygon.io"
    volumes: Dict[str, float] = field(default_factory=dict)
    week52_data: Dict[str, Dict] = field(default_factory=dict)
    min_tradable_stasis: int = 3


config = Config()
config.symbols = list(dict.fromkeys(config.symbols))


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SignalStrength(Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


@dataclass
class BitEntry:
    bit: int
    price: float
    timestamp: datetime


@dataclass
class StasisInfo:
    start_time: datetime
    start_price: float
    peak_stasis: int = 1

    def get_duration(self) -> timedelta:
        return datetime.now() - self.start_time

    def get_duration_str(self) -> str:
        t = int(self.get_duration().total_seconds())
        if t < 60:
            return f"{t}s"
        if t < 3600:
            return f"{t // 60}m {t % 60}s"
        return f"{t // 3600}h {(t % 3600) // 60}m"

    def get_start_date_str(self) -> str:
        return self.start_time.strftime("%m/%d %H:%M")

    def get_price_change_pct(self, p: float) -> float:
        return (p - self.start_price) / self.start_price * 100 if self.start_price else 0


# ============================================================================
# HELPERS
# ============================================================================


def calculate_52week_percentile(price, symbol):
    d = config.week52_data.get(symbol)
    if not d:
        return None
    h, l, r = d.get('high'), d.get('low'), d.get('range')
    if not all([h, l, r]) or r <= 0:
        return None
    return max(0, min(100, ((price - l) / r) * 100))


def fmt_rr(rr):
    if rr is None:
        return "—"
    return "0:1" if rr <= 0 else (f"{rr:.2f}:1" if rr < 10 else f"{rr:.0f}:1")


# ============================================================================
# PORTFOLIO
# ============================================================================


class Portfolio:
    def __init__(self, starting_balance=STARTING_BALANCE):
        self.balance = starting_balance
        self.starting_balance = starting_balance
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.total_realized_pnl = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self._lock = threading.Lock()

    def place_bet(self, symbol, side, amount, buy_price, sell_price, direction, stock_price):
        with self._lock:
            if amount <= 0:
                return {'success': False, 'error': 'Invalid amount'}
            if amount > self.balance:
                return {'success': False, 'error': f'Insufficient (${self.balance:.2f})'}
            entry_price = buy_price if side == 'YES' else sell_price
            if entry_price <= 0:
                return {'success': False, 'error': 'Invalid price'}
            shares = amount / entry_price
            pid = f"{symbol}_{side}_{uuid.uuid4().hex[:6]}"
            self.positions[pid] = {
                'id': pid, 'symbol': symbol, 'side': side, 'direction': direction,
                'shares': shares, 'entry_price': entry_price, 'cost_basis': amount,
                'stock_price_at_entry': stock_price, 'entry_time': datetime.now(),
            }
            self.balance -= amount
            self.trade_history.append({
                'id': uuid.uuid4().hex[:8], 'position_id': pid, 'symbol': symbol,
                'side': side, 'direction': direction, 'action': 'BUY',
                'amount': amount, 'shares': shares, 'price': entry_price,
                'timestamp': datetime.now(),
            })
            return {'success': True, 'position_id': pid, 'new_balance': self.balance}

    def close_position(self, position_id, current_buy, current_sell):
        with self._lock:
            if position_id not in self.positions:
                return {'success': False, 'error': 'Not found'}
            pos = self.positions[position_id]
            exit_price = current_sell if pos['side'] == 'YES' else (1.0 - current_buy)
            proceeds = pos['shares'] * exit_price
            pnl = proceeds - pos['cost_basis']
            self.balance += proceeds
            self.total_realized_pnl += pnl
            if pnl >= 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            self.trade_history.append({
                'id': uuid.uuid4().hex[:8], 'position_id': position_id,
                'symbol': pos['symbol'], 'side': pos['side'], 'direction': pos['direction'],
                'action': 'CLOSE', 'shares': pos['shares'], 'entry_price': pos['entry_price'],
                'exit_price': exit_price, 'pnl': pnl,
                'pnl_pct': (pnl / pos['cost_basis'] * 100) if pos['cost_basis'] > 0 else 0,
                'timestamp': datetime.now(),
            })
            del self.positions[position_id]
            return {'success': True, 'pnl': pnl, 'new_balance': self.balance}

    def get_stats(self, market_prices=None):
        with self._lock:
            unrealized = 0.0
            positions_value = 0.0
            if market_prices:
                for pid, pos in self.positions.items():
                    m = market_prices.get(pos['symbol'], {})
                    if pos['side'] == 'YES':
                        val = pos['shares'] * m.get('sell_price', pos['entry_price'])
                    else:
                        val = pos['shares'] * (1.0 - m.get('buy_price', 1.0 - pos['entry_price']))
                    positions_value += val
                    unrealized += val - pos['cost_basis']
            total_trades = self.winning_trades + self.losing_trades
            return {
                'balance': self.balance,
                'portfolio_value': self.balance + positions_value,
                'unrealized_pnl': unrealized,
                'total_pnl': self.total_realized_pnl + unrealized,
                'positions_count': len(self.positions),
                'win_rate': (self.winning_trades / total_trades * 100) if total_trades > 0 else 0,
            }

    def get_positions_list(self):
        with self._lock:
            return list(self.positions.values())

    def get_recent_trades(self, n=10):
        with self._lock:
            return list(reversed(self.trade_history[-n:]))

    def reset(self):
        with self._lock:
            self.balance = self.starting_balance
            self.positions.clear()
            self.trade_history.clear()
            self.total_realized_pnl = 0.0
            self.winning_trades = 0
            self.losing_trades = 0


portfolio = Portfolio()


# ============================================================================
# DATA FETCHERS
# ============================================================================


def fetch_52_week_data():
    print("📊 Fetching 52-week data...")
    w52 = {}
    end = datetime.now()
    start = end - timedelta(days=365)
    ok = fail = 0
    for i, sym in enumerate(config.symbols):
        try:
            url = (f"{config.polygon_rest_url}/v2/aggs/ticker/{sym}/range/1/day/"
                   f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
                   f"?adjusted=true&sort=asc&limit=365&apiKey={config.polygon_api_key}")
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                res = r.json().get('results', [])
                if res:
                    hv = max(b['h'] for b in res)
                    lv = min(b['l'] for b in res)
                    w52[sym] = {'high': hv, 'low': lv, 'range': hv - lv, 'current': res[-1]['c']}
                    ok += 1
                else:
                    w52[sym] = {'high': None, 'low': None, 'range': None, 'current': None}
                    fail += 1
            else:
                w52[sym] = {'high': None, 'low': None, 'range': None, 'current': None}
                fail += 1
            if (i + 1) % 50 == 0:
                print(f"   52W: {i + 1}/{len(config.symbols)} (✓{ok} ✗{fail})")
            time.sleep(0.12)
        except:
            w52[sym] = {'high': None, 'low': None, 'range': None, 'current': None}
            fail += 1
    print(f"✅ 52-week: {ok} ok, {fail} failed\n")
    return w52


def fetch_volume_data():
    print("📊 Fetching volume data...")
    vols = {}
    end = datetime.now()
    start = end - timedelta(days=45)
    for i, sym in enumerate(config.symbols):
        try:
            url = (f"{config.polygon_rest_url}/v2/aggs/ticker/{sym}/range/1/day/"
                   f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
                   f"?adjusted=true&sort=desc&limit=30&apiKey={config.polygon_api_key}")
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                res = r.json().get('results', [])
                if res:
                    vols[sym] = (sum(b['v'] for b in res) / len(res)) / 1e6
                else:
                    vols[sym] = 10.0
            else:
                vols[sym] = 10.0
            if (i + 1) % 50 == 0:
                print(f"   Vol: {i + 1}/{len(config.symbols)}")
            time.sleep(0.12)
        except:
            vols[sym] = 10.0
    print("✅ Volume loaded\n")
    return vols


def fetch_historical_bars(sym, days=5):
    bars = []
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        url = (f"{config.polygon_rest_url}/v2/aggs/ticker/{sym}/range/1/minute/"
               f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
               f"?adjusted=true&sort=asc&limit=50000&apiKey={config.polygon_api_key}")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            res = r.json().get('results', [])
            bars = [{'timestamp': datetime.fromtimestamp(b['t'] / 1000), 'close': b['c']}
                    for b in res]
    except:
        pass
    return bars


# ============================================================================
# BITSTREAM
# ============================================================================


class Bitstream:
    def __init__(self, symbol, threshold, initial_price, volume):
        self.symbol = symbol
        self.threshold = threshold
        self.initial_price = initial_price
        self.volume = volume
        self.is_etf = symbol in config.etf_symbols
        self.reference_price = initial_price
        self.current_live_price = initial_price
        self.last_price_update = datetime.now()
        self._update_bands()
        self.bits: deque = deque(maxlen=500)
        self.current_stasis = 0
        self.last_bit = None
        self.direction = None
        self.signal_strength = None
        self.stasis_info: Optional[StasisInfo] = None
        self.total_bits = 0
        self._lock = threading.Lock()

    def _update_bands(self):
        self.band_width = self.threshold * self.reference_price
        self.upper_band = self.reference_price + self.band_width
        self.lower_band = self.reference_price - self.band_width

    def process_price(self, price, timestamp):
        with self._lock:
            self.current_live_price = price
            self.last_price_update = timestamp
            if self.lower_band < price < self.upper_band:
                return
            if self.band_width <= 0:
                return
            x = int((price - self.reference_price) / self.band_width)
            if x > 0:
                for _ in range(x):
                    self.bits.append(BitEntry(1, price, timestamp))
                    self.total_bits += 1
                self.reference_price = price
                self._update_bands()
            elif x < 0:
                for _ in range(abs(x)):
                    self.bits.append(BitEntry(0, price, timestamp))
                    self.total_bits += 1
                self.reference_price = price
                self._update_bands()
            self._update_stasis(timestamp)

    def _update_stasis(self, ts):
        if len(self.bits) < 2:
            self.current_stasis = len(self.bits)
            self.last_bit = self.bits[-1].bit if self.bits else None
            self.direction = None
            self.signal_strength = None
            return
        bl = list(self.bits)
        sc = 1
        si = len(bl) - 1
        for i in range(len(bl) - 1, 0, -1):
            if bl[i].bit != bl[i - 1].bit:
                sc += 1
                si = i - 1
            else:
                break
        prev = self.current_stasis
        self.current_stasis = sc
        self.last_bit = bl[-1].bit
        if prev < 2 and sc >= 2 and 0 <= si < len(bl):
            self.stasis_info = StasisInfo(bl[si].timestamp, bl[si].price, sc)
        elif sc >= 2 and self.stasis_info and sc > self.stasis_info.peak_stasis:
            self.stasis_info.peak_stasis = sc
        elif prev >= 2 and sc < 2:
            self.stasis_info = None
        if sc >= 2:
            self.direction = Direction.LONG if self.last_bit == 0 else Direction.SHORT
            if sc >= 10:
                self.signal_strength = SignalStrength.VERY_STRONG
            elif sc >= 7:
                self.signal_strength = SignalStrength.STRONG
            elif sc >= 5:
                self.signal_strength = SignalStrength.MODERATE
            elif sc >= 3:
                self.signal_strength = SignalStrength.WEAK
            else:
                self.signal_strength = None
        else:
            self.direction = None
            self.signal_strength = None

    def get_snapshot(self, live_price=None):
        with self._lock:
            p = live_price if live_price is not None else self.current_live_price
            si = self.stasis_info
            tp = sl = rr = None
            if self.direction and self.current_stasis >= 2:
                if self.direction == Direction.LONG:
                    tp, sl = self.upper_band, self.lower_band
                    reward, risk = tp - p, p - sl
                else:
                    tp, sl = self.lower_band, self.upper_band
                    reward, risk = p - tp, sl - p
                if risk > 0 and reward > 0:
                    rr = reward / risk
                elif risk > 0:
                    rr = 0.0
            return {
                'symbol': self.symbol, 'is_etf': self.is_etf,
                'threshold': self.threshold, 'threshold_pct': self.threshold * 100,
                'stasis': self.current_stasis, 'total_bits': self.total_bits,
                'current_price': p,
                'direction': self.direction.value if self.direction else None,
                'signal_strength': self.signal_strength.value if self.signal_strength else None,
                'is_tradable': (self.current_stasis >= config.min_tradable_stasis
                                and self.direction is not None and self.volume > 1.0),
                'stasis_duration_str': si.get_duration_str() if si else "—",
                'duration_seconds': si.get_duration().total_seconds() if si else 0,
                'take_profit': tp, 'stop_loss': sl, 'risk_reward': rr,
                'volume': self.volume,
            }


# ============================================================================
# PRICE FEED
# ============================================================================


class PolygonPriceFeed:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_prices = {s: None for s in config.symbols}
        self.is_running = False
        self.ws = None
        self.message_count = 0

    def start(self):
        self.is_running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print("✅ WebSocket starting...")

    def _loop(self):
        while self.is_running:
            try:
                self._connect()
            except Exception as e:
                print(f"WS err: {e}")
                time.sleep(5)

    def _connect(self):
        def on_msg(ws, msg):
            try:
                data = json.loads(msg)
                for m in (data if isinstance(data, list) else [data]):
                    self._proc(m)
            except:
                pass

        def on_open(ws):
            print("✅ WS connected")
            ws.send(json.dumps({"action": "auth", "params": config.polygon_api_key}))

        self.ws = websocket.WebSocketApp(config.polygon_ws_url,
                                         on_open=on_open, on_message=on_msg)
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def _proc(self, msg):
        if msg.get('ev') == 'status' and msg.get('status') == 'auth_success':
            self._sub()
        elif msg.get('ev') in ('A', 'AM', 'T', 'Q'):
            sym = msg.get('sym', '') or msg.get('S', '')
            price = msg.get('c') or msg.get('vw') or msg.get('p') or msg.get('bp')
            if price and sym in self.current_prices:
                with self.lock:
                    self.current_prices[sym] = float(price)
                    self.message_count += 1

    def _sub(self):
        for i in range(0, len(config.symbols), 50):
            batch = config.symbols[i:i + 50]
            self.ws.send(json.dumps({"action": "subscribe",
                                     "params": ",".join(f"A.{s}" for s in batch)}))
            time.sleep(0.1)
        print(f"📡 Subscribed {len(config.symbols)} symbols")

    def get_prices(self):
        with self.lock:
            return {k: v for k, v in self.current_prices.items() if v}

    def get_status(self):
        with self.lock:
            return {'connected': sum(1 for v in self.current_prices.values() if v),
                    'total': len(config.symbols), 'messages': self.message_count}


price_feed = PolygonPriceFeed()


# ============================================================================
# BITSTREAM MANAGER
# ============================================================================


class BitstreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.streams: Dict[Tuple[str, float], Bitstream] = {}
        self.is_running = False
        self.cached_pm_merit: Dict[str, Dict] = {}
        self.cached_pm_market: Dict[str, Dict] = {}
        self.cache_lock = threading.Lock()
        self.initialized = False
        self.backfill_complete = False
        self.backfill_progress = 0

    def backfill(self):
        print("\n" + "=" * 60 + "\n📜 BACKFILLING\n" + "=" * 60)
        hist = {}
        for i, sym in enumerate(config.symbols):
            bars = fetch_historical_bars(sym, config.history_days)
            if bars:
                hist[sym] = bars
            self.backfill_progress = int((i + 1) / len(config.symbols) * 100)
            if (i + 1) % 25 == 0:
                print(f"   📊 {i + 1}/{len(config.symbols)} ({self.backfill_progress}%)")
            time.sleep(0.12)
        with self.lock:
            for sym, bars in hist.items():
                if not bars:
                    continue
                vol = config.volumes.get(sym, 10.0)
                for th in config.thresholds:
                    key = (sym, th)
                    self.streams[key] = Bitstream(sym, th, bars[0]['close'], vol)
                    for bar in bars:
                        self.streams[key].process_price(bar['close'], bar['timestamp'])
        self.initialized = True
        self.backfill_complete = True
        tradable = sum(1 for s in self.streams.values()
                       if s.current_stasis >= config.min_tradable_stasis
                       and s.direction is not None and s.volume > 1.0)
        print(f"✅ Streams: {len(self.streams)} | Tradable: {tradable}")
        print("=" * 60)

    def start(self):
        self.is_running = True
        threading.Thread(target=self._process, daemon=True).start()
        threading.Thread(target=self._cache, daemon=True).start()

    def _process(self):
        while self.is_running:
            time.sleep(0.1)
            if not self.backfill_complete:
                continue
            prices = price_feed.get_prices()
            ts = datetime.now()
            with self.lock:
                for sym, p in prices.items():
                    for th in config.thresholds:
                        k = (sym, th)
                        if k in self.streams:
                            self.streams[k].process_price(p, ts)

    def _cache(self):
        while self.is_running:
            time.sleep(config.cache_refresh_interval)
            if not self.initialized:
                continue
            prices = price_feed.get_prices()
            snaps = []
            with self.lock:
                for s in self.streams.values():
                    snaps.append(s.get_snapshot(prices.get(s.symbol)))
            pm_merit = self._calc_merit(snaps)
            pm_market = {}
            for sym, m in pm_merit.items():
                pm_market[sym] = self._calc_market(m, prices.get(sym, 0))
            with self.cache_lock:
                self.cached_pm_merit = pm_merit
                self.cached_pm_market = pm_market

    def _calc_merit(self, snaps):
        WEIGHTS = {0.000625: 1, 0.00125: 1.5, 0.0025: 2, 0.005: 3,
                   0.01: 5, 0.02: 8, 0.03: 12, 0.04: 16, 0.05: 20, 0.10: 30}
        by_sym = defaultdict(list)
        for s in snaps:
            if s['threshold'] in config.pm_thresholds:
                by_sym[s['symbol']].append(s)
        result = {}
        for sym, ss in by_sym.items():
            long_lvl, short_lvl, all_st = [], [], []
            for snap in ss:
                st = snap['stasis']
                d = snap['direction']
                th = snap['threshold']
                all_st.append(st)
                if st >= config.min_tradable_stasis and d:
                    w = (st ** 1.3) * WEIGHTS.get(th, 1)
                    (long_lvl if d == 'LONG' else short_lvl).append(
                        {'thresh': th, 'stasis': st, 'weight': w})
            lw = sum(l['weight'] for l in long_lvl)
            sw = sum(l['weight'] for l in short_lvl)
            tl = len(long_lvl) + len(short_lvl)
            tw = lw + sw
            if tl == 0:
                result[sym] = {'stasis_levels': 0, 'dominant_direction': None,
                               'direction_alignment': 0, 'weighted_score': 0,
                               'max_stasis': max(all_st) if all_st else 0,
                               'long_levels': 0, 'short_levels': 0}
                continue
            if lw > sw:
                dd, nw = 'LONG', lw - sw
            elif sw > lw:
                dd, nw = 'SHORT', sw - lw
            else:
                dd, nw = None, 0
            align = (max(lw, sw) / tw * 100) if tw > 0 else 0
            score = nw * (1 + 0.2 * (tl - 1)) if align == 100 and tl >= 2 else nw
            result[sym] = {
                'stasis_levels': tl, 'dominant_direction': dd,
                'direction_alignment': round(align, 0),
                'weighted_score': round(max(0, score), 1),
                'max_stasis': max(all_st) if all_st else 0,
                'long_levels': len(long_lvl), 'short_levels': len(short_lvl),
            }
        return result

    def _calc_market(self, merit, stock_price):
        levels = merit.get('stasis_levels', 0)
        align = merit.get('direction_alignment', 0)
        direction = merit.get('dominant_direction')
        max_st = merit.get('max_stasis', 0)
        score = merit.get('weighted_score', 0)
        if levels == 0 or direction is None:
            return {'probability': 0.50, 'buy_price': 0.50, 'sell_price': 0.50,
                    'edge': 0, 'direction': None, 'payout': 1.0, 'tier': None,
                    'emoji': '⬜', 'heat': 0, 'stock_price': stock_price}
        base = 0.50
        lc = min(0.30, levels * 0.04)
        sc = min(0.10, max_st * 0.01)
        wc = min(0.05, score / 1000)
        if align == 100:
            prob = base + lc + sc + wc
        else:
            prob = base + (lc * align / 100 * 0.5) + (sc * 0.5)
        prob = max(0.52, min(0.95, prob))
        spread = max(0.005, 0.02 - 0.01 * min(1, levels / 5))
        buy = min(0.98, prob + spread / 2)
        sell = max(0.02, 1 - prob + spread / 2)
        edge = (prob - 0.50) * 100
        payout = 1 / buy if buy > 0 else 1
        heat = min(100, int((levels / 10) * 50 + (align / 100) * 30 + min(20, max_st * 2)))
        tier, emoji = None, '⬜'
        for tn, ti in JACKPOT_TIERS.items():
            if levels >= ti['min_levels'] and align >= ti['min_alignment']:
                tier, emoji = tn, ti['emoji']
                break
        return {'probability': round(prob, 4), 'buy_price': round(buy, 4),
                'sell_price': round(sell, 4), 'edge': round(edge, 2),
                'direction': direction, 'payout': round(payout, 2),
                'tier': tier, 'emoji': emoji, 'heat': heat,
                'stock_price': stock_price}

    def get_pm_market(self):
        with self.cache_lock:
            return copy.deepcopy(self.cached_pm_market)

    def get_pm_merit(self):
        with self.cache_lock:
            return copy.deepcopy(self.cached_pm_merit)


manager = BitstreamManager()


# ============================================================================
# DASH APP
# ============================================================================

PM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600&display=swap');
body { background: linear-gradient(180deg, #0a0a12 0%, #10101e 50%, #0a0a12 100%) !important; }
.title-font { font-family: 'Orbitron', sans-serif !important; }
.neon-green { color: #00ff88; text-shadow: 0 0 5px #00ff88; }
.neon-cyan { color: #00ffff; text-shadow: 0 0 5px #00ffff; }
.neon-gold { color: #ffd700; text-shadow: 0 0 5px #ffd700; }
.pm-market-row {
    background: rgba(18,18,34,0.95); border: 1px solid #2a2a4e;
    border-radius: 5px; padding: 6px 10px; margin-bottom: 4px;
    transition: border-color 0.15s;
}
.pm-market-row:hover { border-color: #00ffff; }
.btn-yes {
    background: linear-gradient(135deg, #00aa44, #00ff88) !important;
    border: none !important; color: #000 !important;
    font-weight: bold !important; font-size: 9px !important;
    padding: 3px 8px !important; border-radius: 3px !important;
}
.btn-no {
    background: linear-gradient(135deg, #aa2200, #ff4444) !important;
    border: none !important; color: #fff !important;
    font-weight: bold !important; font-size: 9px !important;
    padding: 3px 8px !important; border-radius: 3px !important;
}
.btn-close-pos {
    background: #444 !important; border: 1px solid #666 !important;
    color: #fff !important; font-size: 8px !important; padding: 1px 6px !important;
}
.amount-btn {
    background: #2a2a4e !important; border: 1px solid #444 !important;
    color: #fff !important; font-size: 9px !important; padding: 3px 6px !important;
    margin: 1px !important; border-radius: 3px !important;
}
.amount-btn.selected {
    background: linear-gradient(135deg, #0066aa, #00aaff) !important;
    border-color: #00ffff !important;
}
.pm-portfolio {
    background: linear-gradient(135deg, #12192a, #0a1018);
    border: 1px solid #00ffff; border-radius: 8px; padding: 8px;
}
.pm-position { background: rgba(18,24,34,0.9); border: 1px solid #333;
               border-radius: 4px; padding: 6px; margin: 3px 0; }
.pm-position-long { border-left: 3px solid #00ff88; }
.pm-position-short { border-left: 3px solid #ff4444; }
.sym-clickable {
    cursor: pointer; color: #00ff88; font-weight: bold; font-size: 11px;
    text-decoration: none; border-bottom: 1px dotted rgba(0,255,136,0.3);
    transition: all 0.15s;
}
.sym-clickable:hover {
    color: #00ffff; text-shadow: 0 0 8px #00ffff;
    border-bottom-color: #00ffff;
}
"""

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
                suppress_callback_exceptions=True)
app.title = "🎰 STASIS PM"
server = app.server

app.index_string = '''<!DOCTYPE html>
<html><head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>''' + PM_CSS + '''</style>
</head><body>
{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body></html>'''

app.layout = html.Div([
    dcc.Store(id='pm-bet-amount', data=100),
    dcc.Store(id='pm-trigger', data=0),
    dcc.Store(id='pm-filter-mode', data='signals'),
    dcc.Store(id='pm-selected-symbol', data=''),       # NEW: store for selected symbol
    dcc.Interval(id='tick', interval=1000, n_intervals=0),

    # NEW: hidden div for symbol bridge (like AM)
    html.Div(id='_pm_sym_bridge', style={'display': 'none'}),

    dbc.Toast(id="pm-toast", header="", is_open=False, duration=3500, dismissable=True,
              style={"position": "fixed", "top": 10, "left": "50%",
                     "transform": "translateX(-50%)", "width": 300, "zIndex": 9999}),

    html.Div([
        html.Div([
            html.Span("🎰", style={'fontSize': '28px'}),
            html.Span(" STASIS PM", className="title-font neon-green ms-2",
                      style={'fontSize': '20px', 'fontWeight': '700', 'letterSpacing': '3px'}),
            html.Span(" — PREDICTION MARKETS", className="title-font",
                      style={'fontSize': '10px', 'color': '#666', 'letterSpacing': '1px'}),
        ], className="mb-1"),

        html.Div(id='pm-status', style={'fontSize': '9px', 'marginBottom': '4px'}),
        html.Div(id='pm-summary', style={'fontSize': '9px', 'marginBottom': '4px'}),

        # Filters
        html.Div([
            dbc.ButtonGroup([
                dbc.Button("ALL", id="pm-f-all", size="sm", color="secondary", outline=True,
                           className="title-font", style={'fontSize': '8px'}),
                dbc.Button("SIGNALS", id="pm-f-signals", size="sm", color="success", outline=True,
                           active=True, className="title-font", style={'fontSize': '8px'}),
                dbc.Button("JACKPOT", id="pm-f-jackpot", size="sm", color="warning", outline=True,
                           className="title-font", style={'fontSize': '8px'}),
            ], size="sm", className="me-2"),
            dcc.Dropdown(id='pm-f-dir', options=[
                {'label': 'ALL', 'value': 'ALL'}, {'label': 'LONG', 'value': 'LONG'},
                {'label': 'SHORT', 'value': 'SHORT'}],
                value='ALL', clearable=False,
                style={'width': '70px', 'fontSize': '9px', 'display': 'inline-block',
                       'verticalAlign': 'middle'}),
        ], className="d-flex align-items-center mb-1"),

        # Bet amount
        html.Div([
            html.Span("BET: ", className="title-font",
                       style={'fontSize': '9px', 'color': '#ffd700'}),
            *[dbc.Button(f"${a}", id={'type': 'pm-amt', 'amount': a},
                         className=f"amount-btn {'selected' if a == 100 else ''}",
                         size="sm") for a in BET_SIZES],
        ], className="d-flex align-items-center flex-wrap mb-1"),

        # NEW: Click hint
        html.Div("💡 Click any symbol name to look up on SA/RH/TT",
                 style={'fontSize': '9px', 'color': '#aa8800', 'padding': '2px 0 4px 0'}),

        # Markets
        html.Div(id='pm-market-list', style={'height': '50vh', 'overflowY': 'auto'}),

        html.Hr(style={'borderColor': '#222', 'margin': '6px 0'}),

        # Portfolio
        html.Div([
            html.Div([
                html.Span("💰 ACCOUNT", className="title-font neon-cyan",
                           style={'fontSize': '10px'}),
                dbc.Button("Reset", id="pm-reset", size="sm", color="secondary",
                           style={'fontSize': '7px', 'padding': '1px 5px', 'marginLeft': '8px'}),
            ], className="d-flex align-items-center mb-1"),
            html.Div(id='pm-portfolio'),
        ], className="pm-portfolio mb-2"),

        html.Div([
            html.Span("📊 POSITIONS", className="title-font",
                       style={'fontSize': '9px', 'color': '#00ffff'}),
            html.Div(id='pm-positions', style={'maxHeight': '15vh', 'overflowY': 'auto'}),
        ], className="mb-2"),

        html.Div([
            html.Span("📜 HISTORY", className="title-font",
                       style={'fontSize': '9px', 'color': '#ffd700'}),
            html.Div(id='pm-history', style={'maxHeight': '12vh', 'overflowY': 'auto'}),
        ]),

        html.Div("© 2026 Truth Communications LLC • STASIS PM",
                 className="text-center",
                 style={'fontSize': '8px', 'color': '#444', 'padding': '8px'}),
    ], style={'padding': '8px', 'minHeight': '100vh'}),
])


# ============================================================================
# NEW: Symbol bridge callback — writes selected symbol to window.name
# so the desktop app can read it via polling
# ============================================================================

app.clientside_callback(
    """function(sym) {
        if (sym && sym.length > 0) {
            window.name = 'STASIS_SYM:' + sym;
        }
        return '';
    }""",
    Output('_pm_sym_bridge', 'children'),
    Input('pm-selected-symbol', 'data'),
    prevent_initial_call=True
)


# NEW: Callback to handle symbol clicks from the market list
@app.callback(
    Output('pm-selected-symbol', 'data'),
    Input({'type': 'pm-sym-click', 'symbol': ALL}, 'n_clicks'),
    prevent_initial_call=True)
def pm_symbol_clicked(clicks):
    ctx = callback_context
    if not ctx.triggered or not any(clicks):
        return no_update
    try:
        prop = ctx.triggered[0]['prop_id']
        d = json.loads(prop.rsplit('.', 1)[0])
        sym = d.get('symbol', '')
        if sym:
            return sym
    except:
        pass
    return no_update


# ============================================================================
# CALLBACKS
# ============================================================================


@app.callback(Output('pm-status', 'children'), Input('tick', 'n_intervals'))
def update_status(n):
    if not manager.backfill_complete:
        return html.Span(f"⏳ INITIALIZING... {manager.backfill_progress}%",
                         style={'color': '#ffaa00'})
    st = price_feed.get_status()
    if st['connected'] == 0:
        return html.Span("🔴 CONNECTING...", style={'color': '#ffaa00'})
    return html.Span(
        f"🟢 LIVE {st['connected']}/{st['total']} | 📨 {st['messages']:,}",
        style={'color': '#00ff88'})


@app.callback(Output('pm-summary', 'children'), Input('tick', 'n_intervals'))
def pm_summary(n):
    if not manager.backfill_complete:
        return ""
    mkt = manager.get_pm_market()
    sigs = sum(1 for m in mkt.values() if m.get('direction'))
    hi = sum(1 for m in mkt.values() if m.get('edge', 0) >= 10)
    jp = sum(1 for m in mkt.values() if m.get('tier'))
    return html.Div([
        html.Span(f"📈 {sigs} signals", style={'color': '#00ff88'}),
        html.Span(f" | 🔥 {hi} high-edge", style={'color': '#ff8800'}),
        html.Span(f" | 🎰 {jp} jackpots", style={'color': '#ffd700'}),
    ])


@app.callback(
    [Output('pm-f-all', 'active'), Output('pm-f-signals', 'active'),
     Output('pm-f-jackpot', 'active'), Output('pm-filter-mode', 'data')],
    [Input('pm-f-all', 'n_clicks'), Input('pm-f-signals', 'n_clicks'),
     Input('pm-f-jackpot', 'n_clicks')],
    prevent_initial_call=True)
def pm_toggle_filter(n1, n2, n3):
    ctx = callback_context
    if not ctx.triggered:
        return False, True, False, 'signals'
    b = ctx.triggered[0]['prop_id'].split('.')[0]
    if b == 'pm-f-all':
        return True, False, False, 'all'
    if b == 'pm-f-jackpot':
        return False, False, True, 'jackpot'
    return False, True, False, 'signals'


@app.callback(
    Output('pm-bet-amount', 'data'),
    Input({'type': 'pm-amt', 'amount': ALL}, 'n_clicks'),
    State('pm-bet-amount', 'data'),
    prevent_initial_call=True)
def pm_set_amount(clicks, cur):
    ctx = callback_context
    if not ctx.triggered or not any(clicks):
        return cur
    try:
        return json.loads(ctx.triggered[0]['prop_id'].rsplit('.', 1)[0])['amount']
    except:
        return cur


@app.callback(
    [Output({'type': 'pm-amt', 'amount': a}, 'className') for a in BET_SIZES],
    Input('pm-bet-amount', 'data'))
def pm_highlight(sel):
    return [f"amount-btn {'selected' if a == sel else ''}" for a in BET_SIZES]


@app.callback(
    Output('pm-market-list', 'children'),
    [Input('tick', 'n_intervals'), Input('pm-filter-mode', 'data'),
     Input('pm-f-dir', 'value'), Input('pm-bet-amount', 'data'),
     Input('pm-trigger', 'data')])
def pm_market_list(n, fmode, fdir, bet, trigger):
    if not manager.backfill_complete:
        return html.Div("Loading…", className="text-muted text-center p-3",
                         style={'color': '#666'})
    mkt = manager.get_pm_market()
    mer = manager.get_pm_merit()
    items = []
    for sym in config.symbols:
        m = mkt.get(sym, {})
        mr = mer.get(sym, {})
        d = m.get('direction')
        edge = m.get('edge', 0)
        tier = m.get('tier')
        if fmode == 'signals' and not d:
            continue
        if fmode == 'jackpot' and not tier:
            continue
        if fdir == 'LONG' and d != 'LONG':
            continue
        if fdir == 'SHORT' and d != 'SHORT':
            continue
        items.append({'symbol': sym, 'm': m, 'mr': mr, 'edge': edge})
    items.sort(key=lambda x: x['edge'], reverse=True)
    if not items:
        return html.Div("No markets match", className="text-muted text-center p-3",
                         style={'color': '#666'})
    rows = []
    for i, it in enumerate(items[:60]):
        sym = it['symbol']
        m = it['m']
        mr = it['mr']
        d = m.get('direction', '—')
        bp = m.get('buy_price', .5)
        sp = m.get('sell_price', .5)
        edge = m.get('edge', 0)
        payout = m.get('payout', 1)
        emoji = m.get('emoji', '⬜')
        lvl = mr.get('stasis_levels', 0)
        align = mr.get('direction_alignment', 0)
        dc = '#00ff88' if d == 'LONG' else '#ff4444' if d == 'SHORT' else '#555'
        rows.append(html.Div([
            dbc.Row([
                dbc.Col([
                    html.Span(f"#{i + 1}",
                              style={'color': '#ffd700' if i < 3 else '#555', 'fontSize': '9px'}),
                    html.Span(f" {emoji} ", style={'fontSize': '12px'}),
                    # CHANGED: symbol is now a clickable button with pattern-matching ID
                    html.Button(
                        sym,
                        id={'type': 'pm-sym-click', 'symbol': sym},
                        className="sym-clickable",
                        style={'background': 'none', 'border': 'none', 'padding': '0',
                               'cursor': 'pointer', 'color': '#00ff88', 'fontWeight': 'bold',
                               'fontSize': '11px', 'borderBottom': '1px dotted rgba(0,255,136,0.3)'},
                    ),
                ], width=3),
                dbc.Col([
                    html.Span(d, style={'color': dc, 'fontWeight': 'bold', 'fontSize': '9px'}),
                    html.Span(f" {lvl}L {align:.0f}%",
                              style={'color': '#666', 'fontSize': '8px'}),
                ], width=2),
                dbc.Col([
                    html.Span(f"${bp:.2f}", style={'color': '#00ff88', 'fontWeight': 'bold',
                                                     'fontSize': '10px'}),
                    html.Span("/", style={'color': '#333'}),
                    html.Span(f"${sp:.2f}", style={'color': '#ff4444', 'fontWeight': 'bold',
                                                     'fontSize': '10px'}),
                    html.Br(),
                    html.Span(f"+{edge:.1f}% {payout:.1f}x",
                              style={'color': '#ffd700', 'fontSize': '8px'}),
                ], width=3),
                dbc.Col([
                    html.Button(f"YES ${bp:.2f}",
                                id={'type': 'pm-buy-yes', 'symbol': sym},
                                className="btn-yes me-1", disabled=not d),
                    html.Button(f"NO ${sp:.2f}",
                                id={'type': 'pm-buy-no', 'symbol': sym},
                                className="btn-no", disabled=not d),
                ], width=4, className="text-end"),
            ], className="align-items-center"),
        ], className="pm-market-row"))
    return html.Div(rows)


@app.callback(Output('pm-portfolio', 'children'),
              [Input('tick', 'n_intervals'), Input('pm-trigger', 'data')])
def pm_portfolio_display(n, t):
    mkt = manager.get_pm_market()
    st = portfolio.get_stats(mkt)
    pnl = st['total_pnl']
    pc = '#00ff88' if pnl >= 0 else '#ff4444'
    return html.Div([
        html.Div(f"${st['portfolio_value']:,.2f}", className="neon-cyan",
                 style={'fontSize': '14px', 'fontWeight': 'bold'}),
        html.Div([
            html.Span(f"Cash ${st['balance']:,.2f}",
                       style={'fontSize': '9px', 'color': '#888'}),
            html.Span(f" | P&L {'+' if pnl >= 0 else ''}${pnl:,.2f}",
                       style={'fontSize': '9px', 'color': pc}),
            html.Span(f" | Win {st['win_rate']:.0f}%",
                       style={'fontSize': '9px',
                              'color': '#00ff88' if st['win_rate'] >= 50 else '#ff4444'}),
        ]),
    ])


@app.callback(Output('pm-positions', 'children'),
              [Input('tick', 'n_intervals'), Input('pm-trigger', 'data')])
def pm_positions_display(n, t):
    pos = portfolio.get_positions_list()
    mkt = manager.get_pm_market()
    if not pos:
        return html.Div("No positions", style={'fontSize': '9px', 'color': '#555'})
    items = []
    for p in pos:
        m = mkt.get(p['symbol'], {})
        if p['side'] == 'YES':
            cv = p['shares'] * m.get('sell_price', p['entry_price'])
        else:
            cv = p['shares'] * (1 - m.get('buy_price', 1 - p['entry_price']))
        pnl = cv - p['cost_basis']
        pc = '#00ff88' if pnl >= 0 else '#ff4444'
        sc = 'pm-position-long' if p['side'] == 'YES' else 'pm-position-short'
        items.append(html.Div([
            html.Div([
                html.Span(p['symbol'], style={'color': '#00ff88', 'fontWeight': 'bold',
                                               'fontSize': '10px'}),
                html.Span(f" {p['side']}",
                           style={'color': '#00ff88' if p['side'] == 'YES' else '#ff4444',
                                  'fontSize': '9px'}),
                html.Button("✕", id={'type': 'pm-close', 'id': p['id']},
                             className="btn-close-pos ms-2"),
            ], className="d-flex align-items-center justify-content-between"),
            html.Span(f"P&L {'+' if pnl >= 0 else ''}${pnl:.2f}",
                       style={'color': pc, 'fontSize': '9px'}),
        ], className=f"pm-position {sc}"))
    return html.Div(items)


@app.callback(Output('pm-history', 'children'),
              [Input('tick', 'n_intervals'), Input('pm-trigger', 'data')])
def pm_history_display(n, t):
    trades = portfolio.get_recent_trades(6)
    if not trades:
        return html.Div("No trades", style={'fontSize': '9px', 'color': '#555'})
    items = []
    for tr in trades:
        if tr['action'] == 'CLOSE':
            pnl = tr.get('pnl', 0)
            pc = '#00ff88' if pnl >= 0 else '#ff4444'
            items.append(html.Div([
                html.Span(f"CLOSE {tr['symbol']} {tr['side']}",
                           style={'color': '#888', 'fontSize': '9px'}),
                html.Span(f" {'+' if pnl >= 0 else ''}${pnl:.2f}",
                           style={'color': pc, 'fontSize': '9px'}),
            ], style={'padding': '2px 0'}))
        else:
            sc = '#00ff88' if tr['side'] == 'YES' else '#ff4444'
            items.append(html.Div([
                html.Span(f"BUY {tr['side']} {tr['symbol']} ${tr['amount']:.0f}",
                           style={'color': sc, 'fontSize': '9px'}),
            ], style={'padding': '2px 0'}))
    return html.Div(items)


@app.callback(
    [Output('pm-toast', 'is_open'), Output('pm-toast', 'header'),
     Output('pm-toast', 'children'), Output('pm-toast', 'style'),
     Output('pm-trigger', 'data')],
    [Input({'type': 'pm-buy-yes', 'symbol': ALL}, 'n_clicks'),
     Input({'type': 'pm-buy-no', 'symbol': ALL}, 'n_clicks'),
     Input({'type': 'pm-close', 'id': ALL}, 'n_clicks'),
     Input('pm-reset', 'n_clicks')],
    [State('pm-bet-amount', 'data'), State('pm-trigger', 'data')],
    prevent_initial_call=True)
def pm_execute(yes_c, no_c, close_c, reset_c, bet, trigger):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update, no_update
    prop = ctx.triggered[0]['prop_id']
    val = ctx.triggered[0]['value']
    base = {"position": "fixed", "top": 10, "left": "50%",
            "transform": "translateX(-50%)", "width": 300, "zIndex": 9999,
            "backgroundColor": "#1a1a2e"}
    if 'pm-reset' in prop and val:
        portfolio.reset()
        return True, "🔄 RESET", \
               html.Span(f"${STARTING_BALANCE:,.2f}", style={'color': '#00ffff'}), \
               {**base, 'border': '2px solid #00ffff'}, trigger + 1
    if 'pm-close' in prop and val:
        try:
            d = json.loads(prop.rsplit('.', 1)[0])
            pid = d.get('id')
            pos = next((p for p in portfolio.get_positions_list() if p['id'] == pid), None)
            if pos:
                mkt = manager.get_pm_market()
                m = mkt.get(pos['symbol'], {})
                res = portfolio.close_position(pid, m.get('buy_price', .5),
                                                m.get('sell_price', .5))
                if res['success']:
                    pnl = res['pnl']
                    pc = '#00ff88' if pnl >= 0 else '#ff4444'
                    return True, "✅ CLOSED", \
                           html.Span(f"{pos['symbol']} {'+' if pnl >= 0 else ''}${pnl:.2f}",
                                     style={'color': pc, 'fontWeight': 'bold'}), \
                           {**base, 'border': f'2px solid {pc}'}, trigger + 1
        except:
            pass
        return no_update, no_update, no_update, no_update, no_update
    if ('pm-buy-yes' in prop or 'pm-buy-no' in prop) and val:
        try:
            d = json.loads(prop.rsplit('.', 1)[0])
            sym = d.get('symbol')
            side = 'YES' if 'buy-yes' in prop else 'NO'
            mkt = manager.get_pm_market()
            m = mkt.get(sym, {})
            if not m.get('direction'):
                return True, "❌", html.Span("No signal", style={'color': '#ff4444'}), \
                       {**base, 'border': '2px solid #ff4444'}, trigger
            res = portfolio.place_bet(sym, side, bet, m.get('buy_price', .5),
                                       m.get('sell_price', .5), m.get('direction'),
                                       m.get('stock_price', 0))
            if res['success']:
                sc = '#00ff88' if side == 'YES' else '#ff4444'
                return True, "✅ BET", \
                       html.Div([
                           html.Span(f"{side} {sym} ${bet}",
                                     style={'color': sc, 'fontWeight': 'bold'}),
                           html.Br(),
                           html.Span(f"Bal: ${res['new_balance']:,.2f}",
                                     style={'color': '#888', 'fontSize': '10px'}),
                       ]), {**base, 'border': f'2px solid {sc}'}, trigger + 1
            else:
                return True, "❌", \
                       html.Span(res.get('error', ''), style={'color': '#ff4444'}), \
                       {**base, 'border': '2px solid #ff4444'}, trigger
        except Exception as e:
            return True, "❌", html.Span(str(e), style={'color': '#ff4444'}), \
                   {**base, 'border': '2px solid #ff4444'}, trigger
    return no_update, no_update, no_update, no_update, no_update


# ============================================================================
# API
# ============================================================================


@server.route('/api/health')
def health():
    return json.dumps({
        'status': 'ok', 'app': 'stasis_pm',
        'initialized': manager.initialized,
        'backfill_complete': manager.backfill_complete,
        'backfill_progress': manager.backfill_progress,
    })


# ============================================================================
# INITIALIZATION
# ============================================================================

_init_done = False
_init_lock = threading.Lock()


def initialize():
    global _init_done
    with _init_lock:
        if _init_done:
            return
        print("=" * 70)
        print("  STASIS PM SERVER — PREDICTION MARKETS")
        print("  © 2026 Truth Communications LLC")
        print("=" * 70)
        print(f"\n🎯 Symbols: {len(config.symbols)}")
        config.week52_data = fetch_52_week_data()
        config.volumes = fetch_volume_data()
        manager.backfill()
        price_feed.start()
        manager.start()
        print(f"\n✅ STASIS PM READY")
        print("=" * 70)
        _init_done = True


_init_thread = threading.Thread(target=initialize, daemon=True)
_init_thread.start()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    _init_thread.join()
    port = int(os.environ.get('PORT', 8051))
    print(f"\n🟢 http://0.0.0.0:{port}\n")
    app.run(debug=False, host='0.0.0.0', port=port)
