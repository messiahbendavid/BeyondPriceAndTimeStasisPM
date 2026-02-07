# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 12:26:33 2026

@author: brcum
"""

# -*- coding: utf-8 -*-
"""
STASIS PM SERVER v2.1 — JACKPOT EDITION
Standalone web server for Stasis PM - Multi-Level Alignment Detector
Deploy to: stasisPM.beyondpriceandtime.com
Copyright © 2026 Truth Communications LLC. All Rights Reserved.
"""

import sys
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
import signal
import traceback

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import websocket
import ssl
import requests

# ============================================================================
# API KEYS & SERVER CONFIG
# ============================================================================

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "PnzhJOXEJO7tSpHr0ct2zjFKi6XO0yGi")
PORT = int(os.environ.get("PORT", 8051))
HOST = os.environ.get("HOST", "0.0.0.0")

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
AM_SERVER_URL = os.environ.get("AM_SERVER_URL", "https://stasisAM.beyondpriceandtime.com")

# ============================================================================
# GAMIFICATION CONSTANTS
# ============================================================================

JACKPOT_TIERS = {
    'GRAND_JACKPOT': {'min_levels': 8, 'min_alignment': 100, 'emoji': '🎰💎🎰',
                      'color': '#ff00ff', 'multiplier': '1000x'},
    'MEGA_JACKPOT': {'min_levels': 6, 'min_alignment': 100, 'emoji': '🎰🎰🎰',
                     'color': '#ffff00', 'multiplier': '100x'},
    'SUPER_JACKPOT': {'min_levels': 5, 'min_alignment': 100, 'emoji': '💰💰💰',
                      'color': '#00ffff', 'multiplier': '50x'},
    'JACKPOT': {'min_levels': 4, 'min_alignment': 100, 'emoji': '🍀🍀🍀',
                'color': '#00ff88', 'multiplier': '25x'},
    'BIG_WIN': {'min_levels': 3, 'min_alignment': 100, 'emoji': '⭐⭐⭐',
                'color': '#88ff88', 'multiplier': '10x'},
    'WIN': {'min_levels': 2, 'min_alignment': 100, 'emoji': '✨✨',
            'color': '#aaffaa', 'multiplier': '5x'},
    'NEAR_MISS': {'min_levels': 2, 'min_alignment': 75, 'emoji': '🎯',
                  'color': '#ffaa00', 'multiplier': '2x'},
}

ACHIEVEMENTS = {
    'PERFECT_10': {'desc': '10 levels aligned', 'emoji': '🏆', 'rarity': 'LEGENDARY'},
    'SEVEN_SEVEN_SEVEN': {'desc': '7+ levels, 7+ stasis', 'emoji': '🎰', 'rarity': 'EPIC'},
    'BLACKJACK': {'desc': '21+ total stasis aligned', 'emoji': '🃏', 'rarity': 'RARE'},
    'FULL_HOUSE': {'desc': '5 levels same direction', 'emoji': '🏠', 'rarity': 'UNCOMMON'},
    'TRIPLE_THREAT': {'desc': '3 high-threshold alignments', 'emoji': '🔥', 'rarity': 'COMMON'},
}

RARITY_COLORS = {
    'LEGENDARY': '#ff8000',
    'EPIC': '#a335ee',
    'RARE': '#0070dd',
    'UNCOMMON': '#1eff00',
    'COMMON': '#ffffff',
}

# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class Config:
    symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLU", "XLK",
        "XLP", "XLB", "XLV", "XLI", "XLY", "XLC", "XLRE", "KRE",
        "SMH", "XBI", "GDX",
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META',
        'TSLA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'INTC'
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
        0.000625, 0.00125, 0.0025, 0.005, 0.01, 0.02, 0.03,
        0.04, 0.05, 0.10
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
# HELPER FUNCTIONS
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


def format_bits(bits):
    return "".join(str(b) for b in bits) if bits else "—"


def format_band(threshold_pct):
    if threshold_pct < 0.1:
        return f"{threshold_pct:.4f}%"
    elif threshold_pct < 1:
        return f"{threshold_pct:.3f}%"
    else:
        return f"{threshold_pct:.2f}%"


# ============================================================================
# JACKPOT CALCULATOR
# ============================================================================


def calculate_jackpot_status(merit_data: Dict) -> Dict:
    levels = merit_data.get('stasis_levels', 0)
    alignment = merit_data.get('direction_alignment', 0)
    total_stasis = merit_data.get('total_stasis', 0)
    max_stasis = merit_data.get('max_stasis', 0)

    result = {
        'tier': None,
        'tier_name': 'NO SIGNAL',
        'emoji': '⬜',
        'color': '#333333',
        'multiplier': '0x',
        'is_jackpot': False,
        'achievements': [],
        'slot_display': ['⬜', '⬜', '⬜'],
        'heat_level': 0,
        'vector_strength': 0,
    }

    if levels == 0:
        return result

    if alignment == 100:
        result['vector_strength'] = levels * 10 + total_stasis
    else:
        result['vector_strength'] = (levels * 10 + total_stasis) * (alignment / 100) * 0.5

    heat = min(100, (levels / 10) * 50 + (alignment / 100) * 30 + min(20, max_stasis * 2))
    result['heat_level'] = int(heat)

    for tier_name, tier_info in JACKPOT_TIERS.items():
        if levels >= tier_info['min_levels'] and alignment >= tier_info['min_alignment']:
            result['tier'] = tier_name
            result['tier_name'] = tier_name.replace('_', ' ')
            result['emoji'] = tier_info['emoji']
            result['color'] = tier_info['color']
            result['multiplier'] = tier_info['multiplier']
            result['is_jackpot'] = 'JACKPOT' in tier_name
            break

    direction = merit_data.get('dominant_direction', None)
    if direction == 'LONG':
        base_symbol = '🟢'
        alt_symbol = '🔴'
    elif direction == 'SHORT':
        base_symbol = '🔴'
        alt_symbol = '🟢'
    else:
        base_symbol = '⬜'
        alt_symbol = '⬜'

    if alignment == 100 and levels >= 3:
        result['slot_display'] = [base_symbol, base_symbol, base_symbol]
    elif alignment >= 75:
        result['slot_display'] = [base_symbol, base_symbol, alt_symbol]
    elif alignment >= 50:
        result['slot_display'] = [base_symbol, alt_symbol, base_symbol]
    else:
        result['slot_display'] = [base_symbol, alt_symbol, alt_symbol]

    if levels >= 10 and alignment == 100:
        result['achievements'].append('PERFECT_10')
    if levels >= 7 and max_stasis >= 7 and alignment == 100:
        result['achievements'].append('SEVEN_SEVEN_SEVEN')
    if total_stasis >= 21 and alignment == 100:
        result['achievements'].append('BLACKJACK')
    if levels >= 5 and alignment == 100:
        result['achievements'].append('FULL_HOUSE')
    thresholds = merit_data.get('thresholds_in_stasis', [])
    high_thresholds = [t for t in thresholds if t >= 0.02]
    if len(high_thresholds) >= 3 and alignment == 100:
        result['achievements'].append('TRIPLE_THREAT')

    return result


def get_heat_color(heat_level: int) -> str:
    if heat_level >= 90:
        return '#ff0000'
    elif heat_level >= 75:
        return '#ff4400'
    elif heat_level >= 60:
        return '#ff8800'
    elif heat_level >= 45:
        return '#ffcc00'
    elif heat_level >= 30:
        return '#ffff00'
    elif heat_level >= 15:
        return '#88ff00'
    else:
        return '#00ff88'


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
                d = r.json()
                res = d.get('results', [])
                if res:
                    hv = max(b['h'] for b in res)
                    lv = min(b['l'] for b in res)
                    w52[sym] = {'high': hv, 'low': lv, 'range': hv - lv,
                                'current': res[-1]['c']}
                    ok += 1
                else:
                    w52[sym] = {'high': None, 'low': None, 'range': None, 'current': None}
                    fail += 1
            else:
                w52[sym] = {'high': None, 'low': None, 'range': None, 'current': None}
                fail += 1
            if (i + 1) % 50 == 0:
                print(f"   52W: {i + 1}/{len(config.symbols)} (✓{ok} ✗{fail})")
            time.sleep(0.13)
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
            time.sleep(0.13)
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
            self.direction = Direction.LONG if self.last_bit == 1 else Direction.SHORT
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
            distance_to_tp_pct = None
            distance_to_sl_pct = None
            stasis_price_change_pct = None

            if si is not None:
                stasis_price_change_pct = si.get_price_change_pct(p)

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
                else:
                    rr = None
                if p > 0:
                    distance_to_tp_pct = (abs(tp - p) / p) * 100
                    distance_to_sl_pct = (abs(sl - p) / p) * 100

            recent_bits = [b.bit for b in list(self.bits)[-15:]]

            return {
                'symbol': self.symbol,
                'is_etf': self.is_etf,
                'threshold': self.threshold,
                'threshold_pct': self.threshold * 100,
                'stasis': self.current_stasis,
                'total_bits': self.total_bits,
                'recent_bits': recent_bits,
                'current_price': p,
                'anchor_price': si.start_price if si else None,
                'direction': self.direction.value if self.direction else None,
                'signal_strength': self.signal_strength.value if self.signal_strength else None,
                'is_tradable': (self.current_stasis >= config.min_tradable_stasis
                                and self.direction is not None and self.volume > 1.0),
                'stasis_start_str': si.get_start_date_str() if si else "—",
                'stasis_duration_str': si.get_duration_str() if si else "—",
                'duration_seconds': si.get_duration().total_seconds() if si else 0,
                'stasis_price_change_pct': stasis_price_change_pct,
                'take_profit': tp,
                'stop_loss': sl,
                'risk_reward': rr,
                'distance_to_tp_pct': distance_to_tp_pct,
                'distance_to_sl_pct': distance_to_sl_pct,
                'week52_percentile': calculate_52week_percentile(p, self.symbol),
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
        print("🔌 WebSocket starting...")

    def _loop(self):
        while self.is_running:
            try:
                self._connect()
            except Exception as e:
                print(f"WS reconnect err: {e}")
                time.sleep(5)

    def _connect(self):
        def on_msg(ws, raw):
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    for m in data:
                        self._proc(m)
                else:
                    self._proc(data)
            except:
                pass

        def on_open(ws):
            print("✅ WS connected, authenticating...")
            ws.send(json.dumps({"action": "auth", "params": config.polygon_api_key}))

        def on_error(ws, err):
            print(f"WS error: {err}")

        self.ws = websocket.WebSocketApp(
            config.polygon_ws_url,
            on_open=on_open, on_message=on_msg, on_error=on_error
        )
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def _proc(self, msg):
        ev = msg.get('ev')
        if ev == 'status':
            status = msg.get('status', '')
            print(f"   WS status: {status} - {msg.get('message', '')}")
            if status == 'auth_success':
                self._subscribe()
            elif status == 'auth_failed':
                print("   ❌ AUTH FAILED — check API key")
        elif ev in ('A', 'AM', 'T', 'Q'):
            sym = msg.get('sym', '') or msg.get('S', '')
            price = msg.get('c') or msg.get('vw') or msg.get('p') or msg.get('bp')
            if price and sym in self.current_prices:
                with self.lock:
                    self.current_prices[sym] = float(price)
                    self.message_count += 1

    def _subscribe(self):
        syms = list(config.symbols)
        for i in range(0, len(syms), 50):
            batch = syms[i:i + 50]
            self.ws.send(json.dumps({
                "action": "subscribe",
                "params": ",".join(f"A.{s}" for s in batch)
            }))
            time.sleep(0.1)
        print(f"📡 Subscribed to {len(syms)} symbols")

    def get_prices(self):
        with self.lock:
            return {k: v for k, v in self.current_prices.items() if v is not None}

    def get_status(self):
        with self.lock:
            return {
                'connected': sum(1 for v in self.current_prices.values() if v is not None),
                'total': len(config.symbols),
                'messages': self.message_count
            }


price_feed = PolygonPriceFeed()

# ============================================================================
# BITSTREAM MANAGER (PM-only version)
# ============================================================================


class BitstreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.streams: Dict[Tuple[str, float], Bitstream] = {}
        self.is_running = False
        self.cached_pm_data: List[Dict] = []
        self.cached_pm_merit: Dict[str, Dict] = {}
        self.cached_pm_jackpots: Dict[str, Dict] = {}
        self.cache_lock = threading.Lock()
        self.initialized = False
        self.backfill_complete = False
        self.backfill_progress = 0
        self.stream_count = 0
        self.tradable_count = 0
        self.jackpot_history: List[Dict] = []
        self.recent_jackpots: deque = deque(maxlen=10)

    def backfill(self):
        print("\n" + "=" * 60 + "\n📜 BACKFILLING HISTORICAL DATA\n" + "=" * 60)
        hist = {}
        for i, sym in enumerate(config.symbols):
            bars = fetch_historical_bars(sym, config.history_days)
            if bars:
                hist[sym] = bars
            self.backfill_progress = int((i + 1) / len(config.symbols) * 100)
            if (i + 1) % 25 == 0:
                print(f"   📊 {i + 1}/{len(config.symbols)} ({self.backfill_progress}%)"
                      f" — {len(hist)} with data")
            time.sleep(0.13)

        print(f"\n   Building bitstreams from {len(hist)} symbols...")
        with self.lock:
            for sym, bars in hist.items():
                if not bars or len(bars) < 2:
                    continue
                vol = config.volumes.get(sym, 10.0)
                for th in config.thresholds:
                    key = (sym, th)
                    self.streams[key] = Bitstream(sym, th, bars[0]['close'], vol)
                    for bar in bars:
                        self.streams[key].process_price(bar['close'], bar['timestamp'])

        self.stream_count = len(self.streams)
        self.tradable_count = sum(1 for s in self.streams.values()
                                  if s.current_stasis >= config.min_tradable_stasis
                                  and s.direction is not None and s.volume > 1.0)
        self.initialized = True
        self.backfill_complete = True
        print(f"✅ Streams: {self.stream_count} | Tradable: {self.tradable_count}")
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

            pm_merit = self._calculate_pm_merit_scores(snaps)

            pm_jackpots = {}
            for symbol, merit_data in pm_merit.items():
                jackpot_status = calculate_jackpot_status(merit_data)
                pm_jackpots[symbol] = jackpot_status

                if jackpot_status['is_jackpot']:
                    jackpot_entry = {
                        'symbol': symbol,
                        'tier': jackpot_status['tier'],
                        'levels': merit_data.get('stasis_levels', 0),
                        'direction': merit_data.get('dominant_direction'),
                        'timestamp': datetime.now(),
                    }
                    existing = [j for j in self.recent_jackpots if j['symbol'] == symbol]
                    if not existing or existing[-1]['tier'] != jackpot_status['tier']:
                        self.recent_jackpots.append(jackpot_entry)

            self.tradable_count = sum(1 for s in snaps
                                      if s.get('is_tradable')
                                      and s['threshold'] in config.pm_thresholds)

            with self.cache_lock:
                self.cached_pm_data = snaps
                self.cached_pm_merit = pm_merit
                self.cached_pm_jackpots = pm_jackpots

    def _calculate_pm_merit_scores(self, snapshots):
        THRESHOLD_WEIGHTS = {
            0.000625: 1.0, 0.00125: 1.5, 0.0025: 2.0, 0.005: 3.0,
            0.01: 5.0, 0.02: 8.0, 0.03: 12.0, 0.04: 16.0, 0.05: 20.0, 0.10: 30.0,
        }

        def get_threshold_weight(threshold):
            if threshold in THRESHOLD_WEIGHTS:
                return THRESHOLD_WEIGHTS[threshold]
            return max(1.0, threshold * 200)

        def calculate_level_weight(stasis, threshold):
            stasis_factor = stasis ** 1.3
            threshold_factor = get_threshold_weight(threshold)
            return stasis_factor * threshold_factor

        merit_data = defaultdict(lambda: {
            'stasis_levels': 0, 'total_stasis': 0, 'weighted_score': 0,
            'directions': [], 'dominant_direction': None, 'direction_alignment': 0,
            'conflict_penalty': 0, 'alignment_bonus': 0, 'avg_stasis': 0,
            'max_stasis': 0, 'max_threshold_in_stasis': 0, 'long_levels': 0,
            'short_levels': 0, 'long_weight': 0, 'short_weight': 0,
            'levels_detail': [], 'thresholds_in_stasis': [], 'net_direction': None,
            'confidence': 0, 'highest_aligned_threshold': 0,
        })

        symbol_snapshots = defaultdict(list)
        for snap in snapshots:
            if snap['threshold'] in config.pm_thresholds:
                symbol_snapshots[snap['symbol']].append(snap)

        for symbol, snaps in symbol_snapshots.items():
            long_levels = []
            short_levels = []
            all_stasis = []
            all_thresholds_in_stasis = []

            for snap in snaps:
                stasis = snap['stasis']
                direction = snap['direction']
                threshold = snap['threshold']
                all_stasis.append(stasis)

                if stasis >= config.min_tradable_stasis and direction:
                    weight = calculate_level_weight(stasis, threshold)
                    level_data = {
                        'threshold': threshold,
                        'threshold_pct': threshold * 100,
                        'stasis': stasis,
                        'direction': direction,
                        'weight': weight,
                        'threshold_weight': get_threshold_weight(threshold),
                    }
                    all_thresholds_in_stasis.append(threshold)
                    if direction == 'LONG':
                        long_levels.append(level_data)
                    else:
                        short_levels.append(level_data)

            long_weight = sum(level['weight'] for level in long_levels)
            short_weight = sum(level['weight'] for level in short_levels)
            long_stasis_sum = sum(level['stasis'] for level in long_levels)
            short_stasis_sum = sum(level['stasis'] for level in short_levels)
            total_levels = len(long_levels) + len(short_levels)
            total_weight = long_weight + short_weight
            max_long_threshold = max([l['threshold'] for l in long_levels]) if long_levels else 0
            max_short_threshold = max([l['threshold'] for l in short_levels]) if short_levels else 0
            max_threshold_in_stasis = max(max_long_threshold, max_short_threshold)

            if total_levels == 0:
                merit_data[symbol] = {
                    'stasis_levels': 0, 'total_stasis': sum(all_stasis), 'weighted_score': 0,
                    'directions': [], 'dominant_direction': None, 'direction_alignment': 0,
                    'conflict_penalty': 0, 'alignment_bonus': 0, 'avg_stasis': 0,
                    'max_stasis': max(all_stasis) if all_stasis else 0,
                    'max_threshold_in_stasis': 0, 'long_levels': 0, 'short_levels': 0,
                    'long_weight': 0, 'short_weight': 0, 'levels_detail': [],
                    'thresholds_in_stasis': [], 'net_direction': None, 'confidence': 0,
                    'highest_aligned_threshold': 0,
                }
                continue

            if long_weight > short_weight:
                dominant_direction = 'LONG'
                net_weight = long_weight - short_weight
                aligned_levels = long_levels
                opposing_levels = short_levels
                aligned_weight = long_weight
                opposing_weight = short_weight
                highest_aligned_threshold = max_long_threshold
            elif short_weight > long_weight:
                dominant_direction = 'SHORT'
                net_weight = short_weight - long_weight
                aligned_levels = short_levels
                opposing_levels = long_levels
                aligned_weight = short_weight
                opposing_weight = long_weight
                highest_aligned_threshold = max_short_threshold
            else:
                dominant_direction = None
                net_weight = 0
                aligned_levels = []
                opposing_levels = long_levels + short_levels
                aligned_weight = 0
                opposing_weight = total_weight
                highest_aligned_threshold = 0

            if total_weight > 0:
                alignment_pct = (max(long_weight, short_weight) / total_weight) * 100
            else:
                alignment_pct = 0

            base_score = net_weight
            num_aligned = len(aligned_levels)
            num_opposing = len(opposing_levels)

            if alignment_pct == 100 and num_aligned >= 2:
                threshold_multiplier = 1 + (get_threshold_weight(highest_aligned_threshold) / 30)
                alignment_bonus = base_score * (0.2 * (num_aligned - 1)) * threshold_multiplier
                if num_aligned >= 3:
                    alignment_bonus *= 1.3
                if num_aligned >= 4:
                    alignment_bonus *= 1.2
                if num_aligned >= 5:
                    alignment_bonus *= 1.2
            elif alignment_pct >= 75:
                alignment_bonus = base_score * (0.1 * (num_aligned - 1))
            elif alignment_pct >= 60:
                alignment_bonus = base_score * 0.05
            else:
                alignment_bonus = 0

            if num_opposing > 0 and total_weight > 0:
                conflict_ratio = opposing_weight / total_weight
                max_opposing_threshold = max(
                    [l['threshold'] for l in opposing_levels]) if opposing_levels else 0
                high_threshold_conflict = max_opposing_threshold >= 0.01
                if conflict_ratio > 0.4:
                    conflict_penalty = base_score * 0.6
                    if high_threshold_conflict:
                        conflict_penalty *= 1.3
                elif conflict_ratio > 0.25:
                    conflict_penalty = base_score * 0.35
                    if high_threshold_conflict:
                        conflict_penalty *= 1.2
                elif conflict_ratio > 0.1:
                    conflict_penalty = base_score * 0.15
                else:
                    conflict_penalty = base_score * 0.05
            else:
                conflict_penalty = 0

            high_threshold_bonus = 0
            for level in aligned_levels:
                if level['threshold'] >= 0.05:
                    high_threshold_bonus += level['weight'] * 0.25
                elif level['threshold'] >= 0.02:
                    high_threshold_bonus += level['weight'] * 0.1
                elif level['threshold'] >= 0.01:
                    high_threshold_bonus += level['weight'] * 0.05

            max_stasis = max(all_stasis) if all_stasis else 0
            max_aligned_stasis = max(
                [l['stasis'] for l in aligned_levels]) if aligned_levels else 0

            if max_aligned_stasis >= 10:
                stasis_depth_bonus = base_score * 0.2
            elif max_aligned_stasis >= 7:
                stasis_depth_bonus = base_score * 0.1
            elif max_aligned_stasis >= 5:
                stasis_depth_bonus = base_score * 0.05
            else:
                stasis_depth_bonus = 0

            weighted_score = max(0, base_score + alignment_bonus + high_threshold_bonus
                                 + stasis_depth_bonus - conflict_penalty)

            if dominant_direction and aligned_levels:
                alignment_component = (alignment_pct / 100) * 40
                stasis_component = min(max_aligned_stasis / 10, 1.0) * 30
                threshold_component = min(
                    get_threshold_weight(highest_aligned_threshold) / 20, 1.0) * 30
                confidence = alignment_component + stasis_component + threshold_component
            else:
                confidence = 0

            all_directions = [l['direction'] for l in long_levels + short_levels]
            all_levels = sorted(long_levels + short_levels,
                                key=lambda x: x['threshold'], reverse=True)

            merit_data[symbol] = {
                'stasis_levels': total_levels,
                'total_stasis': sum(all_stasis),
                'weighted_score': round(weighted_score, 1),
                'directions': all_directions,
                'dominant_direction': dominant_direction,
                'direction_alignment': round(alignment_pct, 0),
                'conflict_penalty': round(conflict_penalty, 1),
                'alignment_bonus': round(alignment_bonus + high_threshold_bonus
                                         + stasis_depth_bonus, 1),
                'avg_stasis': round((long_stasis_sum + short_stasis_sum) / total_levels,
                                    1) if total_levels > 0 else 0,
                'max_stasis': max_stasis,
                'max_threshold_in_stasis': max_threshold_in_stasis,
                'max_threshold_pct': max_threshold_in_stasis * 100,
                'long_levels': len(long_levels),
                'short_levels': len(short_levels),
                'long_weight': round(long_weight, 1),
                'short_weight': round(short_weight, 1),
                'levels_detail': all_levels,
                'thresholds_in_stasis': sorted(all_thresholds_in_stasis, reverse=True),
                'net_direction': dominant_direction,
                'confidence': round(confidence, 0),
                'highest_aligned_threshold': highest_aligned_threshold,
                'highest_aligned_threshold_pct': highest_aligned_threshold * 100,
            }

        return dict(merit_data)

    def get_pm_data(self):
        with self.cache_lock:
            return copy.deepcopy(self.cached_pm_data)

    def get_pm_merit(self):
        with self.cache_lock:
            return copy.deepcopy(self.cached_pm_merit)

    def get_pm_jackpots(self):
        with self.cache_lock:
            return copy.deepcopy(self.cached_pm_jackpots)

    def get_recent_jackpots(self):
        return list(self.recent_jackpots)


manager = BitstreamManager()

# ============================================================================
# SELECTED SYMBOL STATE
# ============================================================================

_selected_symbol = {'symbol': None, 'lock': threading.Lock()}


def set_selected_symbol(sym):
    with _selected_symbol['lock']:
        _selected_symbol['symbol'] = sym


def get_selected_symbol():
    with _selected_symbol['lock']:
        return _selected_symbol['symbol']


# ============================================================================
# DASH PM APP — STASIS PM (JACKPOT EDITION)
# ============================================================================

PM_CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

body { 
    background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0a0a0f 100%) !important;
    background-attachment: fixed !important;
}

body::before {
    content: '';
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: 
        radial-gradient(ellipse at 20% 80%, rgba(255, 0, 255, 0.1) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(0, 255, 136, 0.1) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(0, 255, 255, 0.05) 0%, transparent 70%);
    pointer-events: none; z-index: -1;
}

.title-font { font-family: 'Orbitron', sans-serif !important; }
.arcade-font { font-family: 'Press Start 2P', cursive !important; }
.data-font { font-family: 'Roboto Mono', monospace !important; }
h1, h2, h3, h4, h5, h6, .btn, label, .nav-link, .card-header { font-family: 'Orbitron', sans-serif !important; }
td, input, .form-control, pre, code { font-family: 'Roboto Mono', monospace !important; }

@keyframes jackpot-glow {
    0%, 100% { box-shadow: 0 0 5px #ffd700, 0 0 10px #ffd700, 0 0 15px #ffd700; }
    50% { box-shadow: 0 0 10px #ffd700, 0 0 20px #ffd700, 0 0 30px #ffd700, 0 0 40px #ffd700; }
}
@keyframes mega-jackpot {
    0%, 100% { box-shadow: 0 0 10px #ff00ff, 0 0 20px #ff00ff, 0 0 30px #ff00ff; transform: scale(1); }
    50% { box-shadow: 0 0 20px #ff00ff, 0 0 40px #ff00ff, 0 0 60px #ff00ff, 0 0 80px #ff00ff; transform: scale(1.02); }
}
@keyframes rainbow-border {
    0% { border-color: #ff0000; } 17% { border-color: #ff8800; }
    33% { border-color: #ffff00; } 50% { border-color: #00ff00; }
    67% { border-color: #0088ff; } 83% { border-color: #8800ff; }
    100% { border-color: #ff0000; }
}
@keyframes flash-celebration { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

.jackpot-card { animation: jackpot-glow 1.5s ease-in-out infinite; border: 2px solid #ffd700 !important; }
.mega-jackpot-card { animation: mega-jackpot 1s ease-in-out infinite; border: 3px solid #ff00ff !important; }
.grand-jackpot-card { animation: mega-jackpot 0.5s ease-in-out infinite, rainbow-border 2s linear infinite; border: 4px solid #ff00ff !important; }

.slot-container { display: inline-flex; gap: 4px; padding: 4px 8px; background: linear-gradient(180deg, #2a2a4e 0%, #1a1a2e 100%); border-radius: 8px; border: 2px solid #ffd700; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5); }
.slot-reel { width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-size: 18px; background: #0a0a0f; border-radius: 4px; border: 1px solid #444; }
.achievement-badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 9px; font-weight: bold; margin: 1px; }
.achievement-legendary { background: linear-gradient(135deg, #ff8000, #ffd700); color: #000; animation: jackpot-glow 1s ease-in-out infinite; }
.achievement-epic { background: linear-gradient(135deg, #a335ee, #cc77ff); color: #fff; }
.achievement-rare { background: linear-gradient(135deg, #0070dd, #00aaff); color: #fff; }

.leaderboard-item { background: linear-gradient(90deg, rgba(0,255,136,0.1) 0%, transparent 100%); border-left: 3px solid #00ff88; padding: 8px 12px; margin: 4px 0; transition: all 0.3s ease; }
.leaderboard-item:hover { background: linear-gradient(90deg, rgba(0,255,136,0.2) 0%, transparent 100%); transform: translateX(5px); }
.leaderboard-rank { font-size: 18px; font-weight: bold; color: #ffd700; }

.neon-green { color: #00ff88; text-shadow: 0 0 5px #00ff88, 0 0 10px #00ff88; }
.neon-red { color: #ff4444; text-shadow: 0 0 5px #ff4444, 0 0 10px #ff4444; }
.neon-gold { color: #ffd700; text-shadow: 0 0 5px #ffd700, 0 0 10px #ffd700, 0 0 15px #ffd700; }
.neon-purple { color: #ff00ff; text-shadow: 0 0 5px #ff00ff, 0 0 10px #ff00ff, 0 0 15px #ff00ff; }
.hover-glow:hover { box-shadow: 0 0 15px rgba(0, 255, 136, 0.5); transition: box-shadow 0.3s ease; }

@keyframes count-up { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.stat-value { animation: count-up 0.3s ease-out; }
"""

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.CYBORG],
    title="🎰 STASIS PM",
)

server = app.server

# Add CORS headers
@server.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', ALLOWED_ORIGINS)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>''' + PM_CUSTOM_CSS + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


def pm_get_table_data():
    data = manager.get_pm_data()
    merit_scores = manager.get_pm_merit()
    jackpots = manager.get_pm_jackpots()

    if not data:
        return pd.DataFrame()

    rows = []
    for d in data:
        if d['threshold'] not in config.pm_thresholds:
            continue
        symbol = d['symbol']
        merit = merit_scores.get(symbol, {})
        jackpot = jackpots.get(symbol, {})

        chg_str = "—"
        if d.get('stasis_price_change_pct') is not None:
            sign = "+" if d['stasis_price_change_pct'] >= 0 else ""
            chg_str = f"{sign}{d['stasis_price_change_pct']:.2f}%"

        w52_str = "—"
        if d.get('week52_percentile') is not None:
            w52_str = f"{d['week52_percentile']:.0f}%"

        merit_score = merit.get('weighted_score', 0)
        stasis_levels = merit.get('stasis_levels', 0)
        alignment = merit.get('direction_alignment', 0)
        jackpot_emoji = jackpot.get('emoji', '⬜')
        heat_level = jackpot.get('heat_level', 0)

        if stasis_levels > 0:
            merit_str = f"{merit_score:.0f}"
        else:
            merit_str = "—"

        type_str = "ETF" if d['is_etf'] else "STK"

        rows.append({
            'Type': type_str,
            'Symbol': symbol,
            'Jackpot': jackpot_emoji,
            'Jackpot_Tier': jackpot.get('tier_name', 'NO SIGNAL'),
            'Heat': heat_level,
            'Merit': merit_str,
            'Merit_Val': merit_score,
            'Levels': stasis_levels,
            'Align': f"{alignment:.0f}%" if stasis_levels > 0 else "—",
            'Align_Val': alignment,
            'Band': format_band(d['threshold_pct']),
            'Band_Val': d['threshold'],
            'Stasis': d['stasis'],
            'Dir': d['direction'] or '—',
            'Str': d['signal_strength'] or '—',
            'Current': f"${d['current_price']:.2f}" if d['current_price'] else "—",
            'Current_Val': d['current_price'] or 0,
            'Anchor': f"${d['anchor_price']:.2f}" if d.get('anchor_price') else "—",
            'TP': f"${d['take_profit']:.2f}" if d.get('take_profit') else "—",
            'SL': f"${d['stop_loss']:.2f}" if d.get('stop_loss') else "—",
            'R:R': fmt_rr(d.get('risk_reward')),
            'RR_Val': d['risk_reward'] if d.get('risk_reward') is not None else -1,
            '→TP': f"{d['distance_to_tp_pct']:.3f}%" if d.get('distance_to_tp_pct') else "—",
            '→SL': f"{d['distance_to_sl_pct']:.3f}%" if d.get('distance_to_sl_pct') else "—",
            'Started': d['stasis_start_str'],
            'Duration': d['stasis_duration_str'],
            'Dur_Val': d['duration_seconds'],
            'Chg': chg_str,
            'Chg_Val': d.get('stasis_price_change_pct') if d.get('stasis_price_change_pct') else 0,
            '52W': w52_str,
            '52W_Val': d['week52_percentile'] if d.get('week52_percentile') is not None else -1,
            'Bits': d.get('total_bits', 0),
            'Recent': format_bits(d.get('recent_bits', [])),
            'Tradable': '✅' if d['is_tradable'] else '',
            'Is_Tradable': d['is_tradable'],
            'Is_ETF': d['is_etf'],
            'Is_Jackpot': jackpot.get('is_jackpot', False),
        })

    return pd.DataFrame(rows)


# Main PM layout
app.layout = dbc.Container([
    dbc.Toast(
        id="jackpot-toast", header="🎰 JACKPOT! 🎰", icon="warning",
        is_open=False, dismissable=True, duration=8000,
        style={
            "position": "fixed", "top": 66, "right": 10, "width": 400, "zIndex": 9999,
            "backgroundColor": "#1a1a2e", "border": "3px solid #ffd700",
            "boxShadow": "0 0 30px rgba(255, 215, 0, 0.5)",
        },
        header_style={"color": "#ffd700", "fontFamily": "Orbitron",
                       "fontWeight": "bold", "fontSize": "16px"}
    ),

    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H1("🎰", style={'fontSize': '40px', 'marginRight': '10px'}),
                    html.Div([
                        html.H2("STASIS PM", className="neon-green mb-0 title-font",
                                style={'fontSize': '20px', 'fontWeight': '700',
                                       'letterSpacing': '3px'}),
                        html.P("JACKPOT EDITION • MULTI-LEVEL ALIGNMENT DETECTOR",
                               className="text-warning arcade-font",
                               style={'fontSize': '8px', 'letterSpacing': '1px',
                                      'marginBottom': '0'}),
                    ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
                ], style={'display': 'flex', 'alignItems': 'center'})
            ])
        ], width=5),
        dbc.Col([html.Div(id='jackpot-counter', className="text-center")], width=3),
        dbc.Col([html.Div(id='connection-status', className="text-end")], width=2),
        dbc.Col([html.Div(id='stats-summary', className="text-end",
                          style={'fontSize': '10px'})], width=2)
    ], className="mb-2 mt-2", style={'alignItems': 'center'}),

    # Jackpot Display Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H3("🎰 JACKPOT SCANNER 🎰",
                                className="text-center neon-gold arcade-font mb-0",
                                style={'fontSize': '16px', 'letterSpacing': '2px'}),
                        html.P("Detecting multi-level alignment...",
                               className="text-center text-muted",
                               style={'fontSize': '9px', 'marginBottom': '0'}),
                    ], className="mb-2"),
                    html.Div(id='current-jackpots', className="text-center"),
                ], style={'padding': '10px'})
            ], style={
                'backgroundColor': 'rgba(26, 26, 46, 0.9)',
                'border': '2px solid #ffd700', 'borderRadius': '10px',
                'boxShadow': '0 0 20px rgba(255, 215, 0, 0.3)',
            })
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🏆 LEADERBOARD", className="title-font neon-gold mb-2",
                             style={'fontSize': '12px'}),
                    html.Div(id='leaderboard-display'),
                ], style={'padding': '10px', 'maxHeight': '300px', 'overflowY': 'auto'})
            ], style={
                'backgroundColor': 'rgba(26, 26, 46, 0.9)',
                'border': '1px solid #ffd700', 'borderRadius': '8px',
            })
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🏅 ACHIEVEMENTS", className="title-font text-info mb-2",
                             style={'fontSize': '12px'}),
                    html.Div(id='achievements-display'),
                ], style={'padding': '10px'})
            ], style={
                'backgroundColor': 'rgba(26, 26, 46, 0.9)',
                'border': '1px solid #00ffff', 'borderRadius': '8px',
            })
        ], width=3),
    ], className="mb-2"),

    # Stats Bar
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div(id='stats-display', className="text-center")
                ], style={'padding': '6px'})
            ], style={'backgroundColor': 'rgba(26, 42, 58, 0.9)',
                      'border': '1px solid #00ff88'})
        ])
    ], className="mb-2"),

    # Filters
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("ALL", id="btn-all", color="secondary", outline=True, size="sm",
                           className="title-font",
                           style={'letterSpacing': '1px', 'fontSize': '9px', 'padding': '4px 8px'}),
                dbc.Button("🎰 JACKPOT", id="btn-jackpot", color="warning", outline=True,
                           size="sm", className="title-font",
                           style={'letterSpacing': '1px', 'fontSize': '9px', 'padding': '4px 8px'}),
                dbc.Button("TRADE", id="btn-tradable", color="success", outline=True, size="sm",
                           active=True, className="title-font",
                           style={'letterSpacing': '1px', 'fontSize': '9px', 'padding': '4px 8px'}),
            ], size="sm")
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-type',
                         options=[{'label': 'ALL', 'value': 'ALL'},
                                  {'label': 'ETF', 'value': 'ETF'},
                                  {'label': 'STK', 'value': 'STK'}],
                         value='ALL', clearable=False,
                         style={'fontSize': '10px', 'minWidth': '70px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-symbol',
                         options=[{'label': 'ALL', 'value': 'ALL'}] +
                                 [{'label': s, 'value': s} for s in config.symbols],
                         value='ALL', clearable=False,
                         style={'fontSize': '10px', 'minWidth': '80px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-threshold',
                         options=[{'label': 'ALL BANDS', 'value': 'ALL'}] +
                                 [{'label': format_band(t * 100), 'value': t}
                                  for t in config.pm_thresholds],
                         value='ALL', clearable=False,
                         style={'fontSize': '10px', 'minWidth': '90px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Input(id='filter-stasis', type='number', value=3, min=0,
                      placeholder="Min",
                      style={'width': '55px', 'fontSize': '10px',
                             'fontFamily': 'Roboto Mono', 'padding': '5px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-direction',
                         options=[{'label': 'DIR', 'value': 'ALL'},
                                  {'label': '📈 LONG', 'value': 'LONG'},
                                  {'label': '📉 SHORT', 'value': 'SHORT'}],
                         value='ALL', clearable=False,
                         style={'fontSize': '10px', 'minWidth': '80px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-merit',
                         options=[{'label': 'MERIT', 'value': 0},
                                  {'label': '≥10', 'value': 10},
                                  {'label': '≥25', 'value': 25},
                                  {'label': '≥50', 'value': 50},
                                  {'label': '≥100', 'value': 100}],
                         value=0, clearable=False,
                         style={'fontSize': '10px', 'minWidth': '70px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-alignment',
                         options=[{'label': 'ALIGN', 'value': 0},
                                  {'label': '100%', 'value': 100},
                                  {'label': '≥75%', 'value': 75},
                                  {'label': '≥50%', 'value': 50}],
                         value=0, clearable=False,
                         style={'fontSize': '10px', 'minWidth': '70px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-rows',
                         options=[{'label': '50', 'value': 50},
                                  {'label': '100', 'value': 100},
                                  {'label': '250', 'value': 250},
                                  {'label': 'ALL', 'value': 5000}],
                         value=100, clearable=False,
                         style={'fontSize': '10px', 'minWidth': '60px'})
        ], width="auto", style={'paddingRight': '5px'}),
        dbc.Col([
            dcc.Dropdown(id='filter-sort',
                         options=[{'label': '🔥 HEAT↓', 'value': 'heat'},
                                  {'label': 'MERIT↓', 'value': 'merit'},
                                  {'label': 'STASIS↓', 'value': 'stasis'},
                                  {'label': 'R:R↓', 'value': 'rr'}],
                         value='merit', clearable=False,
                         style={'fontSize': '10px', 'minWidth': '80px'})
        ], width="auto"),
    ], className="mb-2", style={'flexWrap': 'nowrap', 'overflowX': 'auto'}),

    html.Div("💡 Click any row → Desktop app navigates SA, RH & TT to that stock",
             style={'fontSize': '9px', 'color': '#aa6600', 'padding': '2px 6px',
                    'background': 'rgba(26, 26, 46, 0.9)', 'marginBottom': '4px'}),

    # Table
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='main-table',
                row_selectable='single',
                columns=[
                    {'name': '🎰', 'id': 'Jackpot', 'sortable': False},
                    {'name': 'T', 'id': 'Type', 'sortable': True},
                    {'name': 'SYM', 'id': 'Symbol', 'sortable': True},
                    {'name': '🔥', 'id': 'Heat', 'sortable': True},
                    {'name': 'MERIT', 'id': 'Merit', 'sortable': True},
                    {'name': 'LVL', 'id': 'Levels', 'sortable': True},
                    {'name': 'ALIGN', 'id': 'Align', 'sortable': True},
                    {'name': 'BAND', 'id': 'Band', 'sortable': True},
                    {'name': 'STS', 'id': 'Stasis', 'sortable': True},
                    {'name': 'DIR', 'id': 'Dir', 'sortable': True},
                    {'name': 'STR', 'id': 'Str', 'sortable': True},
                    {'name': 'CURRENT', 'id': 'Current', 'sortable': True},
                    {'name': 'TP', 'id': 'TP', 'sortable': True},
                    {'name': 'SL', 'id': 'SL', 'sortable': True},
                    {'name': 'R:R', 'id': 'R:R', 'sortable': True},
                    {'name': 'DUR', 'id': 'Duration', 'sortable': True},
                    {'name': 'CHG', 'id': 'Chg', 'sortable': True},
                    {'name': '52W', 'id': '52W', 'sortable': True},
                    {'name': 'BITS', 'id': 'Recent', 'sortable': False},
                ],
                sort_action='native', sort_mode='multi',
                sort_by=[{'column_id': 'Merit', 'direction': 'desc'}],
                style_table={'height': '50vh', 'overflowY': 'auto'},
                style_cell={
                    'backgroundColor': '#1a1a2e', 'color': 'white',
                    'padding': '3px 5px', 'fontSize': '10px',
                    'fontFamily': 'Roboto Mono, monospace',
                    'whiteSpace': 'nowrap', 'textAlign': 'right', 'minWidth': '40px',
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'Jackpot'}, 'textAlign': 'center',
                     'fontSize': '14px', 'minWidth': '50px'},
                    {'if': {'column_id': 'Type'}, 'textAlign': 'center',
                     'fontWeight': '600', 'minWidth': '35px'},
                    {'if': {'column_id': 'Symbol'}, 'textAlign': 'left',
                     'fontWeight': '700', 'color': '#00ff88'},
                    {'if': {'column_id': 'Heat'}, 'textAlign': 'center', 'fontWeight': '700'},
                    {'if': {'column_id': 'Merit'}, 'textAlign': 'center', 'fontWeight': '700'},
                    {'if': {'column_id': 'Levels'}, 'textAlign': 'center', 'fontWeight': '600'},
                    {'if': {'column_id': 'Align'}, 'textAlign': 'center'},
                    {'if': {'column_id': 'Dir'}, 'textAlign': 'center'},
                    {'if': {'column_id': 'Str'}, 'textAlign': 'center'},
                    {'if': {'column_id': 'Recent'}, 'textAlign': 'left', 'minWidth': '90px'},
                ],
                style_header={
                    'backgroundColor': '#2a2a4e', 'color': '#ffd700',
                    'fontWeight': '700', 'fontSize': '9px',
                    'fontFamily': 'Orbitron, sans-serif',
                    'borderBottom': '2px solid #ffd700',
                    'textAlign': 'center', 'letterSpacing': '0.5px',
                },
                style_data_conditional=[
                    {'if': {'filter_query': '{Type} = "ETF"', 'column_id': 'Type'},
                     'color': '#00ffff'},
                    {'if': {'filter_query': '{Type} = "STK"', 'column_id': 'Type'},
                     'color': '#ffaa00'},
                    {'if': {'filter_query': '{Is_Jackpot} = true'},
                     'backgroundColor': 'rgba(255, 215, 0, 0.15)',
                     'border': '1px solid #ffd700'},
                    {'if': {'filter_query': '{Heat} >= 80', 'column_id': 'Heat'},
                     'color': '#ff0000', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Heat} >= 60 && {Heat} < 80', 'column_id': 'Heat'},
                     'color': '#ff8800'},
                    {'if': {'filter_query': '{Heat} >= 40 && {Heat} < 60', 'column_id': 'Heat'},
                     'color': '#ffff00'},
                    {'if': {'filter_query': '{Heat} >= 20 && {Heat} < 40', 'column_id': 'Heat'},
                     'color': '#88ff00'},
                    {'if': {'filter_query': '{Heat} < 20', 'column_id': 'Heat'},
                     'color': '#00ff88'},
                    {'if': {'filter_query': '{Merit_Val} >= 100', 'column_id': 'Merit'},
                     'color': '#ff00ff', 'fontWeight': 'bold', 'backgroundColor': '#3d2a4d'},
                    {'if': {'filter_query': '{Merit_Val} >= 50 && {Merit_Val} < 100',
                            'column_id': 'Merit'}, 'color': '#ffff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Merit_Val} >= 25 && {Merit_Val} < 50',
                            'column_id': 'Merit'}, 'color': '#00ffff', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Levels} >= 5', 'column_id': 'Levels'},
                     'color': '#ff00ff', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Levels} >= 3 && {Levels} < 5',
                            'column_id': 'Levels'}, 'color': '#ffff00'},
                    {'if': {'filter_query': '{Align_Val} = 100', 'column_id': 'Align'},
                     'color': '#00ff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Align_Val} >= 75 && {Align_Val} < 100',
                            'column_id': 'Align'}, 'color': '#88ff00'},
                    {'if': {'filter_query': '{Merit_Val} >= 50'}, 'backgroundColor': '#2d3a4d'},
                    {'if': {'filter_query': '{Stasis} >= 10'}, 'backgroundColor': '#2d4a2d'},
                    {'if': {'filter_query': '{Stasis} >= 7 && {Stasis} < 10'},
                     'backgroundColor': '#2a3a2a'},
                    {'if': {'filter_query': '{Dir} = "LONG"', 'column_id': 'Dir'},
                     'color': '#00ff00', 'fontWeight': 'bold'},
                    {'if': {'filter_query': '{Dir} = "SHORT"', 'column_id': 'Dir'},
                     'color': '#ff4444', 'fontWeight': 'bold'},
                    {'if': {'column_id': 'Current'}, 'color': '#00ffff', 'fontWeight': '600'},
                    {'if': {'column_id': 'TP'}, 'color': '#00ff00'},
                    {'if': {'column_id': 'SL'}, 'color': '#ff4444'},
                    {'if': {'filter_query': '{RR_Val} >= 2', 'column_id': 'R:R'},
                     'color': '#00ff00', 'fontWeight': '600'},
                    {'if': {'filter_query': '{Chg} contains "+"', 'column_id': 'Chg'},
                     'color': '#00ff00'},
                    {'if': {'filter_query': '{Chg} contains "-"', 'column_id': 'Chg'},
                     'color': '#ff4444'},
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#151520'},
                ]
            )
        ])
    ]),

    # Footer
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span("🎰 JACKPOT",
                           style={'color': '#ffd700', 'fontSize': '9px', 'fontWeight': 'bold'}),
                html.Span(" = ALL LEVELS ALIGNED | ", style={'fontSize': '8px'}),
                html.Span("🔥 HEAT", style={'color': '#ff8800', 'fontSize': '9px'}),
                html.Span(" = Alignment Intensity | ", style={'fontSize': '8px'}),
                html.Span("📈 LONG", style={'color': '#00ff00', 'fontSize': '8px'}),
                html.Span("/", style={'fontSize': '8px'}),
                html.Span("📉 SHORT", style={'color': '#ff4444', 'fontSize': '8px'}),
                html.Span(" | ", style={'fontSize': '8px'}),
                html.Span("🏆 Hit 7-7-7 for MEGA JACKPOT!",
                           style={'color': '#ff00ff', 'fontSize': '8px'}),
            ], className="text-center text-muted mt-1"),
            html.Hr(style={'borderColor': '#333', 'margin': '5px 0'}),
            html.P("© 2026 TRUTH COMMUNICATIONS LLC • STASIS PM",
                   className="text-muted text-center mb-0 title-font",
                   style={'fontSize': '8px', 'letterSpacing': '1px'}),
        ])
    ]),

    dcc.Store(id='view-mode', data='tradable'),
    dcc.Store(id='last-jackpot', data=None),
    dcc.Interval(id='refresh-interval', interval=500, n_intervals=0),

], fluid=True, className="p-2", style={'backgroundColor': 'transparent'})


# ============================================================================
# PM CALLBACKS
# ============================================================================

@app.callback(
    [Output('btn-all', 'active'), Output('btn-jackpot', 'active'),
     Output('btn-tradable', 'active'), Output('view-mode', 'data')],
    [Input('btn-all', 'n_clicks'), Input('btn-jackpot', 'n_clicks'),
     Input('btn-tradable', 'n_clicks')],
    [State('view-mode', 'data')]
)
def pm_toggle_view(n1, n2, n3, current):
    ctx = callback_context
    if not ctx.triggered:
        return False, False, True, 'tradable'
    btn = ctx.triggered[0]['prop_id'].split('.')[0]
    if btn == 'btn-all':
        return True, False, False, 'all'
    elif btn == 'btn-jackpot':
        return False, True, False, 'jackpot'
    return False, False, True, 'tradable'


@app.callback(
    Output('connection-status', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def pm_update_status(n):
    if not manager.backfill_complete:
        return html.Span(f"⏳ LOADING {manager.backfill_progress}%",
                         className="text-warning title-font", style={'fontSize': '11px'})
    status = price_feed.get_status()
    if status['connected'] == 0:
        return html.Span("🔴 CONNECTING", className="text-warning",
                         style={'fontSize': '11px'})
    elif status['connected'] < status['total']:
        return html.Span(f"🟡 {status['connected']}/{status['total']}",
                         className="text-info data-font", style={'fontSize': '11px'})
    return html.Span(f"🟢 LIVE {status['messages']:,}",
                     className="text-success data-font", style={'fontSize': '11px'})


@app.callback(
    Output('jackpot-counter', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def pm_update_jackpot_counter(n):
    if not manager.backfill_complete:
        return ""
    jackpots = manager.get_pm_jackpots()
    grand = sum(1 for j in jackpots.values() if j.get('tier') == 'GRAND_JACKPOT')
    mega = sum(1 for j in jackpots.values() if j.get('tier') == 'MEGA_JACKPOT')
    super_jp = sum(1 for j in jackpots.values() if j.get('tier') == 'SUPER_JACKPOT')
    jackpot = sum(1 for j in jackpots.values() if j.get('tier') == 'JACKPOT')
    big_win = sum(1 for j in jackpots.values() if j.get('tier') == 'BIG_WIN')
    total_jackpots = grand + mega + super_jp + jackpot + big_win
    return html.Div([
        html.Span("🎰 ACTIVE JACKPOTS: ", className="title-font",
                   style={'fontSize': '10px', 'color': '#ffd700'}),
        html.Span(f"{total_jackpots}", className="neon-gold",
                  style={'fontSize': '18px', 'fontWeight': 'bold',
                         'fontFamily': 'Orbitron'}),
        html.Div([
            html.Span(f"💎{grand} ",
                       style={'color': '#ff00ff', 'fontSize': '9px'}) if grand > 0 else None,
            html.Span(f"🎰{mega} ",
                       style={'color': '#ffff00', 'fontSize': '9px'}) if mega > 0 else None,
            html.Span(f"💰{super_jp} ",
                       style={'color': '#00ffff', 'fontSize': '9px'}) if super_jp > 0 else None,
            html.Span(f"🍀{jackpot} ",
                       style={'color': '#00ff88', 'fontSize': '9px'}) if jackpot > 0 else None,
            html.Span(f"⭐{big_win}",
                       style={'color': '#88ff88', 'fontSize': '9px'}) if big_win > 0 else None,
        ], style={'marginTop': '2px'})
    ], className="text-center")


@app.callback(
    Output('current-jackpots', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def pm_update_current_jackpots(n):
    if not manager.backfill_complete:
        return html.Span("Scanning...", className="text-muted")

    jackpots = manager.get_pm_jackpots()
    merit_scores = manager.get_pm_merit()

    hot_jackpots = [(sym, j, merit_scores.get(sym, {}))
                    for sym, j in jackpots.items()
                    if j.get('is_jackpot') or j.get('heat_level', 0) >= 50]
    hot_jackpots.sort(key=lambda x: x[1].get('heat_level', 0), reverse=True)
    hot_jackpots = hot_jackpots[:3]

    if not hot_jackpots:
        return html.Div([
            html.Div([
                html.Span("⬜", className="slot-reel"),
                html.Span("⬜", className="slot-reel"),
                html.Span("⬜", className="slot-reel"),
            ], className="slot-container"),
            html.P("Scanning for alignment patterns...",
                   className="text-muted mt-2", style={'fontSize': '10px'}),
        ])

    cards = []
    for symbol, jackpot_data, merit in hot_jackpots:
        tier = jackpot_data.get('tier')
        tier_name = jackpot_data.get('tier_name', 'NO SIGNAL')
        emoji = jackpot_data.get('emoji', '⬜')
        color = jackpot_data.get('color', '#333')
        heat = jackpot_data.get('heat_level', 0)
        slot_display = jackpot_data.get('slot_display', ['⬜', '⬜', '⬜'])
        achievements = jackpot_data.get('achievements', [])
        direction = merit.get('dominant_direction', '—')
        levels = merit.get('stasis_levels', 0)
        alignment = merit.get('direction_alignment', 0)

        card_class = ""
        if tier == 'GRAND_JACKPOT':
            card_class = "grand-jackpot-card"
        elif tier == 'MEGA_JACKPOT':
            card_class = "mega-jackpot-card"
        elif tier and 'JACKPOT' in tier:
            card_class = "jackpot-card"

        direction_icon = "📈" if direction == "LONG" else "📉" if direction == "SHORT" else "➖"
        direction_color = "#00ff00" if direction == "LONG" else "#ff4444" if direction == "SHORT" else "#888"

        cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span(symbol,
                                  style={'color': color, 'fontWeight': 'bold', 'fontSize': '14px'}),
                        html.Span(f" {emoji}", style={'fontSize': '16px'}),
                    ], className="mb-1"),
                    html.Div([
                        html.Span(s, className="slot-reel") for s in slot_display
                    ], className="slot-container mb-1"),
                    html.Div(tier_name,
                             style={'color': color, 'fontSize': '9px', 'fontWeight': 'bold'}),
                    html.Div([
                        html.Span(f"{direction_icon} {direction}",
                                  style={'color': direction_color, 'fontSize': '10px'}),
                        html.Span(f" | {levels}LVL | {alignment:.0f}%",
                                  style={'color': '#888', 'fontSize': '9px'}),
                    ], className="mt-1"),
                    html.Div([
                        html.Div(style={
                            'width': f'{heat}%', 'height': '4px',
                            'backgroundColor': get_heat_color(heat),
                            'borderRadius': '2px', 'transition': 'width 0.3s ease',
                        })
                    ], style={
                        'width': '100%', 'height': '4px',
                        'backgroundColor': '#1a1a2e', 'borderRadius': '2px',
                        'marginTop': '4px',
                    }),
                    html.Div([
                        html.Span(
                            f"{ACHIEVEMENTS[a]['emoji']} {a.replace('_', ' ')}",
                            className=f"achievement-badge achievement-{ACHIEVEMENTS[a]['rarity'].lower()}"
                        ) for a in achievements[:2]
                    ], className="mt-1") if achievements else None,
                ], style={'padding': '8px', 'textAlign': 'center'})
            ], className=f"m-1 hover-glow {card_class}", style={
                'display': 'inline-block', 'minWidth': '150px',
                'backgroundColor': 'rgba(26, 26, 46, 0.9)',
                'border': f'2px solid {color}', 'borderRadius': '8px',
            })
        )

    return html.Div(cards, style={'display': 'flex', 'justifyContent': 'center',
                                   'flexWrap': 'wrap'})


@app.callback(
    Output('leaderboard-display', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def pm_update_leaderboard(n):
    if not manager.backfill_complete:
        return html.Span("Loading...", className="text-muted")

    merit_scores = manager.get_pm_merit()
    jackpots = manager.get_pm_jackpots()

    sorted_symbols = sorted(merit_scores.items(),
                            key=lambda x: x[1].get('weighted_score', 0),
                            reverse=True)[:10]

    items = []
    for rank, (symbol, merit) in enumerate(sorted_symbols, 1):
        score = merit.get('weighted_score', 0)
        if score == 0:
            continue

        levels = merit.get('stasis_levels', 0)
        direction = merit.get('dominant_direction', '—')
        alignment = merit.get('direction_alignment', 0)
        jp = jackpots.get(symbol, {})
        emoji = jp.get('emoji', '')
        heat = jp.get('heat_level', 0)

        direction_icon = "📈" if direction == "LONG" else "📉" if direction == "SHORT" else ""
        direction_color = "#00ff00" if direction == "LONG" else "#ff4444" if direction == "SHORT" else "#888"

        rank_style = {
            1: {'color': '#ffd700', 'fontWeight': 'bold'},
            2: {'color': '#c0c0c0', 'fontWeight': 'bold'},
            3: {'color': '#cd7f32', 'fontWeight': 'bold'},
        }.get(rank, {'color': '#888'})

        items.append(
            html.Div([
                html.Span(f"#{rank}", className="leaderboard-rank me-2", style=rank_style),
                html.Span(f"{symbol}",
                          style={'color': '#00ff88', 'fontWeight': 'bold', 'fontSize': '11px'}),
                html.Span(f" {emoji}", style={'fontSize': '12px'}),
                html.Span(f" {direction_icon}", style={'color': direction_color}),
                html.Div([
                    html.Span(f"Score: {score:.0f}",
                              style={'color': '#ffd700', 'fontSize': '9px'}),
                    html.Span(f" | {levels}LVL",
                              style={'color': '#888', 'fontSize': '9px'}),
                    html.Span(f" | {alignment:.0f}%",
                              style={'color': '#00ff88' if alignment == 100 else '#888',
                                     'fontSize': '9px'}),
                ], style={'marginTop': '2px'}),
            ], className="leaderboard-item", style={
                'borderLeftColor': get_heat_color(heat),
            })
        )

    if not items:
        return html.Span("No signals yet...", className="text-muted",
                         style={'fontSize': '10px'})

    return html.Div(items)


@app.callback(
    Output('achievements-display', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def pm_update_achievements(n):
    if not manager.backfill_complete:
        return html.Span("Loading...", className="text-muted")

    jackpots = manager.get_pm_jackpots()

    all_achievements = []
    for symbol, jp in jackpots.items():
        for achievement in jp.get('achievements', []):
            all_achievements.append({
                'symbol': symbol,
                'achievement': achievement,
                'info': ACHIEVEMENTS.get(achievement, {}),
            })

    if not all_achievements:
        return html.Div([
            html.P("🏅 No achievements unlocked yet",
                   className="text-muted text-center", style={'fontSize': '10px'}),
            html.P("Keep watching for perfect alignments!",
                   className="text-muted text-center", style={'fontSize': '9px'}),
        ])

    by_rarity = defaultdict(list)
    for a in all_achievements:
        rarity = a['info'].get('rarity', 'COMMON')
        by_rarity[rarity].append(a)

    sections = []
    for rarity in ['LEGENDARY', 'EPIC', 'RARE', 'UNCOMMON', 'COMMON']:
        if rarity in by_rarity:
            items = by_rarity[rarity]
            sections.append(
                html.Div([
                    html.Div([
                        html.Span(
                            f"{a['info'].get('emoji', '')} {a['symbol']}",
                            className=f"achievement-badge achievement-{rarity.lower()} me-1"
                        ) for a in items[:5]
                    ])
                ], className="mb-1")
            )

    return html.Div(sections)


@app.callback(
    Output('stats-summary', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def pm_update_stats_summary(n):
    if not manager.backfill_complete:
        return ""
    data = manager.get_pm_data()
    if not data:
        return ""
    tradable = sum(1 for d in data
                   if d['is_tradable'] and d['threshold'] in config.pm_thresholds)
    etf_count = len(config.etf_symbols)
    stock_count = len(config.symbols) - etf_count
    return html.Span([
        html.Span(f"ETF:{etf_count} ", style={'color': '#00ffff'}),
        html.Span(f"STK:{stock_count} ", style={'color': '#ffaa00'}),
        html.Span(f"SIG:{tradable}", style={'color': '#00ff88', 'fontWeight': 'bold'}),
    ], className="data-font")


@app.callback(
    Output('stats-display', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def pm_update_stats(n):
    if not manager.backfill_complete:
        return html.Span(
            f"⏳ LOADING {len(config.symbols)} symbols... {manager.backfill_progress}%",
            className="text-warning title-font")

    data = manager.get_pm_data()
    merit_scores = manager.get_pm_merit()
    jackpots = manager.get_pm_jackpots()

    if not data:
        return html.Span("LOADING...", className="text-muted title-font")

    pm_data = [d for d in data if d['threshold'] in config.pm_thresholds]
    tradable = [d for d in pm_data if d['is_tradable']]
    long_count = sum(1 for d in tradable if d['direction'] == 'LONG')
    short_count = sum(1 for d in tradable if d['direction'] == 'SHORT')
    max_stasis = max([d['stasis'] for d in pm_data]) if pm_data else 0

    top_merit = max(merit_scores.values(),
                    key=lambda x: x['weighted_score'],
                    default={'weighted_score': 0})
    top_merit_symbol = [k for k, v in merit_scores.items()
                        if v['weighted_score'] == top_merit['weighted_score']]
    top_merit_symbol = top_merit_symbol[0] if top_merit_symbol else "—"

    perfect_align = sum(1 for m in merit_scores.values()
                        if m['direction_alignment'] == 100 and m['stasis_levels'] >= 2)

    hottest = max(jackpots.items(),
                  key=lambda x: x[1].get('heat_level', 0),
                  default=(None, {'heat_level': 0}))
    hottest_symbol = hottest[0] if hottest[0] else "—"
    hottest_heat = hottest[1].get('heat_level', 0)

    return html.Div([
        html.Span("🎯 SIGNALS: ", className="title-font", style={'fontSize': '10px'}),
        html.Span(f"{len(tradable)}", className="data-font text-success stat-value",
                  style={'fontSize': '11px', 'fontWeight': '600'}),
        html.Span("  🏆 TOP: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{top_merit_symbol}({top_merit['weighted_score']:.0f})",
                  className="data-font text-warning stat-value",
                  style={'fontSize': '11px', 'fontWeight': '600'}),
        html.Span("  🔥 HOT: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{hottest_symbol}({hottest_heat})",
                  className="data-font stat-value",
                  style={'fontSize': '11px', 'color': get_heat_color(hottest_heat)}),
        html.Span("  ✅ ALIGNED: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{perfect_align}", className="data-font text-info stat-value",
                  style={'fontSize': '11px'}),
        html.Span("  📈 L: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{long_count}", className="data-font text-success stat-value",
                  style={'fontSize': '11px'}),
        html.Span("  📉 S: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{short_count}", className="data-font text-danger stat-value",
                  style={'fontSize': '11px'}),
        html.Span("  ⚡ MAX: ", className="title-font ms-2", style={'fontSize': '10px'}),
        html.Span(f"{max_stasis}", className="data-font text-warning stat-value",
                  style={'fontSize': '11px'}),
    ])


@app.callback(
    [Output('jackpot-toast', 'is_open'),
     Output('jackpot-toast', 'children'),
     Output('last-jackpot', 'data')],
    [Input('refresh-interval', 'n_intervals')],
    [State('last-jackpot', 'data')]
)
def pm_show_jackpot_notification(n, last_jackpot):
    if not manager.backfill_complete:
        return False, "", last_jackpot

    recent_jackpots = manager.get_recent_jackpots()
    if not recent_jackpots:
        return False, "", last_jackpot

    latest = recent_jackpots[-1]

    if (last_jackpot and latest['symbol'] == last_jackpot.get('symbol')
            and latest['tier'] == last_jackpot.get('tier')):
        return False, "", last_jackpot

    tier_info = JACKPOT_TIERS.get(latest['tier'], {})

    content = html.Div([
        html.Div([
            html.Span(tier_info.get('emoji', '🎰'), style={'fontSize': '30px'}),
        ], className="text-center mb-2"),
        html.Div([
            html.Span(latest['symbol'],
                      style={'color': '#00ff88', 'fontWeight': 'bold', 'fontSize': '20px'}),
            html.Span(f" hit ", style={'color': 'white'}),
            html.Span(latest['tier'].replace('_', ' '),
                      style={'color': tier_info.get('color', '#ffd700'),
                             'fontWeight': 'bold', 'fontSize': '16px'}),
        ], className="text-center"),
        html.Div([
            html.Span(f"{latest['levels']} levels aligned ",
                      style={'color': '#ffff00', 'fontSize': '12px'}),
            html.Span(
                f"{'📈 LONG' if latest['direction'] == 'LONG' else '📉 SHORT'}",
                style={'color': '#00ff00' if latest['direction'] == 'LONG' else '#ff4444',
                       'fontSize': '12px'}),
        ], className="text-center mt-2"),
        html.Div([
            html.Span(f"Multiplier: {tier_info.get('multiplier', '?x')}",
                      style={'color': '#ff00ff', 'fontSize': '14px', 'fontWeight': 'bold'}),
        ], className="text-center mt-2"),
    ], className="data-font")

    return True, content, {'symbol': latest['symbol'], 'tier': latest['tier']}


@app.callback(
    Output('main-table', 'data'),
    [Input('refresh-interval', 'n_intervals'),
     Input('view-mode', 'data'),
     Input('filter-type', 'value'),
     Input('filter-symbol', 'value'),
     Input('filter-threshold', 'value'),
     Input('filter-stasis', 'value'),
     Input('filter-direction', 'value'),
     Input('filter-merit', 'value'),
     Input('filter-alignment', 'value'),
     Input('filter-rows', 'value'),
     Input('filter-sort', 'value')]
)
def pm_update_table(n, view_mode, type_filter, sym, thresh, stasis,
                    direction, merit, alignment, rows, sort):
    df = pm_get_table_data()
    if df.empty:
        return []

    if view_mode == 'tradable':
        df = df[df['Is_Tradable'] == True]
    elif view_mode == 'jackpot':
        df = df[df['Is_Jackpot'] == True]

    if type_filter == 'ETF':
        df = df[df['Is_ETF'] == True]
    elif type_filter == 'STK':
        df = df[df['Is_ETF'] == False]

    if sym != 'ALL':
        df = df[df['Symbol'] == sym]
    if thresh != 'ALL':
        df = df[df['Band_Val'] == thresh]
    if stasis and stasis > 0:
        df = df[df['Stasis'] >= stasis]
    if direction != 'ALL':
        df = df[df['Dir'] == direction]
    if merit is not None and merit > 0:
        df = df[df['Merit_Val'] >= merit]
    if alignment is not None and alignment > 0:
        df = df[df['Align_Val'] >= alignment]

    if sort == 'heat':
        df = df.sort_values(['Heat', 'Merit_Val'], ascending=[False, False])
    elif sort == 'merit':
        df = df.sort_values(['Merit_Val', 'Stasis'], ascending=[False, False])
    elif sort == 'stasis':
        df = df.sort_values(['Stasis', 'Merit_Val'], ascending=[False, False])
    elif sort == 'rr':
        df = df.sort_values(['RR_Val', 'Merit_Val'], ascending=[False, False])

    df = df.head(rows)

    drop_cols = ['Band_Val', 'Current_Val', 'Anchor', 'RR_Val', 'Dur_Val', 'Chg_Val',
                 '52W_Val', 'Is_Tradable', 'Merit_Val', 'Bits', 'Is_ETF', 'Is_Jackpot',
                 'Align_Val', 'Jackpot_Tier', '→TP', '→SL']
    df = df.drop(columns=drop_cols, errors='ignore')

    return df.to_dict('records')


# Symbol bridge — clicking a row notifies desktop app
app.clientside_callback(
    """function(rows, data) {
        if (!rows || !rows.length || !data) return '';
        var sym = data[rows[0]]['Symbol'];
        if (sym) {
            fetch('/api/symbol/' + sym);
            try {
                if (window.parent && window.parent !== window) {
                    window.parent.postMessage({type: 'symbolSelected', symbol: sym}, '*');
                }
            } catch(e) {}
        }
        return sym;
    }""",
    Output('connection-status', 'title'),
    Input('main-table', 'selected_rows'), State('main-table', 'data'),
    prevent_initial_call=True)


@server.route('/api/symbol/<symbol>')
def pm_set_symbol_api(symbol):
    set_selected_symbol(symbol)
    return json.dumps({'ok': True, 'symbol': symbol})


@server.route('/api/symbol')
def pm_get_symbol_api():
    sym = get_selected_symbol()
    return json.dumps({'symbol': sym})


@server.route('/api/health')
def health_check():
    return json.dumps({
        'status': 'ok',
        'app': 'stasis_pm',
        'initialized': manager.initialized,
        'backfill_complete': manager.backfill_complete,
        'backfill_progress': manager.backfill_progress,
        'stream_count': manager.stream_count,
        'tradable_count': manager.tradable_count,
        'price_feed': price_feed.get_status(),
    })


@server.route('/api/status')
def status_api():
    return json.dumps({
        'backfill_complete': manager.backfill_complete,
        'backfill_progress': manager.backfill_progress,
        'tradable_count': manager.tradable_count,
        'price_feed': price_feed.get_status(),
    })


# ============================================================================
# INITIALIZATION
# ============================================================================

_init_done = False


def initialize_data():
    global _init_done
    if _init_done:
        return
    print("=" * 70)
    print("  STASIS PM SERVER v2.1 — JACKPOT EDITION")
    print("  © 2026 Truth Communications LLC")
    print("=" * 70)
    print(f"\n🎯 {len(config.symbols)} symbols to process\n")

    config.week52_data = fetch_52_week_data()
    config.volumes = fetch_volume_data()
    manager.backfill()
    price_feed.start()
    manager.start()

    print(f"\n✅ STASIS PM READY")
    print(f"   Streams: {manager.stream_count}")
    print(f"   Tradable: {manager.tradable_count}")
    _init_done = True


init_thread = threading.Thread(target=initialize_data, daemon=True)
init_thread.start()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print(f"\n🚀 LAUNCHING STASIS PM SERVER on {HOST}:{PORT}\n")
    app.run(debug=False, host=HOST, port=PORT, use_reloader=False)
