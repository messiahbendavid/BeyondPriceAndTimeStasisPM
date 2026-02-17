# -*- coding: utf-8 -*-
"""
STASIS AM — Alpha Markets Server
Deploy to: stasisAM.beyondpriceandtime.com
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

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import websocket
import ssl
import requests

# ============================================================================
# CONFIG
# ============================================================================

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "PnzhJOXEJO7tSpHr0ct2zjFKi6XO0yGi")


@dataclass
class Config:
    symbols: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLU", "XLK",
        "XLP", "XLB", "XLV", "XLI", "XLY", "XLC", "XLRE", "KRE",
        "SMH", "XBI", "GDX",
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META',
        'TSLA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'INTC', 'CSCO'
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
    am_thresholds: List[float] = field(default_factory=lambda: [
        0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05
    ])
    update_interval_ms: int = 1000
    cache_refresh_interval: float = 0.5
    history_days: int = 5
    polygon_api_key: str = POLYGON_API_KEY
    polygon_ws_url: str = "wss://delayed.polygon.io/stocks"
    polygon_rest_url: str = "https://api.polygon.io"
    volumes: Dict[str, float] = field(default_factory=dict)
    week52_data: Dict[str, Dict] = field(default_factory=dict)
    fundamental_data: Dict[str, Dict] = field(default_factory=dict)
    fundamental_slopes: Dict[str, Dict] = field(default_factory=dict)
    correlation_data: Dict[str, Dict] = field(default_factory=dict)
    min_tradable_stasis: int = 3
    # minimum quarters needed for correlation window
    corr_window_quarters: int = 5


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


def fmt_slope(v):
    return "—" if v is None else f"{'+' if v >= 0 else ''}{v * 100:.1f}%"


def fmt_rr(rr):
    if rr is None:
        return "—"
    return "0:1" if rr <= 0 else (f"{rr:.2f}:1" if rr < 10 else f"{rr:.0f}:1")


def fmt_corr(v):
    """Format correlation value for display."""
    if v is None:
        return "—"
    return f"{v:+.3f}"


def fmt_corr_delta(v):
    """Format correlation delta — negative means decorrelating."""
    if v is None:
        return "—"
    arrow = "↘" if v < -0.05 else ("↗" if v > 0.05 else "→")
    return f"{arrow}{v:+.3f}"


# ============================================================================
# PRICE:REVENUE CORRELATION ENGINE
# ============================================================================


# ============================================================================
# PRICE:REVENUE CORRELATION ENGINE
# ============================================================================


def fetch_prices_for_correlation(symbol: str, num_quarters: int = 8) -> List[Dict]:
    """
    Fetch DAILY closing prices going back far enough to cover the quarterly
    filing dates. Returns list of {'date': 'YYYY-MM-DD', 'close': float}.
    
    Using daily bars instead of monthly — monthly bars have alignment issues
    where filing dates fall between bar timestamps.
    """
    try:
        end = datetime.now()
        # Go back enough: ~90 days per quarter plus buffer
        start = end - timedelta(days=num_quarters * 95 + 30)

        url = (f"{config.polygon_rest_url}/v2/aggs/ticker/{symbol}/range/1/day/"
               f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
               f"?adjusted=true&sort=asc&limit=5000&apiKey={config.polygon_api_key}")

        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            print(f"      ⚠️ {symbol} daily bars HTTP {resp.status_code}")
            return []

        results = resp.json().get('results', [])
        if not results:
            print(f"      ⚠️ {symbol} no daily bar results")
            return []

        bars = []
        for bar in results:
            dt = datetime.fromtimestamp(bar['t'] / 1000)
            bars.append({
                'date': dt.strftime('%Y-%m-%d'),
                'date_obj': dt,
                'close': bar['c']
            })
        return bars

    except Exception as e:
        print(f"      ⚠️ {symbol} daily price fetch failed: {e}")
        return []


def find_price_on_date(daily_bars: List[Dict], target_date_str: str,
                       max_lookback_days: int = 10) -> Optional[float]:
    """
    Find the closing price on or just before a target date.
    Uses binary-search style lookup on sorted daily bars.
    
    Much more reliable than monthly bar alignment.
    """
    if not daily_bars or not target_date_str:
        return None

    try:
        target = datetime.strptime(target_date_str[:10], '%Y-%m-%d')
    except (ValueError, TypeError):
        return None

    best_price = None
    best_gap = max_lookback_days + 1

    for bar in daily_bars:
        gap = (target - bar['date_obj']).days
        # We want bars ON or BEFORE the target date
        if 0 <= gap < best_gap:
            best_gap = gap
            best_price = bar['close']
        # Also accept bars 1 day AFTER (for weekends/holidays)
        elif -1 <= gap < 0 and best_price is None:
            best_price = bar['close']

    return best_price


def _pearson_corr(prices: np.ndarray, revenues: np.ndarray) -> Optional[float]:
    """Safe Pearson correlation between two arrays of equal length."""
    if len(prices) < 3 or len(revenues) < 3:
        return None
    if len(prices) != len(revenues):
        return None
    if np.std(prices) < 1e-10 or np.std(revenues) < 1e-10:
        return None
    try:
        corr = np.corrcoef(prices, revenues)[0, 1]
        if np.isnan(corr) or np.isinf(corr):
            return None
        return round(float(corr), 4)
    except Exception:
        return None


def calculate_price_revenue_correlation(
        symbol: str, current_live_price: Optional[float] = None) -> Dict:
    """
    Calculate the 5-quarter price:revenue correlation in two states:

    1. corr_at_earnings — correlation using the PRICE ON THE LAST EARNINGS DATE
       paired with 5 trailing quarters of revenue.

    2. corr_now — same 5 trailing quarters of revenue, but replacing the most
       recent quarter's paired price with TODAY'S LIVE PRICE.

    3. corr_delta = corr_now - corr_at_earnings
       Negative → price is decorrelating from revenue since last earnings.
       Positive → price is re-correlating toward revenue.
    """
    result = {
        'corr_at_earnings': None,
        'corr_now': None,
        'corr_delta': None,
        'decorrelation_score': None,
        'price_vs_rev_divergence': None,
        'quarters_available': 0,
        'aligned_quarters': 0,
        'earnings_date_price': None,
        'current_price_used': None,
        'latest_rev_change_pct': None,
        'price_change_since_earnings_pct': None,
        'correlation_history': [],
        'debug': '',
    }

    # --- Get fundamental data (already fetched) ---
    fund = config.fundamental_data.get(symbol)
    if not fund:
        result['debug'] = 'no_fundamental_data'
        return result

    revenues = fund.get('revenue', [])
    dates = fund.get('dates', [])
    window = config.corr_window_quarters  # 5

    result['quarters_available'] = len(revenues)

    if len(revenues) < window:
        result['debug'] = f'insufficient_quarters:{len(revenues)}<{window}'
        return result

    if len(dates) < window:
        result['debug'] = f'insufficient_dates:{len(dates)}<{window}'
        return result

    # --- Fetch DAILY prices for this symbol ---
    daily_bars = fetch_prices_for_correlation(symbol, len(revenues) + 2)
    if not daily_bars:
        result['debug'] = 'no_daily_bars'
        return result

    # --- Align prices to each quarter's filing date ---
    aligned_prices = []
    aligned_revenues = []
    aligned_dates = []
    skipped = []

    for i in range(len(revenues)):
        if i >= len(dates):
            break

        rev = revenues[i]
        date_str = dates[i]

        # Skip quarters with zero or missing revenue
        if rev is None or rev == 0:
            skipped.append(f"q{i}:zero_rev")
            continue

        # Skip quarters with empty date
        if not date_str or len(date_str) < 8:
            skipped.append(f"q{i}:bad_date")
            continue

        price = find_price_on_date(daily_bars, date_str)

        if price is None:
            skipped.append(f"q{i}:{date_str}:no_price")
            continue

        aligned_prices.append(price)
        aligned_revenues.append(rev)
        aligned_dates.append(date_str)

    n = len(aligned_prices)
    result['aligned_quarters'] = n

    if n < window:
        result['debug'] = (f'alignment_failed:{n}_aligned_of_'
                           f'{len(revenues)}_quarters|skipped={skipped}')
        return result

    # --- Use the last `window` quarters ---
    trail_prices = np.array(aligned_prices[-window:], dtype=float)
    trail_revs = np.array(aligned_revenues[-window:], dtype=float)

    # The price on the most recent earnings date
    earnings_date_price = float(trail_prices[-1])
    result['earnings_date_price'] = round(earnings_date_price, 2)

    # =====================================================================
    # CORRELATION AT EARNINGS DATE
    # Correlation as it stood when the last earnings were filed.
    # Uses the actual price on the filing date for the most recent quarter.
    # =====================================================================
    result['corr_at_earnings'] = _pearson_corr(trail_prices, trail_revs)

    if result['corr_at_earnings'] is None:
        result['debug'] = (f'pearson_failed_at_earnings|'
                           f'prices={trail_prices.tolist()}|'
                           f'revs={trail_revs.tolist()}')
        return result

    # =====================================================================
    # CORRELATION NOW (with live price)
    # Same trailing revenue, but swap the most recent quarter's price
    # with today's live price.
    # =====================================================================
    live_price = current_live_price
    if live_price is None:
        w = config.week52_data.get(symbol, {})
        if w.get('current'):
            live_price = w['current']
        elif w.get('high') and w.get('low'):
            live_price = (w['high'] + w['low']) / 2

    if live_price is not None and live_price > 0:
        result['current_price_used'] = round(float(live_price), 2)

        # Replace most recent quarter's price with live price
        live_prices = trail_prices.copy()
        live_prices[-1] = float(live_price)

        result['corr_now'] = _pearson_corr(live_prices, trail_revs)

        # Price change since earnings
        if earnings_date_price > 0:
            result['price_change_since_earnings_pct'] = round(
                ((live_price - earnings_date_price) / earnings_date_price) * 100, 2
            )
    else:
        result['debug'] = f'no_live_price|earnings_corr={result["corr_at_earnings"]}'
        # Still have corr_at_earnings, just no delta
        result['corr_now'] = result['corr_at_earnings']

    # =====================================================================
    # CORRELATION DELTA = corr_now - corr_at_earnings
    # =====================================================================
    if result['corr_at_earnings'] is not None and result['corr_now'] is not None:
        result['corr_delta'] = round(
            result['corr_now'] - result['corr_at_earnings'], 4
        )

    # --- Rolling correlation history ---
    if n >= window:
        for end_idx in range(window, n + 1):
            start_idx = end_idx - window
            c = _pearson_corr(
                np.array(aligned_prices[start_idx:end_idx], dtype=float),
                np.array(aligned_revenues[start_idx:end_idx], dtype=float))
            result['correlation_history'].append(c)

    # --- Latest quarter revenue change ---
    if n >= 2:
        rev_prev = aligned_revenues[-2]
        rev_curr = aligned_revenues[-1]
        if rev_prev and rev_prev != 0:
            result['latest_rev_change_pct'] = round(
                ((rev_curr - rev_prev) / abs(rev_prev)) * 100, 2
            )

    # --- Decorrelation Score (0 to 1) ---
    corr_now = result['corr_now']
    corr_delta = result['corr_delta']

    if corr_now is not None:
        # Component 1: low absolute correlation (0 to 0.5)
        abs_decorr = max(0, (1.0 - abs(corr_now)) * 0.5)

        # Component 2: correlation dropping since earnings (0 to 0.5)
        delta_score = 0.0
        if corr_delta is not None:
            delta_score = max(0, min(0.5, (-corr_delta) * 0.5))

        result['decorrelation_score'] = round(abs_decorr + delta_score, 4)

    # --- Price vs Revenue divergence direction ---
    price_chg = result.get('price_change_since_earnings_pct')
    rev_chg = result.get('latest_rev_change_pct')

    if price_chg is not None and rev_chg is not None:
        if price_chg > rev_chg + 10:
            result['price_vs_rev_divergence'] = 'PRICE_AHEAD'
        elif rev_chg > price_chg + 10:
            result['price_vs_rev_divergence'] = 'PRICE_BEHIND'
        else:
            result['price_vs_rev_divergence'] = 'ALIGNED'
    elif price_chg is not None:
        if abs(price_chg) > 15:
            result['price_vs_rev_divergence'] = (
                'PRICE_AHEAD' if price_chg > 0 else 'PRICE_BEHIND')
        else:
            result['price_vs_rev_divergence'] = 'ALIGNED'

    result['debug'] = 'ok'
    return result


def calculate_all_correlations():
    """
    Calculate price:revenue correlations for all non-ETF symbols.
    """
    print("\n📊 CALCULATING PRICE:REVENUE CORRELATIONS...")
    print(f"   Window: {config.corr_window_quarters} quarters")
    ok = fail = skip = 0
    debug_summary = defaultdict(list)

    for i, sym in enumerate(config.symbols):
        # Skip ETFs
        if sym in config.etf_symbols:
            skip += 1
            continue

        # Skip if no fundamental data
        if sym not in config.fundamental_data:
            skip += 1
            debug_summary['no_fundamentals'].append(sym)
            continue

        try:
            w = config.week52_data.get(sym, {})
            init_price = w.get('current')

            corr_data = calculate_price_revenue_correlation(sym, init_price)
            config.correlation_data[sym] = corr_data

            dbg = corr_data.get('debug', '')

            if corr_data['corr_at_earnings'] is not None:
                ok += 1
                delta_str = fmt_corr_delta(corr_data.get('corr_delta'))
                ep = corr_data.get('earnings_date_price', '?')
                cp = corr_data.get('current_price_used', '?')
                pchg = corr_data.get('price_change_since_earnings_pct', '?')

                # Always print results during init for verification
                decor = corr_data.get('decorrelation_score', 0)
                flag = "🔔" if decor and decor > 0.3 else "  "
                print(f"   {flag} {sym:6s}: "
                      f"@earn={corr_data['corr_at_earnings']:+.3f} "
                      f"@now={corr_data.get('corr_now', '?'):+.3f}"
                      f"  Δ={delta_str:>8s}"
                      f"  (${ep}→${cp} {pchg:+.1f}%)"
                      f"  decor={decor:.3f}"
                      f"  [{corr_data.get('price_vs_rev_divergence', '?'):>12s}]"
                      f"  aligned={corr_data.get('aligned_quarters', 0)}q")
            else:
                fail += 1
                reason = dbg.split('|')[0] if dbg else 'unknown'
                debug_summary[reason].append(sym)
                print(f"      ✗ {sym:6s}: {dbg}")

        except Exception as e:
            print(f"      ⚠️ {sym} correlation error: {e}")
            import traceback
            traceback.print_exc()
            fail += 1

        if (i + 1) % 10 == 0:
            print(f"   --- Progress: {i + 1}/{len(config.symbols)} "
                  f"(✓{ok} ✗{fail} ⏭{skip})")

        time.sleep(0.15)

    # Print summary of failures for debugging
    if debug_summary:
        print(f"\n   📋 FAILURE SUMMARY:")
        for reason, syms in sorted(debug_summary.items()):
            print(f"      {reason}: {', '.join(syms[:10])}"
                  f"{'...' if len(syms) > 10 else ''} ({len(syms)})")

    print(f"\n✅ Correlations: {ok} calculated, {fail} failed, {skip} skipped\n")
# ============================================================================
# FUNDAMENTAL DATA
# ============================================================================


def fetch_fundamental_data_polygon(sym):
    try:
        url = (f"{config.polygon_rest_url}/vX/reference/financials"
               f"?ticker={sym}&timeframe=quarterly&limit=24"
               f"&sort=filing_date&order=desc&apiKey={config.polygon_api_key}")
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        results = resp.json().get('results', [])
        if not results:
            return None
        fund = {k: [] for k in [
            'dates', 'revenue', 'net_income', 'operating_cash_flow',
            'capex', 'fcf', 'total_assets', 'total_liabilities',
            'shareholders_equity', 'current_assets', 'current_liabilities',
            'total_debt', 'eps']}
        for r in results:
            try:
                fi = r.get('financials', {})
                inc = fi.get('income_statement', {})
                cf = fi.get('cash_flow_statement', {})
                bs = fi.get('balance_sheet', {})
                rev = inc.get('revenues', {}).get('value', 0) or 0
                ni = inc.get('net_income_loss', {}).get('value', 0) or 0
                eps = inc.get('basic_earnings_per_share', {}).get('value', 0) or 0
                ocf = cf.get('net_cash_flow_from_operating_activities', {}).get('value', 0) or 0
                cx = cf.get('net_cash_flow_from_investing_activities', {}).get('value', 0) or 0
                ta = bs.get('assets', {}).get('value', 0) or 0
                tl = bs.get('liabilities', {}).get('value', 0) or 0
                eq = bs.get('equity', {}).get('value', 0) or 0
                ca = bs.get('current_assets', {}).get('value', 0) or 0
                cl = bs.get('current_liabilities', {}).get('value', 0) or 0
                ltd = bs.get('long_term_debt', {}).get('value', 0) or 0
                std = bs.get('short_term_debt', {}).get('value', 0) or 0
                fund['dates'].append(r.get('filing_date', ''))
                fund['revenue'].append(rev)
                fund['net_income'].append(ni)
                fund['operating_cash_flow'].append(ocf)
                fund['capex'].append(abs(cx))
                fund['fcf'].append(ocf + cx)
                fund['total_assets'].append(ta)
                fund['total_liabilities'].append(tl)
                fund['shareholders_equity'].append(eq)
                fund['current_assets'].append(ca)
                fund['current_liabilities'].append(cl)
                fund['total_debt'].append(ltd + std)
                fund['eps'].append(eps)
            except:
                continue
        for k in fund:
            fund[k] = fund[k][::-1]
        return fund
    except:
        return None


def calculate_slopes(series, ss=4, sl=20):
    if not series or len(series) < 5:
        return None, None
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan)
    s5 = s20 = None
    try:
        if len(s.dropna()) >= 5:
            e = s.ewm(span=ss, adjust=False).mean()
            if abs(e.iloc[-5]) > 0.0001:
                s5 = (e.iloc[-1] - e.iloc[-5]) / abs(e.iloc[-5])
    except:
        pass
    try:
        if len(s.dropna()) >= 21:
            e = s.ewm(span=sl, adjust=False).mean()
            if abs(e.iloc[-21]) > 0.0001:
                s20 = (e.iloc[-1] - e.iloc[-21]) / abs(e.iloc[-21])
    except:
        pass
    return s5, s20


def calculate_all_slopes(fund, ratios):
    sl = {}
    sl['Rev_Slope_5'], sl['Rev_Slope_20'] = calculate_slopes(fund.get('revenue', []))
    sl['FCF_Slope_5'], sl['FCF_Slope_20'] = calculate_slopes(fund.get('fcf', []))
    for n, k in [('P/E Ratio', 'pe_ratio'), ('Return on Equity', 'roe'),
                 ('Net Profit Margin', 'net_profit_margin'),
                 ('Debt to Equity Ratio', 'debt_to_equity')]:
        sl[f'{n}_Slope_5'], sl[f'{n}_Slope_20'] = calculate_slopes(ratios.get(k, []))
    fl = ratios.get('fcfy', [])
    sl['FCFY'] = fl[-1] if fl and fl[-1] is not None else None
    return sl


def fetch_all_fundamental_data():
    print("\n📊 FETCHING FUNDAMENTAL DATA...")
    ok = fail = 0
    for i, sym in enumerate(config.symbols):
        try:
            fund = fetch_fundamental_data_polygon(sym)
            if fund and len(fund.get('revenue', [])) >= 4:
                price = 100
                w = config.week52_data.get(sym, {})
                if w.get('high') and w.get('low'):
                    price = (w['high'] + w['low']) / 2
                eq = fund['shareholders_equity'][-1]
                mcap = eq * 2 if eq and eq > 0 else 1e9
                ratios = {k: [] for k in ['pe_ratio', 'roe', 'net_profit_margin',
                                           'debt_to_equity', 'fcfy']}
                for j in range(len(fund['revenue'])):
                    try:
                        eps = fund['eps'][j]
                        ratios['pe_ratio'].append(price / eps if eps > 0 else None)
                        eq_j = fund['shareholders_equity'][j]
                        ratios['roe'].append(fund['net_income'][j] / eq_j if eq_j > 0 else None)
                        rev_j = fund['revenue'][j]
                        ratios['net_profit_margin'].append(
                            fund['net_income'][j] / rev_j if rev_j else None)
                        ratios['debt_to_equity'].append(
                            fund['total_debt'][j] / eq_j if eq_j > 0 else None)
                        if j >= 3:
                            ratios['fcfy'].append(
                                sum(fund['fcf'][max(0, j - 3):j + 1]) / mcap if mcap else None)
                        else:
                            ratios['fcfy'].append(None)
                    except:
                        for k in ratios:
                            ratios[k].append(None)
                slopes = calculate_all_slopes(fund, ratios)
                config.fundamental_data[sym] = fund
                config.fundamental_slopes[sym] = slopes
                ok += 1
            else:
                fail += 1
        except:
            fail += 1
        if (i + 1) % 25 == 0:
            print(f"   📈 {i + 1}/{len(config.symbols)} (✓{ok} ✗{fail})")
        time.sleep(0.15)
    print(f"✅ Fundamentals: {ok} ok, {fail} failed\n")


# ============================================================================
# MERIT SCORING (UPDATED WITH CORRELATION)
# ============================================================================


def calculate_stasis_merit_score(snap):
    ms = 0
    st = snap.get('stasis', 0)
    for t, p in [(15, 10), (12, 9), (10, 8), (8, 7), (7, 6),
                 (6, 5), (5, 4), (4, 3), (3, 2), (2, 1)]:
        if st >= t:
            ms += p
            break
    rr = snap.get('risk_reward')
    if rr:
        for t, p in [(3, 5), (2.5, 4), (2, 3), (1.5, 2), (1, 1)]:
            if rr >= t:
                ms += p
                break
    ms += {'VERY_STRONG': 4, 'STRONG': 3, 'MODERATE': 2, 'WEAK': 1}.get(
        snap.get('signal_strength', ''), 0)
    dur = snap.get('duration_seconds', 0)
    if dur >= 3600:
        ms += 3
    elif dur >= 1800:
        ms += 2
    elif dur >= 900:
        ms += 1
    return ms


def calculate_correlation_merit_score(symbol: str, direction: Optional[str]) -> Tuple[int, Dict]:
    """
    Calculate merit score contribution from price:revenue decorrelation.
    
    Scoring logic:
    - High decorrelation score: up to +6 points
    - Correlation delta (how fast it's decorrelating): up to +4 points  
    - Direction alignment bonus: up to +3 points
      * If PRICE_BEHIND and direction=LONG → bonus (price should catch up)
      * If PRICE_AHEAD and direction=SHORT → bonus (price should correct)
    
    Returns: (score, details_dict)
    """
    score = 0
    details = {
        'corr_5q': None,
        'corr_delta': None,
        'decor_score': None,
        'divergence': None,
        'corr_merit': 0,
    }
    
    corr = config.correlation_data.get(symbol)
    if not corr or corr.get('corr_5q') is None:
        return score, details
    
    details['corr_5q'] = corr['corr_5q']
    details['corr_delta'] = corr.get('corr_delta')
    details['decor_score'] = corr.get('decorrelation_score')
    details['divergence'] = corr.get('price_vs_rev_divergence')
    details['latest_rev_chg'] = corr.get('latest_rev_change_pct')
    details['latest_price_chg'] = corr.get('latest_price_change_pct')
    
    # --- Component 1: Decorrelation Score (0-1) → 0-6 points ---
    decor = corr.get('decorrelation_score', 0) or 0
    for threshold, points in [
        (0.7, 6), (0.55, 5), (0.4, 4), (0.3, 3), (0.2, 2), (0.1, 1)
    ]:
        if decor >= threshold:
            score += points
            break
    
    # --- Component 2: Correlation Delta (rate of decorrelation) → 0-4 points ---
    delta = corr.get('corr_delta')
    if delta is not None:
        # More negative delta = faster decorrelation = more points
        for threshold, points in [
            (-0.4, 4), (-0.25, 3), (-0.15, 2), (-0.05, 1)
        ]:
            if delta <= threshold:
                score += points
                break
    
    # --- Component 3: Direction Alignment Bonus → 0-3 points ---
    divergence = corr.get('price_vs_rev_divergence')
    if divergence and direction:
        # Price lagging behind revenue growth + LONG signal = strong buy signal
        if divergence == 'PRICE_BEHIND' and direction == 'LONG':
            score += 3
        # Price running ahead of revenue + SHORT signal = strong sell signal
        elif divergence == 'PRICE_AHEAD' and direction == 'SHORT':
            score += 3
        # Misaligned: price ahead but going long, or price behind but shorting
        elif divergence == 'PRICE_AHEAD' and direction == 'LONG':
            score -= 1  # Slight penalty
        elif divergence == 'PRICE_BEHIND' and direction == 'SHORT':
            score -= 1  # Slight penalty
    
    details['corr_merit'] = max(0, score)  # Floor at 0
    return max(0, score), details


def calculate_fundamental_merit_score(symbol, w52_pct):
    ms = 0
    sd = {}
    slopes = config.fundamental_slopes.get(symbol, {})
    if not slopes:
        if w52_pct is not None:
            for t, p in [(5, 8), (15, 7), (25, 6), (35, 5), (45, 4),
                         (55, 3), (65, 2), (75, 1)]:
                if w52_pct <= t:
                    ms += p
                    break
        return ms, sd
    for lbl, key, tps in [
        ('Rev_5', 'Rev_Slope_5', [(0.30, 4), (0.20, 3), (0.10, 2), (0.05, 1)]),
        ('FCF_5', 'FCF_Slope_5', [(0.40, 4), (0.25, 3), (0.10, 2), (0.05, 1)]),
        ('ROE_5', 'Return on Equity_Slope_5', [(0.20, 2), (0.10, 1)]),
        ('NPM_5', 'Net Profit Margin_Slope_5', [(0.20, 2), (0.10, 1)])]:
        v = slopes.get(key)
        sd[lbl] = v
        if v is not None:
            for t, p in tps:
                if v >= t:
                    ms += p
                    break
    for lbl, key, tps in [
        ('PE_5', 'P/E Ratio_Slope_5', [(-0.25, 3), (-0.15, 2), (-0.05, 1)]),
        ('DE_5', 'Debt to Equity Ratio_Slope_5', [(-0.20, 2), (-0.10, 1)])]:
        v = slopes.get(key)
        sd[lbl] = v
        if v is not None:
            for t, p in tps:
                if v <= t:
                    ms += p
                    break
    if w52_pct is not None:
        for t, p in [(5, 8), (15, 7), (25, 6), (35, 5), (45, 4),
                     (55, 3), (65, 2), (75, 1)]:
            if w52_pct <= t:
                ms += p
                break
    fcfy = slopes.get('FCFY')
    sd['FCFY'] = fcfy
    if fcfy is not None:
        if fcfy >= 0.15:
            ms += 3
        elif fcfy >= 0.10:
            ms += 2
        elif fcfy >= 0.05:
            ms += 1
    return ms, sd


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
            distance_to_tp_pct = distance_to_sl_pct = stasis_price_change_pct = None
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
                if p > 0:
                    distance_to_tp_pct = (abs(tp - p) / p) * 100
                    distance_to_sl_pct = (abs(sl - p) / p) * 100
            return {
                'symbol': self.symbol, 'is_etf': self.is_etf,
                'threshold': self.threshold, 'threshold_pct': self.threshold * 100,
                'stasis': self.current_stasis, 'total_bits': self.total_bits,
                'current_price': p, 'anchor_price': si.start_price if si else None,
                'direction': self.direction.value if self.direction else None,
                'signal_strength': self.signal_strength.value if self.signal_strength else None,
                'is_tradable': (self.current_stasis >= config.min_tradable_stasis
                                and self.direction is not None and self.volume > 1.0),
                'stasis_start_str': si.get_start_date_str() if si else "—",
                'stasis_duration_str': si.get_duration_str() if si else "—",
                'duration_seconds': si.get_duration().total_seconds() if si else 0,
                'stasis_price_change_pct': stasis_price_change_pct,
                'take_profit': tp, 'stop_loss': sl, 'risk_reward': rr,
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
# BITSTREAM MANAGER (UPDATED)
# ============================================================================


class BitstreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.streams: Dict[Tuple[str, float], Bitstream] = {}
        self.is_running = False
        self.cached_am_data: List[Dict] = []
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
            am = self._build_am(snaps)
            with self.cache_lock:
                self.cached_am_data = am

    def _build_am(self, snaps):
        rows = []
        for s in snaps:
            if s['threshold'] not in config.am_thresholds:
                continue
            
            sms = calculate_stasis_merit_score(s)
            fms, sd = calculate_fundamental_merit_score(
                s['symbol'], s.get('week52_percentile'))
            
            # Calculate correlation merit score
            cms, cd = calculate_correlation_merit_score(
                s['symbol'], s.get('direction'))
            
            # Total merit score now includes correlation
            tms = sms + fms + cms
            
            rows.append({
                **s,
                'sms': sms,
                'fms': fms,
                'cms': cms,
                'tms': tms,
                'slope_details': sd,
                'corr_details': cd,
            })
        return rows

    def get_am_data(self):
        with self.cache_lock:
            return copy.deepcopy(self.cached_am_data)


manager = BitstreamManager()

# ============================================================================
# DASH APP (UPDATED WITH CORRELATION COLUMNS & FILTERS)
# ============================================================================

AM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600&display=swap');
body { background: #f5f0e8 !important; }
.title-font { font-family: 'Orbitron', sans-serif !important; }
"""

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True)
app.title = "STASIS AM"
server = app.server

app.index_string = f'''<!DOCTYPE html>
<html><head>
{{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
<style>{AM_CSS}</style>
</head><body>
{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body></html>'''

app.layout = html.Div([
    dcc.Store(id='fmode', data='tradable'),
    dcc.Interval(id='tick', interval=1000, n_intervals=0),

    # Header
    html.Div([
        html.Span("📈", style={'fontSize': '22px'}),
        html.Span(" STASIS AM", className="title-font ms-2",
                  style={'fontSize': '16px', 'fontWeight': '700', 'color': '#1a5c2a',
                         'letterSpacing': '2px'}),
        html.Span(" — ALPHA MARKETS", className="title-font",
                  style={'fontSize': '9px', 'color': '#888', 'letterSpacing': '1px'}),
    ], style={'padding': '8px'}),

    html.Div(id='status', style={'fontSize': '10px', 'padding': '4px 8px',
                                  'background': '#e8f5e9', 'fontWeight': 'bold'}),
    # Filters
    html.Div([
        dbc.ButtonGroup([
            dbc.Button("ALL", id="f-all", size="sm", outline=True,
                       style={'fontSize': '9px'}),
            dbc.Button("TRADABLE", id="f-trad", size="sm", outline=True,
                       active=True, style={'fontSize': '9px', 'color': '#1a5c2a'}),
            dbc.Button("DECORR", id="f-decorr", size="sm", outline=True,
                       style={'fontSize': '9px', 'color': '#8b4513'},
                       className="ms-1"),
        ], size="sm", className="me-2"),
        dcc.Dropdown(id='f-dir',
                     options=[{'label': x, 'value': x}
                              for x in ['ALL', 'LONG', 'SHORT']],
                     value='ALL', clearable=False,
                     style={'width': '80px', 'fontSize': '10px',
                            'display': 'inline-block'}),
        dcc.Dropdown(id='f-sort', options=[
            {'label': 'TMS ↓', 'value': 'tms'},
            {'label': 'FMS ↓', 'value': 'fms'},
            {'label': 'CMS ↓', 'value': 'cms'},
            {'label': 'DECORR ↓', 'value': 'decorr'},
            {'label': 'Δ CORR ↑', 'value': 'corr_delta'},
            {'label': 'STASIS ↓', 'value': 'stasis'},
            {'label': '52W ↑', 'value': '52w'},
        ],
            value='tms', clearable=False,
            style={'width': '100px', 'fontSize': '10px', 'display': 'inline-block',
                   'marginLeft': '4px'}),
    ], className="d-flex align-items-center p-1",
       style={'background': '#f5f0e8'}),

    # Table — updated columns to include correlation data
    dash_table.DataTable(
        id='tbl', columns=[{'name': c, 'id': c} for c in [
            '✓', 'SYM', 'BAND', 'STS', 'DIR',
            'SMS', 'FMS', 'CMS', 'TMS',
            'CORR', 'ΔCOR', 'DCOR', 'DIV',
            'REV5', 'FCF5', 'FCFY', '52W',
            'PRICE', 'TP', 'SL', 'R:R', 'DUR']],
        sort_action='native',
        style_table={'overflowY': 'auto', 'height': '80vh'},
        style_cell={
            'backgroundColor': '#faf7f0', 'color': '#1a1a1a',
            'padding': '3px 4px', 'fontSize': '10px',
            'fontFamily': 'Consolas, monospace',
            'whiteSpace': 'nowrap', 'textAlign': 'right',
            'border': '1px solid #ddd'},
        style_cell_conditional=[
            {'if': {'column_id': 'SYM'}, 'textAlign': 'left',
             'fontWeight': '700', 'color': '#1a5c2a'},
            {'if': {'column_id': 'DIR'}, 'textAlign': 'center'},
            {'if': {'column_id': 'DIV'}, 'textAlign': 'center',
             'fontSize': '8px'},
        ],
        style_header={
            'backgroundColor': '#1a5c2a', 'color': '#fff',
            'fontWeight': '700', 'fontSize': '9px', 'textAlign': 'center'},
        style_data_conditional=[
            # Direction colors
            {'if': {'filter_query': '{DIR} = "LONG"', 'column_id': 'DIR'},
             'color': '#1a8c3a', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{DIR} = "SHORT"', 'column_id': 'DIR'},
             'color': '#cc2200', 'fontWeight': 'bold'},
            # Stasis highlighting
            {'if': {'filter_query': '{STS} >= 10'},
             'backgroundColor': '#e8f5e9'},
            {'if': {'filter_query': '{STS} >= 7 && {STS} < 10'},
             'backgroundColor': '#f1f8e9'},
            # Price column
            {'if': {'column_id': 'PRICE'},
             'color': '#0055aa', 'fontWeight': '600'},
            {'if': {'column_id': 'TP'}, 'color': '#1a8c3a'},
            {'if': {'column_id': 'SL'}, 'color': '#cc2200'},
            # TMS highlighting
            {'if': {'filter_query': '{TMS} >= 35', 'column_id': 'TMS'},
             'backgroundColor': '#1a8c3a', 'color': '#fff'},
            {'if': {'filter_query': '{TMS} >= 25 && {TMS} < 35',
                    'column_id': 'TMS'},
             'backgroundColor': '#4caf50', 'color': '#fff'},
            {'if': {'filter_query': '{TMS} >= 15 && {TMS} < 25',
                    'column_id': 'TMS'},
             'backgroundColor': '#81c784', 'color': '#fff'},
            # CMS (correlation merit) highlighting
            {'if': {'filter_query': '{CMS} >= 8', 'column_id': 'CMS'},
             'backgroundColor': '#e65100', 'color': '#fff'},
            {'if': {'filter_query': '{CMS} >= 5 && {CMS} < 8',
                    'column_id': 'CMS'},
             'backgroundColor': '#f57c00', 'color': '#fff'},
            {'if': {'filter_query': '{CMS} >= 3 && {CMS} < 5',
                    'column_id': 'CMS'},
             'backgroundColor': '#ffb74d', 'color': '#000'},
            # Decorrelation score highlighting
            {'if': {'filter_query': '{_dcor_raw} >= 0.5',
                    'column_id': 'DCOR'},
             'backgroundColor': '#d32f2f', 'color': '#fff'},
            {'if': {'filter_query': '{_dcor_raw} >= 0.3 && {_dcor_raw} < 0.5',
                    'column_id': 'DCOR'},
             'backgroundColor': '#ff7043', 'color': '#fff'},
            # Divergence highlighting
            {'if': {'filter_query': '{DIV} = "P>R"', 'column_id': 'DIV'},
             'backgroundColor': '#fff3e0', 'color': '#e65100'},
            {'if': {'filter_query': '{DIV} = "P<R"', 'column_id': 'DIV'},
             'backgroundColor': '#e8f5e9', 'color': '#1b5e20'},
            # Alternating rows
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f0ebe0'},
        ],
        tooltip_header={
            'CORR': 'Price:Revenue 5-quarter correlation',
            'ΔCOR': 'Correlation change (negative = decorrelating)',
            'DCOR': 'Decorrelation score (0-1, higher = more decorrelated)',
            'CMS': 'Correlation Merit Score',
            'DIV': 'Price vs Revenue divergence direction',
        },
    ),

    # Footer
    html.Div("© 2026 Truth Communications LLC • STASIS AM",
             className="text-center",
             style={'fontSize': '8px', 'color': '#888', 'padding': '4px'}),

], style={'background': '#f5f0e8', 'minHeight': '100vh'})


# ============================================================================
# CALLBACKS (UPDATED)
# ============================================================================


@app.callback(Output('status', 'children'), Input('tick', 'n_intervals'))
def update_status(n):
    if not manager.backfill_complete:
        return html.Span(
            f"⏳ Initializing... {manager.backfill_progress}%",
            style={'color': '#aa6600'})
    st = price_feed.get_status()
    am_data = manager.get_am_data()
    tradable = sum(1 for d in am_data if d.get('is_tradable'))
    corr_count = sum(1 for v in config.correlation_data.values()
                     if v.get('corr_5q') is not None)
    decorr_count = sum(
        1 for v in config.correlation_data.values()
        if v.get('decorrelation_score') is not None
        and v['decorrelation_score'] > 0.3)
    if st['connected'] == 0:
        return html.Span(
            f"🔴 Connecting... | {tradable} tradable",
            style={'color': '#aa6600'})
    return html.Span(
        f"🟢 LIVE {st['connected']}/{st['total']} | "
        f"📨 {st['messages']:,} msgs | "
        f"📊 {len(config.fundamental_slopes)} fundamentals | "
        f"🔗 {corr_count} correlations ({decorr_count} decorrelating) | "
        f"🎯 {tradable} tradable",
        style={'color': '#1a5c2a'})


@app.callback(
    [Output('f-all', 'active'), Output('f-trad', 'active'),
     Output('f-decorr', 'active'), Output('fmode', 'data')],
    [Input('f-all', 'n_clicks'), Input('f-trad', 'n_clicks'),
     Input('f-decorr', 'n_clicks')],
    prevent_initial_call=True)
def toggle_filter(n1, n2, n3):
    ctx = callback_context
    tid = ctx.triggered[0]['prop_id']
    if 'f-all' in tid:
        return True, False, False, 'all'
    elif 'f-decorr' in tid:
        return False, False, True, 'decorr'
    return False, True, False, 'tradable'


@app.callback(
    Output('tbl', 'data'),
    [Input('tick', 'n_intervals'), Input('fmode', 'data'),
     Input('f-dir', 'value'), Input('f-sort', 'value')])
def update_table(n, fm, fd, fs):
    if not manager.backfill_complete:
        return []
    data = manager.get_am_data()
    if not data:
        return []
    rows = []
    for d in data:
        if fm == 'tradable' and not d.get('is_tradable'):
            continue
        if fm == 'decorr':
            # Only show stocks with significant decorrelation
            cd = d.get('corr_details', {})
            dcor = cd.get('decor_score')
            if dcor is None or dcor < 0.15:
                continue
        if fd != 'ALL' and d.get('direction') != fd:
            continue

        sd = d.get('slope_details', {})
        cd = d.get('corr_details', {})
        w52 = d.get('week52_percentile')

        # Format correlation columns
        corr_5q = cd.get('corr_5q')
        corr_delta = cd.get('corr_delta')
        decor_score = cd.get('decor_score')
        divergence = cd.get('divergence')

        # Divergence display
        div_display = '—'
        if divergence == 'PRICE_AHEAD':
            div_display = 'P>R'
        elif divergence == 'PRICE_BEHIND':
            div_display = 'P<R'
        elif divergence == 'ALIGNED':
            div_display = '≈'

        rows.append({
            '✓': '✅' if d.get('is_tradable') else '',
            'SYM': d['symbol'],
            'BAND': f"{d['threshold_pct']:.2f}%",
            'STS': d['stasis'],
            'DIR': d.get('direction') or '—',
            'SMS': d.get('sms', 0),
            'FMS': d.get('fms', 0),
            'CMS': d.get('cms', 0),
            'TMS': d.get('tms', 0),
            'CORR': fmt_corr(corr_5q),
            'ΔCOR': fmt_corr_delta(corr_delta),
            'DCOR': (f"{decor_score:.2f}"
                     if decor_score is not None else '—'),
            'DIV': div_display,
            'REV5': fmt_slope(sd.get('Rev_5')),
            'FCF5': fmt_slope(sd.get('FCF_5')),
            'FCFY': (f"{sd['FCFY'] * 100:.1f}%"
                     if sd.get('FCFY') else '—'),
            '52W': f"{w52:.0f}%" if w52 is not None else '—',
            'PRICE': (f"${d['current_price']:.2f}"
                      if d.get('current_price') else '—'),
            'TP': (f"${d['take_profit']:.2f}"
                   if d.get('take_profit') else '—'),
            'SL': (f"${d['stop_loss']:.2f}"
                   if d.get('stop_loss') else '—'),
            'R:R': fmt_rr(d.get('risk_reward')),
            'DUR': d.get('stasis_duration_str', '—'),
            # Hidden sort columns
            '_tms': d.get('tms', 0),
            '_fms': d.get('fms', 0),
            '_cms': d.get('cms', 0),
            '_stasis': d['stasis'],
            '_52w': w52 if w52 is not None else 999,
            '_dcor_raw': decor_score if decor_score is not None else -1,
            '_corr_delta': (corr_delta if corr_delta is not None
                            else 999),
        })
    if not rows:
        return []
    df = pd.DataFrame(rows)
    sort_map = {
        'tms': ('_tms', False),
        'fms': ('_fms', False),
        'cms': ('_cms', False),
        'decorr': ('_dcor_raw', False),
        'corr_delta': ('_corr_delta', True),  # ascending: most negative first
        'stasis': ('_stasis', False),
        '52w': ('_52w', True),
    }
    col, asc = sort_map.get(fs, ('_tms', False))
    df = df.sort_values(col, ascending=asc).head(200)
    df = df.drop(
        columns=['_tms', '_fms', '_cms', '_stasis', '_52w',
                 '_dcor_raw', '_corr_delta'],
        errors='ignore')
    return df.to_dict('records')


# ============================================================================
# API ENDPOINTS
# ============================================================================


@server.route('/api/health')
def health():
    return json.dumps({
        'status': 'ok', 'app': 'stasis_am',
        'initialized': manager.initialized,
        'backfill_complete': manager.backfill_complete,
        'backfill_progress': manager.backfill_progress,
        'correlations_calculated': len(config.correlation_data),
    })


@server.route('/api/correlations')
def api_correlations():
    """API endpoint to get all correlation data."""
    result = {}
    for sym, data in config.correlation_data.items():
        if data.get('corr_5q') is not None:
            result[sym] = {
                'corr_5q': data['corr_5q'],
                'corr_4q_prev': data.get('corr_4q_prev'),
                'corr_delta': data.get('corr_delta'),
                'decorrelation_score': data.get('decorrelation_score'),
                'divergence': data.get('price_vs_rev_divergence'),
                'quarters_available': data.get('quarters_available'),
                'latest_rev_change_pct': data.get('latest_rev_change_pct'),
                'latest_price_change_pct': data.get('latest_price_change_pct'),
            }
    return json.dumps(result, indent=2)


@server.route('/api/decorrelating')
def api_decorrelating():
    """API endpoint: stocks sorted by decorrelation score."""
    items = []
    for sym, data in config.correlation_data.items():
        if data.get('decorrelation_score') is not None:
            items.append({
                'symbol': sym,
                'decorrelation_score': data['decorrelation_score'],
                'corr_5q': data['corr_5q'],
                'corr_delta': data.get('corr_delta'),
                'divergence': data.get('price_vs_rev_divergence'),
            })
    items.sort(key=lambda x: x['decorrelation_score'], reverse=True)
    return json.dumps(items, indent=2)


# ============================================================================
# INITIALIZATION (UPDATED)
# ============================================================================

_init_done = False
_init_lock = threading.Lock()


def initialize():
    global _init_done
    with _init_lock:
        if _init_done:
            return
        print("=" * 70)
        print("  STASIS AM SERVER")
        print("  © 2026 Truth Communications LLC")
        print("  Now with Price:Revenue Correlation Analysis")
        print("=" * 70)
        print(f"\n🎯 Symbols: {len(config.symbols)}")

        # Step 1: 52-week data
        config.week52_data = fetch_52_week_data()

        # Step 2: Volume data
        config.volumes = fetch_volume_data()

        # Step 3: Fundamental data (revenue, FCF, etc.)
        fetch_all_fundamental_data()

        # Step 4: Price:Revenue Correlations (NEW)
        calculate_all_correlations()

        # Step 5: Backfill bitstreams
        manager.backfill()

        # Step 6: Start live feeds
        price_feed.start()
        manager.start()

        corr_count = sum(
            1 for v in config.correlation_data.values()
            if v.get('corr_5q') is not None)
        decorr_count = sum(
            1 for v in config.correlation_data.values()
            if v.get('decorrelation_score') is not None
            and v['decorrelation_score'] > 0.3)

        print(f"\n✅ READY")
        print(f"   📊 {len(config.fundamental_slopes)} fundamentals")
        print(f"   🔗 {corr_count} correlations calculated")
        print(f"   🔔 {decorr_count} significantly decorrelating")
        print("=" * 70)
        _init_done = True


_init_thread = threading.Thread(target=initialize, daemon=True)
_init_thread.start()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    _init_thread.join()
    port = int(os.environ.get('PORT', 8050))
    print(f"\n🟢 http://0.0.0.0:{port}\n")
    app.run(debug=False, host='0.0.0.0', port=port)
