#!/usr/bin/env python3
"""
paper_trader.py - backtest AND forward-test (paper trade) crypto strategies; HTML dashboard.

Commands
    python paper_trader.py backtest # replay ~2 years of 4h history with the SAME rules as live
    python paper_trader.py run      # one cycle (use with cron / GitHub Actions)
    python paper_trader.py loop     # run forever, waking after each candle close
    python paper_trader.py report   # rebuild docs/index.html from the journal

Everything is configurable through environment variables (see Config).
Strategies include long, short, and regime-switching setups, risk a fixed % of paper
equity per trade, cap correlated directional exposure, and include exchange fees + slippage.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

log = logging.getLogger("paper_trader")

TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "1d": 86_400_000,
}


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Config:
    exchange: str = os.getenv("EXCHANGE", "okx")
    symbols: tuple = tuple(
        s.strip()
        for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,ADA/USDT").split(",")
        if s.strip()
    )
    timeframe: str = os.getenv("TIMEFRAME", "4h")             # 4h default to beat fee drag
    candles: int = _env_int("CANDLES", 300)                   # history fetched per symbol
    start_equity: float = _env_float("START_EQUITY", 1000.0)  # per strategy, in quote currency
    risk_pct: float = _env_float("RISK_PCT", 0.01)            # equity risked per trade
    max_pos_pct: float = _env_float("MAX_POS_PCT", 0.20)      # max notional per position
    max_same_side: int = _env_int("MAX_SAME_SIDE", 2)         # max concurrent LONG or SHORT per strategy
    fee: float = _env_float("FEE", 0.001)                     # per side (0.1%)
    slippage: float = _env_float("SLIPPAGE", 0.0005)          # per side (0.05%)
    max_bars: int = _env_int("MAX_BARS", 48)                  # base time stop
    min_notional: float = _env_float("MIN_NOTIONAL", 10.0)
    db_path: str = os.getenv("DB_PATH", "trades.db")
    report_path: str = os.getenv("REPORT_PATH", "docs/index.html")
    min_trades_ready: int = _env_int("MIN_TRADES_READY", 30)
    backtest_days: int = _env_int("BACKTEST_DAYS", 730)
    backtest_path: str = os.getenv("BACKTEST_PATH", "backtest_4h.json")


# --------------------------------------------------------------------------- #
# Time helpers (module-level so tests can patch now_ms)
# --------------------------------------------------------------------------- #
def now_ms() -> int:
    return int(time.time() * 1000)


def fmt_ts(ms: int | float | None) -> str:
    if ms is None or (isinstance(ms, float) and math.isnan(ms)):
        return "-"
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
def make_exchange(cfg: Config):
    import ccxt  # imported lazily so report/tests work without it

    if not hasattr(ccxt, cfg.exchange):
        raise SystemExit(f"Unknown ccxt exchange '{cfg.exchange}'")
    return getattr(ccxt, cfg.exchange)({"enableRateLimit": True, "timeout": 20_000})


def fetch_candles(ex, symbol: str, timeframe: str, limit: int, retries: int = 3) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
            return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        except Exception as e:
            last_err = e
            log.warning("fetch %s attempt %d/%d failed: %s", symbol, attempt, retries, e)
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not fetch {symbol}: {last_err}")


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = atr(df, n)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / n, adjust=False).mean() / tr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0.0)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema50"] = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)
    d["ema200_prev5"] = d["ema200"].shift(5)               # 5-bar slope reference
    d["rsi"] = rsi(d["close"], 14)
    d["atr"] = atr(d, 14)
    d["adx"] = adx(d, 14)                                   # trend strength filter
    d["hh20"] = d["high"].rolling(20).max().shift(1)        # prior 20-bar high
    d["vol_ma"] = d["volume"].rolling(20).mean().shift(1)   # prior 20-bar avg volume
    d["hh55"] = d["high"].rolling(55).max().shift(1)        # prior 55-bar high
    d["ll20"] = d["low"].rolling(20).min().shift(1)         # prior 20-bar low
    return d


# --------------------------------------------------------------------------- #
# Strategies (evaluated on the last CLOSED candle; shared by live & backtest)
# --------------------------------------------------------------------------- #
@dataclass
class Signal:
    strategy: str
    symbol: str
    signal_ts: int
    entry: float
    stop: float
    target: float
    reason: str
    side: str = "LONG"


@dataclass
class Scan:
    checks: list                      # [(label, passed, detail), ...]
    signal: Signal | None = None

    @property
    def passed(self) -> int:
        return sum(1 for _, ok, _ in self.checks if ok)


IND_COLS = ["ema50", "ema200", "ema200_prev5", "rsi", "atr", "adx", "hh20", "vol_ma", "hh55", "ll20"]


def _not_ready(last, n: int) -> Scan | None:
    if n < 210 or any(math.isnan(getattr(last, c)) for c in IND_COLS):
        return Scan([("Enough candle history (210+)", False, f"{n} candles")])
    return None


def _finish(name, symbol, last, checks, stop, target, reason, side="LONG") -> Scan:
    sig = None
    if all(ok for _, ok, _ in checks):
        sig = Signal(name, symbol, int(last.ts), float(last.close), float(stop), float(target), reason, side)
    return Scan(checks, sig)


def strat_pullback(symbol, last, prev, n) -> Scan:
    """Uptrend pullback: EMA50 > rising EMA200, ADX >= 18, RSI turns up through 42 on green candle. Stop 1.5 ATR, 2.5R."""
    if (nr := _not_ready(last, n)) is not None:
        return nr
    rising = last.ema200 > last.ema200_prev5
    checks = [
        ("Uptrend: EMA50 > rising EMA200", bool(last.ema50 > last.ema200 and rising),
         f"EMA50 {last.ema50:.6g} vs EMA200 {last.ema200:.6g}"),
        ("Trend strength: ADX >= 18", bool(last.adx >= 18), f"ADX {last.adx:.1f}"),
        ("Price above EMA200", bool(last.close > last.ema200), f"price {last.close:.6g}"),
        ("RSI turned up through 42", bool(prev.rsi < 42 <= last.rsi), f"RSI {prev.rsi:.0f} -> {last.rsi:.0f}"),
        ("Last candle is green", bool(last.close > last.open), "green" if last.close > last.open else "red"),
    ]
    risk = 1.5 * last.atr
    return _finish("pullback", symbol, last, checks, last.close - risk, last.close + 2.5 * risk,
                   f"RSI {prev.rsi:.0f}->{last.rsi:.0f} in rising uptrend")


def strat_breakout(symbol, last, prev, n) -> Scan:
    """Volume breakout: close above prior 20-bar high on 1.5x volume with ADX >= 20 & rising EMA200. Stop 2 ATR, 2.5R."""
    if (nr := _not_ready(last, n)) is not None:
        return nr
    vr = last.volume / last.vol_ma if last.vol_ma else 0.0
    rising = last.ema200 > last.ema200_prev5
    checks = [
        ("Uptrend: EMA50 > rising EMA200", bool(last.ema50 > last.ema200 and rising),
         f"EMA50 {last.ema50:.6g} vs EMA200 {last.ema200:.6g}"),
        ("Trend strength: ADX >= 20", bool(last.adx >= 20), f"ADX {last.adx:.1f}"),
        ("Close above prior 20-candle high", bool(last.close > last.hh20),
         f"close {last.close:.6g} vs high {last.hh20:.6g}"),
        ("Volume at least 1.5x average", bool(vr > 1.5), f"{vr:.1f}x average"),
    ]
    risk = 2.0 * last.atr
    return _finish("breakout", symbol, last, checks, last.close - risk, last.close + 2.5 * risk,
                   f"20-candle breakout, volume {vr:.1f}x")


def strat_trend(symbol, last, prev, n) -> Scan:
    """Donchian trend entry: fresh close above 55-bar high above rising EMA200 & ADX >= 20. Trailed stop, 4R ceiling."""
    if (nr := _not_ready(last, n)) is not None:
        return nr
    rising = last.ema200 > last.ema200_prev5
    checks = [
        ("Price above rising EMA200", bool(last.close > last.ema200 and rising),
         f"price {last.close:.6g} vs EMA200 {last.ema200:.6g}"),
        ("Trend strength: ADX >= 20", bool(last.adx >= 20), f"ADX {last.adx:.1f}"),
        ("Close above prior 55-candle high", bool(last.close > last.hh55),
         f"close {last.close:.6g} vs high {last.hh55:.6g}"),
        ("Fresh breakout (previous candle was not one)", bool(prev.close <= prev.hh55),
         "new breakout" if prev.close <= prev.hh55 else "already above the high"),
    ]
    risk = 2.5 * last.atr
    return _finish("trend", symbol, last, checks, last.close - risk, last.close + 4.0 * risk,
                   "55-candle trend breakout (trailed)")


def strat_short_pullback(symbol, last, prev, n) -> Scan:
    """SHORT: downtrend rally. EMA50 < falling EMA200, ADX >= 18, RSI turns down through 58 on red candle. Stop 1.5 ATR, 2.2R."""
    if (nr := _not_ready(last, n)) is not None:
        return nr
    falling = last.ema200 < last.ema200_prev5
    checks = [
        ("Downtrend: EMA50 < falling EMA200", bool(last.ema50 < last.ema200 and falling),
         f"EMA50 {last.ema50:.6g} vs EMA200 {last.ema200:.6g}"),
        ("Trend strength: ADX >= 18", bool(last.adx >= 18), f"ADX {last.adx:.1f}"),
        ("Price below EMA200", bool(last.close < last.ema200), f"price {last.close:.6g}"),
        ("RSI turned down through 58", bool(prev.rsi > 58 >= last.rsi), f"RSI {prev.rsi:.0f} -> {last.rsi:.0f}"),
        ("Last candle is red", bool(last.close < last.open), "red" if last.close < last.open else "green"),
    ]
    risk = 1.5 * last.atr
    return _finish("short_pullback", symbol, last, checks, last.close + risk, last.close - 2.2 * risk,
                   f"RSI {prev.rsi:.0f}->{last.rsi:.0f} in downtrend", "SHORT")


def strat_short_breakdown(symbol, last, prev, n) -> Scan:
    """SHORT: close below prior 20-bar low on 1.5x volume in falling downtrend with ADX >= 20. Stop 2 ATR, 2.2R."""
    if (nr := _not_ready(last, n)) is not None:
        return nr
    vr = last.volume / last.vol_ma if last.vol_ma else 0.0
    falling = last.ema200 < last.ema200_prev5
    checks = [
        ("Downtrend: EMA50 < falling EMA200", bool(last.ema50 < last.ema200 and falling),
         f"EMA50 {last.ema50:.6g} vs EMA200 {last.ema200:.6g}"),
        ("Trend strength: ADX >= 20", bool(last.adx >= 20), f"ADX {last.adx:.1f}"),
        ("Close below prior 20-candle low", bool(last.close < last.ll20),
         f"close {last.close:.6g} vs low {last.ll20:.6g}"),
        ("Volume at least 1.5x average", bool(vr > 1.5), f"{vr:.1f}x average"),
    ]
    risk = 2.0 * last.atr
    return _finish("short_breakdown", symbol, last, checks, last.close + risk, last.close - 2.2 * risk,
                   f"20-candle breakdown, volume {vr:.1f}x", "SHORT")


def strat_regime_switch(symbol, last, prev, n) -> Scan:
    """ADAPTIVE: Trades Breakout in bull regimes (EMA50 >= EMA200) and Short Pullback in bear regimes."""
    if (nr := _not_ready(last, n)) is not None:
        return nr
    if last.ema50 >= last.ema200:
        sc = strat_breakout(symbol, last, prev, n)
    else:
        sc = strat_short_pullback(symbol, last, prev, n)
    if sc.signal is not None:
        sc.signal.strategy = "regime_switch"
    return sc


STRATEGIES = {
    "regime_switch": strat_regime_switch,
    "short_pullback": strat_short_pullback,
    "breakout": strat_breakout,
    "trend": strat_trend,
    "short_breakdown": strat_short_breakdown,
    "pullback": strat_pullback,
}
INFO = {
    "regime_switch": "ADAPTIVE: Long breakouts in bull regime, short rallies in bear regime",
    "short_pullback": "SHORT: sells rallies inside a falling downtrend",
    "breakout": "LONG: buys strong volume breakouts in rising uptrend",
    "trend": "LONG: rides 55-candle breakouts with trailing stop",
    "short_breakdown": "SHORT: sells strong volume breakdowns",
    "pullback": "LONG: buys dips inside a rising uptrend",
}


# --------------------------------------------------------------------------- #
# Journal (SQLite)
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
    signal_ts INTEGER NOT NULL, entry_ts INTEGER NOT NULL,
    entry_price REAL NOT NULL, stop REAL NOT NULL, target REAL NOT NULL,
    qty REAL NOT NULL, notional REAL NOT NULL, risk_amt REAL NOT NULL, entry_fee REAL NOT NULL,
    reason TEXT, side TEXT NOT NULL DEFAULT 'LONG',
    status TEXT NOT NULL DEFAULT 'OPEN',
    exit_ts INTEGER, exit_price REAL, exit_reason TEXT, outcome TEXT,
    pnl REAL, r_multiple REAL, bars_held INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_trade ON trades(strategy, symbol, signal_ts);
CREATE TABLE IF NOT EXISTS scans (
    strategy TEXT NOT NULL, symbol TEXT NOT NULL, ts INTEGER NOT NULL, candle_ts INTEGER,
    price REAL, status TEXT NOT NULL, passed INTEGER, total INTEGER, checks TEXT,
    PRIMARY KEY (strategy, symbol)
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL
);
"""


def connect(path: str) -> sqlite3.Connection:
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    if "side" not in [r[1] for r in con.execute("PRAGMA table_info(trades)")]:
        con.execute("ALTER TABLE trades ADD COLUMN side TEXT NOT NULL DEFAULT 'LONG'")
        con.commit()
    return con


def strategy_equity(con, cfg: Config, strategy: str) -> float:
    r = con.execute(
        "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE strategy=? AND status='CLOSED'", (strategy,)
    ).fetchone()
    return cfg.start_equity + r[0]


def open_notional(con, strategy: str) -> float:
    r = con.execute(
        "SELECT COALESCE(SUM(notional),0) FROM trades WHERE strategy=? AND status='OPEN'", (strategy,)
    ).fetchone()
    return r[0]


def open_same_side_count(con, strategy: str, side: str) -> int:
    r = con.execute(
        "SELECT COUNT(*) FROM trades WHERE strategy=? AND status='OPEN' AND side=?", (strategy, side)
    ).fetchone()
    return int(r[0])


def gapped(side: str, o: float, stop: float, target: float) -> bool:
    """True if the entry price is already beyond the stop or the target (setup invalidated)."""
    return (o <= stop or o >= target) if side == "LONG" else (o >= stop or o <= target)


def entry_fill(side: str, o: float, cfg: Config) -> float:
    return o * (1 + cfg.slippage) if side == "LONG" else o * (1 - cfg.slippage)


def size_position(cfg: Config, equity: float, open_notional_: float, fill: float, stop: float, side: str = "LONG"):
    """Shared by live and backtest: risk-based size, capped per position and by free equity."""
    per_unit_risk = (fill - stop) if side == "LONG" else (stop - fill)
    if per_unit_risk <= 0 or equity <= 0:
        return None
    qty = equity * cfg.risk_pct / per_unit_risk
    cap = min(cfg.max_pos_pct * equity, equity - open_notional_)
    if qty * fill > cap:
        qty = cap / fill
    notional = qty * fill
    if notional < cfg.min_notional:
        return None
    return qty, notional, qty * per_unit_risk


def open_position(con, cfg: Config, sig: Signal, next_open: float) -> sqlite3.Row | None:
    """Enter at the NEXT candle's open (plus adverse slippage). Enforces correlation cap."""
    if open_same_side_count(con, sig.strategy, sig.side) >= cfg.max_same_side:
        return None
    if gapped(sig.side, next_open, sig.stop, sig.target):
        return None
    fill = entry_fill(sig.side, next_open, cfg)
    sz = size_position(cfg, strategy_equity(con, cfg, sig.strategy), open_notional(con, sig.strategy),
                       fill, sig.stop, sig.side)
    if sz is None:
        return None
    qty, notional, risk_amt = sz
    entry_ts = sig.signal_ts + TF_MS[cfg.timeframe]
    cur = con.execute(
        """INSERT OR IGNORE INTO trades
           (strategy,symbol,timeframe,signal_ts,entry_ts,entry_price,stop,target,qty,notional,
            risk_amt,entry_fee,reason,side)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sig.strategy, sig.symbol, cfg.timeframe, sig.signal_ts, entry_ts, fill, sig.stop,
         sig.target, qty, notional, risk_amt, notional * cfg.fee, sig.reason, sig.side),
    )
    if cur.rowcount == 0:
        return None
    return con.execute("SELECT * FROM trades WHERE id=?", (cur.lastrowid,)).fetchone()


def simulate_exit(side, entry_price, stop, target, qty, entry_fee, risk_amt, candles, cfg: Config, strategy: str = ""):
    """Unified exit routine for live and backtest.
    - Pessimistic intrabar order: checks stop before target.
    - Ratchets stop to breakeven + fees once unrealized profit reaches 1.2R.
    - Trails stop at prior 20-bar low (ll20) for 'trend' strategy and doubles its max hold window."""
    long = side == "LONG"
    cur_stop = float(stop)
    init_risk = (entry_price - cur_stop) if long else (cur_stop - entry_price)
    fee_buffer = entry_price * (2 * cfg.fee + cfg.slippage)
    max_hold = cfg.max_bars * 2 if strategy == "trend" else cfg.max_bars

    for bars, row in enumerate(candles.itertuples(index=False), start=1):
        close_ts = int(row.ts) + TF_MS[cfg.timeframe]
        if long and row.low <= cur_stop:
            exit_px, reason = min(cur_stop, row.open) * (1 - cfg.slippage), "STOP"
        elif not long and row.high >= cur_stop:
            exit_px, reason = max(cur_stop, row.open) * (1 + cfg.slippage), "STOP"
        elif long and row.high >= target:
            exit_px, reason = max(target, row.open) * (1 - cfg.slippage), "TARGET"
        elif not long and row.low <= target:
            exit_px, reason = min(target, row.open) * (1 + cfg.slippage), "TARGET"
        elif bars >= max_hold:
            exit_px, reason = row.close * ((1 - cfg.slippage) if long else (1 + cfg.slippage)), "TIME"
        else:
            if init_risk > 0:
                if long and (row.high - entry_price) >= 1.2 * init_risk:
                    cur_stop = max(cur_stop, entry_price + fee_buffer)
                elif not long and (entry_price - row.low) >= 1.2 * init_risk:
                    cur_stop = min(cur_stop, entry_price - fee_buffer)
            if strategy == "trend" and long and hasattr(row, "ll20") and not math.isnan(row.ll20):
                cur_stop = max(cur_stop, float(row.ll20))
            continue

        gross = (exit_px - entry_price) * qty if long else (entry_price - exit_px) * qty
        pnl = gross - entry_fee - exit_px * qty * cfg.fee
        return {"exit_ts": close_ts, "exit_price": float(exit_px), "exit_reason": reason, "pnl": float(pnl),
                "r_multiple": float(pnl / risk_amt) if risk_amt > 0 else 0.0,
                "outcome": "WIN" if pnl > 0 else "LOSS", "bars_held": bars}
    return None


def find_exit(trade: sqlite3.Row, candles: pd.DataFrame, cfg: Config):
    after = candles[candles["ts"] > trade["signal_ts"]]
    return simulate_exit(trade["side"], trade["entry_price"], trade["stop"], trade["target"], trade["qty"],
                         trade["entry_fee"], trade["risk_amt"], after, cfg, trade["strategy"])


def close_position(con, trade: sqlite3.Row, res: dict):
    con.execute(
        """UPDATE trades SET status='CLOSED', exit_ts=?, exit_price=?, exit_reason=?, outcome=?,
           pnl=?, r_multiple=?, bars_held=? WHERE id=?""",
        (res["exit_ts"], res["exit_price"], res["exit_reason"], res["outcome"], res["pnl"],
         res["r_multiple"], res["bars_held"], trade["id"]),
    )
    return res["outcome"], res["pnl"], res["r_multiple"]


# --------------------------------------------------------------------------- #
# Notifications (optional Telegram)
# --------------------------------------------------------------------------- #
def notify(text: str) -> None:
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=10).read()
    except Exception as e:
        log.warning("telegram failed: %s", e)


# --------------------------------------------------------------------------- #
# One trading cycle
# --------------------------------------------------------------------------- #
def log_event(con, kind: str, text: str) -> None:
    con.execute("INSERT INTO events (ts, kind, text) VALUES (?,?,?)", (now_ms(), kind, text))
    con.execute("DELETE FROM events WHERE id <= (SELECT MAX(id) FROM events) - 300")


def save_scan(con, name, symbol, scan: Scan, status, price, candle_ts) -> None:
    con.execute(
        "INSERT OR REPLACE INTO scans VALUES (?,?,?,?,?,?,?,?,?)",
        (name, symbol, now_ms(), candle_ts, price, status, scan.passed, len(scan.checks),
         json.dumps(scan.checks)),
    )


def run_once(cfg: Config, con: sqlite3.Connection, ex) -> dict:
    if cfg.timeframe not in TF_MS:
        raise SystemExit(f"Unsupported timeframe {cfg.timeframe}")
    tf = TF_MS[cfg.timeframe]
    summary = {"opened": 0, "closed": 0, "errors": 0, "signals": 0}
    ctx = []

    for symbol in cfg.symbols:
        try:
            raw = fetch_candles(ex, symbol, cfg.timeframe, cfg.candles)
        except Exception as e:
            log.error("%s", e)
            summary["errors"] += 1
            log_event(con, "error", f"Could not fetch {symbol} data from {cfg.exchange}")
            continue

        closed = raw[raw["ts"] + tf <= now_ms()].reset_index(drop=True)
        if len(closed) < 2:
            continue
        data = add_indicators(closed)
        price, candle_ts = float(data.iloc[-1]["close"]), int(data.iloc[-1]["ts"])
        con.execute("INSERT OR IGNORE INTO meta VALUES (?,?)",
                    (f"bh:{symbol}", json.dumps({"ts": candle_ts, "price": price})))

        # 1) manage open trades using enriched indicator dataframe (for ll20 trailing)
        for tr in con.execute("SELECT * FROM trades WHERE status='OPEN' AND symbol=?", (symbol,)).fetchall():
            res = find_exit(tr, data, cfg)
            if res:
                outcome, pnl, r = close_position(con, tr, res)
                summary["closed"] += 1
                msg = f"[{tr['strategy']}] {symbol} {outcome} via {res['exit_reason']}: {pnl:+.2f} ({r:+.2f}R)"
                log.info(msg)
                log_event(con, "close", msg)
                notify(msg)

        ctx.append((symbol, raw, data, price, candle_ts))

    # 2) scan every strategy AFTER all exits are booked
    for symbol, raw, data, price, candle_ts in ctx:
        last, prev, n = data.iloc[-1], data.iloc[-2], len(data)
        fwd = raw[raw["ts"] > candle_ts]
        next_open = float(fwd.iloc[0]["open"]) if not fwd.empty else price
        for name, fn in STRATEGIES.items():
            scan = fn(symbol, last, prev, n)
            busy = con.execute(
                "SELECT 1 FROM trades WHERE status='OPEN' AND strategy=? AND symbol=?", (name, symbol)
            ).fetchone()
            if busy:
                status = "IN_TRADE"
            elif scan.signal is None:
                status = "WAITING"
            else:
                summary["signals"] += 1
                tr = open_position(con, cfg, scan.signal, next_open)
                if tr:
                    status = "SIGNAL"
                    summary["opened"] += 1
                    msg = (f"[{name}] {symbol} {tr['side']} @ {tr['entry_price']:.6g} "
                           f"stop {tr['stop']:.6g} target {tr['target']:.6g} ({scan.signal.reason})")
                    log.info(msg)
                    log_event(con, "open", msg)
                    notify(msg)
                else:
                    status = "SKIPPED"
                    log_event(con, "skip", f"[{name}] {symbol} signal skipped (max_same_side cap, gapped, or cap reached)")
            save_scan(con, name, symbol, scan, status, price, candle_ts)

    log_event(con, "scan", f"Scanned {len(cfg.symbols)} coins: {summary['signals']} signal(s), "
                           f"{summary['opened']} opened, {summary['closed']} closed")
    con.commit()
    return summary


# --------------------------------------------------------------------------- #
# Backtest (true bar-by-bar state machine, identical rules to live)
# --------------------------------------------------------------------------- #
PAGE_SIZE = {"okx": 100, "binance": 1000, "kucoin": 1500, "kraken": 720, "bybit": 1000}


def fetch_history(ex, symbol: str, timeframe: str, days: int, exchange_id: str) -> pd.DataFrame:
    step, end = TF_MS[timeframe], now_ms()
    since = end - days * 86_400_000
    page, rows, pages = PAGE_SIZE.get(exchange_id, 300), [], 0
    while since < end and pages < 1500:
        pages += 1
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=page)
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + step
        if nxt <= since:
            break
        since = nxt
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df[df["ts"] + step <= end].reset_index(drop=True)


def _step_open_trade(tr: dict, row, close_ts: int, cfg: Config) -> bool:
    """Advances an open backtest trade by one candle. Returns True if the trade closed."""
    tr["bars_held"] += 1
    long = tr["side"] == "LONG"
    cur_stop = tr["cur_stop"]
    target = tr["target"]
    entry_price = tr["entry_price"]

    if long and row.low <= cur_stop:
        exit_px, reason = min(cur_stop, row.open) * (1 - cfg.slippage), "STOP"
    elif not long and row.high >= cur_stop:
        exit_px, reason = max(cur_stop, row.open) * (1 + cfg.slippage), "STOP"
    elif long and row.high >= target:
        exit_px, reason = max(target, row.open) * (1 - cfg.slippage), "TARGET"
    elif not long and row.low <= target:
        exit_px, reason = min(target, row.open) * (1 + cfg.slippage), "TARGET"
    elif tr["bars_held"] >= tr["max_hold"]:
        exit_px, reason = row.close * ((1 - cfg.slippage) if long else (1 + cfg.slippage)), "TIME"
    else:
        init_risk = tr["init_risk"]
        if init_risk > 0:
            if long and (row.high - entry_price) >= 1.2 * init_risk:
                tr["cur_stop"] = max(cur_stop, entry_price + tr["fee_buffer"])
            elif not long and (entry_price - row.low) >= 1.2 * init_risk:
                tr["cur_stop"] = min(cur_stop, entry_price - tr["fee_buffer"])
        if tr["strategy"] == "trend" and long and not math.isnan(row.ll20):
            tr["cur_stop"] = max(tr["cur_stop"], float(row.ll20))
        return False

    qty = tr["qty"]
    gross = (exit_px - entry_price) * qty if long else (entry_price - exit_px) * qty
    pnl = gross - tr["entry_fee"] - exit_px * qty * cfg.fee
    tr.update({
        "exit_ts": close_ts, "exit_price": float(exit_px), "exit_reason": reason,
        "pnl": float(pnl), "r_multiple": float(pnl / tr["risk_amt"]) if tr["risk_amt"] > 0 else 0.0,
        "outcome": "WIN" if pnl > 0 else "LOSS",
    })
    for k in ("cur_stop", "init_risk", "fee_buffer", "max_hold", "entry_fee"):
        tr.pop(k, None)
    return True


def portfolio_backtest(cfg: Config, frames: dict[str, pd.DataFrame]):
    """Replays history candle-by-candle without look-ahead pre-computation."""
    tf = TF_MS[cfg.timeframe]
    data = {}
    for sym, df in frames.items():
        if len(df) >= 260:
            d = add_indicators(df).reset_index(drop=True)
            recs = [SimpleNamespace(**r) for r in d.to_dict("records")]
            data[sym] = (d, recs, {int(t): i for i, t in enumerate(d["ts"])})
    if not data:
        return [], None
    trade_from = max(int(d["ts"].iloc[210]) for d, _, _ in data.values())
    times = sorted({int(t) for d, _, _ in data.values() for t in d["ts"] if t >= trade_from})
    realized = {k: 0.0 for k in STRATEGIES}
    open_, done = [], []

    for t in times:
        now_t = t + tf
        # 1) Step all active trades on candle `t` (which just closed at `now_t`)
        keep = []
        for tr in open_:
            _, recs, pos = data[tr["symbol"]]
            idx = pos.get(t)
            if idx is not None and _step_open_trade(tr, recs[idx], now_t, cfg):
                realized[tr["strategy"]] += tr["pnl"]
                done.append(tr)
            else:
                keep.append(tr)
        open_ = keep

        # 2) Scan for new entries filling at candle `t + tf` open (`recs[i + 1].open`)
        for sym, (d, recs, pos) in data.items():
            i = pos.get(t)
            if i is None or i + 1 >= len(recs):
                continue
            for name, fn in STRATEGIES.items():
                if any(o["strategy"] == name and o["symbol"] == sym for o in open_):
                    continue
                sig = fn(sym, recs[i], recs[i - 1], i + 1).signal
                if sig is None:
                    continue
                if sum(1 for o in open_ if o["strategy"] == name and o["side"] == sig.side) >= cfg.max_same_side:
                    continue
                raw_entry = recs[i + 1].open
                if gapped(sig.side, raw_entry, sig.stop, sig.target):
                    continue
                fill = entry_fill(sig.side, raw_entry, cfg)
                taken = sum(o["notional"] for o in open_ if o["strategy"] == name)
                sz = size_position(cfg, cfg.start_equity + realized[name], taken, fill, sig.stop, sig.side)
                if sz is None:
                    continue
                qty, notional, risk_amt = sz
                long = sig.side == "LONG"
                open_.append({
                    "strategy": name, "side": sig.side, "symbol": sym, "signal_ts": sig.signal_ts,
                    "entry_ts": now_t, "entry_price": fill, "stop": sig.stop, "cur_stop": float(sig.stop),
                    "target": sig.target, "qty": qty, "notional": notional, "risk_amt": risk_amt,
                    "entry_fee": notional * cfg.fee,
                    "init_risk": (fill - sig.stop) if long else (sig.stop - fill),
                    "fee_buffer": fill * (2 * cfg.fee + cfg.slippage),
                    "max_hold": cfg.max_bars * 2 if name == "trend" else cfg.max_bars,
                    "bars_held": 0,
                })
    return done, {"data": data, "trade_from": trade_from, "end": int(max(d["ts"].iloc[-1] for d, _, _ in data.values())) + tf}


def _buy_hold(data: dict, t0: int, t1: int) -> float:
    rets = []
    for d, _, _ in data.values():
        ts = d["ts"].to_numpy()
        i0 = min(int(np.searchsorted(ts, t0, side="left")), len(ts) - 1)
        i1 = max(int(np.searchsorted(ts, t1, side="left")) - 1, 0)
        rets.append(d["close"].iloc[i1] / d["open"].iloc[i0] - 1)
    return float(np.mean(rets))


def _clean(st: dict) -> dict:
    out = {}
    for k, v in st.items():
        if k == "curve":
            continue
        v = float(v) if isinstance(v, (np.floating, float)) else v
        out[k] = ("inf" if v > 0 else 0.0) if isinstance(v, float) and not math.isfinite(v) else v
    return out


def run_backtest(cfg: Config, ex) -> dict:
    tf = TF_MS[cfg.timeframe]
    frames, coverage = {}, {}
    expected = (cfg.backtest_days + 10) * 86_400_000 / tf
    for sym in cfg.symbols:
        try:
            log.info("downloading history for %s ...", sym)
            frames[sym] = fetch_history(ex, sym, cfg.timeframe, cfg.backtest_days + 10, cfg.exchange)
            h = frames[sym]
            coverage[sym] = {"candles": len(h), "complete": round(len(h) / expected, 2),
                             "from": fmt_ts(h["ts"].iloc[0]) if len(h) else "-",
                             "to": fmt_ts(h["ts"].iloc[-1]) if len(h) else "-"}
        except Exception as e:
            log.error("history failed for %s: %s", sym, e)
            coverage[sym] = {"candles": 0, "complete": 0, "from": "-", "to": "-", "error": str(e)[:120]}
    trades, meta = portfolio_backtest(cfg, frames)
    res = {"generated_at": now_ms(), "exchange": cfg.exchange, "timeframe": cfg.timeframe,
           "days": cfg.backtest_days, "fee": cfg.fee, "slippage": cfg.slippage,
           "start_equity": cfg.start_equity, "coverage": coverage, "strategies": {}, "benchmark": None,
           "notes": [
               "Simulated history, not a forecast. Costs: exchange fee and slippage on both sides.",
               "Entry is at the next candle's open plus slippage; if stop and target are inside one candle, the stop is assumed first.",
               f"One shared account per strategy, max {cfg.max_same_side} concurrent same-side trades, +1.2R breakeven ratchet.",
               "Candle data cannot show order-book depth, outages or exact price paths inside a candle.",
               "'Earlier 70%' and 'latest 30%' check consistency across market regimes.",
               "Buy and hold is fully invested while each strategy risks only a small slice per trade."]}
    if meta:
        t0, t1 = meta["trade_from"], meta["end"]
        split = int(t0 + 0.7 * (t1 - t0))
        res["window"] = {"from": fmt_ts(t0), "to": fmt_ts(t1), "split": fmt_ts(split)}
        res["benchmark"] = {"all": _buy_hold(meta["data"], t0, t1), "early": _buy_hold(meta["data"], t0, split),
                            "late": _buy_hold(meta["data"], split, t1)}
        for name in STRATEGIES:
            mine = [t for t in trades if t["strategy"] == name]
            sets = {"all": mine, "early": [t for t in mine if t["entry_ts"] < split],
                    "late": [t for t in mine if t["entry_ts"] >= split]}
            sts = {k: compute_stats(pd.DataFrame(v), cfg.start_equity) for k, v in sets.items()}
            res["strategies"][name] = {k: _clean(v) for k, v in sts.items()}
            res["strategies"][name]["curve"] = [float(x) for x in sts["all"]["curve"]]
            res["strategies"][name]["trades"] = sorted(mine, key=lambda x: x["exit_ts"], reverse=True)[:100]
    Path(cfg.backtest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.backtest_path).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


# --------------------------------------------------------------------------- #
# Statistics and dashboard
# --------------------------------------------------------------------------- #
def compute_stats(df: pd.DataFrame, start_equity: float) -> dict:
    n = len(df)
    if n == 0:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "expectancy_r": 0.0, "total_pnl": 0.0, "return_pct": 0.0, "max_dd": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "curve": [start_equity]}
    df = df.sort_values("exit_ts")
    wins, losses = df[df["pnl"] > 0], df[df["pnl"] <= 0]
    gross_win, gross_loss = wins["pnl"].sum(), -losses["pnl"].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    curve = np.concatenate([[start_equity], start_equity + df["pnl"].cumsum().to_numpy()])
    peak = np.maximum.accumulate(curve)
    return {
        "trades": n, "wins": len(wins), "losses": len(losses), "win_rate": len(wins) / n,
        "profit_factor": pf, "expectancy_r": float(df["r_multiple"].mean()),
        "total_pnl": float(df["pnl"].sum()), "return_pct": float(df["pnl"].sum() / start_equity),
        "max_dd": float(((curve - peak) / peak).min()),
        "avg_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
        "curve": curve.tolist(),
    }


def readiness(st: dict, cfg: Config) -> list[tuple[str, bool]]:
    return [
        (f"At least {cfg.min_trades_ready} closed trades ({st['trades']} so far)", st["trades"] >= cfg.min_trades_ready),
        ("Positive expectancy after fees and slippage", st["trades"] > 0 and st["expectancy_r"] > 0),
        ("Profit factor of 1.3 or better", st["profit_factor"] >= 1.3),
        ("Max drawdown shallower than 15%", st["trades"] > 0 and st["max_dd"] > -0.15),
    ]


_COLORS = {
    "regime_switch": "#14b8a6", "short_pullback": "#ef4444", "breakout": "#f59e0b",
    "trend": "#10b981", "short_breakdown": "#8b5cf6", "pullback": "#3b82f6",
}
_STATUS = {"WAITING": ("Waiting", "wait"), "SIGNAL": ("Trade taken", "ok"),
           "IN_TRADE": ("In a trade", "info"), "SKIPPED": ("Signal skipped", "warn")}
esc = html.escape


def _nice(name: str) -> str:
    return name.replace('_', ' ')


CSS = """
:root{--bg:#f6f7fb;--card:#fff;--fg:#0f172a;--mut:#64748b;--line:#e2e8f0;--ok:#059669;--bad:#dc2626;
--warn:#d97706;--info:#2563eb;--soft:#f1f5f9}
@media(prefers-color-scheme:dark){:root{--bg:#0b1020;--card:#131a2e;--fg:#e6e9f2;--mut:#8b97b1;--line:#232c47;
--ok:#34d399;--bad:#f87171;--warn:#fbbf24;--info:#60a5fa;--soft:#1a2340}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1000px;margin:auto;padding:20px 16px 48px}
h1{font-size:24px;margin:0}h2{font-size:17px;margin:32px 0 4px}.sub{color:var(--mut);font-size:13px;margin:0 0 12px}
.top{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;justify-content:space-between}
.meta{color:var(--mut);font-size:13px;margin:6px 0 0}
.pill{display:inline-flex;gap:6px;align-items:center;padding:4px 12px;border-radius:99px;font-size:13px;font-weight:600;
background:var(--soft)}.pill:before{content:"";width:8px;height:8px;border-radius:50%;background:var(--ok)}
.pill.warn:before{background:var(--warn)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
.card h3{margin:0;font-size:16px;text-transform:capitalize}.card p{margin:0;color:var(--mut);font-size:13px}
.big{font-size:30px;font-weight:700;margin:10px 0 2px}.delta{font-size:14px;font-weight:600;margin-left:6px}
.up{color:var(--ok)}.down{color:var(--bad)}
.kv{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0}
.kv div{background:var(--soft);border-radius:10px;padding:8px}.kv b{display:block;font-size:16px}
.kv span{color:var(--mut);font-size:12px}
.bar{height:8px;background:var(--soft);border-radius:9px;overflow:hidden}.bar i{display:block;height:100%;background:var(--info)}
.hint{color:var(--mut);font-size:12px;margin-top:6px}
.chart{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px}
.chart svg{width:100%;height:auto}.grid{stroke:var(--line);stroke-width:1}.axis{fill:var(--mut);font-size:11px}
.base{stroke:var(--mut);stroke-dasharray:4 4;opacity:.5}.legend{color:var(--mut);font-size:13px;padding:0 6px}
.empty{color:var(--mut);padding:28px 8px;text-align:center}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:14px}
table{border-collapse:collapse;width:100%;font-size:14px}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);text-align:left;padding:10px 12px}
td{padding:10px 12px;border-top:1px solid var(--line);vertical-align:top}
.chip{display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;font-weight:600;background:var(--soft)}
.chip.ok{color:var(--ok)}.chip.warn{color:var(--warn)}.chip.info{color:var(--info)}.chip.wait{color:var(--mut)}
.chip.bad{color:var(--bad)}
details summary{cursor:pointer;list-style:none}details summary::-webkit-details-marker{display:none}
details summary small{display:block;color:var(--mut);font-size:12px;margin-top:2px;max-width:300px;white-space:normal}
details ul{list-style:none;margin:8px 0 0;padding:8px 10px;background:var(--soft);border-radius:10px;font-size:13px}
details li{padding:2px 0}details li.y span{color:var(--ok)}details li.n span{color:var(--bad)}
details li em{display:block;color:var(--mut);font-style:normal;font-size:12px;margin-left:18px}
.log{list-style:none;margin:0;padding:0;background:var(--card);border:1px solid var(--line);border-radius:14px}
.log li{display:flex;gap:12px;padding:9px 14px;border-top:1px solid var(--line);font-size:13px}
.log li:first-child{border-top:0}.log time{color:var(--mut);min-width:96px}
.coin{font-weight:600}.banner{background:var(--soft);border-left:4px solid var(--warn);border-radius:8px;padding:10px 14px;font-size:13px;margin:10px 0}
.notes{color:var(--mut);font-size:13px}.scan details{border-top:1px solid var(--line);padding:6px 0}.scan .sh{display:flex;justify-content:space-between;margin-bottom:6px}.mut{color:var(--mut)}
"""

JS = """
document.querySelectorAll('time[data-ts]').forEach(function(e){var d=new Date(+e.dataset.ts);
e.title=e.textContent;e.textContent=d.toLocaleString([],{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'})});
var p=document.getElementById('pill');if(p){var age=Date.now()-(+p.dataset.last),m=Math.round(age/60000);
if(age>(+p.dataset.max)){p.className='pill warn';p.textContent='No scan for '+(m>=120?Math.round(m/60)+' hours':m+' min')+' - check the bot'}}
"""


def _t(ms) -> str:
    return f'<time data-ts="{int(ms)}">{fmt_ts(ms)} UTC</time>' if ms else "-"


def _svg_curves(curves: dict, start: float) -> str:
    if all(len(c) < 2 for c in curves.values()):
        return (f'<div class="empty">The equity curve appears after the first trade closes. '
                f'Each strategy starts with {start:,.0f}.</div>')
    w, h, left, bot, top = 720, 230, 56, 24, 12
    vals = [v for c in curves.values() for v in c]
    lo, hi = min(vals + [start]), max(vals + [start])
    pad = (hi - lo) * 0.1 or hi * 0.01
    lo, hi = lo - pad, hi + pad
    n = max(len(c) for c in curves.values())
    X = lambda k: left + (w - left - 8) * k / max(n - 1, 1)
    Y = lambda v: top + (h - top - bot) * (1 - (v - lo) / (hi - lo))
    grid = "".join(
        f'<line x1="{left}" x2="{w - 8}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
        f'<text x="{left - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" class="axis">{v:,.0f}</text>'
        for v in np.linspace(lo, hi, 5))
    base = f'<line x1="{left}" x2="{w - 8}" y1="{Y(start):.1f}" y2="{Y(start):.1f}" class="base"/>'
    lines = "".join(
        f'<polyline fill="none" stroke="{_COLORS.get(k, "#888")}" stroke-width="2.5" stroke-linejoin="round" '
        f'points="{" ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(c))}"/>'
        for k, c in curves.items())
    legend = " &nbsp; ".join(f'<span style="color:{_COLORS.get(k, "#888")}">&#9679;</span> {_nice(k)}' for k in curves)
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Equity curves">{grid}{base}{lines}</svg>'
            f'<div class="legend">{legend} &nbsp; <span class="mut">x-axis: closed trades, dashed line: starting balance</span></div>')


def _scan_row(name, r) -> str:
    label, cls = _STATUS.get(r["status"], (r["status"], "wait"))
    checks = json.loads(r["checks"])
    miss = next((c for c in checks if not c[1]), None)
    why = {"SIGNAL": "All conditions met", "IN_TRADE": "Managing an open trade",
           "SKIPPED": "Signal found, but no trade opened"}.get(r["status"]) \
        or (f"Missing: {miss[0]}" if miss else "")
    items = "".join(
        f"<li class='{'y' if ok else 'n'}'><span>{'&#10003;' if ok else '&#10007;'}</span> {esc(lbl)}<em>{esc(det)}</em></li>"
        for lbl, ok, det in checks)
    return (f"<details><summary><span class='chip {cls}'>{label}</span> <b>{_nice(name)}</b> "
            f"<span class='mut'>{r['passed']}/{r['total']}</span><small>{esc(why)}</small></summary><ul>{items}</ul></details>")


def _f(x) -> float:
    return float("inf") if x == "inf" else float(x)


def _backtest_html(cfg: Config, bt: dict | None) -> str:
    if not bt or not bt.get("strategies"):
        return ("<h2>Historical backtest</h2><div class='banner'>No backtest yet. On the server run "
                "<code>python paper_trader.py backtest</code> to test the strategies on past candles.</div>")
    sc, bench = bt["start_equity"], bt["benchmark"]
    weak = [k for k, v in bt["coverage"].items() if v.get("complete", 0) < 0.8]
    warn = (f"<div class='banner'>History looks incomplete for {esc(', '.join(weak))} (under 80% of the expected candles). "
            "Treat these results with caution.</div>") if weak else ""
    rows = []
    for name, v in bt["strategies"].items():
        a, e, l = v["all"], v["early"], v["late"]
        checks = [("30+ trades", a["trades"] >= cfg.min_trades_ready),
                  ("positive expectancy", a["trades"] > 0 and a["expectancy_r"] > 0),
                  ("profit factor 1.3+", _f(a["profit_factor"]) >= 1.3),
                  ("drawdown under 15%", a["trades"] > 0 and a["max_dd"] > -0.15),
                  ("positive in both periods", e["trades"] > 0 and l["trades"] > 0 and e["expectancy_r"] > 0 and l["expectancy_r"] > 0)]
        failed = [t for t, ok in checks if not ok]
        cls, label = ("ok", "Passes all 5") if not failed else (("warn", f"Mixed ({5 - len(failed)}/5)") if len(failed) <= 2 else ("bad", f"Not working ({5 - len(failed)}/5)"))
        pf = "inf" if a["profit_factor"] == "inf" else f"{_f(a['profit_factor']):.2f}"
        rr = lambda st: f"{st['return_pct']:+.1%} <span class='mut'>({st['trades']})</span>"
        rows.append(f"<tr><td class='coin'>{_nice(name)}</td><td>{a['trades']}</td><td>{a['win_rate']:.0%}</td><td>{pf}</td>"
                    f"<td class='{'up' if a['total_pnl'] > 0 else 'down'}'>{a['return_pct']:+.1%}</td><td>{a['max_dd']:.1%}</td>"
                    f"<td>{rr(e)}</td><td>{rr(l)}</td><td><span class='chip {cls}'>{label}</span>"
                    f"<div class='mut' style='font-size:12px'>{'Missing: ' + ', '.join(failed) if failed else ''}</div></td></tr>")
    rows.append(f"<tr><td class='coin'>Buy &amp; hold</td><td>-</td><td>-</td><td>-</td><td>{bench['all']:+.1%}</td><td>-</td>"
                f"<td>{bench['early']:+.1%}</td><td>{bench['late']:+.1%}</td><td class='mut'>Same coins, no trading</td></tr>")
    tbl = ("<div class='wrap'><table><tr><th>Strategy</th><th>Trades</th><th>Win rate</th><th>Profit factor</th><th>Return</th>"
           "<th>Max drawdown</th><th>Earlier 70% (trades)</th><th>Latest 30% (trades)</th><th>Verdict</th></tr>" + "".join(rows) + "</table></div>")
    cov = "".join(f"<tr><td>{esc(k)}</td><td>{v['candles']}</td><td>{v['complete']:.0%}</td><td>{esc(v['from'])}</td><td>{esc(v['to'])}</td>"
                  f"<td>{esc(v.get('error', ''))}</td></tr>" for k, v in bt["coverage"].items())
    trs = sorted((dict(t, strategy=n) for n, v in bt["strategies"].items() for t in v["trades"]), key=lambda x: x["exit_ts"], reverse=True)[:100]
    trows = "".join(f"<tr><td>{_nice(t['strategy'])}</td><td class='coin'>{esc(t['symbol'])}</td><td>{_t(t['entry_ts'])}</td><td>{_t(t['exit_ts'])}</td>"
                    f"<td>{t['exit_reason']}</td><td class='{'up' if t['pnl'] > 0 else 'down'}'>{t['pnl']:+.2f}</td><td>{t['r_multiple']:+.2f}R</td></tr>" for t in trs)
    curves = {n: v["curve"] for n, v in bt["strategies"].items()}
    w = bt.get("window", {})
    return f"""<h2>Historical backtest</h2>
<p class="sub">{esc(bt['exchange'])} &middot; {esc(bt['timeframe'])} &middot; {esc(w.get('from', '-'))} to {esc(w.get('to', '-'))} UTC &middot;
fees {bt['fee']:.2%} + slippage {bt['slippage']:.2%} per side &middot; run {_t(bt['generated_at'])}. Each strategy has one {sc:,.0f} account shared across all coins.</p>
{warn}{tbl}<div class="chart" style="margin-top:12px">{_svg_curves(curves, sc)}</div>
<details style="margin-top:10px"><summary class="hint">Data coverage by coin</summary><div class="wrap"><table><tr><th>Coin</th><th>Candles</th><th>Complete</th><th>From</th><th>To</th><th>Error</th></tr>{cov}</table></div></details>
<details><summary class="hint">Latest 100 backtest trades</summary><div class="wrap"><table><tr><th>Strategy</th><th>Coin</th><th>Entered</th><th>Exited</th><th>How it ended</th><th>P&amp;L</th><th>R</th></tr>{trows or "<tr><td colspan='7'>None</td></tr>"}</table></div></details>
<details><summary class="hint">Assumptions and limits</summary><ul class="notes">{"".join(f"<li>{esc(n)}</li>" for n in bt['notes'])}</ul></details>"""


def write_report(cfg: Config, con: sqlite3.Connection) -> None:
    tf = TF_MS[cfg.timeframe]
    trades = pd.read_sql_query("SELECT * FROM trades ORDER BY id", con)
    closed, opened = trades[trades["status"] == "CLOSED"], trades[trades["status"] == "OPEN"]
    scans = {(r["strategy"], r["symbol"]): r for r in con.execute("SELECT * FROM scans")}
    events = con.execute("SELECT * FROM events ORDER BY id DESC LIMIT 25").fetchall()
    prices = {sym: r["price"] for (_, sym), r in scans.items()}
    last_scan = max((r["ts"] for r in scans.values()), default=None)
    stats = {n: compute_stats(closed[closed["strategy"] == n], cfg.start_equity) for n in STRATEGIES}

    cards = []
    for n, st in stats.items():
        pf = "-" if st["trades"] == 0 else ("inf" if st["profit_factor"] == float("inf") else f"{st['profit_factor']:.2f}")
        eq = cfg.start_equity + st["total_pnl"]
        cls = "up" if st["total_pnl"] > 0 else "down" if st["total_pnl"] < 0 else "mut"
        prog = min(100, 100 * st["trades"] / cfg.min_trades_ready)
        checks = "".join(f"<li class='{'y' if ok else 'n'}'><span>{'&#10003;' if ok else '&#10007;'}</span> {esc(t)}</li>"
                         for t, ok in readiness(st, cfg))
        cards.append(f"""<article class="card"><h3>{_nice(n)}</h3><p>{INFO[n]}</p>
<div class="big">{eq:,.2f}<span class="delta {cls}">{st['return_pct']:+.2%}</span></div>
<div class="kv"><div><b>{st['trades']}</b><span>closed trades</span></div>
<div><b>{st['win_rate']:.0%}</b><span>win rate ({st['wins']}W/{st['losses']}L)</span></div>
<div><b>{pf}</b><span>profit factor</span></div>
<div><b>{st['expectancy_r']:+.2f}R</b><span>avg per trade</span></div>
<div><b>{st['max_dd']:.1%}</b><span>max drawdown</span></div>
<div><b>{st['total_pnl']:+,.2f}</b><span>total P&amp;L</span></div></div>
<div class="bar"><i style="width:{prog:.0f}%"></i></div>
<div class="hint">{st['trades']} of {cfg.min_trades_ready} trades needed before the results mean much</div>
<details><summary class="hint">Ready for real money? (not yet unless all four are ticked)</summary>
<ul>{checks}</ul></details></article>""")

    ups = downs = 0
    scan_cards = []
    for sym in cfg.symbols:
        r0 = scans.get(("pullback", sym))
        if r0:
            c0 = json.loads(r0["checks"])[0]
            if c0[0].startswith("Uptrend"):
                ups += 1 if c0[1] else 0
                downs += 0 if c0[1] else 1
        body = "".join(_scan_row(n, scans[(n, sym)]) for n in STRATEGIES if (n, sym) in scans) \
            or "<span class='mut'>waiting for first scan</span>"
        px = f"{prices[sym]:.6g}" if sym in prices else "-"
        scan_cards.append(f"<div class='card scan'><div class='sh'><span class='coin'>{esc(sym)}</span>"
                          f"<span class='mut'>{px}</span></div>{body}</div>")
    regime = ""
    if ups + downs:
        why = " Long strategies require rising uptrends; short & adaptive strategies remain active." if ups == 0 else \
              (" Short strategies wait for falling downtrends." if downs == 0 else "")
        regime = (f"<div class='banner'><b>Market right now:</b> {ups} of {ups + downs} coins are in a rising uptrend "
                  f"and {downs} are in a downtrend/chop.{why}</div>")
    scanner = regime + f"<div class='cards'>{''.join(scan_cards)}</div>"

    orows = []
    for r in opened.itertuples():
        now = prices.get(r.symbol)
        short = r.side == "SHORT"
        chg = None if not now else ((r.entry_price - now) if short else (now - r.entry_price)) / r.entry_price
        pl = "-" if chg is None else f"<span class='{'up' if chg >= 0 else 'down'}'>{chg:+.2%}</span>"
        orows.append(f"<tr><td>{_nice(r.strategy)}</td><td class='coin'>{esc(r.symbol)}</td>"
                     f"<td><span class='chip {'bad' if short else 'ok'}'>{r.side}</span></td><td>{_t(r.entry_ts)}</td>"
                     f"<td>{r.entry_price:.6g}</td><td>{f'{now:.6g}' if now else '-'}</td><td>{pl}</td>"
                     f"<td class='down'>{r.stop:.6g}</td><td class='up'>{r.target:.6g}</td></tr>")
    open_tbl = ("<div class='wrap'><table><tr><th>Strategy</th><th>Coin</th><th>Side</th><th>Entered</th><th>Entry</th>"
                "<th>Now</th><th>Open P/L</th><th>Stop</th><th>Target</th></tr>"
                + ("".join(orows) or "<tr><td colspan='9' class='mut'>No open trades right now.</td></tr>") + "</table></div>")

    crows = []
    for r in closed.sort_values("exit_ts", ascending=False).head(50).itertuples():
        win = r.outcome == "WIN"
        how = {"TARGET": "Hit target", "STOP": "Hit stop", "TIME": "Time limit"}.get(r.exit_reason, r.exit_reason)
        crows.append(f"<tr><td>{_nice(r.strategy)}</td><td class='coin'>{esc(r.symbol)}</td>"
                     f"<td><span class='chip {'bad' if r.side == 'SHORT' else 'ok'}'>{r.side}</span></td>"
                     f"<td>{_t(r.entry_ts)}</td><td>{_t(r.exit_ts)}</td><td>{how}</td>"
                     f"<td><span class='chip {'ok' if win else 'bad'}'>{r.outcome}</span></td>"
                     f"<td class='{'up' if win else 'down'}'>{r.pnl:+.2f}</td><td>{r.r_multiple:+.2f}R</td></tr>")
    closed_tbl = ("<div class='wrap'><table><tr><th>Strategy</th><th>Coin</th><th>Side</th><th>Entered</th><th>Exited</th>"
                  "<th>How it ended</th><th>Result</th><th>P&amp;L</th><th>R</th></tr>"
                  + ("".join(crows) or "<tr><td colspan='9' class='mut'>No closed trades yet. Wins and losses will be listed here.</td></tr>")
                  + "</table></div>")

    dot = {"open": "ok", "close": "info", "skip": "warn", "error": "bad", "scan": "wait"}
    log_html = "<ul class='log'>" + "".join(
        f"<li>{_t(e['ts'])}<span><span class='chip {dot.get(e['kind'], 'wait')}'>{e['kind']}</span> {esc(e['text'])}</span></li>"
        for e in events) + ("" if events else "<li class='mut'>Nothing yet.</li>") + "</ul>"

    bt = None
    try:
        bt = json.loads(Path(cfg.backtest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    bh = []
    since = None
    for sym in cfg.symbols:
        m = con.execute("SELECT v FROM meta WHERE k=?", (f"bh:{sym}",)).fetchone()
        if m and sym in prices:
            m = json.loads(m["v"])
            bh.append(prices[sym] / m["price"] - 1)
            since = m["ts"] if since is None else min(since, m["ts"])
    bench_card = ""
    if bh:
        v = float(np.mean(bh))
        bench_card = (f"<article class='card'><h3>Buy &amp; hold</h3><p>Same coins, equal weight, no trading</p>"
                      f"<div class='big'><span class='{'up' if v >= 0 else 'down'}'>{v:+.2%}</span></div>"
                      f"<div class='hint'>Since {_t(since)}. The bot should beat this to be worth running. It is fully invested, "
                      "while the strategies only risk a small slice per trade, so compare roughly.</div></article>")
    bt_html = _backtest_html(cfg, bt)
    nxt = (last_scan // tf + 1) * tf + 15_000 if last_scan else None
    pill = (f'<span class="pill" id="pill" data-last="{last_scan}" data-max="{int(tf * 2.5)}">Live &middot; last scan {_t(last_scan)}</span>'
            if last_scan else '<span class="pill warn">Waiting for first scan</span>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="300">
<title>Paper trading journal</title><style>{CSS}</style></head><body><main>
<div class="top"><div><h1>Paper trading journal</h1>
<p class="meta">Simulated money only &middot; {esc(cfg.exchange)} &middot; {esc(cfg.timeframe)} candles &middot;
{esc(', '.join(cfg.symbols))}</p></div>{pill}</div>
<p class="meta">Each strategy starts with {cfg.start_equity:,.0f}, risks {cfg.risk_pct:.0%} per trade, and caps same-side exposure at {cfg.max_same_side} trades. Fees
{cfg.fee:.2%} and slippage {cfg.slippage:.2%} per side are included. Next scan around {_t(nxt)}.</p>

<h2>Live paper trading</h2><p class="sub">How each strategy is doing so far on live candles.</p><div class="cards">{''.join(cards)}{bench_card}</div>
<h2>Equity curve</h2><p class="sub">Account value after each closed trade.</p>
<div class="chart">{_svg_curves({n: s['curve'] for n, s in stats.items()}, cfg.start_equity)}</div>

{bt_html}
<h2>Why no trade? Live scanner</h2>
<p class="sub">What each strategy saw on the latest closed candle. Tap a line to see every condition. A trade is only
taken when all conditions are met.</p>{scanner}
<h2>Open trades</h2><p class="sub">Positions waiting to hit their stop or target.</p>{open_tbl}
<h2>Closed trades</h2><p class="sub">Every finished trade, marked win or loss.</p>{closed_tbl}
<h2>Activity</h2><p class="sub">What the bot did recently.</p>{log_html}
<p class="meta" style="margin-top:24px">Simulated results. Doing well on a small sample does not prove a strategy
will work with real money. Times are shown in your local time zone.</p>
</main><script>{JS}</script></body></html>"""
    Path(cfg.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.report_path).write_text(page, encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cycle(cfg: Config, ex) -> None:
    con = connect(cfg.db_path)
    try:
        s = run_once(cfg, con, ex)
        write_report(cfg, con)
        log.info("cycle done: opened=%(opened)d closed=%(closed)d errors=%(errors)d", s)
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Crypto paper-trading forward tester")
    ap.add_argument("command", choices=["run", "loop", "report", "backtest"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config()

    if args.command == "report":
        con = connect(cfg.db_path)
        write_report(cfg, con)
        con.close()
        return 0

    ex = make_exchange(cfg)
    if args.command == "backtest":
        res = run_backtest(cfg, ex)
        con = connect(cfg.db_path)
        write_report(cfg, con)
        con.close()
        print(f"Saved {cfg.backtest_path}; dashboard updated: {cfg.report_path}")
        for name, v in res["strategies"].items():
            a = v["all"]
            print(f"{name:15s} trades={a['trades']:4d} win={a['win_rate']:.0%} PF={a['profit_factor']} "
                  f"return={a['return_pct']:+.1%} maxDD={a['max_dd']:.1%}")
        if res["benchmark"]:
            print(f"{'buy&hold':15s} return={res['benchmark']['all']:+.1%}")
        return 0
    if args.command == "run":
        cycle(cfg, ex)
        return 0

    step = TF_MS[cfg.timeframe]
    while True:
        try:
            cycle(cfg, ex)
        except Exception:
            log.exception("cycle failed; will retry next candle")
        wake = (now_ms() // step + 1) * step + 15_000
        time.sleep(max(1.0, (wake - now_ms()) / 1000))


if __name__ == "__main__":
    sys.exit(main())