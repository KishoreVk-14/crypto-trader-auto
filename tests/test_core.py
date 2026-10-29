import math
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
from unittest.mock import patch

import paper_trader as pt


class PaperTraderCoreTests(unittest.TestCase):
    def setUp(self):
        self.cfg = pt.Config(
            timeframe="4h",
            start_equity=1000.0,
            risk_pct=0.01,
            max_pos_pct=0.20,
            fee=0.001,
            slippage=0.0005,
            max_bars=3,
            min_notional=10.0,
        )

    def test_strategy_allowlist_defaults_to_all_and_accepts_one_strategy(self):
        self.assertEqual(set(pt.active_strategy_names(self.cfg)), set(pt.STRATEGIES))
        only_short_pullback = pt.replace(self.cfg, strategies_enabled="short_pullback")
        self.assertEqual(pt.active_strategy_names(only_short_pullback), ("short_pullback",))

    def test_strategy_allowlist_rejects_unknown_names(self):
        cfg = pt.replace(self.cfg, strategies_enabled="short_pullback,not_a_strategy")
        with self.assertRaisesRegex(ValueError, "not_a_strategy"):
            pt.active_strategy_names(cfg)

    def test_next_candle_open_requires_exact_next_timestamp(self):
        tf = pt.TF_MS["4h"]
        raw = pd.DataFrame([
            {"ts": 1000, "open": 100.0},
            {"ts": 1000 + tf, "open": 101.0},
            {"ts": 1000 + 3 * tf, "open": 103.0},
        ])
        self.assertEqual(pt.next_candle_open(raw, 1000, "4h"), 101.0)
        self.assertIsNone(pt.next_candle_open(raw, 1000 + tf, "4h"))
        self.assertIsNone(pt.next_candle_open(raw.iloc[:1], 1000, "4h"))

    def test_live_cycle_uses_exact_next_candle_open_for_entry(self):
        tf = pt.TF_MS["4h"]
        ts = [i * tf for i in range(266)]
        raw = pd.DataFrame([{
            "ts": t, "open": 101.0 if i == 265 else 100.0,
            "high": 102.0, "low": 99.0, "close": 100.0, "volume": 100.0,
        } for i, t in enumerate(ts)])

        def one_signal(symbol, last, prev, n):
            signal = None
            if int(last.ts) == ts[264]:
                signal = pt.Signal("pullback", symbol, int(last.ts), 100.0, 90.0, 120.0,
                                   "test signal", "LONG")
            return SimpleNamespace(signal=signal, passed=False, checks=[])

        con = pt.connect(":memory:")
        try:
            with patch.object(pt, "fetch_candles", return_value=raw), \
                 patch.object(pt, "now_ms", return_value=ts[265] + 1), \
                 patch.dict(pt.STRATEGIES, {"pullback": one_signal}, clear=True):
                summary = pt.run_once(pt.replace(self.cfg, symbols=("BTC/USDT",)), con, SimpleNamespace())

            trade = con.execute("SELECT entry_price, entry_ts FROM trades").fetchone()
            self.assertEqual(summary["opened"], 1)
            self.assertAlmostEqual(trade["entry_price"], 101.0 * (1 + self.cfg.slippage))
            self.assertEqual(trade["entry_ts"], ts[265])
        finally:
            con.close()

    def test_live_cycle_does_not_fabricate_entry_without_next_candle(self):
        tf = pt.TF_MS["4h"]
        ts = [i * tf for i in range(265)]
        raw = pd.DataFrame([{
            "ts": t, "open": 100.0, "high": 102.0, "low": 99.0,
            "close": 100.0, "volume": 100.0,
        } for t in ts])

        def one_signal(symbol, last, prev, n):
            signal = None
            if int(last.ts) == ts[264]:
                signal = pt.Signal("pullback", symbol, int(last.ts), 100.0, 90.0, 120.0,
                                   "test signal", "LONG")
            return SimpleNamespace(signal=signal, passed=False, checks=[])

        con = pt.connect(":memory:")
        try:
            with patch.object(pt, "fetch_candles", return_value=raw), \
                 patch.object(pt, "now_ms", return_value=ts[264] + tf), \
                 patch.dict(pt.STRATEGIES, {"pullback": one_signal}, clear=True):
                summary = pt.run_once(self.cfg, con, SimpleNamespace())

            count = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            status = con.execute("SELECT status FROM scans WHERE strategy='pullback'").fetchone()[0]
            self.assertEqual(summary["opened"], 0)
            self.assertEqual(count, 0)
            self.assertEqual(status, "WAITING_NEXT_OPEN")
        finally:
            con.close()

    def test_open_trade_persists_across_restart_and_duplicate_signal_is_ignored(self):
        signal = pt.Signal(
            "pullback", "BTC/USDT", 123456, 100.0, 90.0, 120.0,
            "persistence test", "LONG",
        )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(__import__("pathlib").Path(tmp) / "paper.db")
            first = pt.connect(db_path)
            try:
                opened = pt.open_position(first, self.cfg, signal, 100.0)
                self.assertIsNotNone(opened)
                trade_id = opened["id"]
                first.commit()
            finally:
                first.close()

            # Simulate process restart: reconnect to the same SQLite journal.
            restarted = pt.connect(db_path)
            try:
                persisted = restarted.execute(
                    "SELECT id, status, entry_price FROM trades WHERE id=?", (trade_id,)
                ).fetchone()
                self.assertIsNotNone(persisted)
                self.assertEqual(persisted["status"], "OPEN")
                self.assertAlmostEqual(
                    persisted["entry_price"], 100.0 * (1 + self.cfg.slippage)
                )

                # Reprocessing the same signal must not create a duplicate trade.
                duplicate = pt.open_position(restarted, self.cfg, signal, 100.0)
                count = restarted.execute(
                    "SELECT COUNT(*) FROM trades WHERE strategy='pullback' AND symbol='BTC/USDT'"
                ).fetchone()[0]
                self.assertIsNone(duplicate)
                self.assertEqual(count, 1)
            finally:
                restarted.close()

    def test_restarted_cycle_resumes_exit_management_for_persisted_open_trade(self):
        tf = pt.TF_MS["4h"]
        ts = [i * tf for i in range(266)]
        raw = pd.DataFrame([{
            "ts": t, "open": 100.0, "high": 101.0,
            "low": 98.0 if i == 264 else 99.5,
            "close": 100.0, "volume": 100.0,
        } for i, t in enumerate(ts)])
        signal = pt.Signal(
            "pullback", "BTC/USDT", ts[263], 100.0, 99.0, 120.0,
            "restart recovery test", "LONG",
        )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(__import__("pathlib").Path(tmp) / "recovery.db")
            first = pt.connect(db_path)
            try:
                opened = pt.open_position(first, pt.replace(self.cfg, symbols=("BTC/USDT",)), signal, 100.0)
                self.assertIsNotNone(opened)
                trade_id = opened["id"]
                first.commit()
            finally:
                first.close()

            # A new process reconnects and must continue managing the persisted position.
            restarted = pt.connect(db_path)
            try:
                def no_signal(symbol, last, prev, n):
                    return SimpleNamespace(signal=None, passed=False, checks=[])

                with (patch.object(pt, "fetch_candles", return_value=raw),
                      patch.object(pt, "now_ms", return_value=ts[266 - 1] + tf),
                      patch.dict(pt.STRATEGIES, {"pullback": no_signal}, clear=True)):
                    summary = pt.run_once(
                        pt.replace(self.cfg, symbols=("BTC/USDT",)), restarted, SimpleNamespace()
                    )

                trade = restarted.execute(
                    "SELECT status, exit_reason, exit_ts FROM trades WHERE id=?", (trade_id,)
                ).fetchone()
                self.assertEqual(summary["closed"], 1)
                self.assertEqual(trade["status"], "CLOSED")
                self.assertEqual(trade["exit_reason"], "STOP")
                self.assertEqual(trade["exit_ts"], ts[264] + tf)
            finally:
                restarted.close()

    def test_closed_trade_stays_closed_after_restart_and_cycle(self):
        tf = pt.TF_MS["4h"]
        signal = pt.Signal(
            "pullback", "BTC/USDT", 123456, 100.0, 90.0, 120.0,
            "closed persistence test", "LONG",
        )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(__import__("pathlib").Path(tmp) / "closed.db")
            first = pt.connect(db_path)
            try:
                opened = pt.open_position(first, self.cfg, signal, 100.0)
                self.assertIsNotNone(opened)
                trade_id = opened["id"]
                pt.close_position(first, opened, {
                    "exit_ts": signal.signal_ts + tf,
                    "exit_price": 105.0,
                    "exit_reason": "TARGET",
                    "outcome": "WIN",
                    "pnl": 4.0,
                    "r_multiple": 0.4,
                    "bars_held": 1,
                })
                first.commit()
            finally:
                first.close()

            restarted = pt.connect(db_path)
            try:
                trade = restarted.execute(
                    "SELECT status, exit_reason, pnl FROM trades WHERE id=?", (trade_id,)
                ).fetchone()
                self.assertEqual(trade["status"], "CLOSED")
                self.assertEqual(trade["exit_reason"], "TARGET")
                self.assertAlmostEqual(trade["pnl"], 4.0)

                # The same signal remains deduplicated after the closed row is reloaded.
                duplicate = pt.open_position(restarted, self.cfg, signal, 100.0)
                count = restarted.execute(
                    "SELECT COUNT(*) FROM trades WHERE strategy='pullback' AND symbol='BTC/USDT'"
                ).fetchone()[0]
                self.assertIsNone(duplicate)
                self.assertEqual(count, 1)
            finally:
                restarted.close()

    def test_open_position_respects_same_side_risk_limit_after_restart(self):
        cfg = pt.replace(self.cfg, max_same_side=1)
        signal_a = pt.Signal(
            "pullback", "BTC/USDT", 123456, 100.0, 90.0, 120.0,
            "risk limit first", "LONG",
        )
        signal_b = pt.Signal(
            "pullback", "ETH/USDT", 123457, 100.0, 90.0, 120.0,
            "risk limit second", "LONG",
        )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(__import__("pathlib").Path(tmp) / "risk.db")
            first = pt.connect(db_path)
            try:
                opened = pt.open_position(first, cfg, signal_a, 100.0)
                self.assertIsNotNone(opened)
                first.commit()
            finally:
                first.close()

            restarted = pt.connect(db_path)
            try:
                # A second symbol must not bypass the strategy's same-side cap after restart.
                blocked = pt.open_position(restarted, cfg, signal_b, 100.0)
                count = restarted.execute(
                    "SELECT COUNT(*) FROM trades WHERE strategy='pullback' AND status='OPEN'"
                ).fetchone()[0]
                self.assertIsNone(blocked)
                self.assertEqual(count, 1)
            finally:
                restarted.close()

    def test_gap_checks_reject_invalid_long_and_short_entries(self):
        self.assertTrue(pt.gapped("LONG", 94.0, 95.0, 110.0))
        self.assertTrue(pt.gapped("LONG", 111.0, 95.0, 110.0))
        self.assertTrue(pt.gapped("SHORT", 111.0, 110.0, 95.0))
        self.assertTrue(pt.gapped("SHORT", 94.0, 110.0, 95.0))
        self.assertFalse(pt.gapped("LONG", 100.0, 95.0, 110.0))
        self.assertFalse(pt.gapped("SHORT", 100.0, 110.0, 95.0))

    def test_position_sizing_respects_cost_adjusted_risk_and_notional_caps(self):
        size = pt.size_position(self.cfg, equity=1000, open_notional_=0,
                                fill=100, stop=90, side="LONG")
        self.assertIsNotNone(size)
        qty, notional, risk_amt = size
        stop_fill = 90 * (1 - self.cfg.slippage)
        estimated_loss_per_unit = (100 - stop_fill) + 100 * self.cfg.fee + stop_fill * self.cfg.fee
        self.assertAlmostEqual(risk_amt, qty * estimated_loss_per_unit)
        self.assertLessEqual(risk_amt, 1000 * self.cfg.risk_pct + 1e-9)
        self.assertLessEqual(notional, 200.0)

    def test_position_sizing_includes_costs_for_short_stops(self):
        size = pt.size_position(self.cfg, equity=1000, open_notional_=0,
                                fill=100, stop=110, side="SHORT")
        self.assertIsNotNone(size)
        qty, _, risk_amt = size
        stop_fill = 110 * (1 + self.cfg.slippage)
        estimated_loss_per_unit = (stop_fill - 100) + 100 * self.cfg.fee + stop_fill * self.cfg.fee
        self.assertAlmostEqual(risk_amt, qty * estimated_loss_per_unit)
        self.assertLessEqual(risk_amt, 1000 * self.cfg.risk_pct + 1e-9)

    def test_position_sizing_rejects_invalid_risk_and_too_small_notional(self):
        self.assertIsNone(pt.size_position(self.cfg, 1000, 0, 100, 101, "LONG"))
        self.assertIsNone(pt.size_position(self.cfg, 10, 0, 100, 95, "LONG"))

    def test_stop_wins_when_stop_and_target_touch_same_candle(self):
        candles = pd.DataFrame([{
            "ts": 0, "open": 100.0, "high": 111.0, "low": 94.0,
            "close": 105.0, "ll20": math.nan,
        }])
        result = pt.simulate_exit("LONG", 100.0, 95.0, 110.0, 1.0, 0.1, 5.0,
                                  candles, self.cfg, "pullback")
        self.assertIsNotNone(result)
        self.assertEqual(result["exit_reason"], "STOP")
        self.assertLess(result["exit_price"], 95.0)  # adverse slippage is applied

    def test_target_exits_apply_adverse_slippage_for_long_and_short(self):
        long_candles = pd.DataFrame([{
            "ts": 0, "open": 100.0, "high": 111.0, "low": 99.0,
            "close": 110.0, "ll20": math.nan,
        }])
        long_result = pt.simulate_exit(
            "LONG", 100.0, 90.0, 110.0, 1.0, 0.1, 10.0,
            long_candles, self.cfg, "pullback",
        )
        self.assertEqual(long_result["exit_reason"], "TARGET")
        self.assertAlmostEqual(long_result["exit_price"], 110.0 * (1 - self.cfg.slippage))

        short_candles = pd.DataFrame([{
            "ts": 0, "open": 100.0, "high": 101.0, "low": 89.0,
            "close": 90.0, "ll20": math.nan,
        }])
        short_result = pt.simulate_exit(
            "SHORT", 100.0, 110.0, 90.0, 1.0, 0.1, 10.0,
            short_candles, self.cfg, "short_pullback",
        )
        self.assertEqual(short_result["exit_reason"], "TARGET")
        self.assertAlmostEqual(short_result["exit_price"], 90.0 * (1 + self.cfg.slippage))

    def test_backtest_step_applies_target_slippage_for_long_and_short(self):
        long_trade = {
            "side": "LONG", "entry_price": 100.0, "stop": 90.0, "cur_stop": 90.0,
            "target": 110.0, "qty": 1.0, "entry_fee": 0.1, "risk_amt": 10.0,
            "init_risk": 10.0, "fee_buffer": 0.25, "max_hold": 3, "bars_held": 0,
            "strategy": "pullback",
        }
        long_row = SimpleNamespace(
            ts=0, open=100.0, high=111.0, low=99.0, close=110.0, ll20=math.nan,
        )
        self.assertTrue(pt._step_open_trade(long_trade, long_row, 14400000, self.cfg))
        self.assertEqual(long_trade["exit_reason"], "TARGET")
        self.assertAlmostEqual(long_trade["exit_price"], 110.0 * (1 - self.cfg.slippage))

        short_trade = {
            "side": "SHORT", "entry_price": 100.0, "stop": 110.0, "cur_stop": 110.0,
            "target": 90.0, "qty": 1.0, "entry_fee": 0.1, "risk_amt": 10.0,
            "init_risk": 10.0, "fee_buffer": 0.25, "max_hold": 3, "bars_held": 0,
            "strategy": "short_pullback",
        }
        short_row = SimpleNamespace(
            ts=0, open=100.0, high=101.0, low=89.0, close=90.0, ll20=math.nan,
        )
        self.assertTrue(pt._step_open_trade(short_trade, short_row, 14400000, self.cfg))
        self.assertEqual(short_trade["exit_reason"], "TARGET")
        self.assertAlmostEqual(short_trade["exit_price"], 90.0 * (1 + self.cfg.slippage))

    def test_no_exit_before_time_limit_when_bar_does_not_touch_levels(self):
        candles = pd.DataFrame([{
            "ts": 0, "open": 100.0, "high": 102.0, "low": 98.0,
            "close": 101.0, "ll20": math.nan,
        }])
        result = pt.simulate_exit("LONG", 100.0, 95.0, 110.0, 1.0, 0.1, 5.0,
                                  candles, self.cfg, "pullback")
        self.assertIsNone(result)

    def test_open_backtest_trade_is_liquidated_at_end_with_costs(self):
        long_trade = {
            "side": "LONG", "entry_price": 100.0, "qty": 2.0,
            "entry_fee": 0.2, "risk_amt": 10.0, "cur_stop": 95.0,
            "init_risk": 5.0, "fee_buffer": 0.25, "max_hold": 3,
        }
        pt._force_close_at_end(long_trade, SimpleNamespace(close=105.0), 1234, self.cfg)
        expected_long_exit = 105.0 * (1 - self.cfg.slippage)
        expected_long_pnl = (expected_long_exit - 100.0) * 2.0 - 0.2 - expected_long_exit * 2.0 * self.cfg.fee
        self.assertEqual(long_trade["exit_reason"], "END_OF_DATA")
        self.assertEqual(long_trade["exit_ts"], 1234)
        self.assertAlmostEqual(long_trade["exit_price"], expected_long_exit)
        self.assertAlmostEqual(long_trade["pnl"], expected_long_pnl)
        self.assertNotIn("entry_fee", long_trade)

        short_trade = {
            "side": "SHORT", "entry_price": 100.0, "qty": 2.0,
            "entry_fee": 0.2, "risk_amt": 10.0, "cur_stop": 105.0,
            "init_risk": 5.0, "fee_buffer": 0.25, "max_hold": 3,
        }
        pt._force_close_at_end(short_trade, SimpleNamespace(close=95.0), 5678, self.cfg)
        expected_short_exit = 95.0 * (1 + self.cfg.slippage)
        expected_short_pnl = (100.0 - expected_short_exit) * 2.0 - 0.2 - expected_short_exit * 2.0 * self.cfg.fee
        self.assertEqual(short_trade["exit_reason"], "END_OF_DATA")
        self.assertAlmostEqual(short_trade["exit_price"], expected_short_exit)
        self.assertAlmostEqual(short_trade["pnl"], expected_short_pnl)

    def test_mark_to_market_includes_new_entry_costs_immediately(self):
        tf = pt.TF_MS["4h"]
        ts = [i * tf for i in range(265)]
        frame = pd.DataFrame([{
            "ts": t, "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 100.0,
        } for t in ts])

        def one_signal(symbol, last, prev, n):
            signal = None
            if int(last.ts) == ts[210]:
                signal = pt.Signal(
                    "pullback", symbol, int(last.ts), 100.0, 90.0, 120.0,
                    "test signal", "LONG",
                )
            return SimpleNamespace(signal=signal)

        with patch.dict(pt.STRATEGIES, {"pullback": one_signal}, clear=True):
            trades, meta = pt.portfolio_backtest(self.cfg, {"BTC/USDT": frame})

        self.assertIsNotNone(meta)
        curve = meta["equity_curves"]["pullback"]
        entry_ts = ts[210] + tf
        point = next(value for stamp, value in curve if stamp == entry_ts)
        self.assertEqual(trades[0]["entry_ts"], entry_ts)
        # At entry, mark-to-market should already reflect the entry fee and
        # estimated costs to liquidate rather than still showing starting equity.
        self.assertLess(point, self.cfg.start_equity)

    def test_buy_hold_benchmark_includes_round_trip_costs(self):
        frame = pd.DataFrame([
            {"ts": 0, "open": 100.0, "close": 105.0},
            {"ts": 10, "open": 105.0, "close": 110.0},
        ])
        data = {"BTC/USDT": (frame, [], {})}
        result = pt._buy_hold(data, 0, 20, self.cfg)
        entry = 100.0 * (1 + self.cfg.slippage)
        exit_ = 110.0 * (1 - self.cfg.slippage)
        qty = 1.0 / (entry * (1 + self.cfg.fee))
        expected = qty * exit_ * (1 - self.cfg.fee) - 1.0
        self.assertAlmostEqual(result, expected)
        self.assertLess(result, 0.10)  # raw 10% price return ignores all costs

    def test_compute_stats_uses_mark_to_market_curve_for_drawdown(self):
        trades = pd.DataFrame([
            {"exit_ts": 1, "pnl": 10.0, "r_multiple": 1.0},
            {"exit_ts": 2, "pnl": -5.0, "r_multiple": -0.5},
        ])
        stats = pt.compute_stats(
            trades, 1000.0,
            mark_to_market_curve=[1000.0, 1200.0, 800.0, 900.0],
        )
        self.assertAlmostEqual(stats["max_dd"], -1.0 / 3.0)

    def test_compute_stats_reports_net_pnl_and_drawdown(self):
        trades = pd.DataFrame([
            {"exit_ts": 1, "pnl": 10.0, "r_multiple": 1.0},
            {"exit_ts": 2, "pnl": -5.0, "r_multiple": -0.5},
        ])
        stats = pt.compute_stats(trades, 1000.0)
        self.assertEqual(stats["trades"], 2)
        self.assertAlmostEqual(stats["total_pnl"], 5.0)
        self.assertAlmostEqual(stats["return_pct"], 0.005)
        self.assertAlmostEqual(stats["profit_factor"], 2.0)
        self.assertLess(stats["max_dd"], 0.0)


    def test_cost_stress_replays_history_with_higher_fees_and_slippage(self):
        def fake_backtest(cfg, frames):
            pnl = 20.0 - 10000.0 * (cfg.fee + cfg.slippage)
            trades = [{
                "strategy": "short_pullback", "exit_ts": 1,
                "pnl": pnl, "r_multiple": pnl / 10.0,
            }]
            curves = {name: [(0, 1000.0), (1, 1000.0 + pnl)] for name in pt.STRATEGIES}
            return trades, {"equity_curves": curves}

        with patch.object(pt, "portfolio_backtest", side_effect=fake_backtest) as replay:
            result = pt._cost_stress(self.cfg, {"BTC/USDT": pd.DataFrame()})

        self.assertEqual(replay.call_count, 2)
        self.assertAlmostEqual(result["1.5x_costs"]["fee"], self.cfg.fee * 1.5)
        self.assertAlmostEqual(result["2x_costs"]["slippage"], self.cfg.slippage * 2)
        base_return = result["1.5x_costs"]["strategies"]["short_pullback"]["return_pct"]
        stressed_return = result["2x_costs"]["strategies"]["short_pullback"]["return_pct"]
        self.assertGreater(base_return, stressed_return)



    def test_walk_forward_creates_chronological_evaluation_folds(self):
        tf = pt.TF_MS["4h"]
        ts = [i * tf for i in range(800)]
        frame = pd.DataFrame([{
            "ts": t, "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 100.0,
        } for t in ts])

        observed_windows = []

        def fake_backtest(cfg, frames):
            observed_windows.append(frames["BTC/USDT"]["ts"].astype(int).tolist())
            first = int(frames["BTC/USDT"]["ts"].iloc[210])
            end = int(frames["BTC/USDT"]["ts"].iloc[-1]) + tf
            curves = {name: [(first, 1000.0), (end, 1005.0)] for name in pt.STRATEGIES}
            trades = [
                # A simulated warmup trade must not leak into evaluation stats.
                {"strategy": "short_pullback", "entry_ts": first - tf,
                 "exit_ts": first, "pnl": 50.0, "r_multiple": 5.0},
                # This trade belongs to the evaluation fold.
                # Deliberately differs from the mocked equity change: fold
                # return must come from boundary-adjusted equity, not trade P&L.
                {"strategy": "short_pullback", "entry_ts": first + tf,
                 "exit_ts": first + 2 * tf, "pnl": 50.0, "r_multiple": 5.0},
                # A trade outside the fold must not be attributed to it.
                {"strategy": "short_pullback", "entry_ts": end + tf,
                 "exit_ts": end + 2 * tf, "pnl": 100.0, "r_multiple": 10.0},
            ]
            return trades, {
                "trade_from": first,
                "end": end,
                "equity_curves": curves,
            }

        with patch.object(pt, "portfolio_backtest", side_effect=fake_backtest):
            result = pt._walk_forward(self.cfg, {"BTC/USDT": frame}, fold_days=30)

        self.assertEqual(len(observed_windows), len(result["folds"]))
        for i, window in enumerate(observed_windows):
            # Every fold's source slice ends before the next fold begins;
            # only prior candles are included for indicator warmup.
            self.assertEqual(window, sorted(window))
            if i + 1 < len(observed_windows):
                self.assertLess(window[-1], observed_windows[i + 1][210])
        self.assertEqual(result["fold_days"], 30)
        self.assertGreaterEqual(len(result["folds"]), 2)
        for fold in result["folds"]:
            self.assertIn("short_pullback", fold["strategies"])
            stats = fold["strategies"]["short_pullback"]
            self.assertEqual(stats["trades"], 1)
            self.assertAlmostEqual(stats["return_pct"], 0.005)

if __name__ == "__main__":
    unittest.main()
