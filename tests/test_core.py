import math
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


if __name__ == "__main__":
    unittest.main()
