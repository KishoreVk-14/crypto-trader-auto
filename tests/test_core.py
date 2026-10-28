import math
import unittest
from types import SimpleNamespace

import pandas as pd

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
        self.assertTrue(pt.gapped("SHORT", 111.0, 95.0, 110.0))
        self.assertTrue(pt.gapped("SHORT", 94.0, 95.0, 110.0))
        self.assertFalse(pt.gapped("LONG", 100.0, 95.0, 110.0))
        self.assertFalse(pt.gapped("SHORT", 100.0, 95.0, 110.0))

    def test_position_sizing_respects_risk_and_notional_caps(self):
        size = pt.size_position(self.cfg, equity=1000, open_notional_=0,
                                fill=100, stop=95, side="LONG")
        self.assertIsNotNone(size)
        qty, notional, risk_amt = size
        self.assertAlmostEqual(risk_amt, 10.0)
        self.assertLessEqual(notional, 200.0)
        self.assertAlmostEqual(qty * 5, risk_amt)

    def test_position_sizing_rejects_invalid_risk_and_too_small_notional(self):
        self.assertIsNone(pt.size_position(self.cfg, 1000, 0, 100, 101, "LONG"))
        self.assertIsNone(pt.size_position(self.cfg, 1000, 0, 100, 99.99, "LONG"))

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

    def test_no_exit_before_time_limit_when_bar_does_not_touch_levels(self):
        candles = pd.DataFrame([{
            "ts": 0, "open": 100.0, "high": 102.0, "low": 98.0,
            "close": 101.0, "ll20": math.nan,
        }])
        result = pt.simulate_exit("LONG", 100.0, 95.0, 110.0, 1.0, 0.1, 5.0,
                                  candles, self.cfg, "pullback")
        self.assertIsNone(result)

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
