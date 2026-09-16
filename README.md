# SignalForge — Performance Analytics (P4)

**Status: P4 implemented. P5 not started.**

SignalForge is a pure-Python, read-only crypto signal scanner and backtest
evaluator. P4 adds reproducible performance analytics on top of the real P3
evaluator.

## What P4 adds

- Real evaluator: candle-by-candle TP/SL/expiration scanning over CLOSED future
  candles. No stubs.
- R-based metrics: expectancy, profit factor, max drawdown, equity curve,
  consecutive wins/losses, holding times.
- CLI report (Termux-friendly) + machine-readable JSON.
- Deterministic tests with manually calculated expected answers.
- In-sample / OOS separation (walk-forward NOT implemented — explicitly stated).
- Data quality reporting: invalid candles, rejected signals, missing data,
  insufficient sample warnings.
- Gross vs net separation (fee + slippage configurable, never leveraged).

## What P4 does NOT do

- No strategy changes. Trend 40%, Momentum 20%, Structure 20%, Volume 20%.
  TF weights: 1d 25%, 4h 25%, 1h 20%, 15m 20%, 5m 10%.
- No parameter optimization.
- No ML.
- No trading, no API keys, no private Binance endpoints, no orders, no
  Telegram/webhooks, no auto-trading.
- No walk-forward. Explicitly stated.

## Metric definitions

All metrics are R-based. R = (price move) / (initial SL distance). No account
size, leverage, or fixed position size assumed.

- **Expectancy (R)** = mean(gross_R over all completed signals). Formula
  explicitly reported with counts.
- **Win rate** = TP hits / (TP hits + SL hits). NOT the same as profitability.
- **Loss rate** = SL hits / (TP hits + SL hits).
- **Profit factor** = sum(gross_R where R>0) / abs(sum(gross_R where R<0)).
  Zero-loss cases: if no losing R and some winning R, reported as infinity with
  explicit note. If no R ≠ 0 trades at all, reported as None (undefined).
- **Cumulative R** = sum of gross_R across completed signals (chronological).
- **Max drawdown** = maximum peak-to-trough decline in the chronological
  R-based equity curve. Reported with peak, trough, and duration in signals.
- **Average/median holding time** = mean/median of holding_ms across completed
  signals.
- **Consecutive wins/losses** = max run length of R>0 / R<0 in chronological
  order.

### Gross vs net

- **Gross R**: price move / SL distance. No costs.
- **Net R**: gross R - 2 * entry * (fee_pct + slippage_pct) / SL_distance.
  Costs are configurable (default fee 0.1%, slippage 0.1%). Never silently
  applied. Never leveraged.

## Sample size handling

Every report prominently shows:

- total signals
- completed signals
- invalid/rejected signals
- expired signals

Warnings emitted when:

- **< 30 completed signals**: "INSUFFICIENT SAMPLE — exploratory result, not
  statistically reliable."
- **0 signals**: "INSUFFICIENT SAMPLE: no signals evaluated."
- **Invalid signals present**: data quality note.
- **Expirations present**: expiration note.

No statistical confidence is invented. Tiny samples are labeled exploratory.

## OOS methodology

- In-sample and OOS are reported SEPARATELY when both provided.
- They are NEVER combined into one headline number.
- OOS results do NOT influence signal generation, strategy configuration,
  thresholds, or scoring weights.
- No automatic tuning.
- If no OOS split is provided, the report states: "OOS: not evaluated."
- Walk-forward: "WALK-FORWARD: NOT IMPLEMENTED." always stated.

## Evaluator behavior

### LONG

- TP: first future candle where high >= TP.
- SL: first future candle where low <= SL.
- Same-candle TP+SL: conservative — SL wins.

### SHORT

- TP: first future candle where low <= TP.
- SL: first future candle where high >= SL.
- Same-candle TP+SL: conservative — SL wins.

### Expiration

- No TP/SL hit within horizon (default 4h). R = 0.

### Rejection

- Invalid OHLC (high < low): ValueError at construction.
- Missing/NaN fields: ValueError at construction.
- Out-of-order candles: skipped with data quality note.
- Candles with timestamp <= signal timestamp: ignored.

## File layout

```
SignalForge/
  backtest/
    evaluator.py   — real evaluator (Candle, Signal, EvaluationResult, evaluate_signal, evaluate_many)
    metrics.py     — metrics computation, equity curve, drawdown, breakdowns
    report.py      — CLI report + JSON output
    tests/
      test_backtest.py  — deterministic test suite (97 tests)
```

## Running tests

```bash
cd SignalForge
python -m pytest SignalForge/backtest/tests/test_backtest.py -v
```

Expected: 97 passed, 0 failed (deterministic).

## Running a deterministic sample backtest

```bash
cd SignalForge
python -c "
from SignalForge.backtest.evaluator import Candle, Signal, evaluate_signal
from SignalForge.backtest.report import print_report, build_json_report

T0 = 1_000_000
H1 = 3_600_000
H5 = 5 * H1

sig = Signal('demo', 'BTCUSDT', 'LONG', T0, 100, tp=105, sl=95,
             score=60, timeframe='1h', horizon_ms=H5)
future = [Candle(T0+H1, 100, 106, 99, 104)]
res = evaluate_signal(sig, future)
print_report('SignalForge Demo Backtest', [(sig, res)])
import json
print(json.dumps(build_json_report('SignalForge Demo Backtest', [(sig, res)]), indent=2))
"
```

Run twice — output is identical (reproducibility verified by tests).

## CLI example

```
$ python -c "..."  (sample above)
```

Or import `print_report()` from your own driver script.

## Machine-readable output

`build_json_report()` returns a JSON-serializable dict with keys:

- meta, period, outcomes, direction_counts, rates, cumulative, profit_factor,
  drawdown, holding, streaks, gross_vs_net, direction_breakdown,
  score_breakdown, timeframe_breakdown, symbol_breakdown, warnings,
  data_quality_notes, costs, oos, walk_forward, disclaimer.

Sample size is prominent in `period` and `rates.expectancy_count`.

## Limitations

- R-based only. No monetary P&L without external account/leverage assumptions.
- No walk-forward. IS and OOS are manually split by the caller.
- No statistical significance testing. Sample-size warnings only.
- Default horizon is 4h. Configurable per signal.
- Default fee/slippage 0.1% each. Configurable.
- Backtest performance does not guarantee future trading performance.

## Disclaimer

**Backtest performance does not guarantee future trading performance.**

## Commit

P4 changes committed locally only. Not pushed.

## Gate

After P4: SAFE TO PROCEED TO P5 (subject to P5 scope approval).
