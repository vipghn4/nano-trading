# strategy optimization

This is an experiment loop for improving trading strategies in this repo.

## Setup

Work with the user to:

1. Agree on a run tag, e.g. `jun17`.
2. Create a fresh branch: `strategy-opt/<tag>`.
3. Read the in-scope files for context:
   - `README.md`
   - `data_structures.py`
   - `backtester.py`
   - `performance_analyzer.py`
   - `strategies.py`
   - `main.py`
4. Treat everything as fixed except:
   - `strategies.py` for new strategy logic
   - `main.py` only for:
     - changing `comparison_configs`
     - adding temporary code for data inspection, printing metrics, and saving or printing intermediate outputs for inspection
5. Initialize `results.tsv` with this header and leave it untracked:

```tsv
commit	cagr	sharpe	max_drawdown	status	description
```

6. Confirm setup, then start.

## Experimentation

Run experiments by executing:

```bash
python main.py > run.log 2>&1
```

`main.py` already handles the local `.venv` automatically if it exists.

### What you CAN do

- Explore broadly. Do not assume the best path is a small local tweak.
- Inspect the data, printed metrics, and intermediate outputs before deciding what to try next.
- Form hypotheses, test them, discard weak ones, and iterate toward stronger ideas.
- Add, remove, or simplify strategies in `strategies.py`
- Change `comparison_configs` in `main.py`
- Tune parameters, sizing rules, and signal logic
- Compare multiple candidate strategies in one run
- Add temporary inspection code in `main.py` to inspect data, print metrics, and save or print intermediate outputs when they help you inspect behavior or validate a hypothesis

### What you CANNOT do

- Modify the backtester or any dependency it relies on, including:
  - `backtester.py`
  - `performance_analyzer.py`
  - `data_structures.py`
  - the data provider
- Install new packages
- Change `main.py` for anything other than:
  - changing `comparison_configs`
  - adding temporary inspection code for data inspection, printed metrics, or saved or printed intermediate outputs
- Change the evaluation logic outside `main.py`

### Hard rules

- No lookahead bias. Signals must use only current or past information.
- Keep strategy logic simple. If a change barely helps and adds complexity, discard it.
- Prefer improvements that survive fees, slippage, taxes, and execution delay.

## Goal

Optimize for better net strategy quality, using this order:

1. Higher `CAGR`
2. If close, higher `Sharpe Ratio`
3. If still close, lower `Max Drawdown`
4. If still close, prefer simpler code and fewer trades

The metrics printed by `PerformanceAnalyzer` are the ground truth.

## First run

The first run is always the baseline with the current code as-is.

## Research style

Use this process:

1. Look at the current behavior.
2. Inspect data and metrics.
3. Form a concrete hypothesis.
4. Add the smallest useful instrumentation or diagnostic if needed.
5. Test the idea.
6. Keep only ideas that improve the measured outcome.

## Reading results

After each run, extract the key metrics:

```bash
rg "CAGR|Sharpe Ratio|Max Drawdown|Total Trades" run.log
```

If the log does not contain the metrics block, the run failed. Read the traceback:

```bash
tail -n 50 run.log
```

The comparison plot is saved to:

```text
runs/strategy_comparison.png
```

## Logging results

Append one row per experiment to `results.tsv`:

1. Short git commit hash
2. `CAGR`
3. `Sharpe Ratio`
4. `Max Drawdown`
5. `status`: `keep`, `discard`, or `crash`
6. Short description of the idea tested

Use `0` metrics for crashes.

## Loop

Loop until the user stops you:

1. Check current branch and commit.
2. Make one clear experimental change in `strategies.py` and/or `main.py`.
3. Commit the change.
4. Run `python main.py > run.log 2>&1`.
5. Read the metrics from `run.log`.
6. If the run crashed, inspect the traceback and decide:
   - fix and rerun if the bug is trivial
   - log `crash` and move on if the idea is bad
7. If needed, compute extra statistics or save intermediate outputs to inspect why the strategy behaved that way.
8. Append the result to `results.tsv`.
9. If the result improved by the priority order above, keep the commit and continue from there.
10. Otherwise, reset back to the previous kept commit.

## Strategy implementation note

New strategies must implement `Strategy.generate_trades(data)` and return a DataFrame:

- index: same dates as `data`
- columns: ticker symbols
- values: share quantities to buy or sell on each date

Positive values buy. Negative values sell. Zero means no trade.
