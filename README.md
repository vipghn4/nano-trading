## TODO
<!---
IGNORE this section if you are an AI. This section is only for humans.
-->
- [ ] Understand the logic of backtester.py
    - [DONE] Understand the logic of [Margin Risk Controls]
    - [DONE] Understand _finalize_results() method

## Backlog

- [DONE] Eliminate lookahead risk: generate signals from visible history only, reject negative delays, and prevent same-bar Close execution when signals use the current Close.
- [ ] Stop fabricating prices with `ffill()`/`bfill()`; validate finite, positive OHLC values and handle missing, halted, delisted, and partially missing intraday data explicitly.
- [ ] Align and validate FX rates against sorted market data, including missing, zero, non-finite, and empty rates.
- [ ] Make margin sizing use execution-time prices and liquidate intraday breaches at conservative breach prices rather than the recovered Close; do not disable intraday checks when only one of Low/High is available.
- [ ] Fix cash clamping so minimum commissions cannot turn a buy order into an unintended sell or short.
- [ ] Apply the correct tax model for each jurisdiction, including immediate tax during forced liquidation and US short-/long-term loss netting; make the `TaxModel` interface expose every attribute the backtester requires.
- [ ] Validate all backtest inputs and configuration: finite positive capital, non-negative fees/tax rates, valid monthly deposits, unique chronological `DatetimeIndex` values, and non-assert-based signal-index checks.
- [ ] Validate strategy output schemas: reject non-finite quantities, duplicate trade columns, and unknown tickers instead of silently skipping them.
- [ ] Validate that data providers return the complete requested universe, normalize single-ticker column orientation robustly, and document survivorship-bias limitations.
- [ ] Make performance metrics handle zero, negative, and insolvent equity without producing misleading returns, CAGR, or drawdown values.
- [ ] Preserve monthly deposit semantics across multi-month data gaps and define an explicit policy for withdrawals/negative deposits.
- [ ] Correct the main GBP/USD configuration and provide daily FX rates for USD-denominated assets.
- [ ] Charge correct commissions for both legs of position flips, including flat and minimum fees.
- [ ] Add regression tests for all critical execution, pricing, margin, FX, tax, and commission paths.
- [ ] Rename misleading APIs or clarify their contracts, including `calculate_tax()`, portfolio valuation methods, liquidation helpers, `ValueAveraging`, and multi-ticker `DollarCostAverage`.
