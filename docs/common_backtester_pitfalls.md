---
title: Common backtester pitfalls
tags: Finance, Backtesting
---

<!-- TOC titleSize:1 tabSpaces:2 depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 skip:0 title:1 charForUnorderedList:* -->
# Table of Contents
- [Table of Contents](#table-of-contents)
- [Common backtester pitfalls](#common-backtester-pitfalls)
  - [Purpose](#purpose)
  - [Core rule](#core-rule)
  - [Hard principles](#hard-principles)
    - [Time correctness](#time-correctness)
    - [Universe correctness](#universe-correctness)
    - [Execution correctness](#execution-correctness)
    - [Cost correctness](#cost-correctness)
    - [Accounting correctness](#accounting-correctness)
    - [Resolution correctness](#resolution-correctness)
  - [Do not do these](#do-not-do-these)
  - [Review checklist](#review-checklist)
<!-- /TOC -->

# Common backtester pitfalls
## Purpose
**Purpose**. This file is a working contract for how `backtester.py` should behave.

**Scope**. It is not a broad tutorial on backtesting. It is a short set of principles that prevent fake alpha from entering the simulator.

## Core rule
**Backtester rule**. At each simulated decision point, the engine must only allow:
* *Information that was available then*
* *Orders that could actually have been placed then*
* *Fills that could actually have happened then*

$\to$ If any of the three is too generous, the backtest is overstated.

## Hard principles
### Time correctness
**No lookahead**. Signals must never use future data.
* *Rule*. If a signal uses bar close at time $T$, it cannot also fill at that same close unless the system explicitly models that capability without lookahead.
* *Default*. Prefer signal on current bar, execute on next open or next bar.

**Explicit event ordering**. Signal generation, order submission, execution, valuation, and margin checks must happen in a fixed chronological order.

### Universe correctness
**No survivorship bias by default**. The engine should assume the tradeable universe changes through time.
* *Rule*. Do not evaluate a strategy as if today’s surviving assets were the full historical universe.
* *Fallback*. If point-in-time universe data is unavailable, document the limitation clearly in results.

### Execution correctness
**No instantaneous trading assumption**. The engine must not assume that seeing a price implies filling at that same price with zero delay.

**No free passive fills**. Limit-order logic must not assume neutral fills whenever price touches the level.
* *Reason*. Passive fills are exposed to adverse selection.

**Conservative execution defaults**. When exact microstructure is unknown, choose the more pessimistic fill assumption.

### Cost correctness
**All major frictions must be explicit**. Fees are part of the engine, not an afterthought.
* *Include*. Commission, spread or slippage, financing, borrow costs, taxes, FX fees, and market-specific charges when relevant.
* *Rule*. If costs vary with volatility, liquidity, or size, the model should support that dependence.

### Accounting correctness
**Cash must be real**. Cash-only strategies cannot spend cash they do not have.

**Margin must be real**. Margin-enabled strategies must respect leverage limits and maintenance rules.

**Taxes must be timed correctly**. Immediate tax payment and deferred tax payment produce different compounding paths.

**Portfolio value must not be double-counted**. Liquidation proceeds, holdings value, and equity must reconcile exactly after each event.

>**NOTE**. Many apparent strategy improvements are actually ledger bugs.

### Resolution correctness
**Data frequency must match the mechanism being tested**.
* *Rule*. If the strategy depends on intraday stops, queues, or book behavior, daily bars are not enough.
* *Rule*. If the exchange data cannot reconstruct the needed execution path, the backtest cannot make strong claims about that mechanism.

## Do not do these
**Forbidden shortcuts**.
* *Do not* use indicator values through time $T$ to predict and fill at time $T$ unless the timing is explicitly valid.
* *Do not* assume the strategy can always trade at observed prices without latency.
* *Do not* ignore delistings when the strategy outcome depends on cross-sectional selection.
* *Do not* model transaction costs as zero unless the strategy is explicitly being tested in a frictionless toy setup.
* *Do not* let margin systems bypass cash, leverage, or liquidation constraints.
* *Do not* use data with lower resolution than the trading mechanism requires.

## Review checklist
**Pre-merge checklist**. Changes to `backtester.py` should satisfy all of the following:
* *No future data*. Signal timing is strictly causal.
* *Chronology is explicit*. Signal, order, fill, valuation, and risk checks happen in a consistent order.
* *Universe bias is addressed*. Delistings or universe limitations are either modeled or disclosed.
* *Execution is realistic enough*. The fill model is conservative for the strategy horizon.
* *Frictions are included*. Relevant costs are applied in the right place.
* *Ledger is internally consistent*. Cash, holdings, fees, taxes, and equity reconcile after each step.
* *Margin logic is enforceable*. Leverage and maintenance constraints can actually bind.
* *Data resolution is sufficient*. The input data can support the claimed simulation behavior.
* *Great results are challenged*. If a result looks unusually strong, assume a bug first and try to falsify it.
