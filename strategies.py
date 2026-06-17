import pandas as pd
import numpy as np
from typing import Optional
from data_structures import Strategy

class MovingAverageCrossover(Strategy):
    def __init__(self, short_window: int = 50, long_window: int = 200, trade_quantity: float = 1.0):
        """
        Moving average crossover strategy.
        """
        self.short_window = short_window
        self.long_window = long_window
        self.trade_quantity = trade_quantity

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates trade quantities based on short/long moving average crossovers."""
        trade_signals = pd.DataFrame(index=data.index)
        tickers = data.columns.get_level_values(0).unique()
        
        for ticker in tickers:
            trade_signals[ticker] = self._calculate_ticker_trades(data[ticker])
            
        return trade_signals

    def _calculate_ticker_trades(self, ticker_data: pd.DataFrame) -> pd.Series:
        """Calculates trade signals for a single ticker."""
        close_prices = ticker_data['Close']
        short_moving_avg = close_prices.rolling(window=self.short_window, min_periods=1).mean()
        long_moving_avg = close_prices.rolling(window=self.long_window, min_periods=1).mean()
        
        target_position = (short_moving_avg > long_moving_avg).astype(float) * self.trade_quantity
        return target_position.diff().fillna(0.0)


class DollarCostAverage(Strategy):
    def __init__(self, monthly_share_quantity: Optional[float] = None, monthly_investment_amount: Optional[float] = None):
        """
        Dollar Cost Averaging strategy buying a fixed quantity of shares or a fixed cash amount monthly.
        """
        if monthly_share_quantity is None and monthly_investment_amount is None:
            raise ValueError("Must specify either monthly_share_quantity or monthly_investment_amount")
        self.monthly_share_quantity = monthly_share_quantity
        self.monthly_investment_amount = monthly_investment_amount

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generates DCA buy orders at the beginning of each calendar month."""
        tickers = data.columns.get_level_values(0).unique()
        trade_signals = pd.DataFrame(0.0, index=data.index, columns=tickers)
        
        is_first_day_of_month = trade_signals.index.to_series().dt.to_period('M') != \
                                trade_signals.index.to_series().shift(1).dt.to_period('M')
        
        for ticker in tickers:
            if self.monthly_investment_amount is not None:
                prices = data[ticker]['Close']
                # To prevent division by zero or NaN prices
                valid_prices = prices.where(prices > 0.0, np.nan)
                trade_signals.loc[is_first_day_of_month, ticker] = self.monthly_investment_amount / valid_prices.loc[is_first_day_of_month]
                trade_signals[ticker] = trade_signals[ticker].fillna(0.0)
            else:
                trade_signals.loc[is_first_day_of_month, ticker] = self.monthly_share_quantity
                
        return trade_signals


class ValueAveraging(Strategy):
    def __init__(self, monthly_target_increment: float = 2000.0, allow_fractional: bool = True):
        """
        Value Averaging strategy targeting a fixed monthly portfolio value increase.
        """
        self.monthly_target_increment = monthly_target_increment
        self.allow_fractional = allow_fractional

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generates value averaging trades to hit monthly target increments."""
        tickers = data.columns.get_level_values(0).unique()
        trade_signals = pd.DataFrame(0.0, index=data.index, columns=tickers)
        
        is_first_day_of_month = data.index.to_series().dt.to_period('M') != \
                                data.index.to_series().shift(1).dt.to_period('M')
        monthly_dates = data.index[is_first_day_of_month]
        
        shares_owned = {ticker: 0.0 for ticker in tickers}
        
        for month_idx, date in enumerate(monthly_dates):
            target_value = (month_idx + 1) * self.monthly_target_increment
            self._process_monthly_trade(date, target_value, tickers, data, shares_owned, trade_signals)
                
        return trade_signals

    def _process_monthly_trade(self, date: pd.Timestamp, target_value: float, tickers: list,
                               data: pd.DataFrame, shares_owned: dict, trade_signals: pd.DataFrame):
        """Calculates and records the monthly trade required to meet target value."""
        for ticker in tickers:
            price = data.loc[date, (ticker, 'Close')]
            current_holding_value = shares_owned[ticker] * price
            
            shortfall = target_value - current_holding_value
            if self.allow_fractional:
                shares_to_purchase = shortfall / price if price > 0.0 else 0.0
            else:
                shares_to_purchase = shortfall // price if price > 0.0 else 0.0
            
            trade_signals.loc[date, ticker] = shares_to_purchase
            shares_owned[ticker] += shares_to_purchase


class PullbackDollarCostAverage(Strategy):
    def __init__(self, base_monthly_amount: float, extra_monthly_amount: float,
                 pullback_window: int = 20, pullback_pct: float = 0.0):
        if base_monthly_amount <= 0.0:
            raise ValueError("base_monthly_amount must be positive")
        if extra_monthly_amount < 0.0:
            raise ValueError("extra_monthly_amount must be non-negative")
        if pullback_window < 1:
            raise ValueError("pullback_window must be at least 1")
        if pullback_pct < 0.0:
            raise ValueError("pullback_pct must be non-negative")

        self.base_monthly_amount = base_monthly_amount
        self.extra_monthly_amount = extra_monthly_amount
        self.pullback_window = pullback_window
        self.pullback_pct = pullback_pct

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        tickers = data.columns.get_level_values(0).unique()
        trade_signals = pd.DataFrame(0.0, index=data.index, columns=tickers)
        is_first_day_of_month = data.index.to_series().dt.to_period('M') != \
                                data.index.to_series().shift(1).dt.to_period('M')

        for ticker in tickers:
            prices = data[ticker]['Close']
            rolling_mean = prices.rolling(window=self.pullback_window, min_periods=1).mean()
            monthly_amount = pd.Series(self.base_monthly_amount, index=data.index)
            pullback_trigger = prices <= (rolling_mean * (1.0 - self.pullback_pct))
            monthly_amount = monthly_amount + (pullback_trigger.astype(float) * self.extra_monthly_amount)
            valid_prices = prices.where(prices > 0.0, np.nan)
            shares = (monthly_amount / valid_prices).where(is_first_day_of_month, 0.0)
            trade_signals[ticker] = shares.fillna(0.0)

        return trade_signals


class SingleTickerStrategy(Strategy):
    def __init__(self, ticker: str, inner_strategy: Strategy):
        if not ticker:
            raise ValueError("ticker is required")
        self.ticker = ticker
        self.inner_strategy = inner_strategy

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        trades = self.inner_strategy.generate_trades(data)
        if self.ticker not in trades.columns:
            raise ValueError(f"{self.ticker} not found in trade columns")
        filtered = trades.copy()
        for column in filtered.columns:
            if column != self.ticker:
                filtered[column] = 0.0
        return filtered


class MonthlyRotationStrategy(Strategy):
    def __init__(self, initial_capital: float, monthly_deposit: float,
                 lookback_days: int = 126, trend_window: int = 200,
                 cash_buffer_pct: float = 0.02, top_n: int = 1,
                 mode: str = 'momentum'):
        if initial_capital <= 0.0:
            raise ValueError("initial_capital must be positive")
        if monthly_deposit < 0.0:
            raise ValueError("monthly_deposit must be non-negative")
        if lookback_days < 1 or trend_window < 1 or top_n < 1:
            raise ValueError("lookback_days, trend_window, and top_n must be positive")
        if mode not in {'momentum', 'contrarian'}:
            raise ValueError("mode must be 'momentum' or 'contrarian'")
        self.initial_capital = initial_capital
        self.monthly_deposit = monthly_deposit
        self.lookback_days = lookback_days
        self.trend_window = trend_window
        self.cash_buffer_pct = cash_buffer_pct
        self.top_n = top_n
        self.mode = mode

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        closes = data.xs('Close', level=1, axis=1)
        tickers = closes.columns.tolist()
        trades = pd.DataFrame(0.0, index=closes.index, columns=tickers)
        holdings = {ticker: 0.0 for ticker in tickers}
        cash = self.initial_capital
        month_start = closes.index.to_series().dt.to_period('M') != closes.index.to_series().shift(1).dt.to_period('M')

        for date in closes.index:
            if not month_start.loc[date]:
                continue
            cash += self.monthly_deposit
            prices = closes.loc[date]
            targets = self._select_targets(closes, date)
            cash = self._sell_non_targets(trades, date, prices, holdings, targets, cash)
            if not targets:
                continue
            cash = self._buy_targets(trades, date, prices, holdings, targets, cash)

        return trades

    def _select_targets(self, closes: pd.DataFrame, date: pd.Timestamp) -> list:
        window = closes.loc[:date]
        returns = window.pct_change(self.lookback_days).iloc[-1].dropna()
        trend = window.rolling(self.trend_window, min_periods=1).mean().iloc[-1]
        eligible = returns[window.iloc[-1] >= trend]
        if eligible.empty:
            return []
        ordered = eligible.sort_values(ascending=(self.mode == 'contrarian'))
        return ordered.head(self.top_n).index.tolist()

    def _sell_non_targets(self, trades: pd.DataFrame, date: pd.Timestamp, prices: pd.Series,
                          holdings: dict, targets: list, cash: float) -> float:
        for ticker, shares in holdings.items():
            if shares <= 0.0 or ticker in targets:
                continue
            trades.loc[date, ticker] -= shares
            cash += shares * prices[ticker]
            holdings[ticker] = 0.0
        return cash

    def _buy_targets(self, trades: pd.DataFrame, date: pd.Timestamp, prices: pd.Series,
                     holdings: dict, targets: list, cash: float) -> float:
        budget = cash * (1.0 - self.cash_buffer_pct)
        allocation = budget / len(targets)
        for ticker in targets:
            price = prices[ticker]
            if price <= 0.0:
                continue
            target_shares = allocation / price
            shares_to_buy = max(0.0, target_shares - holdings[ticker])
            trades.loc[date, ticker] += shares_to_buy
            cash -= shares_to_buy * price
            holdings[ticker] += shares_to_buy
        return cash
