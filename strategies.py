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
        portfolio_value = cash + sum(holdings[ticker] * prices[ticker] for ticker in targets)
        allocation = (portfolio_value * (1.0 - self.cash_buffer_pct)) / len(targets)
        for ticker in targets:
            price = prices[ticker]
            if price <= 0.0:
                continue
            target_shares = allocation / price
            shares_delta = target_shares - holdings[ticker]
            trades.loc[date, ticker] += shares_delta
            cash -= shares_delta * price
            holdings[ticker] += shares_delta
        return cash


class RegimeSwitchStrategy(Strategy):
    def __init__(self, initial_capital: float, monthly_deposit: float,
                 growth_tickers: list, defensive_ticker: str,
                 lookback_days: int = 126, trend_window: int = 200,
                 cash_buffer_pct: float = 0.02):
        if not growth_tickers or not defensive_ticker:
            raise ValueError("growth_tickers and defensive_ticker are required")
        self.initial_capital = initial_capital
        self.monthly_deposit = monthly_deposit
        self.growth_tickers = growth_tickers
        self.defensive_ticker = defensive_ticker
        self.lookback_days = lookback_days
        self.trend_window = trend_window
        self.cash_buffer_pct = cash_buffer_pct

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        closes = data.xs('Close', level=1, axis=1)
        trades = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        holdings = {ticker: 0.0 for ticker in closes.columns}
        cash = self.initial_capital
        month_start = closes.index.to_series().dt.to_period('M') != closes.index.to_series().shift(1).dt.to_period('M')

        for date in closes.index:
            if not month_start.loc[date]:
                continue
            cash += self.monthly_deposit
            target = self._select_target(closes, date)
            prices = closes.loc[date]
            for ticker, shares in holdings.items():
                if shares > 0.0 and ticker != target:
                    trades.loc[date, ticker] -= shares
                    cash += shares * prices[ticker]
                    holdings[ticker] = 0.0
            if target is None or prices[target] <= 0.0:
                continue
            investable_cash = cash * (1.0 - self.cash_buffer_pct)
            shares_to_buy = investable_cash / prices[target]
            trades.loc[date, target] += shares_to_buy
            holdings[target] += shares_to_buy
            cash -= shares_to_buy * prices[target]

        return trades

    def _select_target(self, closes: pd.DataFrame, date: pd.Timestamp) -> Optional[str]:
        history = closes.loc[:date]
        current = history.iloc[-1]
        trend = history.rolling(self.trend_window, min_periods=1).mean().iloc[-1]
        growth_returns = history[self.growth_tickers].pct_change(self.lookback_days).iloc[-1].dropna()
        growth_returns = growth_returns[current[growth_returns.index] >= trend[growth_returns.index]]
        if not growth_returns.empty:
            return growth_returns.sort_values(ascending=False).index[0]
        if current[self.defensive_ticker] >= trend[self.defensive_ticker]:
            return self.defensive_ticker
        return None


class SelectiveContributionStrategy(Strategy):
    def __init__(self, initial_capital: float, monthly_deposit: float,
                 lookback_days: int = 126, trend_window: int = 200,
                 pullback_window: int = 10):
        self.initial_capital = initial_capital
        self.monthly_deposit = monthly_deposit
        self.lookback_days = lookback_days
        self.trend_window = trend_window
        self.pullback_window = pullback_window

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        closes = data.xs('Close', level=1, axis=1)
        trades = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        month_start = closes.index.to_series().dt.to_period('M') != closes.index.to_series().shift(1).dt.to_period('M')

        for date in closes.index:
            if not month_start.loc[date]:
                continue
            ticker = self._select_ticker(closes, date)
            if ticker is None:
                continue
            price = closes.loc[date, ticker]
            if price <= 0.0:
                continue
            deposit = self.monthly_deposit
            if self._is_pullback(closes[ticker].loc[:date]):
                deposit += self.monthly_deposit * 0.25
            trades.loc[date, ticker] = deposit / price

        first_date = closes.index[0]
        first_ticker = self._select_ticker(closes, first_date) or closes.columns[0]
        first_price = closes.loc[first_date, first_ticker]
        trades.loc[first_date, first_ticker] += self.initial_capital / first_price
        return trades

    def _select_ticker(self, closes: pd.DataFrame, date: pd.Timestamp) -> Optional[str]:
        history = closes.loc[:date]
        current = history.iloc[-1]
        trend = history.rolling(self.trend_window, min_periods=1).mean().iloc[-1]
        scores = history.pct_change(self.lookback_days).iloc[-1].dropna()
        scores = scores[current[scores.index] >= trend[scores.index]]
        if scores.empty:
            return None
        return scores.sort_values(ascending=False).index[0]

    def _is_pullback(self, history: pd.Series) -> bool:
        rolling_mean = history.rolling(self.pullback_window, min_periods=1).mean().iloc[-1]
        return history.iloc[-1] <= rolling_mean


class MomentumContributionStrategy(Strategy):
    def __init__(self, initial_capital: float, monthly_deposit: float,
                 lookback_days: int = 126, top_n: int = 1,
                 fallback_ticker: str = 'QQQ'):
        if initial_capital <= 0.0 or monthly_deposit < 0.0 or lookback_days < 1 or top_n < 1:
            raise ValueError("invalid momentum contribution inputs")
        self.initial_capital = initial_capital
        self.monthly_deposit = monthly_deposit
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.fallback_ticker = fallback_ticker

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        closes = data.xs('Close', level=1, axis=1)
        trades = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        month_start = closes.index.to_series().dt.to_period('M') != closes.index.to_series().shift(1).dt.to_period('M')

        initial_price = closes.loc[closes.index[0], self.fallback_ticker]
        trades.loc[closes.index[0], self.fallback_ticker] += self.initial_capital / initial_price

        for date in closes.index:
            if not month_start.loc[date]:
                continue
            targets = self._select_targets(closes, date)
            if not targets:
                targets = [self.fallback_ticker]
            allocation = self.monthly_deposit / len(targets)
            for ticker in targets:
                price = closes.loc[date, ticker]
                if price > 0.0:
                    trades.loc[date, ticker] += allocation / price
        return trades

    def _select_targets(self, closes: pd.DataFrame, date: pd.Timestamp) -> list:
        scores = closes.loc[:date].pct_change(self.lookback_days).iloc[-1].dropna()
        if scores.empty:
            return []
        return scores.sort_values(ascending=False).head(self.top_n).index.tolist()


class DefensiveContributionStrategy(Strategy):
    def __init__(self, initial_capital: float, monthly_deposit: float,
                 growth_ticker: str = 'QQQ', defensive_ticker: str = 'SPY',
                 trend_window: int = 200):
        self.initial_capital = initial_capital
        self.monthly_deposit = monthly_deposit
        self.growth_ticker = growth_ticker
        self.defensive_ticker = defensive_ticker
        self.trend_window = trend_window

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        closes = data.xs('Close', level=1, axis=1)
        trades = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        month_start = closes.index.to_series().dt.to_period('M') != closes.index.to_series().shift(1).dt.to_period('M')
        first_price = closes.loc[closes.index[0], self.growth_ticker]
        trades.loc[closes.index[0], self.growth_ticker] += self.initial_capital / first_price

        for date in closes.index:
            if not month_start.loc[date]:
                continue
            target = self._select_target(closes, date)
            price = closes.loc[date, target]
            if price > 0.0:
                trades.loc[date, target] += self.monthly_deposit / price
        return trades

    def _select_target(self, closes: pd.DataFrame, date: pd.Timestamp) -> str:
        history = closes.loc[:date, [self.growth_ticker, self.defensive_ticker]]
        trend = history.rolling(self.trend_window, min_periods=1).mean().iloc[-1]
        current = history.iloc[-1]
        if current[self.growth_ticker] >= trend[self.growth_ticker]:
            return self.growth_ticker
        if current[self.defensive_ticker] >= trend[self.defensive_ticker]:
            return self.defensive_ticker
        return self.growth_ticker


class BufferedMomentumContributionStrategy(Strategy):
    def __init__(self, initial_capital: float, monthly_deposit: float,
                 lookback_days: int = 126, trend_window: int = 200,
                 pullback_window: int = 10, core_ticker: str = 'QQQ'):
        self.initial_capital = initial_capital
        self.monthly_deposit = monthly_deposit
        self.lookback_days = lookback_days
        self.trend_window = trend_window
        self.pullback_window = pullback_window
        self.core_ticker = core_ticker

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        closes = data.xs('Close', level=1, axis=1)
        trades = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        buffer_cash = 0.0
        month_start = closes.index.to_series().dt.to_period('M') != closes.index.to_series().shift(1).dt.to_period('M')
        first_price = closes.loc[closes.index[0], self.core_ticker]
        trades.loc[closes.index[0], self.core_ticker] += self.initial_capital / first_price

        for date in closes.index:
            if not month_start.loc[date]:
                continue
            buffer_cash += self.monthly_deposit * 0.4
            leader = self._select_leader(closes, date)
            if leader is None:
                leader = self.core_ticker
            self._buy(trades, closes, date, leader, self.monthly_deposit * 0.6)
            opportunity = self._select_pullback_asset(closes, date)
            if opportunity is not None and buffer_cash > 0.0:
                self._buy(trades, closes, date, opportunity, buffer_cash)
                buffer_cash = 0.0

        return trades

    def _select_leader(self, closes: pd.DataFrame, date: pd.Timestamp) -> Optional[str]:
        history = closes.loc[:date]
        current = history.iloc[-1]
        trend = history.rolling(self.trend_window, min_periods=1).mean().iloc[-1]
        scores = history.pct_change(self.lookback_days).iloc[-1].dropna()
        scores = scores[current[scores.index] >= trend[scores.index]]
        if scores.empty:
            return None
        return scores.sort_values(ascending=False).index[0]

    def _select_pullback_asset(self, closes: pd.DataFrame, date: pd.Timestamp) -> Optional[str]:
        history = closes.loc[:date]
        current = history.iloc[-1]
        trend = history.rolling(self.trend_window, min_periods=1).mean().iloc[-1]
        short_mean = history.rolling(self.pullback_window, min_periods=1).mean().iloc[-1]
        scores = history.pct_change(self.lookback_days).iloc[-1].dropna()
        scores = scores[current[scores.index] >= trend[scores.index]]
        if scores.empty:
            return None
        candidates = []
        for ticker in scores.sort_values(ascending=False).head(2).index:
            pullback = (short_mean[ticker] - current[ticker]) / short_mean[ticker]
            if pullback >= 0.0:
                candidates.append((pullback, ticker))
        if not candidates:
            return None
        return max(candidates)[1]

    def _buy(self, trades: pd.DataFrame, closes: pd.DataFrame, date: pd.Timestamp,
             ticker: str, amount: float):
        price = closes.loc[date, ticker]
        if price > 0.0 and amount > 0.0:
            trades.loc[date, ticker] += amount / price


class CoreSatelliteContributionStrategy(Strategy):
    def __init__(self, initial_capital: float, monthly_deposit: float,
                 core_ticker: str = 'QQQ', satellite_tickers: Optional[list] = None,
                 lookback_days: int = 126):
        self.initial_capital = initial_capital
        self.monthly_deposit = monthly_deposit
        self.core_ticker = core_ticker
        self.satellite_tickers = satellite_tickers or ['SPY', 'VUG']
        self.lookback_days = lookback_days

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        closes = data.xs('Close', level=1, axis=1)
        trades = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        month_start = closes.index.to_series().dt.to_period('M') != closes.index.to_series().shift(1).dt.to_period('M')
        core_price = closes.loc[closes.index[0], self.core_ticker]
        trades.loc[closes.index[0], self.core_ticker] += (self.initial_capital * 0.75) / core_price
        satellite = self.satellite_tickers[0]
        satellite_price = closes.loc[closes.index[0], satellite]
        trades.loc[closes.index[0], satellite] += (self.initial_capital * 0.25) / satellite_price

        for date in closes.index:
            if not month_start.loc[date]:
                continue
            self._buy_monthly(trades, closes, date, self.core_ticker, self.monthly_deposit * 0.7)
            satellite = self._select_satellite(closes, date)
            self._buy_monthly(trades, closes, date, satellite, self.monthly_deposit * 0.3)

        return trades

    def _select_satellite(self, closes: pd.DataFrame, date: pd.Timestamp) -> str:
        scores = closes.loc[:date, self.satellite_tickers].pct_change(self.lookback_days).iloc[-1].dropna()
        if scores.empty:
            return self.satellite_tickers[0]
        return scores.sort_values(ascending=False).index[0]

    def _buy_monthly(self, trades: pd.DataFrame, closes: pd.DataFrame, date: pd.Timestamp,
                     ticker: str, amount: float):
        price = closes.loc[date, ticker]
        if price > 0.0:
            trades.loc[date, ticker] += amount / price


class EventDrivenSwitchStrategy(Strategy):
    def __init__(self, initial_capital: float, monthly_deposit: float,
                 growth_ticker: str = 'QQQ', defensive_ticker: str = 'SPY',
                 trend_window: int = 200, relative_window: int = 63):
        self.initial_capital = initial_capital
        self.monthly_deposit = monthly_deposit
        self.growth_ticker = growth_ticker
        self.defensive_ticker = defensive_ticker
        self.trend_window = trend_window
        self.relative_window = relative_window

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        closes = data.xs('Close', level=1, axis=1)
        trades = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        holdings = {ticker: 0.0 for ticker in closes.columns}
        cash = self.initial_capital
        current_target = None
        month_start = closes.index.to_series().dt.to_period('M') != closes.index.to_series().shift(1).dt.to_period('M')

        for date in closes.index:
            if month_start.loc[date]:
                cash += self.monthly_deposit
            target = self._select_target(closes, date)
            prices = closes.loc[date]
            if target != current_target:
                for ticker, shares in holdings.items():
                    if shares > 0.0:
                        trades.loc[date, ticker] -= shares
                        cash += shares * prices[ticker]
                        holdings[ticker] = 0.0
                current_target = target
            if current_target is not None and cash > 0.0 and prices[current_target] > 0.0:
                shares_to_buy = cash / prices[current_target]
                trades.loc[date, current_target] += shares_to_buy
                holdings[current_target] += shares_to_buy
                cash = 0.0

        return trades

    def _select_target(self, closes: pd.DataFrame, date: pd.Timestamp) -> Optional[str]:
        history = closes.loc[:date, [self.growth_ticker, self.defensive_ticker]]
        current = history.iloc[-1]
        trend = history.rolling(self.trend_window, min_periods=1).mean().iloc[-1]
        relative = history.pct_change(self.relative_window).iloc[-1].dropna()
        if current[self.growth_ticker] >= trend[self.growth_ticker]:
            if current[self.defensive_ticker] < trend[self.defensive_ticker]:
                return self.growth_ticker
            if relative.get(self.growth_ticker, 0.0) >= relative.get(self.defensive_ticker, 0.0):
                return self.growth_ticker
        if current[self.defensive_ticker] >= trend[self.defensive_ticker]:
            return self.defensive_ticker
        return None


class EventDrivenGrowthRotationStrategy(Strategy):
    def __init__(self, initial_capital: float, monthly_deposit: float,
                 growth_tickers: list, defensive_ticker: str = 'SPY',
                 trend_window: int = 200, relative_window: int = 63):
        self.initial_capital = initial_capital
        self.monthly_deposit = monthly_deposit
        self.growth_tickers = growth_tickers
        self.defensive_ticker = defensive_ticker
        self.trend_window = trend_window
        self.relative_window = relative_window

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        closes = data.xs('Close', level=1, axis=1)
        trades = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        holdings = {ticker: 0.0 for ticker in closes.columns}
        cash = self.initial_capital
        current_target = None
        month_start = closes.index.to_series().dt.to_period('M') != closes.index.to_series().shift(1).dt.to_period('M')

        for date in closes.index:
            if month_start.loc[date]:
                cash += self.monthly_deposit
            target = self._select_target(closes, date)
            prices = closes.loc[date]
            if target != current_target:
                for ticker, shares in holdings.items():
                    if shares > 0.0:
                        trades.loc[date, ticker] -= shares
                        cash += shares * prices[ticker]
                        holdings[ticker] = 0.0
                current_target = target
            if current_target is not None and cash > 0.0 and prices[current_target] > 0.0:
                shares_to_buy = cash / prices[current_target]
                trades.loc[date, current_target] += shares_to_buy
                holdings[current_target] += shares_to_buy
                cash = 0.0

        return trades

    def _select_target(self, closes: pd.DataFrame, date: pd.Timestamp) -> Optional[str]:
        history = closes.loc[:date]
        current = history.iloc[-1]
        trend = history.rolling(self.trend_window, min_periods=1).mean().iloc[-1]
        growth = history[self.growth_tickers].pct_change(self.relative_window).iloc[-1].dropna()
        growth = growth[current[growth.index] >= trend[growth.index]]
        if not growth.empty:
            return growth.sort_values(ascending=False).index[0]
        if current[self.defensive_ticker] >= trend[self.defensive_ticker]:
            return self.defensive_ticker
        return None


class RelativePullbackContributionStrategy(Strategy):
    def __init__(self, initial_capital: float, monthly_deposit: float,
                 long_window: int = 126, short_window: int = 21,
                 trend_window: int = 200, top_n: int = 1,
                 fallback_ticker: str = 'QQQ'):
        self.initial_capital = initial_capital
        self.monthly_deposit = monthly_deposit
        self.long_window = long_window
        self.short_window = short_window
        self.trend_window = trend_window
        self.top_n = top_n
        self.fallback_ticker = fallback_ticker

    def generate_trades(self, data: pd.DataFrame) -> pd.DataFrame:
        closes = data.xs('Close', level=1, axis=1)
        trades = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        month_start = closes.index.to_series().dt.to_period('M') != closes.index.to_series().shift(1).dt.to_period('M')
        initial_price = closes.loc[closes.index[0], self.fallback_ticker]
        trades.loc[closes.index[0], self.fallback_ticker] += self.initial_capital / initial_price

        for date in closes.index:
            if not month_start.loc[date]:
                continue
            targets = self._select_targets(closes, date)
            if not targets:
                targets = [self.fallback_ticker]
            allocation = self.monthly_deposit / len(targets)
            for ticker in targets:
                price = closes.loc[date, ticker]
                if price > 0.0:
                    trades.loc[date, ticker] += allocation / price

        return trades

    def _select_targets(self, closes: pd.DataFrame, date: pd.Timestamp) -> list:
        history = closes.loc[:date]
        current = history.iloc[-1]
        trend = history.rolling(self.trend_window, min_periods=1).mean().iloc[-1]
        long_return = history.pct_change(self.long_window).iloc[-1].dropna()
        short_return = history.pct_change(self.short_window).iloc[-1].dropna()
        aligned = long_return.index.intersection(short_return.index)
        scores = (long_return[aligned] - short_return[aligned])
        scores = scores[current[scores.index] >= trend[scores.index]]
        if scores.empty:
            return []
        return scores.sort_values(ascending=False).head(self.top_n).index.tolist()
