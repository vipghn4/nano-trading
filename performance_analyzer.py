import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from backtester import Backtester
from data_structures import Strategy

class PerformanceAnalyzer:
    """Calculate performance metrics for a completed backtest.

    The metrics are reliable only when all of the following conditions hold:

    - ``portfolio_results`` contains aligned, numeric ``returns`` and
      ``total_equity`` columns on a sorted, unique ``DatetimeIndex`` with at
      least two observations and a positive elapsed time.
    - ``returns`` contains finite, simple (not logarithmic) returns for real
      periods. Results produced by ``Backtester`` include a real first period:
      the change from initial capital plus the first start-of-period deposit to
      the end of the first bar. It is included in TWR and risk metrics.
    - Returns already remove deposits and withdrawals using their actual
      timing. For a start-of-period deposit, the return denominator must
      include that deposit; this class does not correct cash-flow timing.
    - Equity remains positive and finite throughout the period. Zero,
      negative, or insolvent equity makes return-based metrics, CAGR, and
      drawdown undefined or misleading.
    - ``risk_free_rate`` is a finite annual effective decimal rate in the same
      currency (for example, ``0.04`` for 4%). Risk metrics use log-return
      increments normalized by each period's duration. CAGR uses elapsed
      timestamps; the first Backtester bar is one day because ``_simulate``
      explicitly uses ``days_elapsed=1`` there.
    - ``backtester.state`` reflects the completed run when ``Total Trades`` is
      requested. The count represents executed transaction legs, including
      separate legs of a position flip or liquidation.

    ``Time-Weighted Return`` is the canonical cumulative return. The legacy
    ``Total Return`` key remains as an alias for compatibility. Drawdown is
    measured from the raw account-equity high-water mark, so deposits and
    withdrawals are not neutralized for that metric.
    """

    def __init__(self, portfolio_results: pd.DataFrame, backtester: Backtester, strategy: Strategy, data: pd.DataFrame, risk_free_rate: float = 0.0):
        self.portfolio_results = portfolio_results
        self.backtester = backtester
        self.strategy = strategy
        self.data = data
        self._validate_risk_free_rate(risk_free_rate)
        self.risk_free_rate = risk_free_rate
        self.metrics = {}

    @staticmethod
    def _validate_risk_free_rate(risk_free_rate: float) -> None:
        if not np.isfinite(risk_free_rate) or risk_free_rate <= -1.0:
            raise ValueError("risk_free_rate must be a finite annual rate greater than -100%")

    def analyze(self, save_path: Optional[str] = None) -> plt.Figure:
        """Public entry point for full analysis."""
        self._calculate_all_metrics()
        fig = self.plot_performance()
        self._maybe_save_performance_plot(fig, save_path)
        return fig

    def _calculate_all_metrics(self):
        """Orchestrates the calculation of all metric groups."""
        self.metrics.clear()
        df = self.portfolio_results
        if len(df) < 2:
            print("Insufficient data for analysis.")
            return

        self.metrics.update(self._get_return_metrics(df))
        self.metrics.update(self._get_risk_metrics(df))
        self.metrics.update(self._get_drawdown_metrics(df))
        self.metrics.update(self._get_activity_metrics())

        print("\n--- Verified Performance Metrics ---")
        print(pd.Series(self.metrics))

    def _get_return_metrics(self, df: pd.DataFrame) -> dict:
        metric_data = self._get_valid_metric_data(df)
        if metric_data is None:
            return self._unavailable_metrics(
                "Time-Weighted Return", "Total Return", "CAGR"
            )

        returns, _ = metric_data

        growth = float((returns + 1.0).prod())
        total_ret = growth - 1.0
        cagr = self._get_cagr(growth, df.index)

        formatted_total = self._format_percent(total_ret)
        return {
            "Time-Weighted Return": formatted_total,
            "Total Return": formatted_total,
            "CAGR": self._format_percent(cagr),
        }

    def _get_risk_metrics(self, df: pd.DataFrame) -> dict:
        metric_data = self._get_valid_metric_data(df)
        if metric_data is None:
            return self._unavailable_metrics("Annualized Vol", "Sharpe Ratio")

        returns, _ = metric_data

        timing = self._get_period_timing(df.index)
        if timing is None:
            return self._unavailable_metrics("Annualized Vol", "Sharpe Ratio")

        risk_statistics = self._get_risk_statistics(returns, timing)
        if risk_statistics is None:
            return self._unavailable_metrics("Annualized Vol", "Sharpe Ratio")

        ann_vol, sharpe = risk_statistics

        return {
            "Annualized Vol": self._format_percent(ann_vol),
            "Sharpe Ratio": self._format_number(sharpe),
        }

    def _get_drawdown_metrics(self, df: pd.DataFrame) -> dict:
        metric_data = self._get_valid_metric_data(df)
        if metric_data is None:
            return self._unavailable_metrics("Max Drawdown")

        _, equity = metric_data

        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / rolling_max
        return {"Max Drawdown": self._format_percent(float(drawdowns.min()))}

    def _get_cagr(self, growth: float, index: pd.Index) -> Optional[float]:
        timing = self._get_period_timing(index)
        if timing is None or not np.isfinite(growth):
            return None
        if growth == 0.0:
            return -1.0

        _, elapsed_days = timing
        log_cagr = np.log(growth) * (365.25 / elapsed_days)
        if log_cagr >= np.log(np.finfo(float).max):
            return None
        return float(np.expm1(log_cagr))

    def _get_risk_statistics(
        self, returns: pd.Series, timing: Tuple[np.ndarray, float]
    ) -> Optional[Tuple[float, Optional[float]]]:
        durations, elapsed_days = timing
        log_returns = np.log1p(returns.to_numpy(dtype=float))
        if not np.isfinite(log_returns).all():
            return None

        mean_log_rate = float(log_returns.sum() / elapsed_days)
        residuals = log_returns - mean_log_rate * durations
        variance_rate = float(np.dot(residuals, residuals) / elapsed_days)
        ann_vol = float(np.sqrt(max(variance_rate, 0.0) * 365.25))
        sharpe = self._get_sharpe_ratio(
            log_returns, durations, elapsed_days, ann_vol
        )
        return ann_vol, sharpe

    def _get_sharpe_ratio(
        self,
        log_returns: np.ndarray,
        durations: np.ndarray,
        elapsed_days: float,
        ann_vol: float,
    ) -> Optional[float]:
        if not np.isfinite(ann_vol):
            return None

        risk_free_log_rate = np.log1p(self.risk_free_rate) / 365.25
        excess_log_rate = float(
            (log_returns - risk_free_log_rate * durations).sum() / elapsed_days
        )
        if ann_vol <= 1e-9:
            return 0.0 if abs(excess_log_rate) <= 1e-12 else None
        return float((excess_log_rate * 365.25) / ann_vol)

    @staticmethod
    def _unavailable_metrics(*names: str) -> dict:
        return {name: "N/A" for name in names}

    @classmethod
    def _get_valid_metric_data(
        cls, df: pd.DataFrame
    ) -> Optional[Tuple[pd.Series, pd.Series]]:
        if not cls._has_valid_chronological_index(df.index):
            return None

        returns = pd.to_numeric(df['returns'], errors='coerce')
        equity = pd.to_numeric(df['total_equity'], errors='coerce')
        if (
            not np.isfinite(returns).all()
            or (returns < -1.0).any()
            or not np.isfinite(equity).all()
            or (equity <= 0.0).any()
        ):
            return None
        return returns, equity

    @staticmethod
    def _format_percent(value: Optional[float]) -> str:
        return "N/A" if value is None or not np.isfinite(value) else f"{value:.2%}"

    @staticmethod
    def _format_number(value: Optional[float]) -> str:
        return "N/A" if value is None or not np.isfinite(value) else f"{value:.2f}"

    @staticmethod
    def _has_valid_chronological_index(index: pd.Index) -> bool:
        return (
            isinstance(index, pd.DatetimeIndex)
            and index.is_unique
            and index.is_monotonic_increasing
            and len(index) >= 2
        )

    @staticmethod
    def _get_period_timing(index: pd.Index) -> Optional[Tuple[np.ndarray, float]]:
        """Return each period's duration and total duration in days."""
        if not PerformanceAnalyzer._has_valid_chronological_index(index):
            return None

        deltas = (
            pd.Series(index)
            .diff()
            .dt.total_seconds()
            .to_numpy(dtype=float)[1:]
            / 86400.0
        )
        if not np.isfinite(deltas).all() or (deltas <= 0.0).any():
            return None

        durations = np.concatenate(([1.0], deltas))
        elapsed_days = float(durations.sum())
        if elapsed_days <= 0.0:
            return None

        return durations, elapsed_days

    def _get_activity_metrics(self) -> dict:
        count = self.backtester.state.executed_trades_count if hasattr(self.backtester, 'state') else 0
        return {"Total Trades": int(count)}

    def plot_performance(self) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        self._plot_benchmarks(ax1)
        self._plot_equity_curve(ax2)
        plt.tight_layout()
        return fig

    def _maybe_save_performance_plot(
        self, fig: plt.Figure, save_path: Optional[str]
    ) -> None:
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)

    @classmethod
    def plot_comparison(cls, analyzers: dict) -> plt.Figure:
        """
        Generates a consolidated comparison plot of equity curves and drawdowns
        across multiple PerformanceAnalyzer instances.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Subplot 1: Absolute Equity Curve Comparison
        for name, analyzer in analyzers.items():
            ax1.plot(analyzer.portfolio_results['total_equity'], label=name, alpha=0.8)
            
        # Plot Cumulative Invested Capital as a baseline using parameters from the first analyzer
        if analyzers:
            first_analyzer = next(iter(analyzers.values()))
            market_data = first_analyzer.data
            is_new_month = market_data.index.to_series().dt.to_period('M') != market_data.index.to_series().shift(1).dt.to_period('M')
            config = first_analyzer.backtester.config
            initial_usd = config.initial_capital
            monthly_usd = config.monthly_deposit * config.deposit_fx_rate
            invested_capital = (is_new_month.astype(float) * monthly_usd).cumsum() + initial_usd
            ax1.plot(invested_capital, label='Cumulative Invested Capital', color='black', linestyle='--', alpha=0.7)
            
        ax1.set_title('Portfolio Equity Comparison ($ USD)')
        ax1.set_ylabel('Total Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Under-water Drawdown Comparison
        for name, analyzer in analyzers.items():
            rolling_max = analyzer.portfolio_results['total_equity'].cummax()
            drawdowns = (analyzer.portfolio_results['total_equity'] - rolling_max) / rolling_max
            ax2.plot(drawdowns * 100.0, label=name, alpha=0.8)
            
        ax2.set_title('Portfolio Drawdown Comparison (%)')
        ax2.set_ylabel('Drawdown %')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def _plot_benchmarks(self, ax):
        tickers = self.data.columns.get_level_values(0).unique()
        for t in tickers:
            bench = self.data[t]['Close'] / self.data[t]['Close'].iloc[0]
            ax.plot(bench, label=f'{t}', alpha=0.5)
        ax.set_title('Normalized Benchmark Growth')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_equity_curve(self, ax):
        ax.plot(self.portfolio_results['total_equity'], color='green', label='Strategy Equity')
        ax.set_title('Portfolio Equity Curve (Net of Fees & Taxes)')
        ax.fill_between(self.portfolio_results.index, self.portfolio_results['total_equity'],
                         self.backtester.config.initial_capital, alpha=0.1, color='green')
        ax.legend()
        ax.grid(True, alpha=0.3)
