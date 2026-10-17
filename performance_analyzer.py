from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from backtester import Backtester
from data_structures import Strategy
from performance_metrics import PerformanceMetricsCalculator


class PerformanceAnalyzer:
    """Coordinate metric calculation and performance plotting for a backtest."""

    def __init__(
        self,
        portfolio_results: pd.DataFrame,
        backtester: Backtester,
        strategy: Strategy,
        data: pd.DataFrame,
        risk_free_rate: float = 0.0,
    ):
        self.portfolio_results = portfolio_results
        self.backtester = backtester
        self.strategy = strategy
        self.data = data
        self.metrics_calculator = PerformanceMetricsCalculator(risk_free_rate)
        self.risk_free_rate = risk_free_rate
        self.metrics = {}

    def analyze(self, save_path: Optional[str] = None) -> plt.Figure:
        """Public entry point for full analysis."""
        self._calculate_all_metrics()
        fig = self.plot_performance()
        self._maybe_save_performance_plot(fig, save_path)
        return fig

    def _calculate_all_metrics(self):
        """Orchestrates the calculation of all metric groups."""
        self.metrics.clear()
        total_trades = self._get_executed_trade_count()
        calculated_metrics = self.metrics_calculator.calculate(
            self.portfolio_results, total_trades
        )
        if calculated_metrics is None:
            print("Insufficient data for analysis.")
            return

        self.metrics.update(calculated_metrics.to_display_dict())

        print("\n--- Verified Performance Metrics ---")
        print(pd.Series(self.metrics))

    def _get_executed_trade_count(self) -> int:
        state = getattr(self.backtester, "state", None)
        return int(getattr(state, "executed_trades_count", 0))

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
