from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricData:
    returns: pd.Series
    equity: pd.Series
    timing: Tuple[np.ndarray, float]


@dataclass(frozen=True)
class PerformanceMetrics:
    time_weighted_return: Optional[float]
    cagr: Optional[float]
    annualized_volatility: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    total_trades: int

    @classmethod
    def unavailable(cls, total_trades: int) -> "PerformanceMetrics":
        return cls(None, None, None, None, None, total_trades)

    def to_display_dict(self) -> dict:
        formatted_return = self._format_percent(self.time_weighted_return)
        return {
            "Time-Weighted Return": formatted_return,
            "Total Return": formatted_return,
            "CAGR": self._format_percent(self.cagr),
            "Annualized Vol": self._format_percent(self.annualized_volatility),
            "Sharpe Ratio": self._format_number(self.sharpe_ratio),
            "Max Drawdown": self._format_percent(self.max_drawdown),
            "Total Trades": self.total_trades,
        }

    @staticmethod
    def _format_percent(value: Optional[float]) -> str:
        return "N/A" if value is None or not np.isfinite(value) else f"{value:.2%}"

    @staticmethod
    def _format_number(value: Optional[float]) -> str:
        return "N/A" if value is None or not np.isfinite(value) else f"{value:.2f}"


class PerformanceMetricsCalculator:
    """Calculate validated performance metrics without plotting dependencies.

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
      include that deposit; this calculator does not correct cash-flow timing.
    - Equity remains positive and finite throughout the period. Zero,
      negative, or insolvent equity makes return-based metrics, CAGR, and
      drawdown undefined or misleading.
    - ``risk_free_rate`` is a finite annual effective decimal rate in the same
      currency (for example, ``0.04`` for 4%). Risk metrics use log-return
      increments normalized by each period's duration. CAGR uses elapsed
      timestamps; the first Backtester bar is one day because ``_simulate``
      explicitly uses ``days_elapsed=1`` there.
    """

    def __init__(self, risk_free_rate: float = 0.0):
        self._validate_risk_free_rate(risk_free_rate)
        self.risk_free_rate = risk_free_rate

    def calculate(
        self, portfolio_results: pd.DataFrame, total_trades: int
    ) -> Optional[PerformanceMetrics]:
        if len(portfolio_results) < 2:
            return None

        metric_data = self._get_metric_data(portfolio_results)
        if metric_data is None:
            return PerformanceMetrics.unavailable(total_trades)

        total_return = self._calculate_time_weighted_return(metric_data.returns)
        cagr = self._calculate_cagr(total_return, metric_data.timing)
        log_returns = self._to_log_returns(metric_data.returns)
        annualized_volatility = None
        sharpe_ratio = None
        if log_returns is not None:
            annualized_volatility = self._calculate_annualized_volatility(
                log_returns, metric_data.timing
            )
            sharpe_ratio = self._calculate_sharpe_ratio(
                log_returns, metric_data.timing, annualized_volatility
            )

        return PerformanceMetrics(
            time_weighted_return=total_return,
            cagr=cagr,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self._calculate_max_drawdown(metric_data.equity),
            total_trades=total_trades,
        )

    @staticmethod
    def _calculate_time_weighted_return(returns: pd.Series) -> float:
        return float((returns + 1.0).prod() - 1.0)

    def _calculate_cagr(
        self,
        total_return: float,
        timing: Tuple[np.ndarray, float],
    ) -> Optional[float]:
        growth = 1.0 + total_return
        if not np.isfinite(growth):
            return None
        if growth == 0.0:
            return -1.0

        _, elapsed_days = timing
        log_cagr = np.log(growth) * (365.25 / elapsed_days)
        if log_cagr >= np.log(np.finfo(float).max):
            return None
        return float(np.expm1(log_cagr))

    @staticmethod
    def _to_log_returns(returns: pd.Series) -> Optional[np.ndarray]:
        log_returns = np.log1p(returns.to_numpy(dtype=float))
        return log_returns if np.isfinite(log_returns).all() else None

    @staticmethod
    def _calculate_annualized_volatility(
        log_returns: np.ndarray, timing: Tuple[np.ndarray, float]
    ) -> float:
        durations, elapsed_days = timing
        mean_log_rate = float(log_returns.sum() / elapsed_days)
        residuals = log_returns - mean_log_rate * durations
        variance_rate = float(np.dot(residuals, residuals) / elapsed_days)
        return float(np.sqrt(max(variance_rate, 0.0) * 365.25))

    def _calculate_sharpe_ratio(
        self,
        log_returns: np.ndarray,
        timing: Tuple[np.ndarray, float],
        annualized_volatility: float,
    ) -> Optional[float]:
        if not np.isfinite(annualized_volatility):
            return None

        durations, elapsed_days = timing
        risk_free_log_rate = np.log1p(self.risk_free_rate) / 365.25
        excess_log_rate = float(
            (log_returns - risk_free_log_rate * durations).sum() / elapsed_days
        )
        if annualized_volatility <= 1e-9:
            return 0.0 if abs(excess_log_rate) <= 1e-12 else None
        return float((excess_log_rate * 365.25) / annualized_volatility)

    @staticmethod
    def _calculate_max_drawdown(equity: pd.Series) -> float:
        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / rolling_max
        return float(drawdowns.min())

    @staticmethod
    def _validate_risk_free_rate(risk_free_rate: float) -> None:
        if not np.isfinite(risk_free_rate) or risk_free_rate <= -1.0:
            raise ValueError(
                "risk_free_rate must be a finite annual rate greater than -100%"
            )

    @classmethod
    def _get_metric_data(cls, df: pd.DataFrame) -> Optional[MetricData]:
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

        timing = cls._get_period_timing(df.index)
        if timing is None:
            return None
        return MetricData(returns, equity, timing)

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
        if not PerformanceMetricsCalculator._has_valid_chronological_index(index):
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
