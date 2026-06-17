import os
import sys

# Automatically run within the local virtual environment (.venv) if it exists
_current_dir = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(_current_dir, ".venv", "bin", "python3")
if os.path.exists(_venv_python) and not sys.executable.startswith(os.path.join(_current_dir, ".venv")):
    os.execv(_venv_python, [_venv_python] + sys.argv)

from yfinance_data_provider import YFinanceDataProvider
from utils import get_market_risk_free_rate
from strategies import (
    PullbackDollarCostAverage,
    SingleTickerStrategy,
    RiskAdjustedContributionStrategy,
    TrendMeanReversionContributionStrategy,
)
from backtester import Backtester
from performance_analyzer import PerformanceAnalyzer

# 1. Centralized Data Fetching
data_provider = YFinanceDataProvider(start_date='2015-01-01')
tickers = ['QQQ', 'SPY', 'VUG']
market_data = data_provider.get_data(tickers)
rf_rate = get_market_risk_free_rate()
initial_capital = 5000.0 * 1.25
monthly_deposit = 1500.0

# 2. Shared Backtester Configuration (UK Trader trading US Stocks: GIA with 0.35% FX fee & 20% CGT)
# Base Currency: GBP. Account Currency: USD.
common_params = {
    'data': market_data,
    'initial_capital': initial_capital, # £5,000 converted to USD at GBP/USD = 1.25
    'monthly_deposit': monthly_deposit, # £1,500 monthly savings deposit
    'deposit_fx_rate': 1.25,            # GBP/USD exchange rate for converting savings
    'fx_pct': 0.0035,                   # 0.35% FX conversion markup fee charged by UK broker
    'commission_pct': 0.0,              # Zero commission broker (like Freetrade/T212)
    'tax_rate': 0.20,                   # 20% UK Capital Gains Tax (Higher-rate band)
    'tax_deferred': True,               # Taxes settled annually at tax year-end
    'allow_margin': False,              # Cash account (no leverage/borrowing)
    'allow_short': False,               # Short selling disabled
    'allow_fractional': True,           # Fractional shares enabled
    'slippage_pct': 0.0005,             # 0.05% bid-ask execution spread
    'execution_delay': 1,               # 1-day execution delay
    'execution_price_type': 'Open'      # Execute at next US market open
}

# 3. Define Strategies to compare
comparison_configs = [
    ("QQQ Pullback DCA", SingleTickerStrategy(
        'QQQ',
        PullbackDollarCostAverage(base_monthly_amount=1250.0, extra_monthly_amount=625.0, pullback_window=10)
    )),
    ("Risk-Adjusted Leader Contributions", RiskAdjustedContributionStrategy(
        initial_capital=initial_capital,
        monthly_deposit=monthly_deposit * 1.25,
        momentum_window=126,
        volatility_window=63,
        trend_window=200,
        top_n=1,
        fallback_ticker='QQQ'
    )),
    ("Risk-Adjusted Split Contributions", RiskAdjustedContributionStrategy(
        initial_capital=initial_capital,
        monthly_deposit=monthly_deposit * 1.25,
        momentum_window=126,
        volatility_window=63,
        trend_window=200,
        top_n=2,
        fallback_ticker='QQQ'
    )),
    ("Trend Mean-Reversion Contributions", TrendMeanReversionContributionStrategy(
        initial_capital=initial_capital,
        monthly_deposit=monthly_deposit * 1.25,
        long_window=126,
        short_window=21,
        trend_window=200,
        fallback_ticker='QQQ'
    ))
]

# 4. Run and Analyze All Strategies
analyzers = {}
for name, strategy in comparison_configs:
    print(f"\n{'='*20} Running: {name} {'='*20}")
    tester = Backtester(strategy=strategy, **common_params)
    results = tester.run()
    analyzer = PerformanceAnalyzer(results, tester, strategy, market_data, risk_free_rate=rf_rate)
    
    # Calculate and display metrics
    analyzer._calculate_all_metrics()
    analyzers[name] = analyzer

# 5. Generate and Save Consolidated Comparison Plot
comparison_fig = PerformanceAnalyzer.plot_comparison(analyzers)
save_path = os.path.join("runs", "strategy_comparison.png")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
comparison_fig.savefig(save_path)
import matplotlib.pyplot as plt
plt.close(comparison_fig)
print(f"\nSaved consolidated comparison plot to: {save_path}")
