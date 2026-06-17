import os
import sys

# Automatically run within the local virtual environment (.venv) if it exists
_current_dir = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(_current_dir, ".venv", "bin", "python3")
if os.path.exists(_venv_python) and not sys.executable.startswith(os.path.join(_current_dir, ".venv")):
    os.execv(_venv_python, [_venv_python] + sys.argv)

from yfinance_data_provider import YFinanceDataProvider
from utils import get_market_risk_free_rate
from strategies import ValueAveraging, PullbackDollarCostAverage
from backtester import Backtester
from performance_analyzer import PerformanceAnalyzer

# 1. Centralized Data Fetching
data_provider = YFinanceDataProvider(start_date='2015-01-01')
qqq_data = data_provider.get_data(['QQQ'])
rf_rate = get_market_risk_free_rate()

# 2. Shared Backtester Configuration (UK Trader trading US Stocks: GIA with 0.35% FX fee & 20% CGT)
# Base Currency: GBP. Account Currency: USD.
common_params = {
    'data': qqq_data,
    'initial_capital': 5000.0 * 1.25,   # £5,000 converted to USD at GBP/USD = 1.25
    'monthly_deposit': 1500.0,          # £1,500 monthly savings deposit
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
    ("Value Averaging (2500)", ValueAveraging(monthly_target_increment=2500.0)),
    ("Pullback DCA (1250 + 625, 20d mean)", PullbackDollarCostAverage(
        base_monthly_amount=1250.0,
        extra_monthly_amount=625.0,
        pullback_window=20,
        pullback_pct=0.0
    )),
    ("Pullback DCA (1000 + 875, 20d 3%)", PullbackDollarCostAverage(
        base_monthly_amount=1000.0,
        extra_monthly_amount=875.0,
        pullback_window=20,
        pullback_pct=0.03
    )),
    ("Pullback DCA (750 + 1125, 50d 5%)", PullbackDollarCostAverage(
        base_monthly_amount=750.0,
        extra_monthly_amount=1125.0,
        pullback_window=50,
        pullback_pct=0.05
    ))
]

# 4. Run and Analyze All Strategies
analyzers = {}
for name, strategy in comparison_configs:
    print(f"\n{'='*20} Running: {name} {'='*20}")
    tester = Backtester(strategy=strategy, **common_params)
    results = tester.run()
    analyzer = PerformanceAnalyzer(results, tester, strategy, qqq_data, risk_free_rate=rf_rate)
    
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
