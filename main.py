import os

from yfinance_data_provider import YFinanceDataProvider
from utils import get_market_risk_free_rate
from strategies import PullbackDollarCostAverage, SingleTickerStrategy
from backtester import Backtester
from performance_analyzer import PerformanceAnalyzer

# 1. Centralized Data Fetching
data_provider = YFinanceDataProvider(start_date='2015-01-01')
tickers = ['QQQ', 'SPY', 'VUG']
market_data = data_provider.get_data(tickers)
rf_rate = get_market_risk_free_rate()

# 2. Shared Backtester Configuration (UK Trader trading UK Stocks via IBKR UK margin account)
common_params = {
    'data': market_data,
    'initial_capital': 5000.0,          # £5,000 starting balance in a GBP account
    'monthly_deposit': 1500.0,          # £1,500 monthly savings deposit
    'deposit_fx_rate': 1.0,             # No deposit FX conversion for GBP deposits into a GBP account
    'account_currency': 'GBP',          # IBKR UK account funded and tracked in GBP
    'deposit_currency': 'GBP',          # Monthly deposits are made in GBP
    'fx_pct': 0.0,                      # No FX fee when a GBP account buys GBP-denominated UK stocks
    'commission_pct': 0.0005,           # IBKR Fixed pricing for UK stocks: 0.05% of trade value
    'commission_min': 3.0,              # IBKR Fixed pricing minimum: GBP 3.00 per stock order
    'tax_rate': 0.24,                   # UK Capital Gains Tax higher/additional rate band
    'tax_deferred': True,               # Approximate annual tax settlement rather than per-trade withholding
    'allow_margin': True,               # IBKR margin account allows borrowing against portfolio equity
    'annual_margin_rate': 0.05237,      # IBKR GBP margin loan rate for the first tier (0 to GBP 80,000)
    'max_leverage': 2.0,                # Approximate 50% end-of-day stock margin => 2.0x gross leverage cap
    'maintenance_margin_pct': 0.25,     # IBKR UK stock maintenance margin: 25% of stock value
    'allow_short': False,               # Short selling disabled
    'allow_fractional': True,           # IBKR supports fractional trading in eligible European shares/ETFs
    'slippage_pct': 0.0005,             # 0.05% bid-ask execution spread
    'execution_delay': 1,               # 1-day execution delay
    'execution_price_type': 'Open'      # Execute at next market open
}

# 3. Define Strategies to compare
comparison_configs = [
    ("QQQ Pullback DCA", SingleTickerStrategy(
        'QQQ',
        PullbackDollarCostAverage(base_monthly_amount=1250.0, extra_monthly_amount=625.0, pullback_window=10)
    )),
    ("SPY Pullback DCA", SingleTickerStrategy(
        'SPY',
        PullbackDollarCostAverage(base_monthly_amount=1250.0, extra_monthly_amount=625.0, pullback_window=10)
    )),
    ("VUG Pullback DCA", SingleTickerStrategy(
        'VUG',
        PullbackDollarCostAverage(base_monthly_amount=1250.0, extra_monthly_amount=625.0, pullback_window=10)
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
