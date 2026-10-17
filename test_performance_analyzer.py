import unittest

import numpy as np
import pandas as pd

from backtester import Backtester
from data_structures import Strategy
from performance_analyzer import PerformanceAnalyzer


class TestPerformanceMetrics(unittest.TestCase):
    def test_start_of_period_deposit_is_included_in_return_denominator(self):
        dates = pd.to_datetime(['2026-01-01', '2026-01-02', '2026-02-01'])
        columns = pd.MultiIndex.from_tuples(
            [('ABC', 'Open'), ('ABC', 'Close')], names=['Ticker', 'Price']
        )
        data = pd.DataFrame(
            [[100.0, 100.0], [100.0, 110.0], [110.0, 220.0]],
            index=dates,
            columns=columns,
        )

        signals = pd.DataFrame(0.0, index=dates, columns=['ABC'])
        signals.iloc[0] = 1.0

        class BuyOnceStrategy(Strategy):
            def generate_trades(self, visible_data):
                return signals.loc[visible_data.index]

        tester = Backtester(
            data=data,
            strategy=BuyOnceStrategy(),
            initial_capital=100.0,
            monthly_deposit=50.0,
            slippage_pct=0.0,
            execution_delay=1,
            execution_price_type='Open',
        )

        results = tester.run()

        np.testing.assert_allclose(results['total_equity'], [150.0, 160.0, 320.0])
        self.assertAlmostEqual(results['returns'].iloc[0], 0.0)
        self.assertAlmostEqual(results['returns'].iloc[1], 10.0 / 150.0)
        self.assertAlmostEqual(results['returns'].iloc[2], 110.0 / 210.0)

    def test_first_backtest_row_is_a_real_return_period(self):
        dates = pd.date_range('2026-01-01', periods=2, freq='D')
        columns = pd.MultiIndex.from_tuples(
            [('ABC', 'Open'), ('ABC', 'Close')], names=['Ticker', 'Price']
        )
        data = pd.DataFrame(100.0, index=dates, columns=columns)

        class NoOpStrategy(Strategy):
            def generate_trades(self, visible_data):
                return pd.DataFrame(0.0, index=visible_data.index, columns=['ABC'])

        tester = Backtester(
            data=data,
            strategy=NoOpStrategy(),
            initial_capital=100.0,
            monthly_deposit=10.0,
            execution_delay=1,
        )

        results = tester._finalize_results(
            cash_hist=np.array([120.0, 120.0]),
            val_hist=np.zeros(2),
            deposit_hist=np.array([10.0, 0.0]),
        )

        self.assertAlmostEqual(results['returns'].iloc[0], 10.0 / 110.0)

    def test_return_label_explicitly_identifies_time_weighted_return(self):
        results = pd.DataFrame(
            {
                'returns': [0.10, 0.0],
                'total_equity': [110.0, 110.0],
            },
            index=pd.date_range('2026-01-01', periods=2),
        )
        analyzer = self._analyzer(results)
        metrics = analyzer._get_return_metrics(results)
        expected_cagr = np.expm1(np.log(1.10) * 365.25 / 2.0)

        self.assertEqual(metrics['Time-Weighted Return'], '10.00%')
        self.assertEqual(metrics['Total Return'], metrics['Time-Weighted Return'])
        self.assertEqual(metrics['CAGR'], f'{expected_cagr:.2%}')

    def test_irregular_timestamps_use_elapsed_time_for_risk_annualization(self):
        results = pd.DataFrame(
            {
                'returns': [0.0, 0.01, 0.02],
                'total_equity': [100.0, 101.0, 103.02],
            },
            index=pd.to_datetime(['2026-01-01', '2026-02-01', '2026-02-02']),
        )
        analyzer = self._analyzer(results)
        durations, elapsed_days = analyzer._get_period_timing(results.index)
        np.testing.assert_allclose(durations, [1.0, 31.0, 1.0])
        self.assertEqual(elapsed_days, 33.0)
        log_returns = np.log1p(results['returns'].to_numpy())
        mean_log_rate = log_returns.sum() / elapsed_days
        residuals = log_returns - mean_log_rate * durations
        expected_vol = np.sqrt(np.dot(residuals, residuals) / elapsed_days * 365.25)
        expected_sharpe = mean_log_rate * 365.25 / expected_vol

        metrics = analyzer._get_risk_metrics(results)

        self.assertEqual(metrics['Annualized Vol'], f'{expected_vol:.2%}')
        self.assertEqual(metrics['Sharpe Ratio'], f'{expected_sharpe:.2f}')

    def test_one_day_and_ten_day_moves_have_different_annualized_volatility(self):
        def metrics_for_move(days):
            index = pd.to_datetime(['2026-01-01', f'2026-01-{days + 1:02d}'])
            results = pd.DataFrame(
                {
                    'returns': [0.0, 0.10],
                    'total_equity': [100.0, 110.0],
                },
                index=index,
            )
            return self._analyzer(results)._get_risk_metrics(results)

        one_day = metrics_for_move(1)
        ten_day = metrics_for_move(10)

        self.assertNotEqual(one_day['Annualized Vol'], ten_day['Annualized Vol'])
        self.assertGreater(
            float(one_day['Annualized Vol'].rstrip('%')),
            float(ten_day['Annualized Vol'].rstrip('%')),
        )

    def test_invalid_returns_and_equity_are_reported_as_unavailable(self):
        analyzer = self._analyzer(
            pd.DataFrame(
                {
                    'returns': [0.1, 0.1],
                    'total_equity': [100.0, -50.0],
                },
                index=pd.date_range('2026-01-01', periods=2),
            )
        )

        self.assertEqual(analyzer._get_return_metrics(analyzer.portfolio_results), {
            'Time-Weighted Return': 'N/A',
            'Total Return': 'N/A',
            'CAGR': 'N/A',
        })
        self.assertEqual(analyzer._get_risk_metrics(analyzer.portfolio_results), {
            'Annualized Vol': 'N/A',
            'Sharpe Ratio': 'N/A',
        })
        self.assertEqual(analyzer._get_drawdown_metrics(analyzer.portfolio_results), {
            'Max Drawdown': 'N/A',
        })

    def test_unsorted_and_duplicate_indices_make_all_time_series_metrics_unavailable(self):
        for index in (
            pd.to_datetime(['2026-01-02', '2026-01-01']),
            pd.to_datetime(['2026-01-01', '2026-01-01']),
        ):
            with self.subTest(index=index):
                results = pd.DataFrame(
                    {
                        'returns': [0.10, -0.05],
                        'total_equity': [110.0, 104.5],
                    },
                    index=index,
                )
                analyzer = self._analyzer(results)

                self.assertEqual(analyzer._get_return_metrics(results), {
                    'Time-Weighted Return': 'N/A',
                    'Total Return': 'N/A',
                    'CAGR': 'N/A',
                })
                self.assertEqual(analyzer._get_risk_metrics(results), {
                    'Annualized Vol': 'N/A',
                    'Sharpe Ratio': 'N/A',
                })
                self.assertEqual(analyzer._get_drawdown_metrics(results), {
                    'Max Drawdown': 'N/A',
                })

    def test_invalid_risk_free_rate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'risk_free_rate'):
            PerformanceAnalyzer(
                portfolio_results=pd.DataFrame(),
                backtester=object(),
                strategy=object(),
                data=pd.DataFrame(),
                risk_free_rate=-1.0,
            )

    def test_zero_volatility_does_not_hide_nonzero_excess_return(self):
        results = pd.DataFrame(
            {
                'returns': [0.01, 0.01],
                'total_equity': [100.0, 101.0],
            },
            index=pd.date_range('2026-01-01', periods=2),
        )
        analyzer = self._analyzer(results)

        metrics = analyzer._get_risk_metrics(results)

        self.assertEqual(metrics['Annualized Vol'], '0.00%')
        self.assertEqual(metrics['Sharpe Ratio'], 'N/A')

    def test_recalculation_clears_metrics_when_data_becomes_insufficient(self):
        results = pd.DataFrame(
            {
                'returns': [0.0, 0.01],
                'total_equity': [100.0, 101.0],
            },
            index=pd.date_range('2026-01-01', periods=2),
        )
        analyzer = self._analyzer(results)
        analyzer._calculate_all_metrics()
        self.assertIn('Total Return', analyzer.metrics)

        analyzer.portfolio_results = results.iloc[:1]
        analyzer._calculate_all_metrics()

        self.assertEqual(analyzer.metrics, {})

    @staticmethod
    def _analyzer(results):
        return PerformanceAnalyzer(
            portfolio_results=results,
            backtester=object(),
            strategy=object(),
            data=pd.DataFrame(),
        )


if __name__ == '__main__':
    unittest.main()
