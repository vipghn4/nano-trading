import unittest

import numpy as np

from price_validation import is_valid_price, valid_price_mask, validate_active_prices


class TestPriceValidation(unittest.TestCase):
    def test_valid_price_mask_accepts_only_finite_positive_values(self):
        values = np.array([100.0, np.nan, np.inf, 0.0, -1.0])

        np.testing.assert_array_equal(
            valid_price_mask(values),
            np.array([True, False, False, False, False]),
        )

    def test_is_valid_price_matches_the_shared_rule(self):
        self.assertTrue(is_valid_price(100.0))
        self.assertFalse(is_valid_price(np.nan))
        self.assertFalse(is_valid_price(0.0))

    def test_validate_active_prices_ignores_zero_holdings(self):
        validate_active_prices(
            np.array([100.0, np.nan]),
            np.array([1.0, 0.0]),
        )

    def test_validate_active_prices_rejects_invalid_active_prices(self):
        with self.assertRaisesRegex(ValueError, "Active holdings"):
            validate_active_prices(
                np.array([np.nan, 100.0]),
                np.array([1.0, 0.0]),
            )


if __name__ == "__main__":
    unittest.main()
