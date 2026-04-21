import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ev import (
    american_to_decimal,
    expected_value,
    implied_probability,
    market_overround,
    no_vig_two_way_probabilities,
    probability_to_american,
)


class EvMathTests(unittest.TestCase):
    def test_american_to_decimal_positive(self) -> None:
        self.assertEqual(american_to_decimal(150), 2.5)

    def test_american_to_decimal_negative(self) -> None:
        self.assertEqual(round(american_to_decimal(-150), 4), 1.6667)

    def test_implied_probability_negative(self) -> None:
        self.assertEqual(round(implied_probability(-120), 4), 0.5455)

    def test_positive_ev_case(self) -> None:
        self.assertGreater(round(expected_value(0.60, -120), 4), 0)

    def test_no_vig_two_way_probabilities_remove_margin(self) -> None:
        fair_a, fair_b = no_vig_two_way_probabilities(-120, 102)
        self.assertAlmostEqual(fair_a + fair_b, 1.0, places=6)
        self.assertLess(fair_a, implied_probability(-120))
        self.assertLess(fair_b, implied_probability(102))

    def test_market_overround_positive_for_standard_two_way_market(self) -> None:
        self.assertGreater(market_overround(-120, 102), 0.0)

    def test_probability_to_american_round_trip(self) -> None:
        fair_a, _fair_b = no_vig_two_way_probabilities(-120, 102)
        self.assertAlmostEqual(implied_probability(probability_to_american(fair_a)), fair_a, places=2)


if __name__ == "__main__":
    unittest.main()
