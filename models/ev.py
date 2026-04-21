from __future__ import annotations


def american_to_decimal(american_odds: int) -> float:
    if american_odds > 0:
        return 1 + (american_odds / 100)
    return 1 + (100 / abs(american_odds))


def implied_probability(american_odds: int) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def market_overround(*american_odds: int) -> float:
    if not american_odds:
        raise ValueError("Pass at least one price to compute market overround")
    return sum(implied_probability(odds) for odds in american_odds) - 1


def no_vig_probabilities(*american_odds: int) -> list[float]:
    if not american_odds:
        raise ValueError("Pass at least one price to compute no-vig probabilities")
    implied_probs = [implied_probability(odds) for odds in american_odds]
    total_implied = sum(implied_probs)
    if total_implied <= 0:
        raise ValueError("Total implied probability must be positive")
    return [probability / total_implied for probability in implied_probs]


def no_vig_two_way_probabilities(american_a: int, american_b: int) -> tuple[float, float]:
    fair_probs = no_vig_probabilities(american_a, american_b)
    return fair_probs[0], fair_probs[1]


def probability_to_american(probability: float) -> int:
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be between 0 and 1")
    if probability >= 0.5:
        return int(round(-(probability / (1 - probability)) * 100))
    return int(round(((1 - probability) / probability) * 100))


def expected_value(projected_win_prob: float, american_odds: int) -> float:
    decimal_odds = american_to_decimal(american_odds)
    profit_multiple = decimal_odds - 1
    loss_multiple = 1
    return (projected_win_prob * profit_multiple) - ((1 - projected_win_prob) * loss_multiple)
