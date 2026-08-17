"""M0: constants, time conversion, zone semantics, frozen config."""

import pytest
from dataclasses import FrozenInstanceError

from Q3.domain import (
    SAFETY_MARGIN_KM,
    Config,
    NoFlyZone,
    SolutionMetrics,
    hhmm_to_s,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("08:00", 28800),
        ("09:00", 32400),
        ("10:30", 37800),
        ("13:30", 48600),
        ("17:00", 61200),
        ("23:59", 86340),
    ],
)
def test_hhmm_to_s_is_absolute_seconds(text, expected):
    assert hhmm_to_s(text) == expected
    from Q3.domain import START_CLOCK_S

    assert hhmm_to_s(text) - START_CLOCK_S >= 0


def test_hhmm_to_s_rejects_malformed():
    with pytest.raises(ValueError):
        hhmm_to_s("8x0")
    with pytest.raises(ValueError):
        hhmm_to_s("eight:00")


def test_zone_active_uses_closed_intervals():
    zone = NoFlyZone("Z1", 1.0, 1.0, 0.2, 100, 200)
    assert not zone.active(99)
    assert zone.active(100)
    assert zone.active(150)
    assert zone.active(200)
    assert not zone.active(201)


def test_zone_safe_radius_adds_eta():
    zone = NoFlyZone("Z1", 1.0, 1.0, 3.2, 0, 100)
    assert zone.safe_radius() == pytest.approx(3.2 + SAFETY_MARGIN_KM)


def test_config_defaults_match_plan():
    config = Config()
    assert config.fleet_size == {"Case1": 4, "Case2": 2, "Case3": 5, "Case4": 4}
    assert config.eta_km == 0.01
    assert config.seed == 42
    assert config.time_budget_s_per_case == 1800
    assert config.stall_rounds == 3
    assert config.nine_hour_cap_s is None
    assert config.lex_first_is_N is False
    assert config.allow_empty_routes is False


def test_config_is_frozen():
    config = Config()
    with pytest.raises(FrozenInstanceError):
        config.seed = 1  # type: ignore[misc]


def test_zone_is_frozen():
    zone = NoFlyZone("Z1", 1.0, 1.0, 0.2, 0, 100)
    with pytest.raises(FrozenInstanceError):
        zone.radius_km = 0.5  # type: ignore[misc]


def test_metrics_lex_key_rounds_distance_to_1e6():
    a = SolutionMetrics(1000, 500, 500, 3000, 0, 123.4567891)
    b = SolutionMetrics(1000, 500, 500, 3000, 0, 123.4567894)
    c = SolutionMetrics(1001, 500, 501, 3000, 0, 1.0)
    assert a.lex_key() < c.lex_key()
    # 1e-6 rounding makes near-equal distances tie
    assert a.lex_key()[:-1] == b.lex_key()[:-1]
    assert a.lex_key()[4] == b.lex_key()[4]


def test_metrics_lex_order_primary_is_smax():
    a = SolutionMetrics(1000, 500, 500, 3000, 0, 1.0)
    b = SolutionMetrics(999, 500, 499, 3000, 0, 500.0)
    assert b.lex_key() < a.lex_key()


def test_metrics_lex_secondary_is_delta():
    a = SolutionMetrics(1000, 500, 500, 3000, 0, 1.0)
    b = SolutionMetrics(1000, 700, 300, 3000, 0, 1.0)
    assert b.lex_key() < a.lex_key()
