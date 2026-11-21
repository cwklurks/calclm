"""lab.stats: one-sided Welch t-test (no scipy), checked against Student t table values."""
import math

import pytest

from lab import stats


@pytest.mark.parametrize("t, df, p", [
    (0.0, 4, 0.5), (2.1318, 4, 0.05), (3.7469, 4, 0.01), (1.8331, 9, 0.05), (2.8214, 9, 0.01),
    (1.6449, 1e6, 0.05), (6.3138, 1, 0.05), (1.4759, 5, 0.10),
])
def test_student_t_upper_tail_matches_the_table(t, df, p):
    assert stats.t_sf(t, df) == pytest.approx(p, abs=2e-5)
    assert stats.t_sf(-t, df) == pytest.approx(1 - p, abs=2e-5)


def test_t_upper_tail_at_fractional_degrees_of_freedom_is_monotone():
    ps = [stats.t_sf(2.0, df) for df in (3.0, 3.5, 4.0, 6.7, 30.0)]
    assert all(a > b for a, b in zip(ps, ps[1:]))


def test_welch_statistic_and_degrees_of_freedom():
    a, b = [1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 2.5, 2.0, 3.0, 2.5]
    r = stats.welch_greater(a, b)
    va, vb = 2.5, 0.175
    se = math.sqrt(va / 5 + vb / 5)
    assert r["t"] == pytest.approx((3.0 - 2.4) / se)
    assert r["df"] == pytest.approx((va / 5 + vb / 5) ** 2 / ((va / 5) ** 2 / 4 + (vb / 5) ** 2 / 4))
    assert r["p"] == pytest.approx(stats.t_sf(r["t"], r["df"]))
    assert r["diff"] == pytest.approx(0.6)


def test_welch_direction_and_degenerate_spreads():
    assert stats.welch_greater([5, 6, 7], [1, 2, 3])["p"] < 0.01
    assert stats.welch_greater([1, 2, 3], [5, 6, 7])["p"] > 0.99
    assert stats.welch_greater([2, 2, 2], [1, 1, 1])["p"] == 0.0     # no spread, a > b
    assert stats.welch_greater([1, 1, 1], [1, 1, 1])["p"] == 0.5


def test_paired_test_is_a_one_sample_t_on_the_differences():
    a, b = [3.0, 5.0, 4.0, 7.0, 6.0], [1.0, 2.0, 2.0, 3.0, 3.0]
    d = [x - y for x, y in zip(a, b)]                     # 2, 3, 2, 4, 3
    se = math.sqrt(0.7 / 5)
    r = stats.paired_greater(a, b)
    assert r["t"] == pytest.approx(2.8 / se)
    assert r["df"] == 4
    assert r["p"] == pytest.approx(stats.t_sf(2.8 / se, 4))
    assert r["diff"] == pytest.approx(sum(d) / 5)


def test_paired_margin_tests_a_shifted_hypothesis():
    a, b = [1.0, 0.5, 2.0, 1.5], [2.0, 2.0, 2.5, 2.0]
    shifted = stats.paired_greater([x + 2 for x in a], b)
    r = stats.paired_greater(a, b, margin=-2.0)           # H1: mean(a - b) > -2
    assert (r["t"], r["p"]) == pytest.approx((shifted["t"], shifted["p"]))
    assert r["diff"] == pytest.approx(-0.875)             # the raw difference, not shifted


def test_paired_test_degenerate_spread_and_length_check():
    assert stats.paired_greater([2, 3, 4], [1, 2, 3])["p"] == 0.0
    assert stats.paired_greater([1, 2, 3], [1, 2, 3])["p"] == 0.5
    assert stats.paired_greater([1, 2, 3], [2, 3, 4])["p"] == 1.0
    with pytest.raises(ValueError):
        stats.paired_greater([1, 2], [1, 2, 3])
