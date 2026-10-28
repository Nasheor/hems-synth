"""The cleaning rules on hand-made series: despike, counter to cumulative, minute mean, gaps, median of five."""
import numpy as np
import pandas as pd

from hems_synth.loader import (
    ALIVE_WINDOW_MIN,
    _clean_daily_counter,
    _counter_to_cumulative,
    _dead_minutes,
    _despike,
    _held,
    _minute_mean,
    ev_sessions,
)


def _series(values, start="2026-06-01T10:00:00Z", step="5s"):
    idx = pd.date_range(start, periods=len(values), freq=step).as_unit("ns")
    return pd.Series(values, index=idx, dtype=float)


def test_despike_drops_the_documented_example_and_keeps_the_reset():
    s = _series([10.70, 10.72, 10.73, 12.09, 10.75, 10.76, 0.00, 0.01, 0.02])
    out = _despike(s)
    assert 12.09 not in out.to_numpy()
    assert list(out.to_numpy()) == [10.70, 10.72, 10.73, 10.75, 10.76, 0.00, 0.01, 0.02]


def test_despike_keeps_a_jump_after_a_silent_gap():
    idx = pd.DatetimeIndex(["2026-06-01T10:00:00Z", "2026-06-01T10:00:05Z", "2026-06-01T10:20:05Z", "2026-06-01T10:20:10Z"]).as_unit("ns")
    s = pd.Series([5.00, 5.01, 6.21, 6.22], index=idx)
    assert len(_despike(s)) == 4


def test_counter_to_cumulative_handles_reset_and_glitch():
    s = _series([1.00, 1.50, 2.00, 0.00, 0.30, 0.80, 0.20, 1.10, 1.20])
    cum = _counter_to_cumulative(s)
    # 1.00 start (0), +0.5, +0.5, reset adds 0.00, +0.3, +0.5, glitch 0.20 adds 0, 1.10 adds 0.3 against 0.80, +0.1
    assert np.allclose(cum.to_numpy(), [0.0, 0.5, 1.0, 1.0, 1.3, 1.8, 1.8, 2.1, 2.2])


def test_minute_mean_is_time_weighted_hold_last_value():
    idx = pd.DatetimeIndex(["2026-06-01T10:00:00Z", "2026-06-01T10:00:30Z", "2026-06-01T10:01:00Z"]).as_unit("ns")
    s = pd.Series([100.0, 200.0, 300.0], index=idx)
    grid = pd.date_range("2026-06-01T10:00:00Z", periods=2, freq="min").as_unit("ns")
    out = _minute_mean(s, grid)
    assert out.iloc[0] == 150.0
    assert out.iloc[1] == 300.0


def test_dead_minutes_marks_only_runs_of_fifteen_minutes_or_more():
    times = pd.DatetimeIndex(["2026-06-01T10:00:00Z", "2026-06-01T10:10:00Z", "2026-06-01T10:30:00Z", "2026-06-01T10:30:05Z"]).as_unit("ns")
    grid = pd.date_range("2026-06-01T10:00:00Z", "2026-06-01T10:30:00Z", freq="min").as_unit("ns")
    dead = _dead_minutes(times, grid)
    assert ALIVE_WINDOW_MIN == 15
    assert not dead[:11].any()
    assert dead[11:30].all()
    assert not dead[30]


def test_clean_daily_counter_removes_dips_and_two_reading_spikes():
    raw = _series([0.0, 0.0, 0.0, 16.97, 16.97, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 0.02, 2.5, 3.0, 3.5, 4.0], step="5min")
    clean = _clean_daily_counter(raw)
    assert clean.iloc[3] == 0.0 and clean.iloc[4] == 0.0
    assert clean.iloc[11] >= 1.5
    assert clean.iloc[13] == 3.0


def test_held_is_hold_last_value():
    s = _series([1.0, 2.0], step="5min")
    grid = pd.date_range(s.index[0], periods=10, freq="min").as_unit("ns")
    h = _held(s, grid)
    assert list(h.to_numpy()[:5]) == [1.0] * 5 and list(h.to_numpy()[5:]) == [2.0] * 5


def test_ev_sessions_merge_and_subtract_base_load():
    load = pd.Series(300.0, index=pd.date_range("2026-06-01T22:00:00Z", periods=120, freq="min").as_unit("ns"))
    load.iloc[10:40] = 7700.0
    load.iloc[45:60] = 7700.0
    sessions = ev_sessions(load)
    assert len(sessions) == 1
    assert sessions.iloc[0]["minutes"] == 50
    assert abs(sessions.iloc[0]["kwh"] - (45 * (7700 - 350) / 60 / 1000)) < 1e-6
