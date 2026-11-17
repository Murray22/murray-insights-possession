"""Phase D touch-kink features: a deflection is a kink, a ball rolling past a player is not."""

import numpy as np

from mtah import features_d as D
from mtah.frames import FrameTable
from mtah.models import learned

L, W = 105.0, 68.0


def table(ball_m, players_m, teams):
    n = len(ball_m)
    ball = (np.asarray(ball_m) / [L, W]).astype(np.float32)
    xy = np.repeat((np.asarray(players_m, dtype=np.float64) / [L, W])[None], n, axis=0).astype(np.float32)
    return FrameTable(np.arange(n), np.ones(n, dtype=np.int8), ball, xy, np.array(teams, np.int8),
                      (L, W), None, {})


def test_kink_sees_a_deflection_not_a_roll_past():
    n = 100
    x = 20.0 + 0.4 * np.arange(n)                     # 10 m/s along x
    straight = np.c_[x, np.full(n, 30.0)]
    bounce = straight.copy()
    bounce[50:, 0] = x[50] - 0.4 * np.arange(n - 50)   # reversed at frame 50 (x = 40 m)
    players = [[40.0, 30.5], [80.0, 10.0]]            # home player where the ball turns, away far
    f_s = D.compute_extra(table(straight, players, [0, 1]))
    f_b = D.compute_extra(table(bounce, players, [0, 1]))
    assert np.nanmax(f_s["d_kink8"]) < 0.5
    assert f_b["d_kink8"][50] > 15.0                  # 10 m/s to -10 m/s
    # both pass within 1.5 m of the home player: a touch either way, strong only for the bounce
    assert f_s["d_ttk"][50] == 0.0 and f_b["d_ttk"][50] > 0
    assert f_b["d_st_last"][70] == 1.0 and f_s["d_st_last"][70] == 0.0
    assert 0 < f_b["d_st_since_h"][60] <= 10 and f_b["d_st_since_a"][60] == D.F.CAP_GAP


def test_team_swap_flips_signs_and_swaps_team_columns():
    n = 100
    x = 20.0 + 0.4 * np.arange(n)
    bounce = np.c_[x, np.full(n, 30.0)]
    bounce[50:, 0] = x[50] - 0.4 * np.arange(n - 50)
    t = table(bounce, [[40.0, 30.5], [80.0, 10.0]], [0, 1])
    f, g = D.compute_extra(t), D.compute_extra(learned.swapped(t))
    assert np.allclose(f["d_ttk"], -g["d_ttk"]) and np.allclose(f["d_st_last"], -g["d_st_last"])
    assert np.allclose(f["d_st_since_h"], g["d_st_since_a"])
    assert np.allclose(np.nan_to_num(f["d_kink4"]), np.nan_to_num(g["d_kink4"]))


def test_missing_ball_gives_nan_kink_and_no_touch():
    n = 60
    b = np.c_[20.0 + 0.4 * np.arange(n), np.full(n, 30.0)]
    b[25:35] = np.nan
    f = D.compute_extra(table(b, [[30.0, 30.0], [80.0, 10.0]], [0, 1]))
    assert np.isnan(f["d_kink8"][30]) and f["d_ttk"][30] == 0.0
