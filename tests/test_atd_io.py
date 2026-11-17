"""ATD (FIFA/EPTS) loader: frame offset, slot teams, NaN handling, spec switching, bundle paths."""

import os

import numpy as np
import pytest

from conftest import needs_sample
from mtah import atd_io, paths, sample_io


def write_epts(folder, n_frames=200, slots_per_team=23, specs=((1, 120), (121, 200)),
               skip_ids=(), dup_ids=(), speed=True, stem="game", team_names=("Home FC", "Away FC"),
               bad_line_id=None):
    """A tiny ATD-shaped EPTS pair: 'Track k' slots with ShirtNumber k, x,y(,speed) channels.

    Slot k (1-based) sits at x = k / 100, y = frame_id / 1000 and speed = k; it is NaN on frames
    where (frame_id + k) % 7 == 0. Ball x = frame_id / 1000, y = 0.5, NaN on every 4th frame id.
    """
    S = 2 * slots_per_team
    teams = ["T1", "T2"]
    ch = ["x", "y", "speed"] if speed else ["x", "y"]
    players = "".join(
        '<Player id="P{k}" teamId="{t}"><Name>Track {k}</Name><ShirtNumber>{k}</ShirtNumber></Player>'
        .format(k=k, t=teams[0] if k <= slots_per_team else teams[1]) for k in range(1, S + 1))
    pch = "".join('<PlayerChannel channelId="{c}" id="player{k}_{c}" playerId="P{k}"/>'
                  .format(c=c, k=k) for k in range(1, S + 1) for c in ch)
    spec_xml = ""
    for a, b in specs:
        groups = "".join('<SplitRegister separator=",">' + "".join(
            '<PlayerChannelRef playerChannelId="player{k}_{c}"/>'.format(k=k, c=c) for c in ch)
            + "</SplitRegister>" for k in range(1, S + 1))
        ball = ('<SplitRegister separator=",">' + "".join(
            '<BallChannelRef channelId="{}"/>'.format(c) for c in ch) + "</SplitRegister>")
        spec_xml += ('<DataFormatSpecification endFrame="{b}" separator=":" startFrame="{a}">'
                     '<StringRegister name="frameCount"/><SplitRegister separator=";">{g}'
                     '</SplitRegister>{ball}</DataFormatSpecification>').format(
                         a=a, b=b, g=groups, ball=ball)
    xml = ('<?xml version="1.0"?><main><Metadata><GlobalConfig><FrameRate>25</FrameRate>'
           '<ProviderGlobalParameters><ProviderParameter><Name>first_half_start</Name><Value>11'
           '</Value></ProviderParameter><ProviderParameter><Name>first_half_end</Name><Value>100'
           '</Value></ProviderParameter></ProviderGlobalParameters></GlobalConfig>'
           '<Sessions><Session id="s"><FieldSize><Width>105</Width><Height>68</Height></FieldSize>'
           '</Session></Sessions><Teams><Team id="T1"><Name>{h}</Name></Team><Team id="T2"><Name>'
           '{a}</Name></Team></Teams><Players>{p}</Players><Devices><Device id="d"><Sensors>'
           '<Sensor id="position"><Channels><Channel id="x"/><Channel id="y"/><Channel id="speed"/>'
           '</Channels></Sensor></Sensors></Device></Devices><PlayerChannels>{pch}</PlayerChannels>'
           '</Metadata><DataFormatSpecifications>{s}</DataFormatSpecifications></main>').format(
               h=team_names[0], a=team_names[1], p=players, pch=pch, s=spec_xml)
    lines = []
    for fid in range(1, n_frames + 1):
        if fid in skip_ids:
            continue
        groups = []
        for k in range(1, S + 1):
            vals = [k / 100, fid / 1000, float(k)][:len(ch)]
            if (fid + k) % 7 == 0:
                groups.append(",".join(["NaN"] * len(ch)))
            else:
                groups.append(",".join("{:.5f}".format(v) for v in vals))
        bvals = [fid / 1000, 0.5, 9.0][:len(ch)]
        ball = ",".join(["NaN"] * len(ch)) if fid % 4 == 0 else ",".join(
            "{:.5f}".format(v) for v in bvals)
        line = "{}:{}:{}".format(fid, ";".join(groups), ball)
        if fid == bad_line_id:
            line = line.rsplit(",", 1)[0]          # one field short
        lines.append(line)
        if fid in dup_ids:
            lines.append("{}:{}:{}".format(fid, ";".join([",".join(["0.9"] * len(ch))] * S), ball))
    os.makedirs(folder, exist_ok=True)
    meta, raw = os.path.join(folder, stem + "_atd.xml"), os.path.join(folder, stem + "_atd.txt")
    with open(meta, "w") as fh:
        fh.write(xml)
    with open(raw, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return meta, raw


def test_frame_offset_is_frame_id_minus_one(tmp_path):
    meta, raw = write_epts(str(tmp_path))
    t = atd_io.load_epts(meta, raw)
    assert t.n_frames == 200 and t.frame[0] == 0 and t.frame[-1] == 199
    # ball x = frame_id / 1000, so row r holds frame_id r + 1
    vis = t.ball_visible()
    rows = np.flatnonzero(vis)
    assert np.allclose(t.ball[rows, 0], (rows + 1) / 1000.0, atol=1e-6)
    # frame_id 4 (row 3) has no ball; row 4 (frame_id 5) does
    assert not vis[3] and vis[4]
    # a row-shifted reader would fail exactly this
    assert not np.allclose(t.ball[rows, 0], rows / 1000.0, atol=1e-6)


def test_slots_teams_nans_speed_and_halves(tmp_path):
    meta, raw = write_epts(str(tmp_path))
    t = atd_io.load_epts(meta, raw)
    assert t.n_slots == 46 and t.player_id is None
    assert (t.slot_team[:23] == 0).all() and (t.slot_team[23:] == 1).all()
    assert t.meta["shirt_is_slot_index"] is True
    assert t.meta["slot_names"][0] == "Track 1"
    # slot k (0-based j = k - 1) is NaN where (frame_id + k) % 7 == 0
    fid = np.arange(1, 201)
    for j in (0, 5, 45):
        expect_nan = (fid + j + 1) % 7 == 0
        assert np.array_equal(np.isnan(t.xy[:, j, 0]), expect_nan)
        assert np.allclose(t.xy[~expect_nan, j, 0], (j + 1) / 100.0)
    assert np.allclose(np.nanmax(t.meta["slot_speed"], axis=0), np.arange(1, 47))
    # first_half_start 11 .. end 100 (frame ids) -> rows 10 .. 99
    assert (t.period[10:100] == 1).all() and t.period[9] == 0 and t.period[100] == 0
    assert t.pitch == (105.0, 68.0)


def test_spec_switching_missing_and_duplicate_records(tmp_path):
    meta, raw = write_epts(str(tmp_path), specs=((1, 120), (121, 200)), skip_ids=(50,),
                           dup_ids=(60,))
    t = atd_io.load_epts(meta, raw)
    assert t.meta["n_specs"] == 2
    assert t.meta["missing_records"] == 1 and t.meta["duplicate_records"] == 1
    assert np.isnan(t.xy[49]).all() and np.isnan(t.ball[49]).all()    # frame_id 50 absent
    assert np.isclose(t.xy[59, 0, 0], 0.01)                            # first record of id 60 kept
    assert np.isclose(t.xy[130, 0, 0], 0.01)                           # second spec parsed
    assert t.meta["contiguous_from_1"] is False


def test_field_count_mismatch_raises(tmp_path):
    meta, raw = write_epts(str(tmp_path), bad_line_id=30)
    with pytest.raises(Exception):
        atd_io.load_epts(meta, raw)


def test_competition_paths_follow_mtah_data_dir(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    write_epts(str(root / "tactical"), stem="BAR-ATH_tactical",
               team_names=("FC Barcelona", "Athletic Club"))
    monkeypatch.setenv("MTAH_DATA_DIR", str(root))
    meta, raw = atd_io.competition_paths()
    assert meta == str(root / "tactical" / "BAR-ATH_tactical_atd.xml")
    t = atd_io.load_competition(strict=False)
    assert t.n_frames == 199319 and t.pitch == (105.3, 68.0)
    assert t.meta["missing_records"] == 199319 - 200
    # games.py half blocks, not the metadata's
    assert t.period[11185] == 1 and t.period[11184] == 0 and t.period[110808] == 2
    # the organiser's alignment check refuses a 200-record file
    with pytest.raises(ValueError, match="does not match DATA.md"):
        atd_io.load_competition(strict=True)


def test_competition_refuses_barcelona_as_away(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    write_epts(str(root / "tactical"), stem="BAR-ATH_tactical",
               team_names=("Athletic Club", "FC Barcelona"))
    monkeypatch.setenv("MTAH_DATA_DIR", str(root))
    with pytest.raises(ValueError, match="Barcelona away"):
        atd_io.load_competition(strict=True)


def test_ball_gap_lengths(tmp_path):
    meta, raw = write_epts(str(tmp_path))
    t = atd_io.load_epts(meta, raw)
    gaps = atd_io.ball_gap_lengths(t)
    assert len(gaps) and (gaps == 1).all()        # every 4th frame, singletons


@needs_sample
def test_sample_game_3_matches_kloppy():
    d = paths.sample_game_dir(3)
    t = atd_io.load_epts(os.path.join(d, "Sample_Game_3_metadata.xml"),
                         os.path.join(d, "Sample_Game_3_tracking.txt"), with_player_id=True)
    c = sample_io.clean_tracking(3)                # kloppy's reading, frame_id - 1
    assert np.array_equal(t.frame, c.frame) and np.array_equal(t.period, c.period)
    assert np.array_equal(np.isnan(t.ball), np.isnan(c.ball))
    assert np.nanmax(np.abs(t.ball - c.ball)) == 0.0
    idx = {p: k for k, p in enumerate(t.player_id)}
    for j, pid in enumerate(c.player_id):
        k = idx[pid[2:]]
        assert t.slot_team[k] == c.slot_team[j]
        assert np.array_equal(np.isnan(t.xy[:, k]), np.isnan(c.xy[:, j]))
        assert np.nanmax(np.abs(t.xy[:, k] - c.xy[:, j]), initial=0.0) == 0.0
    assert t.meta["n_specs"] > 1 and t.meta["contiguous_from_1"]
