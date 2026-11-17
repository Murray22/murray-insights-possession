"""Readers for Metrica Sports' public sample games 1-3 (github.com/metrica-sports/sample-data).

Games 1 and 2 are Metrica's CSV format: one tracking CSV per team plus a RawEventsData CSV.
Game 3 is FIFA EPTS (metadata XML + tracking TXT) with a JSON event file. Both are normalised here
into the same two things:

- an event DataFrame (normalised_events) with one row per event and the columns
  team ('H' / 'A' / None), type, subtypes (frozenset of upper-case tokens), period,
  start, end (0-based frame index = tracking frame - 1), player, start_x, start_y, end_x, end_y;
- a clean FrameTable (frames.FrameTable) with one slot per real player.

Source acknowledgement: Metrica Sports sample data, https://github.com/metrica-sports/sample-data.
"""

import json
import os
import uuid

import numpy as np
import pandas as pd

from . import paths
from .frames import AWAY, HOME, FrameTable

CACHE_DIR = os.path.join(paths.RUNS_DIR, "cache")


# -- events --------------------------------------------------------------------------------------

def _tokens_csv(subtype):
    if subtype is None or (isinstance(subtype, float) and np.isnan(subtype)) or subtype == "":
        return frozenset()
    return frozenset(t.strip().upper() for t in str(subtype).split("-") if t.strip())


def _tokens_json(subtypes):
    if not subtypes:
        return frozenset()
    if isinstance(subtypes, dict):
        subtypes = [subtypes]
    return frozenset(s["name"].strip().upper() for s in subtypes)


def _csv_events(game):
    path = os.path.join(paths.sample_game_dir(game), "Sample_Game_{}_RawEventsData.csv".format(game))
    raw = pd.read_csv(path)
    team = raw["Team"].map({"Home": "H", "Away": "A"})
    out = pd.DataFrame({
        "team": team.where(team.notna(), None),
        "type": raw["Type"].str.strip().str.upper(),
        "subtypes": [_tokens_csv(s) for s in raw["Subtype"]],
        "period": raw["Period"].astype(int),
        "start": raw["Start Frame"].astype(int) - 1,
        "end": raw["End Frame"].astype(int) - 1,
        "player": raw["From"],
        "start_x": raw["Start X"], "start_y": raw["Start Y"],
        "end_x": raw["End X"], "end_y": raw["End Y"],
    })
    # Set pieces and cards carry End Frame 0 in the CSV; an end before the start means "instant".
    out.loc[out["end"] < out["start"], "end"] = out["start"]
    return out


def _json_events(game):
    path = os.path.join(paths.sample_game_dir(game), "Sample_Game_{}_events.json".format(game))
    with open(path) as handle:
        data = json.load(handle)["data"]
    home_id = _epts_home_team_id(game)
    rows = []
    for e in data:
        team_id = (e.get("team") or {}).get("id")
        start, end = e["start"], e["end"]
        rows.append({
            "team": None if team_id is None else ("H" if team_id == home_id else "A"),
            "type": e["type"]["name"].strip().upper(),
            "subtypes": _tokens_json(e.get("subtypes")),
            "period": int(e["period"]),
            "start": int(start["frame"]) - 1,
            "end": int(end["frame"] if end["frame"] is not None else start["frame"]) - 1,
            "player": (e.get("from") or {}).get("name"),
            "start_x": start.get("x"), "start_y": start.get("y"),
            "end_x": end.get("x"), "end_y": end.get("y"),
        })
    out = pd.DataFrame(rows)
    out.loc[out["end"] < out["start"], "end"] = out["start"]
    return out


def normalised_events(game):
    """Events of a sample game in the common schema, in file order (which is Metrica's order)."""
    events = _csv_events(game) if game in (1, 2) else _json_events(game)
    events = events.reset_index(drop=True)
    events["order"] = np.arange(len(events))
    return events


def _epts_home_team_id(game):
    import xml.etree.ElementTree as ET
    meta = os.path.join(paths.sample_game_dir(game), "Sample_Game_{}_metadata.xml".format(game))
    root = ET.parse(meta).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    return teams[0]  # FIFA/EPTS convention: home first (kloppy agrees: Team A is home)


# -- tracking ------------------------------------------------------------------------------------

def _csv_tracking(game):
    def read(side):
        path = os.path.join(paths.sample_game_dir(game),
                            "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(game, side))
        raw = pd.read_csv(path, skiprows=2, header=0)
        cols = list(raw.columns)
        players, xy = [], []
        i = 3
        while i < len(cols) - 1:
            name = cols[i]
            if name == "Ball":
                break
            players.append(name.replace(" ", ""))
            xy.append(raw.iloc[:, [i, i + 1]].to_numpy(dtype=np.float32))
            i += 2
        ball_col = cols.index("Ball")
        ball = raw.iloc[:, [ball_col, ball_col + 1]].to_numpy(dtype=np.float32)
        return raw["Frame"].to_numpy(), raw["Period"].to_numpy(), players, np.stack(xy, 1), ball

    f_h, p_h, pl_h, xy_h, ball = read("Home")
    f_a, _, pl_a, xy_a, _ = read("Away")
    if not np.array_equal(f_h, f_a):
        raise ValueError("game {}: home and away tracking frames differ".format(game))
    xy = np.concatenate([xy_h, xy_a], axis=1)
    team = np.array([HOME] * len(pl_h) + [AWAY] * len(pl_a), dtype=np.int8)
    ids = np.array(["H_" + p for p in pl_h] + ["A_" + p for p in pl_a])
    return FrameTable(f_h.astype(np.int64) - 1, p_h.astype(np.int8), ball, xy, team,
                      paths.SAMPLE_PITCH, ids, {"game": game, "kind": "clean"})


def _epts_tracking(game):
    from kloppy import metrica
    d = paths.sample_game_dir(game)
    ds = metrica.load_tracking_epts(
        meta_data=os.path.join(d, "Sample_Game_{}_metadata.xml".format(game)),
        raw_data=os.path.join(d, "Sample_Game_{}_tracking.txt".format(game)))
    df = ds.to_df()
    home, away = ds.metadata.teams[0], ds.metadata.teams[1]
    ids, team, xy = [], [], []
    for t_code, team_obj in ((HOME, home), (AWAY, away)):
        for p in team_obj.players:
            cx, cy = "{}_x".format(p.player_id), "{}_y".format(p.player_id)
            if cx not in df.columns:
                continue
            ids.append(("H_" if t_code == HOME else "A_") + str(p.player_id))
            team.append(t_code)
            xy.append(df[[cx, cy]].to_numpy(dtype=np.float32))
    ball = df[["ball_x", "ball_y"]].to_numpy(dtype=np.float32)
    return FrameTable(df["frame_id"].to_numpy(dtype=np.int64) - 1,
                      df["period_id"].to_numpy(dtype=np.int8), ball, np.stack(xy, 1),
                      np.array(team, dtype=np.int8), paths.SAMPLE_PITCH, np.array(ids),
                      {"game": game, "kind": "clean"})


def clean_tracking(game, use_cache=True):
    """Clean (Metrica-corrected) tracking for a sample game as a FrameTable, one slot per player.

    Cached as .npz under runs/cache (gitignored), because the CSV / EPTS parse takes 5-20 s.
    """
    cache = os.path.join(CACHE_DIR, "sample_game_{}_clean.npz".format(game))
    if use_cache and os.path.isfile(cache):
        table = FrameTable.load_npz(cache)
        table.meta.update({"game": game, "kind": "clean"})
        return table
    table = _csv_tracking(game) if game in (1, 2) else _epts_tracking(game)
    if not np.array_equal(table.frame, np.arange(table.n_frames)):
        raise ValueError("game {}: tracking frames are not contiguous from 1".format(game))
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = "{}.{}.{}.tmp.npz".format(cache[:-4], os.getpid(), uuid.uuid4().hex)   # per writer: training workers
        table.save_npz(tmp)   # on a fresh clone must never see a half-written cache
        os.replace(tmp, cache)
    return table
