"""Read a FIFA/EPTS tracking pair (Metrica's ATD export: metadata XML + raw TXT) into a FrameTable.

The competition tracking arrives as `BAR-ATH_tactical_atd.xml` + `BAR-ATH_tactical_atd.txt`
(ref/DATA.md), the same EPTS family as Metrica's public sample game 3. kloppy reads the pair
(`kloppy.metrica.load_tracking_epts`), but builds a Python dict per frame and takes minutes and
gigabytes on 199,319 frames x 46 slots (ref/starter/load_atd.py says so). This reader follows the
same metadata, the DataFormatSpecification registers, and parses the text in bulk with pandas.
tests/test_atd_io.py checks it against kloppy on sample game 3, value for value.

What it produces (frames.FrameTable):

- frame = frame_id - 1. DATA.md: the ATD is one record per video frame numbered from 1, and a
  submission row is the video frame. Records missing from the file become all-NaN rows and are
  counted in meta["missing_records"]; duplicated frame ids keep the first record.
- ball: x, y normalised 0-1, NaN when missing.
- xy: one slot per <Player> in the metadata, in file order. On the ATD a <Player> is a track slot
  ("Track 1".."Track 46", ShirtNumber = slot index); it is not a person, so player_id stays None.
  The slot names and shirt numbers go to meta for inspection only. A slot that no
  DataFormatSpecification references stays all NaN (sample game 3's substitutes in the first half).
- slot_team: 0 for the first <Team> in the metadata (FIFA/EPTS convention: home first; the
  organiser baseline and kloppy both rely on it), 1 for the second.
- period: from the metadata's first/second half frame-id parameters when present, or from an
  explicit `halves` argument (load_competition passes ref/games.py's published half blocks).
  0 elsewhere.
- pitch: the metadata FieldSize, unless `pitch` is given (load_competition passes games.py's
  105.3 x 68.0 m, measured by the organiser).
- meta["slot_speed"], meta["ball_speed"]: the speed channel (m/s) when the file has one (DATA.md
  lists `1_s`); absent otherwise. Sample game 3 has no speed channel.

Defensive checks, each raising rather than guessing: exactly two teams; every slot's team is one of
them; every channel reference resolves; the number of fields on each line equals the registers of
its DataFormatSpecification; the frame register is the first field. load_competition() also runs
the organiser's alignment test (199,319 records, contiguous, from 1) and refuses a mismatch, as
ref/starter/load_atd.py asks ("do not invent a correction").
"""

import importlib.util
import io
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import paths
from .frames import AWAY, HOME, FrameTable

POSITION_CHANNELS = {"x": 0, "y": 1}
SPEED_CHANNELS = {"s", "speed", "v", "velocity"}


@dataclass
class Spec:
    start: int                     # first frame id (inclusive)
    end: int                       # last frame id (inclusive)
    separators: list               # every separator character used by the spec
    leaves: list                   # ordered fields: ("frame",), ("player", pid, ch), ("ball", ch), ("other", name)


@dataclass
class EptsMeta:
    teams: list                    # [(team_id, name)] in file order; [0] is home
    players: list                  # [(player_id, team_id, name, shirt)] in file order = slot order
    channels: dict                 # player channel ref id -> (player_id, channel_id)
    specs: list                    # [Spec]
    frame_rate: float = 25.0
    field_size: tuple = (105.0, 68.0)
    params: dict = field(default_factory=dict)   # ProviderGlobalParameters, name -> value text


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _iter(root, name):
    return (e for e in root.iter() if _local(e.tag) == name)


def _flatten(elm, leaves, seps):
    for child in elm:
        tag = _local(child.tag)
        if tag == "SplitRegister":
            seps.append(child.get("separator"))
            _flatten(child, leaves, seps)
        elif tag == "StringRegister":
            name = child.get("name") or ""
            leaves.append(("frame",) if name == "frameCount" else ("other", name))
        elif tag == "PlayerChannelRef":
            leaves.append(("pref", child.get("playerChannelId")))
        elif tag == "BallChannelRef":
            leaves.append(("ball", child.get("channelId")))
        else:
            raise ValueError("unknown register <{}> in DataFormatSpecification".format(tag))


def read_metadata(meta_path):
    """Parse the EPTS metadata XML (namespace-agnostic)."""
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), (t.findtext("Name") or "").strip()) for t in _iter(root, "Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    team_ids = {t for t, _ in teams}
    players = []
    for p in _iter(root, "Player"):
        if p.get("teamId") not in team_ids:
            raise ValueError("{}: player {} has unknown team {}".format(
                meta_path, p.get("id"), p.get("teamId")))
        name = next((c.text for c in p if _local(c.tag) == "Name"), None)
        shirt = next((c.text for c in p if _local(c.tag) == "ShirtNumber"), None)
        players.append((p.get("id"), p.get("teamId"), name, shirt))
    if not players:
        raise ValueError("{}: no <Player> entries".format(meta_path))
    if len({p[0] for p in players}) != len(players):
        raise ValueError("{}: duplicate player ids".format(meta_path))
    channels = {c.get("id"): (c.get("playerId"), c.get("channelId"))
                for c in _iter(root, "PlayerChannel")}
    specs = []
    for s in _iter(root, "DataFormatSpecification"):
        leaves, seps = [], [s.get("separator")]
        _flatten(s, leaves, seps)
        resolved = []
        for leaf in leaves:
            if leaf[0] == "pref":
                if leaf[1] not in channels:
                    raise ValueError("{}: unresolved PlayerChannelRef {}".format(meta_path, leaf[1]))
                pid, ch = channels[leaf[1]]
                resolved.append(("player", pid, ch))
            else:
                resolved.append(leaf)
        if not resolved or resolved[0] != ("frame",):
            raise ValueError("{}: the frame register is not the first field".format(meta_path))
        specs.append(Spec(int(s.get("startFrame")), int(s.get("endFrame")),
                          sorted({c for c in seps if c}), resolved))
    if not specs:
        raise ValueError("{}: no DataFormatSpecification".format(meta_path))
    params = {}
    for prm in _iter(root, "ProviderParameter"):
        name = next((c.text for c in prm if _local(c.tag) == "Name"), None)
        value = next((c.text for c in prm if _local(c.tag) == "Value"), None)
        if name is not None and name not in params:
            params[name] = value
    fr = next((e.text for e in _iter(root, "FrameRate")), None)
    fs = next(_iter(root, "FieldSize"), None)
    field_size = (105.0, 68.0)
    if fs is not None:
        w = next((c.text for c in fs if _local(c.tag) == "Width"), None)
        h = next((c.text for c in fs if _local(c.tag) == "Height"), None)
        if w and h:
            field_size = (float(w), float(h))
    return EptsMeta(teams, players, channels, specs, float(fr) if fr else 25.0, field_size, params)


def _halves_from_params(params):
    """[(period, first_frame_id, last_frame_id)] from Metrica's provider parameters, if usable."""
    out = []
    for k, name in ((1, "first_half"), (2, "second_half"), (3, "first_extra_half"),
                    (4, "second_extra_half")):
        a, b = params.get(name + "_start"), params.get(name + "_end")
        try:
            out.append((k, int(a), int(b)))
        except (TypeError, ValueError):
            continue
    return out


def _parse_block(lines, spec):
    """Bulk-parse the lines of one spec into a float64 array (n_lines, n_leaves)."""
    blob = b"\n".join(lines)
    seps = "".join(spec.separators).encode()
    blob = blob.translate(bytes.maketrans(seps, b"," * len(seps)))
    # pandas pads a short line with NaN without a word, so count the fields on every line first
    arr = np.frombuffer(blob, dtype=np.uint8)
    commas = np.concatenate(([0], np.cumsum(arr == ord(","))))
    ends = np.concatenate((np.flatnonzero(arr == ord("\n")), [len(arr)]))
    starts = np.concatenate(([0], ends[:-1] + 1))
    per_line = commas[ends] - commas[starts] + 1
    if (per_line != len(spec.leaves)).any():
        bad = int(np.flatnonzero(per_line != len(spec.leaves))[0])
        raise ValueError("spec {}-{}: line {} of the block has {} fields, metadata describes {}"
                         .format(spec.start, spec.end, bad, int(per_line[bad]), len(spec.leaves)))
    df = pd.read_csv(io.BytesIO(blob), header=None, sep=",", engine="c", dtype=np.float64,
                     na_values=["NaN", "nan", "NAN", "None", "null"], keep_default_na=True)
    if df.shape[1] != len(spec.leaves):
        raise ValueError("spec {}-{}: {} fields per line, metadata describes {}".format(
            spec.start, spec.end, df.shape[1], len(spec.leaves)))
    return df.to_numpy()


def load_epts(meta_path, raw_path, pitch=None, halves=None, n_frames=None, with_player_id=False):
    """An EPTS pair as a FrameTable. See the module docstring for every convention.

    pitch:     (length, width) in metres; default the metadata FieldSize.
    halves:    [(period, first_frame, last_frame)] in 0-based frames (= video frames), inclusive;
               default the metadata's half parameters (frame ids, shifted by one).
    n_frames:  table length; default the largest frame id in the file.
    """
    meta = read_metadata(meta_path)
    with open(raw_path, "rb") as handle:
        lines = [ln.strip() for ln in handle.read().split(b"\n")]
    lines = [ln for ln in lines if ln]
    first_sep = meta.specs[0].separators  # frame id ends at the first separator of any kind
    fids = np.empty(len(lines), dtype=np.int64)
    for i, ln in enumerate(lines):
        cut = min((ln.find(s.encode()) for s in first_sep if ln.find(s.encode()) >= 0),
                  default=len(ln))
        fids[i] = int(ln[:cut])

    n = int(n_frames) if n_frames is not None else int(fids.max())
    slot_of = {p[0]: k for k, p in enumerate(meta.players)}
    S = len(meta.players)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    xy = np.full((n, S, 2), np.nan, dtype=np.float32)
    slot_speed = ball_speed = None
    seen = np.zeros(n, dtype=bool)
    n_dup = n_out = 0
    assigned = np.zeros(len(lines), dtype=bool)
    for spec in meta.specs:
        sel = np.flatnonzero((fids >= spec.start) & (fids <= spec.end) & ~assigned)
        if not len(sel):
            continue
        assigned[sel] = True
        vals = _parse_block([lines[i] for i in sel], spec)
        rows = fids[sel] - 1
        ok = (rows >= 0) & (rows < n)
        n_out += int((~ok).sum())
        # first occurrence wins on duplicated frame ids
        _, first_idx = np.unique(rows, return_index=True)
        keep = np.zeros(len(rows), dtype=bool)
        keep[first_idx] = True
        n_dup += int((~keep).sum())
        keep &= ok & ~seen[np.clip(rows, 0, n - 1)]
        r, v = rows[keep], vals[keep]
        seen[r] = True
        for col, leaf in enumerate(spec.leaves):
            kind = leaf[0]
            if kind == "player":
                pid, ch = leaf[1], (leaf[2] or "").lower()
                if pid not in slot_of:
                    raise ValueError("channel for unknown player {}".format(pid))
                k = slot_of[pid]
                if ch in POSITION_CHANNELS:
                    xy[r, k, POSITION_CHANNELS[ch]] = v[:, col]
                elif ch in SPEED_CHANNELS:
                    if slot_speed is None:
                        slot_speed = np.full((n, S), np.nan, dtype=np.float32)
                    slot_speed[r, k] = v[:, col]
            elif kind == "ball":
                ch = (leaf[1] or "").lower()
                if ch in POSITION_CHANNELS:
                    ball[r, POSITION_CHANNELS[ch]] = v[:, col]
                elif ch in SPEED_CHANNELS:
                    if ball_speed is None:
                        ball_speed = np.full(n, np.nan, dtype=np.float32)
                    ball_speed[r] = v[:, col]
    n_unspecified = int((~assigned).sum())

    # a slot or ball half-present (x without y) is treated as missing
    bad = np.isnan(xy).any(axis=2)
    xy[bad] = np.nan
    ball[np.isnan(ball).any(axis=1)] = np.nan

    period = np.zeros(n, dtype=np.int8)
    if halves is None:
        halves_src = "metadata"
        halves = [(k, a - 1, b - 1) for k, a, b in _halves_from_params(meta.params)]
    else:
        halves_src = "argument"
    for k, a, b in halves:
        period[max(0, a):min(n, b + 1)] = k

    home_id = meta.teams[0][0]
    slot_team = np.array([HOME if p[1] == home_id else AWAY for p in meta.players], dtype=np.int8)
    shirts = [p[3] for p in meta.players]
    shirt_is_index = all(str(s).strip() == str(k + 1) for k, s in enumerate(shirts))
    info = {
        "kind": "atd", "source_meta": os.path.abspath(meta_path),
        "source_raw": os.path.abspath(raw_path),
        "team_names": [t[1] for t in meta.teams], "home_team_id": home_id,
        "slot_xml_ids": [p[0] for p in meta.players], "slot_names": [p[2] for p in meta.players],
        "slot_shirts": shirts, "shirt_is_slot_index": shirt_is_index,
        "n_records": len(lines), "frame_id_range": (int(fids.min()), int(fids.max())),
        "missing_records": int(n - seen.sum()), "duplicate_records": n_dup,
        "records_outside_table": n_out, "records_without_spec": n_unspecified,
        "contiguous_from_1": bool(fids.min() == 1 and len(np.unique(fids)) == fids.max()
                                  and len(fids) == fids.max()),
        "halves": [tuple(int(x) for x in h) for h in halves], "halves_source": halves_src,
        "metadata_field_size": meta.field_size, "frame_rate": meta.frame_rate,
        "n_specs": len(meta.specs),
    }
    if slot_speed is not None:
        info["slot_speed"] = slot_speed
    if ball_speed is not None:
        info["ball_speed"] = ball_speed
    pid = np.array([p[0] for p in meta.players]) if with_player_id else None
    return FrameTable(np.arange(n, dtype=np.int64), period, ball, xy, slot_team,
                      tuple(float(v) for v in (pitch or meta.field_size)), pid, info)


# -- the competition bundle ---------------------------------------------------------------------

def _games_module():
    """The organiser's ref/games.py (stdlib only), loaded by path so nothing is copied."""
    path = os.path.join(paths.REF_DIR, "games.py")
    spec = importlib.util.spec_from_file_location("mtah_ref_games", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def data_dir(explicit=None):
    """Bundle root: the argument, else $MTAH_DATA_DIR, else <project>/data."""
    return explicit or os.environ.get("MTAH_DATA_DIR") or os.path.join(paths.ROOT, "data")


def competition_paths(source="tactical", root=None):
    """(atd_meta, atd_raw) for the competition bundle, via the organiser's games.asset_paths()."""
    games = _games_module()
    p = games.asset_paths(games.GAME, data_dir(root), source)
    return p["atd_meta"], p["atd_raw"]


def load_competition(source="tactical", root=None, strict=True):
    """The competition ATD as a FrameTable, on the submission's frame index.

    Uses games.py for the file names, the frame count, the pitch (105.3 x 68.0 m) and the half
    blocks (published boundaries, not labels). With strict=True a file that is not one record per
    video frame numbered from 1, or whose first team is not the home side named in DATA.md,
    raises instead of being corrected.
    """
    games = _games_module()
    meta_path, raw_path = competition_paths(source, root)
    for pth in (meta_path, raw_path):
        if not os.path.isfile(pth):
            raise IOError("match data not found: {} (set MTAH_DATA_DIR)".format(pth))
    n = games.source_frames(source, games.GAME)
    halves = games.half_video_blocks(games.GAME) if source == "tactical" else None
    table = load_epts(meta_path, raw_path, pitch=games.GAMES[games.GAME]["pitch"],
                      halves=halves, n_frames=n)
    m = table.meta
    if strict:
        problems = []
        if m["n_records"] != n or not m["contiguous_from_1"]:
            problems.append("records {} for {} video frames, ids {}..{}, contiguous from 1: {}"
                            .format(m["n_records"], n, *m["frame_id_range"], m["contiguous_from_1"]))
        if m["records_without_spec"]:
            problems.append("{} records outside every DataFormatSpecification"
                            .format(m["records_without_spec"]))
        names = [s.lower() for s in m["team_names"]]
        if "barcelona" in names[1] or ("athletic" in names[0] or "atletic" in names[0]):
            problems.append("team order {} puts Barcelona away".format(m["team_names"]))
        if problems:
            raise ValueError("ATD does not match DATA.md: " + "; ".join(problems))
    return table


def summary(table):
    """Plain-number profile of an ATD table (counts and distributions only, no labels)."""
    m = table.meta
    vis = table.ball_visible()
    graded = table.period > 0
    present = table.present()
    out = {
        "n_frames": table.n_frames, "n_slots": table.n_slots,
        "slots_per_team": [int((table.slot_team == t).sum()) for t in (HOME, AWAY)],
        "team_names": m.get("team_names"), "shirt_is_slot_index": m.get("shirt_is_slot_index"),
        "ball_visible_all": float(vis.mean()),
        "ball_visible_graded": float(vis[graded].mean()) if graded.any() else None,
        "objects_per_graded_frame": float(present[graded].sum(1).mean()) if graded.any() else None,
        "frames_over_22": float((present[graded].sum(1) > 22).mean()) if graded.any() else None,
        "missing_records": m.get("missing_records"), "has_speed": "slot_speed" in m,
    }
    # ball-missing burst lengths inside graded time: the distribution the degrader takes
    runs = []
    for k in np.unique(table.period[graded]):
        idx = np.flatnonzero(table.period == k)
        miss = ~vis[idx]
        e = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
        runs.extend((np.flatnonzero(e == -1) - np.flatnonzero(e == 1)).tolist())
    if runs:
        r = np.array(runs)
        out["ball_gap_runs"] = int(len(r))
        out["ball_gap_median"] = float(np.median(r))
        out["ball_gap_p90"] = float(np.percentile(r, 90))
    return out


def ball_gap_lengths(table):
    """Lengths of every ball-missing run inside graded frames, for DegradeConfig.ball_burst_lengths."""
    vis = table.ball_visible()
    out = []
    for k in np.unique(table.period[table.period > 0]):
        miss = ~vis[table.period == k]
        e = np.diff(np.concatenate(([0], miss.astype(np.int8), [0])))
        out.extend((np.flatnonzero(e == -1) - np.flatnonzero(e == 1)).tolist())
    return np.array(out, dtype=np.int64)


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) == 3:
        t = load_epts(sys.argv[1], sys.argv[2])
    else:
        t = load_competition()
    print(json.dumps(summary(t), indent=1, default=str))
