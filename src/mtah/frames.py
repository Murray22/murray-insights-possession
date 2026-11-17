"""The one in-memory frame table every model reads, for the sample games and for the real ATD alike.

A FrameTable is a handful of aligned numpy arrays, one row per tracking frame:

    frame      (n,)       int64    0-based frame index. For the ATD this is frame_id - 1, which is
                                   the video frame and so the submission row (DATA.md).
    period     (n,)       int8     1 or 2 (0 where unknown).
    ball       (n, 2)     float32  ball x, y in normalised 0-1 pitch coordinates; NaN when missing.
    xy         (n, S, 2)  float32  per-slot x, y, normalised 0-1; NaN when the slot is empty.
    slot_team  (S,)       int8     0 = home, 1 = away. Fixed per slot, as in the ATD metadata,
                                   where every one of the 46 track slots belongs to one team.
    pitch      (length, width) in metres, to turn normalised distances into metric ones.

Coordinates follow the ATD / Metrica convention: (0, 0) top-left, (1, 1) bottom-right, x along
the length of the pitch. Nothing is flipped per half.

`vball` (n, 2) float32, optional: a second, separate ball channel from our own video detection
(Phase E), normalised like `ball`, NaN where there is none. It is only ever set where the ATD
ball is missing, and it never changes `ball`: features that read `ball` keep their ATD meaning.
On degraded bench tables it is the degrader's simulated video track.

`player_id` is set only on clean sample tracking, where a slot is a person. On degraded tracking
and on the real ATD a slot is an anonymous tracklet and player_id is None; no model may read it.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

HOME, AWAY = 0, 1


@dataclass
class FrameTable:
    frame: np.ndarray
    period: np.ndarray
    ball: np.ndarray
    xy: np.ndarray
    slot_team: np.ndarray
    pitch: tuple = (105.0, 68.0)
    player_id: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)
    vball: Optional[np.ndarray] = None

    def __post_init__(self):
        n = len(self.frame)
        if self.ball.shape != (n, 2):
            raise ValueError("ball shape {} != ({}, 2)".format(self.ball.shape, n))
        if self.xy.ndim != 3 or self.xy.shape[0] != n or self.xy.shape[2] != 2:
            raise ValueError("xy shape {} is not (n, S, 2)".format(self.xy.shape))
        if self.slot_team.shape != (self.xy.shape[1],):
            raise ValueError("slot_team shape {} != (S,)".format(self.slot_team.shape))
        if len(self.period) != n:
            raise ValueError("period length mismatch")
        if self.vball is not None and self.vball.shape != (n, 2):
            raise ValueError("vball shape {} != ({}, 2)".format(self.vball.shape, n))

    @property
    def n_frames(self):
        return len(self.frame)

    @property
    def n_slots(self):
        return self.xy.shape[1]

    def ball_visible(self):
        return ~np.isnan(self.ball[:, 0])

    def present(self):
        """(n, S) bool: slot holds a position on this frame."""
        return ~np.isnan(self.xy[:, :, 0])

    def copy(self):
        return FrameTable(self.frame.copy(), self.period.copy(), self.ball.copy(),
                          self.xy.copy(), self.slot_team.copy(), tuple(self.pitch),
                          None if self.player_id is None else self.player_id.copy(),
                          dict(self.meta), None if self.vball is None else self.vball.copy())

    def save_npz(self, path):
        np.savez_compressed(
            path, frame=self.frame, period=self.period, ball=self.ball, xy=self.xy,
            slot_team=self.slot_team, pitch=np.array(self.pitch, dtype=np.float64),
            player_id=(np.array([], dtype=object) if self.player_id is None
                       else np.asarray(self.player_id, dtype=str)),
            has_player_id=np.array(self.player_id is not None),
            **({} if self.vball is None else {"vball": self.vball}))

    @classmethod
    def load_npz(cls, path):
        z = np.load(path, allow_pickle=False)
        pid = z["player_id"] if bool(z["has_player_id"]) else None
        return cls(z["frame"], z["period"], z["ball"], z["xy"], z["slot_team"],
                   tuple(float(v) for v in z["pitch"]), pid, {},
                   z["vball"] if "vball" in z.files else None)
