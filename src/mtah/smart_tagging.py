"""Smart Tagging (SportsCode XML) to the optional `intervals` input of the models.

DATA.md: 638 automatic instances, each with <start>/<end> in seconds from the first frame of the
tactical video (video frame = round(seconds * 25)), a <code> and <label> tags (group/text). The
Team tag is `Barcelona`, `Atletic Club` (sic) or `N/A` (every throw-in). Codes are upper case,
tag values title case; everything here matches case-insensitively.

Output: [{"start", "end", "code", "kind", "team", "type"}], frames half-open [start, end),
kind = "set_piece" for the SET PIECES code, "phase" otherwise; team = "H" / "A" / None.

Team mapping: a tag value is home if it shares a word with the home club name and not with the
away one (the ATD metadata's first team is home; DATA.md says home is Barcelona). An unmatched
value maps to None rather than to a guess. Smart Tagging is automatic, not ground truth, and the
models only ever add it as weak evidence.
"""

import re
import xml.etree.ElementTree as ET

FPS = 25
SET_PIECE_CODES = {"set pieces"}


def _words(s):
    return {w for w in re.split(r"[^a-z]+", (s or "").lower()) if len(w) > 2} - {"club", "the"}


def team_mapper(home_name, away_name):
    hw, aw = _words(home_name), _words(away_name)

    def to_side(value):
        v = _words(value)
        if not v:
            return None
        h, a = bool(v & hw), bool(v & aw)
        if h and not a:
            return "H"
        if a and not h:
            return "A"
        return None
    return to_side


def load(path, home_name="Barcelona", away_name="Athletic Club", fps=FPS):
    mapper = team_mapper(home_name, away_name)
    # the tags spell the away club "Atletic Club" (DATA.md), which shares no word with "Athletic"
    alias = team_mapper(home_name, away_name + " Atletic")
    out = []
    for inst in ET.parse(path).iter("instance"):
        code = (inst.findtext("code") or "").strip()
        tags = {}
        for lab in inst.findall("label"):
            g, t = lab.findtext("group"), lab.findtext("text")
            if g is not None:
                tags[g.strip().lower()] = (t or "").strip()
        team_val = tags.get("team")
        side = mapper(team_val) or alias(team_val)
        start = int(round(float(inst.findtext("start")) * fps))
        end = int(round(float(inst.findtext("end")) * fps))
        out.append({"start": start, "end": max(end, start + 1), "code": code,
                    "kind": "set_piece" if code.lower() in SET_PIECE_CODES else "phase",
                    "team": side, "type": tags.get("type")})
    return out
