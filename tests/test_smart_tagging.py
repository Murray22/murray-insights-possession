"""Smart Tagging loader: time base, team mapping (including N/A), set-piece kind."""

from mtah import smart_tagging

XML = """<?xml version="1.0"?><file><ALL_INSTANCES>
<instance><ID>1</ID><start>447.4</start><end>452.0</end><code>SET PIECES</code>
 <label><group>Team</group><text>Barcelona</text></label>
 <label><group>Type</group><text>Kick Off (Start)</text></label></instance>
<instance><ID>2</ID><start>500</start><end>510.5</end><code>BUILD UP</code>
 <label><group>Team</group><text>Atletic Club</text></label></instance>
<instance><ID>3</ID><start>600</start><end>603</end><code>Set Pieces</code>
 <label><group>Team</group><text>N/A</text></label>
 <label><group>Type</group><text>Throw-in</text></label></instance>
</ALL_INSTANCES></file>"""


def test_load(tmp_path):
    p = tmp_path / "st.xml"
    p.write_text(XML)
    iv = smart_tagging.load(str(p), "FC Barcelona", "Athletic Club")
    assert [i["start"] for i in iv] == [11185, 12500, 15000]
    assert iv[0]["end"] == 11300
    assert [i["team"] for i in iv] == ["H", "A", None]
    assert [i["kind"] for i in iv] == ["set_piece", "phase", "set_piece"]
    assert iv[2]["type"] == "Throw-in"


def test_unknown_team_is_none_not_a_guess():
    m = smart_tagging.team_mapper("FC Barcelona", "Athletic Club")
    assert m("Barcelona") == "H" and m("Athletic Club") == "A"
    assert m("Real Madrid") is None and m("N/A") is None
