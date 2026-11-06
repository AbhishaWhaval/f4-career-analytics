from f4ca.parse import normalise_name, parse_standings
from synthetic import standings_html

REAL_SHAPE = """
<table class="wikitable"><tbody>
<tr><th rowspan="2">Pos</th><th rowspan="2">Driver</th><th colspan="2">KAR1</th><th rowspan="2">Pts</th></tr>
<tr><th>R1</th><th>R2</th></tr>
<tr><th>1</th><td><span class="flagicon"><a href="/wiki/Kenya">KEN</a></span> <a href="/wiki/Shane_Chandaria">Shane Chandaria</a></td>
    <td style="background:#DFFFDF">2</td><td><b>1</b></td><th>186</th></tr>
<tr><th>2</th><td><span class="flagicon"></span> Sachel Rotgé</td><td><i>Ret</i></td><td>12†</td><th>169</th></tr>
<tr><th>3</th><td>IND Ishaan Madesh<sup>R</sup></td><td>WD</td><td>3</td><th>142</th></tr>
<tr><th>Pos</th><th>Driver</th><th>R1</th><th>R2</th><th>Pts</th></tr>
</tbody></table>"""


def test_real_shaped_table():
    df = parse_standings(REAL_SHAPE)
    assert list(df["pos"]) == [1, 2, 3]
    a, b, c = df.to_dict("records")
    assert a["driver_id"] == "Shane_Chandaria" and a["wins"] == 1 and a["poles"] == 1 and a["podiums"] == 2
    assert b["driver_id"] == "name:sachel rotge" and b["retirements"] == 1 and b["fastest_laps"] == 1
    assert b["finishes"] == 1 and b["entries"] == 2
    assert c["driver"] == "Ishaan Madesh" and c["entries"] == 1 and c["podiums"] == 1
    assert a["points"] == 186


def test_skips_entry_list_and_team_column():
    rows = [("A_B", "A B", [1, 2, 3, 4, 5, 6], 100), ("C_D", "C D", [2, 3, 4, 5, 6, 7], 80),
            ("E_F", "E F", [3, 4, 5, 6, 7, 8], 60)]
    df = parse_standings(standings_html(rows, team_col=True))
    assert len(df) == 3
    assert df.loc[0, "entries"] == 6 and df.loc[0, "wins"] == 1


def test_no_table():
    assert parse_standings("<p>nothing here</p>").empty


def test_normalise():
    assert normalise_name("Sachel Rotgé") == "sachel rotge"
    assert normalise_name("  O'Ward,  Pato ") == "o ward pato"
