"""Parse a drivers' championship table from a rendered Wikipedia season page.

Wikipedia standings tables look like:

    Pos | Driver | R1 | R2 | ... | Pts
    1   | [flag] Name | 1 | Ret | ... | 186

Race cells hold a finishing position ("1", "12†"), a status ("Ret", "DNS", "WD") or
nothing. Bold = pole position, italics = fastest lap. Superscripts are footnotes or
bonus-point markers and are ignored.
"""
import re
import unicodedata
from typing import List, Optional

import pandas as pd
from bs4 import BeautifulSoup, Tag

NOT_ENTERED = {"", "WD", "DNA", "DNP", "C", "EX", "DNPQ", "\u2014", "-", "\u2013"}
RETIRED = {"RET", "NC", "DSQ", "DNF"}
META_COLUMNS = {"team", "entrant", "car", "chassis", "no.", "no", "nat.", "class"}
POS_RE = re.compile(r"^\s*(\d+)\s*$")
RESULT_RE = re.compile(r"^(\d+)")


def normalise_name(name: str) -> str:
    """'Andrea Kimi Antonelli' -> 'andrea kimi antonelli'; strips accents and punctuation."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z ]+", " ", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def _clean_text(cell: Tag) -> str:
    cell = _copy(cell)
    for junk in cell.select("sup, .flagicon, .reference, style, .sortkey"):
        junk.decompose()
    text = cell.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _copy(tag: Tag) -> Tag:
    return BeautifulSoup(str(tag), "lxml").find(tag.name)


def _driver(cell: Tag):
    """Return (driver_id, display_name). driver_id = linked article, else 'name:<normalised>'."""
    link = None
    for a in cell.find_all("a", href=True):
        href = a["href"]
        if a.find_parent(class_="flagicon") or not href.startswith("/wiki/") or ":" in href:
            continue
        link = a
        break
    name = link.get_text(" ", strip=True) if link else _clean_text(cell)
    # flag templates sometimes leave a 3-letter country code in front of the name
    name = re.sub(r"^[A-Z]{3}\s+", "", name)
    name = re.sub(r"[\u2020\u2021*]+", "", name).strip()
    if link:
        driver_id = link["href"].split("#")[0][len("/wiki/"):]
    else:
        driver_id = "name:" + normalise_name(name)
    return driver_id, name


def _direct_rows(table: Tag) -> List[Tag]:
    return [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]


def _header_text(table: Tag) -> str:
    rows = _direct_rows(table)[:3]
    return " ".join(th.get_text(" ", strip=True) for r in rows for th in r.find_all("th"))


def _meta_columns_after_driver(table: Tag) -> int:
    """How many non-race columns (Team, Car...) sit between Driver and the first race."""
    first = _direct_rows(table)[0]
    heads = [c.get_text(" ", strip=True).lower() for c in first.find_all(["th", "td"])]
    try:
        i = next(k for k, h in enumerate(heads) if h.startswith("driver"))
    except StopIteration:
        return 0
    n = 0
    for h in heads[i + 1:]:
        if h.rstrip(".") in META_COLUMNS or h in META_COLUMNS:
            n += 1
        else:
            break
    return n


def find_standings_table(soup: BeautifulSoup) -> Optional[Tag]:
    for table in soup.find_all("table", class_="wikitable"):
        head = _header_text(table)
        if re.search(r"\bDriver", head) and re.search(r"\b(Pts|Points)\b", head):
            if len(parse_table(table)) >= 3:
                return table
    return None


def parse_table(table: Tag) -> List[dict]:
    skip = _meta_columns_after_driver(table)
    rows = []
    for tr in _direct_rows(table):
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) < 3:
            continue
        m = POS_RE.match(_clean_text(cells[0]))
        if not m:
            continue
        driver_id, name = _driver(cells[1])
        if not name:
            continue
        race_cells = cells[2 + skip:-1]
        pts_text = _clean_text(cells[-1]).replace(",", ".")
        try:
            points = float(re.match(r"^[\d.]+", pts_text).group(0))
        except (AttributeError, ValueError):
            points = float("nan")

        entries = finishes = wins = podiums = poles = fls = retirements = 0
        for c in race_cells:
            txt = _clean_text(c).upper().replace("\u2020", "").strip()
            if txt in NOT_ENTERED or txt == "DNS":
                continue
            entries += 1
            r = RESULT_RE.match(txt)
            if r:
                finishes += 1
                pos = int(r.group(1))
                wins += pos == 1
                podiums += pos <= 3
            elif txt in RETIRED:
                retirements += 1
            if c.find(["b", "strong"]):
                poles += 1
            if c.find(["i", "em"]):
                fls += 1
        rows.append(dict(pos=int(m.group(1)), driver_id=driver_id, driver=name, points=points,
                         entries=entries, finishes=finishes, wins=wins, podiums=podiums,
                         poles=poles, fastest_laps=fls, retirements=retirements,
                         race_columns=len(race_cells)))
    return rows


def parse_standings(html: str) -> pd.DataFrame:
    """Drivers' championship of a season page as a DataFrame (empty if not found)."""
    soup = BeautifulSoup(html, "lxml")
    table = find_standings_table(soup)
    if table is None:
        return pd.DataFrame()
    df = pd.DataFrame(parse_table(table))
    # Footer rows sometimes repeat a driver; keep the first (classified) row per driver
    return df.drop_duplicates("driver_id").reset_index(drop=True)
