"""Builds a fake but Wikipedia-shaped dataset so the full pipeline can run offline in tests."""
import numpy as np

from f4ca.wiki import cache_path

def _letters(n):
    """Digit-free fake names, because name matching strips digits like a real surname would."""
    s = ""
    while n:
        n, r = divmod(n, 26)
        s = chr(97 + r) + s
    return s.capitalize()


FLAG = '<span class="flagicon"><span><img alt="" src="x.png"></span></span>'


def standings_html(rows, n_races=6, team_col=False):
    head = "<tr><th>Pos</th><th>Driver</th>" + ("<th>Team</th>" if team_col else "")
    head += "".join(f"<th>R{i+1}</th>" for i in range(n_races)) + "<th>Pts</th></tr>"
    body = ""
    for pos, (did, name, results, pts) in enumerate(rows, 1):
        cells = "".join("<td><b>1</b></td>" if r == 1 else f"<td>{r}</td>" for r in results)
        team = "<td>Some Team</td>" if team_col else ""
        link = f'<a href="/wiki/{did}">{name}</a>' if did else name
        body += f"<tr><th>{pos}</th><td>{FLAG} {link}<sup>R</sup></td>{team}{cells}<th>{pts}</th></tr>"
    entry = '<table class="wikitable"><tr><th>Team</th><th>Driver</th></tr><tr><td>A</td><td>B</td></tr></table>'
    return f'<div>{entry}<h2>Drivers</h2><table class="wikitable">{head}{body}</table></div>'


def make_world(cfg, seed=0, drivers_per_season=16):
    rng = np.random.default_rng(seed)
    series = list(cfg.f4_series)
    seasons = sorted({y for s in cfg.f4_series.values() for y in s})
    ladder = {t: {} for t in cfg.tiers}
    uid = 0
    for y in seasons:
        rookies = []
        for _ in range(drivers_per_season):
            uid += 1
            rookies.append({"id": f"Driver_{uid}", "name": f"Driver {_letters(uid)}", "skill": rng.normal(),
                            "money": rng.random()})
        for s_i, s in enumerate(series):
            field = list(rookies[s_i::len(series)])
            # well-funded drivers also enter a second series -> links series for strength estimation
            field += [d for d in rookies if d["money"] > 0.8 and d not in field]
            field.sort(key=lambda d: -(d["skill"] + rng.normal(0, 0.6) - 0.3 * s_i))
            rows = []
            for pos, d in enumerate(field, 1):
                res = [max(1, pos + int(rng.integers(-2, 3))) if rng.random() > 0.1 else "Ret" for _ in range(6)]
                rows.append((d["id"], d["name"], res, max(0, 120 - 8 * pos)))
            title = cfg.f4_series[s].get(y)
            if title:
                cache_path(cfg.raw_dir, title).write_text(standings_html(rows, team_col=(s_i == 1)), encoding="utf-8")
        for d in rookies:
            prob = 1 / (1 + np.exp(-(1.6 * d["skill"] + 1.5 * d["money"] - 2.5)))
            if rng.random() < prob:
                yr = y + int(rng.integers(1, 4))
                ladder["tier3"].setdefault(yr, []).append(d)
                if rng.random() < 0.4:
                    ladder["f2"].setdefault(yr + 2, []).append(d)
    for tier, series_map in cfg.tiers.items():
        for s, titles in series_map.items():
            for y, title in titles.items():
                ds = ladder.get(tier, {}).get(y, [])
                if len(ds) < 3:
                    ds = ds + [{"id": f"Pro_{tier}_{y}_{k}", "name": f"Pro {tier} {_letters(y * 10 + k)}"} for k in range(3)]
                rows = [(d["id"], d["name"], [i + 1] * 3, 50 - i) for i, d in enumerate(ds)]
                cache_path(cfg.raw_dir, title).write_text(standings_html(rows, 3), encoding="utf-8")
