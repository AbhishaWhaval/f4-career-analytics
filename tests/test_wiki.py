import pandas as pd

from f4ca.build import article_title
from f4ca.features import age_at_debut
from f4ca.wiki import parse_birth_date, title_matches


def test_birth_date_formats():
    assert parse_birth_date("| birth_date = {{birth date and age|2006|8|25|df=yes}}") == "2006-08-25"
    assert parse_birth_date("| birth_date = {{Birth date and age|df=yes|2009|1|4}}") == "2009-01-04"
    assert parse_birth_date("| birth_date = {{birth year and age|2007}}") == "2007"
    assert parse_birth_date("| birth_date = 7 December 1996 (age 29)") == "1996-12-07"
    assert parse_birth_date("| birth_date = March 3, 2005") == "2005-03-03"
    assert parse_birth_date("no infobox here") == ""


def test_age_at_debut():
    ages = age_at_debut(pd.Series(["2006-07-01", "2006", "", "1950-01-01"]), pd.Series([2022] * 4))
    assert abs(ages[0] - 16.0) < 0.01
    assert abs(ages[1] - 16.0) < 0.01          # year only -> assume mid-year
    assert pd.isna(ages[2]) and pd.isna(ages[3])  # missing, and implausible (72 years old)


def test_title_matching():
    assert title_matches("2017\u201318 Formula 4 UAE Championship", 2018, ["UAE"])
    assert title_matches("2019 F4 South East Asia Championship", 2019, ["South East Asia"])
    assert not title_matches("2019 Formula 4 UAE Championship", 2018, ["UAE"])   # wrong year
    assert not title_matches("2019 Italian F4 Championship", 2019, ["UAE"])     # wrong series
    assert not title_matches("Formula 4 UAE Championship", 2019, ["UAE"])       # series article


def test_article_title():
    assert article_title("Sebastian_%C3%98gaard") == "Sebastian \u00d8gaard"
