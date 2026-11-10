"""Input file expansion for the Sprint 2 pipeline.

Leaving out a part of the split national PUMS file silently drops whole
states, so the pipeline expands globs, refuses missing files, and warns about
unlisted parts.
"""

import pytest

from synthpop.sprint2 import expand_paths


@pytest.fixture
def raw(tmp_path):
    for part in "abcd":
        (tmp_path / f"psam_pus{part}.csv").write_text("x\n")
    return tmp_path


def test_glob_picks_up_every_part(raw):
    got = expand_paths([str(raw / "psam_pus*.csv")], "person_csvs")
    assert [p[-13:] for p in got] == [f"psam_pus{c}.csv" for c in "abcd"]


def test_unlisted_parts_are_warned_about(raw, capsys):
    expand_paths([str(raw / "psam_pusa.csv"), str(raw / "psam_pusb.csv")], "person_csvs")
    out = capsys.readouterr().out
    assert "psam_pusc.csv" in out and "psam_pusd.csv" in out and "WARNING" in out


def test_complete_list_gives_no_warning(raw, capsys):
    expand_paths([str(raw / f"psam_pus{c}.csv") for c in "abcd"], "person_csvs")
    assert "WARNING" not in capsys.readouterr().out


def test_missing_file_or_empty_pattern_fails(raw):
    with pytest.raises(FileNotFoundError):
        expand_paths([str(raw / "psam_puse.csv")], "person_csvs")
    with pytest.raises(FileNotFoundError):
        expand_paths([str(raw / "psam_hus*.csv")], "household_csvs")


def test_duplicates_are_dropped(raw):
    got = expand_paths([str(raw / "psam_pusa.csv"), str(raw / "psam_pus*.csv")], "person_csvs")
    assert len(got) == 4
