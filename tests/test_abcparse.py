"""Таймлайн ABC (abcparse.py) на фикстуре диалекта YuE."""
import abcparse

ABC = """X:1
T:Test
M:4/4
L:1/8
Q:1/4=120
K:C
V: Vocal
% intro
C2 D2 E2 G2 | A4 G2 | z4 z4 |
% verse
V: Ins
"Am"c2 e2 g2 e2 | "G"d4 B4 | z8 |"""


def test_parse_abc_basic():
    tl = abcparse.parse_abc(ABC)
    assert tl["tempo_bpm"] == 120.0
    assert tl["key"] == "C"
    assert tl["meter"] == "4/4"
    assert tl["unit"] == 0.125
    assert tl["voice_order"] == ["Vocal", "Ins"]
    assert len(tl["bars"]) == 6


def test_parse_abc_sections_and_seconds():
    tl = abcparse.parse_abc(ABC)
    # L:1/8, 120 BPM: единица = 0.5 четверти = 0.25 с
    # такты: 8+6+8 единиц (Vocal) и 8+8+8 (Ins) = 46 единиц = 11.5 с
    assert tl["duration_sec"] == 11.5
    secs = [b["section"] for b in tl["bars"]]
    assert secs == ["intro"] * 3 + ["verse"] * 3
    assert tl["bars"][0]["voices"] == {"Vocal": 4}
    assert tl["bars"][3]["chords"] == ["Am"]
    assert tl["bars"][2]["rests"] == {"Vocal": 8}


def test_parse_abc_empty():
    tl = abcparse.parse_abc("")
    assert tl["bars"] == []
    assert tl["duration_sec"] == 0.0
