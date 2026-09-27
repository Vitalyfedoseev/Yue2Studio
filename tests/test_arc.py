"""Юнит-тесты драматургии (arc.py) — порт проверок поведения yue-studio."""
import arc

ABC = """X:1
T:t
M:4/4
L:1/8
Q:1/4=120
K:C
V: Vocal
% intro
C2 D2 E2 G2 | A4 G2 |
% chorus
V: Vocal
E2 G2 A2 B2 | c4 B2 |"""


def test_style_with_arc_appends_once():
    s = arc.style_with_arc("rock, 120 BPM", "build")
    assert "dynamic arc" in s
    assert arc.style_with_arc(s, "build") == s  # без дублей


def test_apply_arc_build_tempos():
    out = arc.apply_arc(ABC, "build")
    # intro 0.85 → 102, финальная chorus 1.12 → 134
    assert "Q:1/4=102" in out
    assert "Q:1/4=134" in out
    # build не транспонирует
    assert "C2 D2 E2 G2" in out


def test_apply_arc_burst_octave():
    out = arc.apply_arc(ABC, "burst")
    # финальная chorus на октаву выше у Vocal: E→e, B→b (заглавные ушли)
    lines = out.splitlines()
    assert any(ln.strip().startswith("e2") for ln in lines)
    # intro не тронут
    assert "C2 D2 E2 G2" in out


def test_apply_arc_unknown_and_no_bpm():
    assert arc.apply_arc(ABC, "") == ABC
    no_q = ABC.replace("Q:1/4=120", "Q:1/4=0")
    assert arc.apply_arc(no_q, "build") == no_q
