"""Клампинг параметров и графы DSP-цепочек."""
import dsp_chains


def test_three_chains():
    assert [c.id for c in dsp_chains.CHAINS] == ["wall", "wall-lite", "tape"]


def test_clamp_params_defaults_and_bounds():
    wall = dsp_chains.CHAINS_BY_ID["wall"]
    p = dsp_chains.clamp_params(wall, {"exciter": 99, "noise": -1})
    assert p["exciter"] == 6.0          # клэмп сверху
    assert p["noise"] == 0.0            # клэмп снизу
    assert p["wall"] == 0.5             # дефолт для отсутствующего
    # мусор игнорируется дефолтом
    p2 = dsp_chains.clamp_params(wall, {"wall": "abc"})
    assert p2["wall"] == 0.5


def test_graphs_contain_filters():
    wall = dsp_chains.CHAINS_BY_ID["wall"]
    tape = dsp_chains.CHAINS_BY_ID["tape"]
    g = wall.graph({"exciter": 2.5, "wall": 0.5, "noise": 0.09})
    assert "aexciter=amount=2.50" in g and "acrusher" in g and "anoisesrc=color=white" in g
    g2 = tape.graph({"wow": 0.1, "hiss": 0.018, "cut": 9.5})
    assert "vibrato=f=0.7:d=0.10" in g2 and "lowpass=f=10" in g2 and "color=pink" in g2


def test_wall_lite_defaults_differ():
    lite = dsp_chains.CHAINS_BY_ID["wall-lite"]
    p = dsp_chains.clamp_params(lite, None)
    assert p["exciter"] == 1.2 and p["wall"] == 0.7 and p["noise"] == 0.05
