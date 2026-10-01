"""DSP-цепочки ffmpeg — порт internal/dsp/dsp.go из yue-studio + новые.

«Стена» (wall): highpass → эксайтер → биткрашер → лимитер-стена + подмес
белого шума 0.5–9 кГц; «Кассета» (tape): детюн-вибрато (wow) → срез верха
+ розовый шип. «Перегруз» (overdrive): драйв → биткрашер → тон → лимитер;
«Мастеринг» (master): HP → тепло/яркость EQ → компрессор → лимитер → ширина.
Параметры клампятся в [min, max], отсутствующие берутся из дефолтов.
Превью — фрагмент с 20-й секунды длиной 15 с.
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path

PREVIEW_START = 20.0
PREVIEW_DUR = 15.0


@dataclass
class Param:
    id: str
    label: str
    min: float
    max: float
    step: float
    default: float


@dataclass
class Chain:
    id: str
    name: str
    note: str
    params: list
    graph: callable  # dict[str, float] -> str (filter_complex)


def _wall_graph(p):
    # makeup у acompressor линейный 1–64, поэтому «стену» делает лимитер
    return (
        "[0:a]highpass=f=55,"
        f"aexciter=amount={p['exciter']:.2f}:drive=9:freq=2200:ceil=12000,"
        "acrusher=bits=12:mix=0.25,"
        f"alimiter=limit={p['wall']:.2f}:attack=1:release=10:level=disabled[a];"
        f"anoisesrc=color=white:amplitude={p['noise']:.3f}:seed=42,"
        "highpass=f=500,lowpass=f=9000[n];"
        "[a][n]amix=inputs=2:duration=first:normalize=0,"
        "alimiter=limit=0.97:attack=1:release=20:level=disabled[out]"
    )


def _tape_graph(p):
    return (
        f"[0:a]vibrato=f=0.7:d={p['wow']:.2f},highpass=f=60,"
        f"lowpass=f={p['cut']:.0f},"
        "alimiter=limit=0.9:attack=5:release=50:level=disabled[a];"
        f"anoisesrc=color=pink:amplitude={p['hiss']:.3f}:seed=7,"
        "highpass=f=40,lowpass=f=8500[n];"
        "[a][n]amix=inputs=2:duration=first:normalize=0,"
        "alimiter=limit=0.95:attack=5:release=50:level=disabled[out]"
    )


def _overdrive_graph(p):
    # отдельного фильтра overdrive в сборке нет: гнать уровень в acrusher
    return (
        "[0:a]highpass=f=40,"
        f"volume={p['drive']:.1f}dB,"
        f"acrusher=bits={p['crunch']:.0f}:mix=0.45,"
        f"treble=g={p['tone']:.1f}:f=3000,"
        f"volume={p['out']:.1f}dB,"
        "alimiter=limit=0.95:attack=2:release=40:level=disabled[out]"
    )


def _master_graph(p):
    # makeup у acompressor линейный 1–64: 1.3 ≈ +2 дБ
    return (
        "[0:a]highpass=f=28,"
        f"bass=g={p['warm']:.1f}:f=110,"
        f"treble=g={p['bright']:.1f}:f=4500,"
        f"acompressor=threshold=-18dB:ratio={p['density']:.1f}:attack=12:release=140:makeup=1.3,"
        f"alimiter=limit={p['ceil']:.2f}:attack=3:release=60:level=disabled,"
        f"extrastereo=m={p['width']:.2f}[out]"
    )


_WALL_PARAMS = [
    Param("exciter", "эксайтер", 0.0, 6.0, 0.1, 2.5),
    Param("wall", "стена", 0.15, 0.9, 0.05, 0.5),
    Param("noise", "шум", 0.0, 0.3, 0.005, 0.09),
]

CHAINS = [
    Chain("wall", "Стена", "шум/песок/стена громкости", _WALL_PARAMS, _wall_graph),
    Chain("wall-lite", "Стена (лайт)", "то же, мягче", [
        Param("exciter", "эксайтер", 0.0, 6.0, 0.1, 1.2),
        Param("wall", "стена", 0.15, 0.9, 0.05, 0.7),
        Param("noise", "шум", 0.0, 0.3, 0.005, 0.05),
    ], _wall_graph),
    Chain("tape", "Кассета", "wow-детюн, срез верха, шип", [
        Param("wow", "wow", 0.0, 0.3, 0.01, 0.1),
        Param("hiss", "шип", 0.0, 0.1, 0.002, 0.018),
        Param("cut", "срез, кГц", 5.0, 16.0, 0.5, 9.5),
    ], _tape_graph),
    Chain("overdrive", "Перегруз", "драйв, хруст, тон — сатурация микса", [
        Param("drive", "драйв, дБ", 0.0, 24.0, 0.5, 9.0),
        Param("crunch", "хруст, бит", 4.0, 16.0, 1.0, 11.0),
        Param("tone", "тон, дБ", -12.0, 12.0, 0.5, 2.0),
        Param("out", "выход, дБ", -12.0, 6.0, 0.5, -1.0),
    ], _overdrive_graph),
    Chain("master", "Мастеринг", "компрессия + EQ + лимитер + ширина", [
        Param("density", "плотность", 1.5, 8.0, 0.1, 3.0),
        Param("warm", "тепло, дБ", -6.0, 6.0, 0.5, 1.5),
        Param("bright", "яркость, дБ", -6.0, 6.0, 0.5, 1.5),
        Param("width", "ширина", 0.0, 2.5, 0.05, 1.1),
        Param("ceil", "потолок", 0.5, 1.0, 0.01, 0.95),
    ], _master_graph),
]

CHAINS_BY_ID = {c.id: c for c in CHAINS}


def clamp_params(chain: Chain, params: dict | None) -> dict:
    """Дефолты для отсутствующих, клэмп в [min, max]."""
    out = {}
    for p in chain.params:
        try:
            v = float((params or {}).get(p.id, p.default))
        except (TypeError, ValueError):
            v = p.default
        out[p.id] = min(p.max, max(p.min, v))
    return out


def run_chain(src: Path, dst: Path, chain_id: str, params: dict | None,
              span: tuple[float, float] | None = None) -> dict:
    """Прогнать цепочку. span=(start, dur) — превью-фрагмент.
    Возвращает фактические (клэмпнутые) параметры. Бросает RuntimeError."""
    chain = CHAINS_BY_ID.get(chain_id)
    if chain is None:
        raise KeyError(f"неизвестная цепочка: {chain_id}")
    merged = clamp_params(chain, params)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if span is not None:
        cmd += ["-ss", f"{span[0]:.1f}", "-t", f"{span[1]:.1f}"]
    cmd += ["-i", str(src), "-filter_complex", chain.graph(merged),
            "-map", "[out]", str(dst)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(err[-500:] if err else f"ffmpeg exit {proc.returncode}")
    return merged
