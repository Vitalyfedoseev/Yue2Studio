#!/usr/bin/env python3
"""SheetSage2 (m-a-p, CC-BY-NC 4.0): аудио → нативный ABC YuE2 + MIDI.

Запускается сервером из ОСНОВНОГО .venv (там torch 2.10+cu128 и transformers
4.57.6; доставлены только torchaudio/pretty_midi/mir_eval/mido, см.
third_party/sheetsage2/README.md). Подпроцесс — ради изоляции: VRAM
освобождается при выходе, падение не роняет сервер.

Запуск: .venv/bin/python ss2.py <вход.аудио> <выход.abc> <выход.mid>
stdout — JSON {seconds, device, abc_chars, midi_bytes}; аудио декодируется
ffmpeg-CLI в wav 24 кГц моно (статический ffmpeg без shared-библиотек не
годится для внутреннего декодера torchaudio).
"""
import json
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "SheetSage2"
MAX_SECONDS = 15 * 60


def heal_module_cache() -> None:
    """Transformers копирует remote-код в ~/.cache/huggingface/modules и при
    рекурсивном разборе импортов теряет chord_spelling_sheetsage2.py —
    восстанавливаем копию всех .py из локального снапшота."""
    base = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    for cached in base.glob("*"):
        if not (cached / "modeling_sheetsage2.py").is_file():
            continue
        for f in MODEL_DIR.glob("*.py"):
            if not (cached / f.name).is_file():
                shutil.copy(f, cached / f.name)


def find_ffmpeg() -> str:
    import shutil

    ff = shutil.which("ffmpeg") or str(Path.home() / ".local" / "bin" / "ffmpeg")
    return ff


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit("использование: ss2.py <вход.аудио> <выход.abc> <выход.mid>")
    src, abc_path, mid_path = (Path(a) for a in sys.argv[1:])

    import tempfile

    with tempfile.TemporaryDirectory(prefix="ss2-") as tmp:
        wav = Path(tmp) / "in.wav"
        dec = subprocess.run(
            [find_ffmpeg(), "-v", "error", "-y", "-i", str(src),
             "-ac", "1", "-ar", "24000", str(wav)],
            capture_output=True,
        )
        if dec.returncode != 0:
            err = dec.stderr.decode(errors="replace").strip().splitlines()
            sys.exit("ffmpeg не смог декодировать: " + (err[-1] if err else f"exit {dec.returncode}"))
        with wave.open(str(wav)) as w:
            duration = w.getnframes() / w.getframerate()
        if duration > MAX_SECONDS:
            sys.exit(f"аудио длиннее {MAX_SECONDS // 60} мин")

        import torch
        from transformers import AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        heal_module_cache()
        model = AutoModel.from_pretrained(str(MODEL_DIR), trust_remote_code=True).eval()

        def run(dev: str):
            m = model.to(dev)
            with torch.inference_mode():
                # melody_only: обе мелодии (Vocal/Ins) без аккордов — родной
                # формат каверов YuE2
                return m.transcribe(str(wav), melody_only=True)

        try:
            result = run(device)
        except torch.cuda.OutOfMemoryError:
            device = "cpu"
            result = run(device)

        abc_path.write_text(result["abc"], encoding="utf-8")
        mid_path.write_bytes(result["midi"])

        import pretty_midi

        notes = sum(len(i.notes) for i in pretty_midi.PrettyMIDI(str(mid_path)).instruments)
        print(json.dumps({
            "seconds": round(time.time() - t0, 1),
            "device": device,
            "notes": notes,
            "duration_s": round(duration, 1),
            "abc_chars": len(result["abc"]),
            "midi_bytes": mid_path.stat().st_size,
        }))


if __name__ == "__main__":
    main()
