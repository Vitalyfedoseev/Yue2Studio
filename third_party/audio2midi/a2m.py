#!/usr/bin/env python3
"""Аудио → MIDI (Spotify basic-pitch, ONNX на CPU).

Вызывается сервером (server.py, /api/midi2abc) из отдельного venv — окружение
проекта с жёсткими пинами torch/numpy не трогаем. basic-pitch 0.4.0 поставлен
без зависимостей: его базовый pin tensorflow<2.15 не имеет колёс под py3.12,
для ONNX-инференса TF не нужен.

Запуск: a2m.py <вход.аудио> <выход.mid>
stdout: JSON {notes, duration_s}; ffmpeg-декод в wav 22.05 кГц моно —
basic-pitch обучен на этой сетке.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

MAX_SECONDS = 15 * 60  # длиннее — транскрипция на CPU занимает вечность


def find_ffmpeg() -> str:
    ff = shutil.which("ffmpeg") or str(Path.home() / ".local" / "bin" / "ffmpeg")
    if not shutil.which(ff):
        raise RuntimeError("ffmpeg не найден")
    return ff


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("использование: a2m.py <вход.аудио> <выход.mid>")
    src, out = Path(sys.argv[1]), Path(sys.argv[2])

    with tempfile.TemporaryDirectory(prefix="a2m-") as tmp:
        wav_path = Path(tmp) / "in.wav"
        dec = subprocess.run(
            [find_ffmpeg(), "-v", "error", "-y", "-i", str(src),
             "-ac", "1", "-ar", "22050", str(wav_path)],
            capture_output=True,
        )
        if dec.returncode != 0:
            # последняя строка stderr ffmpeg: «Invalid data found...» и т.п.
            err = dec.stderr.decode(errors="replace").strip().splitlines()
            sys.exit("ffmpeg не смог декодировать: " + (err[-1] if err else f"exit {dec.returncode}"))
        with wave.open(str(wav_path)) as w:
            duration = w.getnframes() / w.getframerate()
        if duration > MAX_SECONDS:
            sys.exit(f"аудио длиннее {MAX_SECONDS // 60} мин — обрежьте")
        if duration < 0.5:
            sys.exit("аудио короче полсекунды — нечего транскрибировать")

        import contextlib
        import io
        import warnings

        warnings.filterwarnings("ignore")
        from basic_pitch.inference import predict

        # predict() печатает "Predicting MIDI for …" в stdout — глушим,
        # чтобы единственной строкой stdout был итоговый JSON
        with contextlib.redirect_stdout(io.StringIO()):
            _, midi, note_events = predict(
                wav_path,
                onset_threshold=0.5,
                frame_threshold=0.3,
                minimum_frequency=30.0,   # отсечь суб-бас-шум, ноты от F#0
            )
        midi.write(str(out))

    print(json.dumps({"notes": len(note_events), "duration_s": round(duration, 1)}))


if __name__ == "__main__":
    main()
