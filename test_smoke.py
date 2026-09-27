#!/usr/bin/env python3
"""Smoke-тест YuE2-3B: загрузка + короткая песня.

Проверяет цепочку до FLAC: from_pretrained → план → семантика → латенты →
VAE-декод → файл. Запуск: .venv/bin/python test_smoke.py
"""
import time
from pathlib import Path

import torch

from yue2 import YuE2Pipeline

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

t0 = time.time()
print("загрузка YuE2-3B + VAE…", flush=True)
pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda", progress=False)
print(f"  загрузка: {time.time()-t0:.0f} c", flush=True)

lyrics = """[verse]
Morning light filtering through the pine
Every quiet street is yours and mine
[chorus]
Softly the world begins to breathe"""
style = (
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building "
    "gently into the chorus. Vocals: soft female lead, close and breathy, light "
    "stacked harmonies in the chorus. Arrangement: fingerpicked guitar and soft "
    "piano; brushed drums and upright bass enter in the chorus."
)

counters = {"tokens": 0, "phase": "?"}
def on_token(*args):
    counters["tokens"] += 1
    for a in args:
        if isinstance(a, str):
            counters["phase"] = a

gen = torch.Generator()  # не используется моделью — seed в запросе
torch.cuda.reset_peak_memory_stats()

t1 = time.time()
plan = pipe.plan(style=style, lyrics=lyrics, seed=7, cot="full", on_token=on_token)
t2 = time.time()
print(f"  план ABC: {t2-t1:.1f} c ({counters['tokens']} токенов)", flush=True)
tokens_plan = counters["tokens"]; counters["tokens"] = 0

semantic = pipe.generate_semantic(plan, on_token=on_token)
t3 = time.time()
print(f"  семантика: {t3-t2:.1f} c ({counters['tokens']} токенов)", flush=True)
tokens_sem = counters["tokens"]

latents = pipe.synthesize(semantic)
t4 = time.time()
print(f"  flow-латенты: {t4-t3:.1f} c", flush=True)

audio = pipe.decode(latents)
peak = torch.cuda.max_memory_allocated() / 2**30

import soundfile as sf
flac = OUT / "smoke_test.flac"
sf.write(flac, audio, 48000, subtype="PCM_24")

import numpy as np
rms = float(np.sqrt((audio.astype(np.float64) ** 2).mean()))
print(f"\n=== результат ===")
print(f"файл: {flac} ({flac.stat().st_size/1e6:.1f} МБ)")
print(f"аудио: {len(audio)/48000:.1f} c, 48000 Гц, {audio.shape[-1] if audio.ndim>1 else 1} кан.")
print(f"RMS: {rms:.4f} {'OK (не тишина)' if rms > 0.01 else 'ПРОБЛЕМА: тишина!'}")
print(f"пик VRAM: {peak:.2f} ГБ")
print(f"время: всего {time.time()-t1:.0f} c (план {t2-t1:.0f} + семантика {t3-t2:.0f} + флоу {t4-t3:.0f} + VAE {time.time()-t4:.0f})")
