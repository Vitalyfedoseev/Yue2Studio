#!/usr/bin/env python3
"""Веб-морда локальной генерации песен: YuE2-3B (m-a-p).

FastAPI-бэкенд с однозадачной GPU-очередью. Модель ~4B в bf16 живёт в VRAM
резидентно (пик ~11–14 ГБ — влезает в 16 ГБ без оффлоадов), VAE fp32.

YuE2: AR–NAR Mixture-of-Transformers пишет ABC-нотный план + семантические
токены, flow-matching даёт акустические латенты, VAE — 48 кГц стерео FLAC.
Длина песни следует за лирикой (лимит 9000 токенов ≈ ~5 мин).
Отмена генерации — родная (pipe.plan/generate_semantic принимают cancelled).
"""
import json
import os
import queue
import random
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import arc
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# Каверы по MIDI: конвертер marmooo/midi2abc, завендорен в third_party/ (MIT).
# Deno не нужен — обёртка cli.mjs работает под Node, зависимость midi-file
# лежит в node_modules рядом. ABC печатается после маркера <<<ABC>>>: до него
# в stdout бывают предупреждения конвертера о нелегальных длительностях.
MIDI2ABC_DIR = BASE / "third_party" / "midi2abc"
MIDI2ABC_CLI = MIDI2ABC_DIR / "cli.mjs"
ABC_MARKER = "<<<ABC>>>"

# Аудио → MIDI: Spotify basic-pitch (ONNX, CPU) в отдельном venv — окружение
# проекта с жёсткими пинами torch/numpy не трогаем. Ставился с обходом пина
# tensorflow<2.15 (нет колёс под py3.12): сначала свежий numpy/librosa/onnxruntime,
# затем basic-pitch --no-deps, см. third_party/audio2midi/README.md.
A2M_DIR = BASE / "third_party" / "audio2midi"
A2M_PY = A2M_DIR / "venv" / "bin" / "python"
A2M_SCRIPT = A2M_DIR / "a2m.py"
A2M_MELODY = A2M_DIR / "melody.py"   # прореживание до моно-мелодии

# Основной движок транскрипции: SheetSage2 (m-a-p) — та же команда, что за
# YuE2; выдаёт СРАЗУ нативный ABC (голоса Vocal/Ins, секции %, честный темп,
# melody_only=True). Работает в основном .venv (torch уже там), подпроцессом.
SS2_DIR = BASE / "third_party" / "sheetsage2"
SS2_SCRIPT = SS2_DIR / "ss2.py"
SS2_MODEL = SS2_DIR / "SheetSage2" / "config.json"

# Партитура делит контекст 24576 со стилем, лирикой и самой песней; полный
# микс легко даёт 25к+ токенов (все инструменты аккордами). Бюджет выше —
# конвертер сам упрощает партитуру до мелодической линии (melody.py).
ABC_TOKEN_BUDGET = 15000


def find_node() -> Optional[str]:
    """Путь до node: PATH, затем nvm (run.sh может стартовать без профиля)."""
    import shutil

    node = shutil.which("node")
    if node:
        return node
    nvm = Path.home() / ".nvm" / "versions" / "node"
    cands = sorted(nvm.glob("v*/bin/node")) if nvm.is_dir() else []
    return str(cands[-1]) if cands else None

MODEL_ID = os.environ.get("YUE2_MODEL", "m-a-p/YuE2-3B")
VAE_ID = os.environ.get("YUE2_VAE", "m-a-p/YuE2-Vae")  # legacy — бенчмарки выше
SAMPLE_RATE = 48000
# Бюджет семантических токенов на песню (дефолт протокола 9000 ≈ 4.5–6 мин).
# Контекст модели 24576 токенов общий; ~22k на песню ≈ 10–14 мин аудио,
# пик VRAM при этом доходит до ~14 ГБ — система позволяет.
MAX_SEM_TOKENS = int(os.environ.get("YUE2_MAX_TOKENS", "22000"))
# Черновик: короткий рендер на прикидку (~15–20 с), как в yue-studio
DRAFT_TOKENS = 450

# транслитерация для имени файла из названия песни
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}


def slugify(title: str) -> str:
    """Название песни → безопасный фрагмент имени файла (латиница, дефисы)."""
    import re

    s = "".join(_TRANSLIT.get(ch, ch) for ch in title.lower())
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:40].rstrip("-")


class YueEngine:
    def __init__(self):
        self.pipe = None
        self.ready = threading.Event()
        self.error: Optional[str] = None
        self.detail = "не загружена"

    def start_load(self):
        threading.Thread(target=self._load_sync, daemon=True, name="yue2-load").start()

    def _load_sync(self):
        try:
            from yue2 import YuE2Pipeline

            self.detail = "загрузка YuE2-3B + VAE (~8 ГБ)"
            self.pipe = YuE2Pipeline.from_pretrained(
                MODEL_ID, vae=VAE_ID, device="cuda", progress=False
            )
            self.detail = "готова (bf16 резидентно)"
        except Exception as e:  # noqa: BLE001 — текст ошибки уходит в /api/health
            import traceback

            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"
            self.detail = f"ошибка загрузки: {self.error}"
        finally:
            self.ready.set()


ENGINE = YueEngine()
QUEUE: "queue.Queue[str]" = queue.Queue()
JOBS: dict = {}
JOBS_LOCK = threading.Lock()
# Одна GPU: генерация держит пайплайн всё время работы, план-only и превью
# фрагментов пытаются взять этот же лок с таймаутом → 409 «занято»
PIPE_LOCK = threading.Lock()

# Кадров латентов YuE2 в секунду (срез превью считается в этих единицах)
LATENT_HZ = 25.0


def _set(jid: str, **kw):
    with JOBS_LOCK:
        JOBS[jid].update(kw)


def _run_job(job: dict):
    t0 = time.time()
    seed = job["seed"]
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)
        _set(job["id"], seed=seed)

    def cancelled() -> bool:
        with JOBS_LOCK:
            return bool(JOBS[job["id"]].get("cancel"))

    counters = {"phase": "план", "tokens": 0, "budget": 0}

    def on_token(*args):
        # бэкенд зовёт (token_phase, token); берём фазу и считаем токены
        for a in args:
            if isinstance(a, str):
                counters["phase"] = "план ABC" if "abc" in a.lower() else "семантика"
        counters["tokens"] += 1

    stop_watch = threading.Event()

    def watch():
        last = 0
        while not stop_watch.is_set():
            elapsed = time.time() - t0
            tps = (counters["tokens"] - last) / 2 if elapsed > 2 else None
            last = counters["tokens"]
            # честный процент есть только на семантике: токены/бюджет
            pct = None
            if counters["phase"] == "семантика" and counters["budget"] > 0:
                pct = min(99, counters["tokens"] * 100 // counters["budget"])
            _set(
                job["id"], stage=counters["phase"], tokens=counters["tokens"],
                elapsed_s=round(elapsed, 1), tok_per_s=round(tps, 1) if tps else None,
                pct=pct,
            )
            stop_watch.wait(2)

    watcher = threading.Thread(target=watch, daemon=True, name=f"watch-{job['id']}")
    watcher.start()

    pipe = ENGINE.pipe
    try:
        torch.cuda.reset_peak_memory_stats()
        kw = dict(
            style=job["style"], lyrics=job["lyrics"], seed=seed,
            cot=job["cot"], cancelled=cancelled,
        )
        if job["cfg_scale"] is not None:
            kw["cfg_scale"] = job["cfg_scale"]
        if job.get("abc"):
            kw["abc"] = job["abc"]  # внешняя партитура: план-генерация пропускается
        if job.get("arc"):
            kw["style"] = arc.style_with_arc(kw["style"], job["arc"])

        with PIPE_LOCK:
            t1 = time.time()
            plan = pipe.plan(**kw, on_token=on_token)
            if job.get("arc"):
                # драматургия правит авторский план (темпы по секциям, октава
                # в финале у burst) — префикс пересобирается повторным планом
                counters["phase"] = "arc-plan"
                _set(job["id"], stage="arc-plan")
                abc2 = arc.apply_arc(plan.abc, job["arc"])
                plan = pipe.plan(**{**kw, "abc": abc2}, on_token=on_token)
                counters["phase"] = "семантика"
            t2 = time.time()
            _set(job["id"], stage="семантика", plan_s=round(t2 - t1, 1))

            from yue2.protocol import CONTEXT, Sampling

            # контекст 24576 общий: префикс (стиль+лирика+ABC-план) + песня ≤ CONTEXT;
            # пайплайн не усекает молча — подгоняем бюджет под фактический префикс
            budget = max(200, min(MAX_SEM_TOKENS, CONTEXT - len(plan.prefix) - 4))
            if job.get("draft"):
                budget = min(budget, DRAFT_TOKENS)
            counters["budget"] = budget
            if job.get("abc") and CONTEXT - len(plan.prefix) - 4 < 1000:
                # длинный MIDI съедает контекст — песня вышла бы на 200 токенов;
                # честная ошибка лучше, чем минута GPU и 5 секунд «песни»
                raise ValueError(
                    f"партитура из MIDI занимает почти весь контекст модели "
                    f"({len(plan.prefix)}/{CONTEXT} ток.) — обрежьте MIDI или уменьшите лирику"
                )
            semantic = pipe.generate_semantic(
                plan, sampling=Sampling(max_tokens=budget),
                cancelled=cancelled, on_token=on_token,
            )
            counters["phase"] = "акустические латенты"
            t3 = time.time()
            _set(job["id"], stage="акустические латенты", semantic_s=round(t3 - t2, 1))

            latents = pipe.synthesize(semantic, cancelled=cancelled)
            counters["phase"] = "VAE-декод"
            t4 = time.time()
            _set(job["id"], stage="VAE-декод", synth_s=round(t4 - t3, 1))

            audio = pipe.decode(latents)
        peak_vram = torch.cuda.max_memory_allocated() / 2**30

        import soundfile as sf

        title = job.get("title") or ""
        stem = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slugify(title) + '-' if slugify(title) else ''}{job['id'][:8]}"
        flac_path = OUT / f"{stem}.flac"
        sf.write(flac_path, audio, SAMPLE_RATE, subtype="PCM_24")
        # партитура и латенты — фундамент превью/овердаба/таймлайна;
        # float16 вдвое легче (~2 МБ/мин), decode вернём во float32
        (OUT / f"{stem}.score.abc").write_text(plan.abc, encoding="utf-8")
        z = latents.detach().cpu().numpy() if hasattr(latents, "detach") else latents
        np.save(OUT / f"{stem}.latent.npy", np.asarray(z, dtype=np.float16))

        meta = {
            "file": flac_path.name,
            "title": title,
            "duration_s": round(len(audio) / SAMPLE_RATE, 2),
            "sample_rate": SAMPLE_RATE,
            "style": job["style"],          # склеенный промпт (как уходит в модель)
            "style_base": job["style_base"],
            "voice": job.get("voice") or "",
            "lyrics": job["lyrics"],
            "cot": job["cot"],
            "cfg_scale": job["cfg_scale"],
            "seed": seed,
            "tokens": counters["tokens"],
            "total_s": round(time.time() - t0, 1),
            "peak_vram_gb": round(peak_vram, 2),
            "ts": datetime.now().isoformat(timespec="seconds"),
            "score_abc_file": f"{stem}.score.abc",
            "latent_file": f"{stem}.latent.npy",
            "arc": job.get("arc") or "",
            "draft": bool(job.get("draft")),
        }
        if job.get("abc"):
            (OUT / f"{stem}.abc").write_text(job["abc"], encoding="utf-8")
            meta["abc_file"] = f"{stem}.abc"   # сама партитура не в мете — бывает >100 КБ
            meta["abc_chars"] = len(job["abc"])
        if job.get("overdub_parent"):
            # овердаб: смешиваем с родителем сразу, ребёнок остаётся и отдельно
            meta["overdub_of"] = job["overdub_parent"]
            try:
                mixed = _mix_overdub(
                    OUT / f"{job['overdub_parent']}.flac", flac_path,
                    job.get("overdub_gain", 0.5), job["overdub_parent"])
                meta["overdub_file"] = mixed["file"]
            except Exception as e:  # noqa: BLE001 — неудачный микс не роняет джобу
                meta["overdub_error"] = f"{type(e).__name__}: {e}"
        (OUT / f"{stem}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        _set(job["id"], status="done", stage="done", **{"result": meta})
    except InterruptedError:
        _set(job["id"], status="canceled", error="остановлено пользователем")
    finally:
        stop_watch.set()


def worker():
    while True:
        jid = QUEUE.get()
        with JOBS_LOCK:
            job = JOBS.get(jid)
        if job is None or job["status"] != "queued":
            continue
        if ENGINE.error:
            _set(jid, status="error", error=f"модель не загрузилась: {ENGINE.error}")
            continue
        if not ENGINE.ready.is_set():
            _set(jid, stage=f"загрузка модели ({ENGINE.detail})")
        ENGINE.ready.wait()
        if ENGINE.error:
            _set(jid, status="error", error=f"модель не загрузилась: {ENGINE.error}")
            continue
        try:
            _set(jid, status="running")
            _run_job(job)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _set(jid, status="error", error="CUDA out of memory — укоротите лирику")
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            _set(jid, status="error", error=f"{type(e).__name__}: {e}")


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app):
    if os.environ.get("YUE2_NO_MODEL"):
        # тестовый режим: GPU не занимаем, генерация недоступна (503)
        ENGINE.detail = "модель не грузится (YUE2_NO_MODEL=1)"
    else:
        ENGINE.start_load()
    threading.Thread(target=worker, daemon=True, name="job-worker").start()
    yield


app = FastAPI(title="YuE2-3B Music Studio", lifespan=lifespan)


class GenRequest(BaseModel):
    style: str = Field(..., min_length=3, description="Стиль: жанр, темп, аранжировка")
    voice: str = Field("", description="Описание голоса — отдельно, клеится к стилю")
    lyrics: str = Field(..., min_length=2, description="Лирика с тегами секций")
    title: str = Field("", description="Название песни (уходит в имя файла)")
    cot: str = Field("full", pattern="^(full|melody|off)$")
    cfg_scale: Optional[float] = Field(None, ge=1.0, le=3.0)
    seed: int = Field(-1, ge=-1)
    abc: Optional[str] = Field(
        None, max_length=400_000,
        description="Готовая ABC-партитура (кавер по MIDI) — модель поёт её, план не пишет",
    )
    draft: bool = Field(False, description="Черновик ~15–20 с (бюджет 450 токенов)")
    arc: str = Field("", pattern="^(|build|wave|burst)$",
                     description="Драматургия: правки темпа/октавы в авторском плане")


@app.get("/api/health")
def health():
    vram_free_gb = None
    if torch.cuda.is_available():
        free_b, _ = torch.cuda.mem_get_info()
        vram_free_gb = round(free_b / 2**30, 1)
    node = find_node()
    try:
        import librosa  # noqa: F401

        metrics_ok = True
    except ImportError:
        metrics_ok = False
    return {
        "ready": ENGINE.pipe is not None and ENGINE.error is None,
        "detail": ENGINE.error or ENGINE.detail,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_free_gb": vram_free_gb,
        "queue": QUEUE.qsize(),
        "sampling_rate": SAMPLE_RATE,
        "metrics_ok": metrics_ok,
        "midi2abc": {
            "ok": node is not None and MIDI2ABC_CLI.is_file(),
            "node": bool(node),
            "ss2": SS2_SCRIPT.is_file() and SS2_MODEL.is_file(),     # аудио → нативный ABC
            "a2m": A2M_PY.is_file() and A2M_SCRIPT.is_file(),  # фолбэк (basic-pitch)
        },
    }


def _estimate_abc_tokens(abc: str) -> int:
    """Токены ABC токенизатором модели; до готовности пайплайна — грубая оценка
    (ASCII-нотация в среднем ~2 символа на токен)."""
    if ENGINE.pipe is not None:
        try:
            return len(ENGINE.pipe.tokenizer.encode(abc))
        except Exception:  # noqa: BLE001 — фолбэк на оценку по символам
            pass
    return len(abc) // 2


def _midi_to_abc(midi: bytes, env: dict, node: str) -> str:
    import subprocess

    try:
        proc = subprocess.run(
            [node, MIDI2ABC_CLI], input=midi, capture_output=True,
            timeout=60, env=env, cwd=MIDI2ABC_DIR,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(500, "конвертер ABC не уложился в 60 с")
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip().splitlines()[-1:]
        raise HTTPException(422, "MIDI не разобран: " + (err[0] if err else f"exit {proc.returncode}"))
    _, _, abc = proc.stdout.decode(errors="replace").partition(ABC_MARKER)
    abc = abc.strip()
    if not abc:
        raise HTTPException(422, "ноты не найдены (партитура пуста)")
    return abc


def _thin_midi(midi: bytes) -> tuple[bytes, Optional[int]]:
    """MIDI → монофоническая мелодия (melody.py в venv basic-pitch)."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="melody-") as tmp:
        src, dst = Path(tmp) / "in.mid", Path(tmp) / "out.mid"
        src.write_bytes(midi)
        try:
            proc = subprocess.run(
                [str(A2M_PY), str(A2M_MELODY), str(src), str(dst)],
                capture_output=True, timeout=60, cwd=A2M_DIR,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(500, "упрощение партитуры не уложилось в 60 с")
        if proc.returncode != 0 or not dst.is_file():
            err = proc.stderr.decode(errors="replace").strip().splitlines()[-1:]
            raise HTTPException(500, "упрощение партитуры не удалось: " + (err[0] if err else f"exit {proc.returncode}"))
        notes = None
        try:
            notes = json.loads(proc.stdout.decode(errors="replace").strip().splitlines()[-1]).get("notes_out")
        except (ValueError, IndexError):
            pass
        return dst.read_bytes(), notes


def _sheetsage2_abc(data: bytes) -> Optional[dict]:
    """Аудио → нативный ABC YuE2 (SheetSage2). None — движок недоступен или
    упал: вызывающий откатывается на basic-pitch."""
    import subprocess
    import sys
    import tempfile

    if not (SS2_SCRIPT.is_file() and SS2_MODEL.is_file()):
        return None
    with tempfile.TemporaryDirectory(prefix="ss2-") as tmp:
        src = Path(tmp) / "in"
        src.write_bytes(data)
        abc_path, mid_path = Path(tmp) / "out.abc", Path(tmp) / "out.mid"
        try:
            proc = subprocess.run(
                [sys.executable, str(SS2_SCRIPT), str(src), str(abc_path), str(mid_path)],
                capture_output=True, timeout=300, cwd=SS2_DIR,
            )
        except subprocess.TimeoutExpired:
            return None
        if proc.returncode != 0 or not abc_path.is_file():
            return None
        try:
            info = json.loads(proc.stdout.decode(errors="replace").strip().splitlines()[-1])
        except (ValueError, IndexError):
            info = {}
        midi = mid_path.read_bytes() if mid_path.is_file() else None
        return {"abc": abc_path.read_text(encoding="utf-8").strip(),
                "notes": info.get("notes"), "duration_s": info.get("duration_s"),
                "midi": midi}


def _convert_to_abc(data: bytes, title: str) -> dict:
    """Синхронная работа threadpool'а: MIDI → ABC (midi2abc), либо аудио →
    SheetSage2 → нативный ABC (фолбэк: basic-pitch → midi2abc, с упрощением
    до мелодии при переполнении бюджета). Поднимает HTTPException."""
    import base64
    import subprocess
    import tempfile

    node = find_node()
    if node is None or not MIDI2ABC_CLI.is_file():
        raise HTTPException(503, "конвертер MIDI→ABC недоступен (нет node или third_party/midi2abc)")
    env = {**os.environ, "MIDI2ABC_TITLE": title}
    result: dict = {}

    midi = data
    if data[:4] != b"MThd":
        # не MIDI — аудио: сначала SheetSage2 (нативная транскрипция YuE2)
        if len(data) > 100 * 2**20:
            raise HTTPException(422, "аудио больше 100 МБ")
        ss2 = _sheetsage2_abc(data)
        if ss2 is not None:
            tokens = _estimate_abc_tokens(ss2["abc"])
            if tokens <= ABC_TOKEN_BUDGET:
                result.update(
                    transcribed=True, engine="sheetsage2", notes=ss2["notes"],
                    duration_s=ss2["duration_s"], abc_tokens=tokens,
                    abc=ss2["abc"], chars=len(ss2["abc"]),
                )
                if ss2["midi"]:
                    result["midi_b64"] = base64.b64encode(ss2["midi"]).decode()
                return result
        # фолбэк: basic-pitch → MIDI → midi2abc
        if not (A2M_PY.is_file() and A2M_SCRIPT.is_file()):
            raise HTTPException(503, "транскрипция аудио недоступна (нет SheetSage2 и third_party/audio2midi/venv)")
        with tempfile.TemporaryDirectory(prefix="a2m-") as tmp:
            src = Path(tmp) / "in"
            src.write_bytes(data)
            mid_path = Path(tmp) / "out.mid"
            try:
                proc = subprocess.run(
                    [str(A2M_PY), str(A2M_SCRIPT), str(src), str(mid_path)],
                    capture_output=True, timeout=300, cwd=A2M_DIR,
                )
            except subprocess.TimeoutExpired:
                raise HTTPException(500, "транскрипция не уложилась в 300 с — возьмите отрывок короче")
            if proc.returncode != 0:
                err = (proc.stderr.decode(errors="replace").strip().splitlines() or
                       proc.stdout.decode(errors="replace").strip().splitlines())
                raise HTTPException(422, "аудио не разобрано: " + (err[-1] if err else f"exit {proc.returncode}"))
            try:
                info = json.loads(proc.stdout.decode(errors="replace").strip().splitlines()[-1])
            except (ValueError, IndexError):
                info = {}
            midi = mid_path.read_bytes()
        result["transcribed"] = True
        result["engine"] = "basic-pitch"
        result["notes"] = info.get("notes")
        result["midi_b64"] = base64.b64encode(midi).decode()
        result["duration_s"] = info.get("duration_s")

    abc = _midi_to_abc(midi, env, node)
    tokens = _estimate_abc_tokens(abc)
    if tokens > ABC_TOKEN_BUDGET:
        # полный микс не лезет в контекст — оставляем ведущую линию и
        # перегоняем заново; скачивание .mid и счётчик нот тоже обновляем
        midi, notes = _thin_midi(midi)
        abc = _midi_to_abc(midi, env, node)
        tokens = _estimate_abc_tokens(abc)
        result["thinned"] = True
        result["notes"] = notes if notes is not None else result.get("notes")
        result["midi_b64"] = base64.b64encode(midi).decode()
    result["abc"] = abc
    result["chars"] = len(abc)
    result["abc_tokens"] = tokens
    return result


@app.post("/api/midi2abc")
async def midi_to_abc(request: Request):
    """MIDI или аудио → ABC. Тело — сырые байты файла (мультипарт не нужен —
    python-multipart не стоит). .mid/.midi идут напрямую через midi2abc; аудио
    сначала транскрибируется SheetSage2 (нативный ABC YuE2; фолбэк basic-pitch,
    ~10–60 с). ?title= — заголовок T: и имя скачанного .mid."""
    from fastapi.concurrency import run_in_threadpool

    data = await request.body()
    if not data:
        raise HTTPException(422, "пустое тело — приложите файл")
    if data[:4] == b"MThd" and len(data) > 20 * 2**20:
        raise HTTPException(422, "MIDI больше 20 МБ")
    title = (request.query_params.get("title") or "").replace("\n", " ").strip()[:60]
    return await run_in_threadpool(_convert_to_abc, data, title)


@app.post("/api/generate")
def generate(req: GenRequest):
    if ENGINE.error:
        raise HTTPException(503, f"модель не загрузилась: {ENGINE.error}")
    abc = (req.abc or "").strip()
    if abc and req.cot == "off":
        raise HTTPException(422, "внешняя партитура несовместима с cot=off — выберите melody или full")
    if req.arc and abc:
        raise HTTPException(422, "драматургия правит авторский план — с внешней партитурой несовместима")
    if req.arc and req.cot == "off":
        raise HTTPException(422, "драматургии нужен нотный план — выберите full или melody")
    jid = uuid.uuid4().hex[:12]
    style_base = req.style.strip()
    voice = req.voice.strip()
    with JOBS_LOCK:
        JOBS[jid] = {
            "id": jid, "status": "queued", "stage": "в очереди",
            "style": (style_base + "\n\n" + voice).strip() if voice else style_base,
            "style_base": style_base, "voice": voice,
            "title": req.title.strip()[:80],
            "lyrics": req.lyrics.strip(),
            "cot": req.cot, "cfg_scale": req.cfg_scale, "seed": req.seed,
            "abc": abc or None, "arc": req.arc or "", "draft": req.draft,
            "elapsed_s": 0.0, "tokens": 0, "tok_per_s": None, "pct": None,
            "cancel": False, "created": time.time(),
        }
    QUEUE.put(jid)
    return {"id": jid}


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    with JOBS_LOCK:
        job = JOBS.get(jid)
        if job is None:
            raise HTTPException(404, "задача не найдена")
        if job["status"] == "queued":
            # ещё не стартовала — снимаем сразу, GPU не занимается
            job.update(status="canceled", stage="отменена", error="отменена до запуска")
        else:
            job["cancel"] = True
    return {"ok": True}


# ---------- plan-only: план без рендера (посмотреть/поправить ABC до генерации) ----------

class PlanRequest(BaseModel):
    style: str = Field(..., min_length=3)
    voice: str = Field("", description="Отдельное описание голоса — клеится к стилю")
    lyrics: str = Field(..., min_length=2)
    cot: str = Field("full", pattern="^(full|melody|off)$")
    seed: int = Field(-1, ge=-1)
    arc: str = Field("", pattern="^(|build|wave|burst)$")


@app.post("/api/plan")
def plan_only(req: PlanRequest):
    """ABC-план по стилю/лирике без рендера песни. Под PIPE_LOCK: план тоже
    занимает GPU; при работающей генерации — 409."""
    if ENGINE.error:
        raise HTTPException(503, f"модель не загрузилась: {ENGINE.error}")
    if not ENGINE.ready.is_set():
        raise HTTPException(503, f"модель ещё грузится ({ENGINE.detail})")
    if req.cot == "off":
        raise HTTPException(422, "cot=off не пишет нотный план")
    if not PIPE_LOCK.acquire(timeout=1):
        raise HTTPException(409, "GPU занят генерацией — постройте план, когда очередь освободится")
    try:
        import abcparse

        style_base = req.style.strip()
        voice = req.voice.strip()
        style = (style_base + "\n\n" + voice).strip() if voice else style_base
        if req.arc:
            style = arc.style_with_arc(style, req.arc)
        seed = req.seed if req.seed >= 0 else random.randint(0, 2**31 - 1)
        plan = ENGINE.pipe.plan(style=style, lyrics=req.lyrics.strip(), cot=req.cot, seed=seed)
        abc_text = plan.abc
        if req.arc:
            abc_text = arc.apply_arc(abc_text, req.arc)
        tl = abcparse.parse_abc(abc_text)
        return {
            "abc": abc_text,
            "seconds": tl.get("duration_sec"),
            "tokens": _estimate_abc_tokens(abc_text),
            "seed": seed,
        }
    finally:
        PIPE_LOCK.release()


# ---------- таймлайн и превью готовых записей ----------

def _clean_stem(stem: str) -> str:
    import re

    if not re.fullmatch(r"[\w.-]+", stem):
        raise HTTPException(422, "недопустимое имя")
    return stem[:-5] if stem.endswith(".flac") else stem


def _rms_sections(flac: Path, bars: list) -> list:
    """Средняя RMS (дБ) по смежным группам тактов одной секции — карта
    громкости для ленты секций. soundfile+numpy, без librosa."""
    import soundfile as sf

    if not flac.is_file() or not bars:
        return []
    try:
        data, sr = sf.read(str(flac), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 — битый flac не должен ломать таймлайн
        return []
    mono = data.mean(axis=1)
    total = len(mono) / sr
    groups: list[dict] = []
    for b in bars:
        if groups and groups[-1]["section"] == b["section"]:
            groups[-1]["end_sec"] = b["end_sec"]
        else:
            groups.append({"section": b["section"], "start_sec": b["start_sec"], "end_sec": b["end_sec"]})
    out = []
    for g in groups:
        a = max(0.0, min(g["start_sec"], total))
        z = min(total, max(a, g["end_sec"]))
        if z - a < 0.05:
            continue
        seg = mono[int(a * sr):int(z * sr)]
        rms = float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0
        out.append({**g, "rms_db": round(20 * np.log10(rms + 1e-9), 1)})
    return out


@app.get("/api/gallery/{stem}/score")
def track_score(stem: str):
    """Таймлайн партитуры (abcparse): такты/голоса/секции/секунды + rms_sections.
    Кеш — outputs/<stem>.d/score.json."""
    stem = _clean_stem(stem)
    abc_path = OUT / f"{stem}.score.abc"
    if not abc_path.is_file():
        abc_path = OUT / f"{stem}.abc"      # кавер или старая запись без score
    if not abc_path.is_file():
        raise HTTPException(404, "партитура не найдена (запись без нотного плана)")
    d = OUT / f"{stem}.d"
    d.mkdir(exist_ok=True)
    cache = d / "score.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    import abcparse

    tl = abcparse.parse_abc(abc_path.read_text(encoding="utf-8"))
    tl["rms_sections"] = _rms_sections(OUT / f"{stem}.flac", tl["bars"])
    cache.write_text(json.dumps(tl, ensure_ascii=False), encoding="utf-8")
    return tl


class PreviewRequest(BaseModel):
    from_sec: float = Field(..., ge=0)
    to_sec: float = Field(...)


@app.post("/api/gallery/{stem}/preview")
def track_preview(stem: str, req: PreviewRequest):
    """Превью фрагмента: срез латентов [from,to] сек → VAE-decode → flac
    в outputs/<stem>.d/. Клэмп по длительности звука (ABC бывает длиннее)."""
    stem = _clean_stem(stem)
    latent = OUT / f"{stem}.latent.npy"
    if not latent.is_file():
        raise HTTPException(404, "латенты не сохранены — запись сделана до появления превью")
    if ENGINE.error or ENGINE.pipe is None:
        raise HTTPException(503, f"модель не готова: {ENGINE.error or ENGINE.detail}")
    meta_path = OUT / f"{stem}.json"
    duration = None
    if meta_path.is_file():
        try:
            duration = json.loads(meta_path.read_text(encoding="utf-8")).get("duration_s")
        except ValueError:
            pass
    z = np.load(latent)
    total = duration if duration else len(z) / LATENT_HZ
    f = max(0.0, min(req.from_sec, total - 1))
    t = min(max(req.to_sec, f + 1), total)
    if t - f < 1:
        raise HTTPException(422, "фрагмент короче секунды")
    if not PIPE_LOCK.acquire(timeout=1):
        raise HTTPException(409, "GPU занят генерацией — превью недоступно, попробуйте позже")
    try:
        import soundfile as sf

        a, b = int(f * LATENT_HZ), max(int(t * LATENT_HZ), int(f * LATENT_HZ) + 1)
        audio = ENGINE.pipe.decode(z[a:b, :].astype(np.float32))
        d = OUT / f"{stem}.d"
        d.mkdir(exist_ok=True)
        out = d / f"preview-{f:.0f}-{t:.0f}.flac"
        sf.write(str(out), audio, SAMPLE_RATE, subtype="PCM_24")
        return {
            "file": f"{stem}.d/{out.name}",
            "url": f"/outputs/{stem}.d/{out.name}",
            "from_sec": round(f, 1), "to_sec": round(t, 1),
            "seconds": round(t - f, 1),
        }
    finally:
        PIPE_LOCK.release()


# ---------- метрики ----------

@app.post("/api/gallery/{stem}/analyze")
def track_analyze(stem: str, fresh: bool = False):
    """librosa-метрики записи (темп, тональность, динамика, полосы, стерео);
    кеш — outputs/<stem>.d/metrics.json, ?fresh=true пересчитать."""
    stem = _clean_stem(stem)
    flac = OUT / f"{stem}.flac"
    if not flac.is_file():
        raise HTTPException(404, "запись не найдена")
    d = OUT / f"{stem}.d"
    cache = d / "metrics.json"
    if cache.is_file() and not fresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    try:
        import audio_metrics
    except ImportError:
        raise HTTPException(503, "librosa не установлена — метрики недоступны")
    try:
        m = audio_metrics.analyze_file(flac)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"анализ не удался: {type(e).__name__}: {e}")
    d.mkdir(exist_ok=True)
    cache.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return m


# ---------- DSP-цепочки (ffmpeg) ----------

class DspRequest(BaseModel):
    chain: str
    params: dict = Field(default_factory=dict)
    preview: bool = Field(False, description="Фрагмент с 20-й секунды, 15 с")


@app.get("/api/dsp/chains")
def dsp_chains_meta():
    from dataclasses import asdict

    import dsp_chains
    return [{"id": c.id, "name": c.name, "note": c.note,
             "params": [asdict(p) for p in c.params]} for c in dsp_chains.CHAINS]


@app.get("/api/gallery/{stem}/dsp")
def dsp_variants(stem: str):
    """Список DSP-вариантов записи (с метриками, если считались)."""
    stem = _clean_stem(stem)
    d = OUT / f"{stem}.d"
    variants = []
    if d.is_dir():
        for p in sorted(d.glob("dsp-*.flac")):
            item = {"file": f"{stem}.d/{p.name}", "url": f"/outputs/{stem}.d/{p.name}"}
            m = p.with_name(p.name + ".metrics.json")
            if m.is_file():
                try:
                    item["metrics"] = json.loads(m.read_text(encoding="utf-8"))
                except ValueError:
                    pass
            variants.append(item)
    return {"variants": variants}


@app.post("/api/gallery/{stem}/dsp")
def dsp_apply(stem: str, req: DspRequest):
    """Прогнать цепочку (wall/wall-lite/tape) по FLAC записи; результат —
    outputs/<stem>.d/dsp[-preview]-<chain>.flac + метрики варианта."""
    stem = _clean_stem(stem)
    flac = OUT / f"{stem}.flac"
    if not flac.is_file():
        raise HTTPException(404, "запись не найдена")
    import dsp_chains

    name = f"dsp-preview-{req.chain}.flac" if req.preview else f"dsp-{req.chain}.flac"
    d = OUT / f"{stem}.d"
    d.mkdir(exist_ok=True)
    dst = d / name
    try:
        merged = dsp_chains.run_chain(
            flac, dst, req.chain, req.params,
            span=(dsp_chains.PREVIEW_START, dsp_chains.PREVIEW_DUR) if req.preview else None,
        )
    except KeyError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(500, f"ffmpeg: {e}")
    metrics = None
    try:
        import audio_metrics

        metrics = audio_metrics.analyze_file(dst)
        (d / f"{name}.metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001 — вариант валиден и без метрик
        pass
    return {"file": f"{stem}.d/{name}", "url": f"/outputs/{stem}.d/{name}",
            "params": merged, "metrics": metrics}


# ---------- овердаб: дочерняя генерация по партитуре родителя ----------

def _mix_overdub(parent_flac: Path, child_flac: Path, gain: float, parent_stem: str) -> dict:
    """Смешать родителя и овердаб (как в yue-studio): gain на дубль, 50-мс
    кроссфейд на обрезе, пик-нормализация; файл в <parent>.d/."""
    import soundfile as sf

    a, sr = sf.read(str(parent_flac), dtype="float32", always_2d=True)
    b, b_sr = sf.read(str(child_flac), dtype="float32", always_2d=True)
    if b_sr != sr:
        raise ValueError(f"частоты записей не совпали: {sr} vs {b_sr}")
    n = max(len(a), len(b))
    mix = np.zeros((n, a.shape[1]), dtype=np.float32)
    mix[:len(a)] += a
    mb = np.zeros((n, a.shape[1]), dtype=np.float32)
    mb[:min(len(b), n)] = b[:n]
    mix[:len(mb)] += mb * gain
    fade = min(int(0.05 * sr), n)
    if fade:
        mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
    peak = float(np.abs(mix).max())
    if peak > 1.0:
        mix /= peak
    d = OUT / f"{parent_stem}.d"
    d.mkdir(exist_ok=True)
    out = d / f"overdub-{child_flac.stem}.flac"
    sf.write(str(out), mix, sr, subtype="PCM_24")
    return {"file": f"{parent_stem}.d/{out.name}", "url": f"/outputs/{parent_stem}.d/{out.name}"}


class OverdubRequest(BaseModel):
    style: str = Field(..., min_length=3, description="Стиль дублирующей партии")
    lyrics: str = Field("", description="Пусто — берётся лирика родителя")
    gain: float = Field(0.5, ge=0.05, le=1.0)
    seed: int = Field(-1, ge=-1)
    title: str = Field("")


@app.post("/api/gallery/{stem}/overdub")
def track_overdub(stem: str, req: OverdubRequest):
    """Овердаб: джоба-потомок по score.abc родителя (или .abc кавера);
    после рендера автоматически смешивается с родителем (gain)."""
    stem = _clean_stem(stem)
    abc_path = OUT / f"{stem}.score.abc"
    if not abc_path.is_file():
        abc_path = OUT / f"{stem}.abc"
    if not abc_path.is_file():
        raise HTTPException(404, "у записи нет партитуры — овердаб невозможен")
    if ENGINE.error:
        raise HTTPException(503, f"модель не загрузилась: {ENGINE.error}")
    lyrics = req.lyrics.strip()
    if not lyrics:
        meta_path = OUT / f"{stem}.json"
        if meta_path.is_file():
            try:
                lyrics = (json.loads(meta_path.read_text(encoding="utf-8"))
                          .get("lyrics") or "").strip()
            except ValueError:
                pass
    if len(lyrics) < 2:
        raise HTTPException(422, "нет лирики ни в запросе, ни у родителя")
    abc_text = abc_path.read_text(encoding="utf-8").strip()
    jid = uuid.uuid4().hex[:12]
    style_base = req.style.strip()
    title = (req.title.strip() or f"овердаб · {stem}")[:80]
    with JOBS_LOCK:
        JOBS[jid] = {
            "id": jid, "status": "queued", "stage": "в очереди",
            "style": style_base, "style_base": style_base, "voice": "",
            "title": title, "lyrics": lyrics,
            "cot": "melody", "cfg_scale": None, "seed": req.seed,
            "abc": abc_text, "arc": "", "draft": False,
            "overdub_parent": stem, "overdub_gain": req.gain,
            "elapsed_s": 0.0, "tokens": 0, "tok_per_s": None, "pct": None,
            "cancel": False, "created": time.time(),
        }
    QUEUE.put(jid)
    return {"id": jid, "parent": stem}


# ---------- переименование (только мета, файлы не трогаем) ----------

class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1)


@app.post("/api/gallery/{stem}/rename")
def rename_track(stem: str, req: RenameRequest):
    stem = _clean_stem(stem)
    meta_path = OUT / f"{stem}.json"
    if not meta_path.is_file():
        raise HTTPException(404, "запись не найдена")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["title"] = req.title.strip()[:80]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"title": meta["title"]}


@app.get("/api/outputs/{name}/wav")
def to_wav(name: str):
    """Конвертация готового FLAC в WAV (PCM 16-bit), с кешем <stem>.wav."""
    import re
    import subprocess

    if not re.fullmatch(r"[\w.-]+\.flac", name):
        raise HTTPException(422, "имя файла должно быть <имя>.flac")
    flac = OUT / name
    if not flac.is_file():
        raise HTTPException(404, "файл не найден")
    wav = OUT / (Path(name).stem + ".wav")
    if not wav.is_file():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(flac), "-codec:a", "pcm_s16le", str(wav)],
            check=True, capture_output=True,
        )
    return FileResponse(wav, filename=wav.name, media_type="audio/wav")


@app.get("/api/outputs/{name}/mp3")
def to_mp3(name: str, q: int = 320):
    """Конвертация готового FLAC в MP3 (320k по умолчанию, ?q=48 — компактный)."""
    import re
    import subprocess

    if q not in (320, 48):
        raise HTTPException(422, "q: 320 или 48")
    if not re.fullmatch(r"[\w.-]+\.flac", name):
        raise HTTPException(422, "имя файла должно быть <имя>.flac")
    flac = OUT / name
    if not flac.is_file():
        raise HTTPException(404, "файл не найден")
    # 320k исторически кешируется как <stem>.mp3, компактный — <stem>.48k.mp3
    mp3 = OUT / (Path(name).stem + (".mp3" if q == 320 else ".48k.mp3"))
    if not mp3.is_file():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(flac), "-codec:a", "libmp3lame", "-b:a", f"{q}k",
             str(mp3)],
            check=True, capture_output=True,
        )
    return FileResponse(mp3, filename=mp3.name, media_type="audio/mpeg")


@app.delete("/api/gallery/{stem}")
def delete_track(stem: str):
    """Удаление записи: flac + json + кеш mp3 + партитура + латенты + производные
    (outputs/<stem>.d/ — превью, DSP-варианты, овердабы, кеши метрик)."""
    import re

    if not re.fullmatch(r"[\w.-]+", stem):
        raise HTTPException(422, "недопустимое имя")
    stem = stem[:-5] if stem.endswith(".flac") else stem
    removed = []
    for suf in (".flac", ".json", ".mp3", ".48k.mp3", ".abc", ".score.abc", ".latent.npy", ".wav"):
        p = OUT / f"{stem}{suf}"
        if p.is_file():
            p.unlink()
            removed.append(p.name)
    d = OUT / f"{stem}.d"
    if d.is_dir():
        shutil.rmtree(d)
        removed.append(f"{d.name}/")
    if not removed:
        raise HTTPException(404, "запись не найдена")
    return {"ok": True, "removed": removed}


@app.post("/api/gallery/{stem}/like")
def like_track(stem: str):
    """Отметка «понравилось»: liked в мете json (галерея его уже отдаёт)."""
    import re

    if not re.fullmatch(r"[\w.-]+", stem):
        raise HTTPException(422, "недопустимое имя")
    stem = stem[:-5] if stem.endswith(".flac") else stem
    meta_path = OUT / f"{stem}.json"
    if not meta_path.is_file():
        raise HTTPException(404, "запись не найдена")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["liked"] = not meta.get("liked", False)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {"liked": meta["liked"]}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    with JOBS_LOCK:
        job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404, "задача не найдена")
    return job


@app.get("/api/gallery")
def gallery():
    items = []
    for meta_path in sorted(OUT.glob("*.json"), reverse=True)[:100]:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["audio_url"] = f"/outputs/{meta['file']}"
            items.append(meta)
        except Exception:  # noqa: BLE001 — битые меты не ломают галерею
            continue
    return items


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
