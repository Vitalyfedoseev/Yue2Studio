#!/usr/bin/env python3
"""Веб-морда локальной генерации песен: YuE2-3B (m-a-p).

FastAPI-бэкенд с однозадачной GPU-очередью. Модель ~4B в bf16 живёт в VRAM
резидентно (пик ~11–14 ГБ — влезает в 16 ГБ без оффлоадов), VAE fp32.

YuE2: AR–NAR Mixture-of-Transformers пишет ABC-нотный план + семантические
токены, flow-matching даёт акустические латенты, VAE — 48 кГц стерео FLAC.
Длина песни следует за лирикой (лимит 9000 токенов ≈ ~5 мин).
Отмена генерации — родная (pipe.plan/generate_semantic принимают cancelled).

Хранилище — SQLite (store.py): мета/партитуры/латенты/кеши в outputs/library.db,
на диске только аудио. Студия-редактор: волна (peaks), правки (обрезка/фейды/
гейн/цепочки), правка ABC, версии песни с флагом «основная», микшер овердаба.
"""
import hashlib
import io
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
import store
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
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
# Контекст модели 24576 общий; ~22k на песню ≈ 10–14 мин аудио,
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
        persist = {k: JOBS[jid].get(k)
                   for k in ("status", "stage", "error") if JOBS[jid].get(k)}
        res = JOBS[jid].get("result")
        if res and res.get("file"):
            persist["result_stem"] = res["file"][:-len(".flac")]
    try:  # история джоб не должна ломать генерацию
        store.job_update(jid, **persist)
    except Exception:  # noqa: BLE001
        pass


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
        stem = _alloc_result_stem(job, title)
        flac_path = OUT / f"{stem}.flac"
        sf.write(flac_path, audio, SAMPLE_RATE, subtype="PCM_24")
        # партитура и латенты — фундамент превью/овердаба/таймлайна;
        # float16 вдвое легче (~2 МБ/мин), decode вернём во float32
        z = latents.detach().cpu().numpy() if hasattr(latents, "detach") else latents
        buf = io.BytesIO()
        np.save(buf, np.asarray(z, dtype=np.float16))

        now = datetime.now().isoformat(timespec="seconds")
        row = {
            "stem": stem, "file": flac_path.name, "title": title,
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
            "ts": now, "created": now,
            "score_abc": plan.abc,
            "latent": buf.getvalue(),
            "arc": job.get("arc") or "",
            "draft": 1 if job.get("draft") else 0,
            "abc_text": (job.get("abc") or "").strip() or None,
            "abc_source": "cover" if job.get("abc") else None,
        }
        if job.get("version_parent"):
            prow = store.get_track(job["version_parent"])
            if prow:  # родитель могли удалить, пока шла генерация
                row["version_of"] = job["version_parent"]
                row["version_root"] = prow.get("version_root") or prow["stem"]
                row["version_label"] = job.get("version_label") or "рендер по ABC"
        if job.get("overdub_parent"):
            # овердаб: смешиваем с родителем сразу, ребёнок остаётся и отдельно
            row["overdub_of"] = job["overdub_parent"]
            try:
                mixed = _mix_overdub(
                    OUT / f"{job['overdub_parent']}.flac", flac_path,
                    job.get("overdub_gain", 0.5), job["overdub_parent"])
                row["overdub_file"] = mixed["file"]
            except Exception as e:  # noqa: BLE001 — неудачный микс не роняет джобу
                row["overdub_error"] = f"{type(e).__name__}: {e}"
        store.upsert_track(row)
        _set(job["id"], status="done", stage="done",
             result=store.track_to_api(store.get_track(stem)), result_stem=stem)
    except InterruptedError:
        _set(job["id"], status="canceled", error="остановлено пользователем")
    finally:
        stop_watch.set()


def _alloc_result_stem(job: dict, title: str) -> str:
    """Стем результата генерации: обычная — метка времени, ре-рендер версии —
    <корень>.v<n> (нумерация при завершении, без коллизий)."""
    if job.get("version_parent"):
        prow = store.get_track(job["version_parent"])
        if prow:
            root = prow.get("version_root") or prow["stem"]
            n = store.next_version_n(root)
            while (OUT / f"{root}.v{n}.flac").exists():
                n += 1
            return f"{root}.v{n}"
    return (f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{slugify(title) + '-' if slugify(title) else ''}{job['id'][:8]}")


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
    try:
        n = store.migrate_legacy()
        if n:
            print(f"[store] импортировано legacy-записей: {n}")
    except Exception as e:  # noqa: BLE001 — миграция не должна ронять сервер
        print(f"[store] миграция не удалась: {type(e).__name__}: {e}")
    try:
        store.mark_interrupted()
    except Exception:  # noqa: BLE001
        pass
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
    store.job_create(jid, {"kind": "generate", "title": req.title.strip()[:80],
                           "seed": req.seed, "cot": req.cot, "draft": req.draft,
                           "abc_chars": len(abc) if abc else 0})
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


def _abc_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _need_track(stem: str) -> dict:
    row = store.get_track(stem)
    if row is None:
        raise HTTPException(404, "запись не найдена")
    if not (OUT / row["file"]).is_file():
        raise HTTPException(404, "аудиофайл записи не найден")
    return row


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
    Правленая партитура (abc_text) приоритетнее плана модели; кеш в БД
    с хешем использованного ABC."""
    stem = _clean_stem(stem)
    row = _need_track(stem)
    abc_text = row.get("abc_text") or row.get("score_abc")
    if not abc_text:
        raise HTTPException(404, "партитура не найдена (запись без нотного плана)")
    h = _abc_hash(abc_text)
    cached = store.get_score_cache(stem, h)
    if cached:
        return cached
    import abcparse

    tl = abcparse.parse_abc(abc_text)
    tl["rms_sections"] = _rms_sections(OUT / row["file"], tl["bars"])
    store.put_score_cache(stem, h, tl)
    return tl


class PreviewRequest(BaseModel):
    from_sec: float = Field(..., ge=0)
    to_sec: float = Field(...)


@app.post("/api/gallery/{stem}/preview")
def track_preview(stem: str, req: PreviewRequest):
    """Превью фрагмента: срез латентов [from,to] сек → VAE-decode → flac
    в outputs/<stem>.d/. Клэмп по длительности звука (ABC бывает длиннее)."""
    stem = _clean_stem(stem)
    row = _need_track(stem)
    blob = row.get("latent")
    if not blob:
        raise HTTPException(404, "латенты не сохранены — запись сделана до появления превью")
    if ENGINE.error or ENGINE.pipe is None:
        raise HTTPException(503, f"модель не готова: {ENGINE.error or ENGINE.detail}")
    z = np.load(io.BytesIO(blob))
    total = row.get("duration_s") or len(z) / LATENT_HZ
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
    кеш в БД, ?fresh=true пересчитать."""
    stem = _clean_stem(stem)
    row = _need_track(stem)
    if not fresh:
        cached = store.get_metrics(stem)
        if cached:
            return cached
    try:
        import audio_metrics
    except ImportError:
        raise HTTPException(503, "librosa не установлена — метрики недоступны")
    try:
        m = audio_metrics.analyze_file(OUT / row["file"])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"анализ не удался: {type(e).__name__}: {e}")
    store.put_metrics(stem, m)
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
    """Legacy: DSP-варианты старого формата в <stem>.d/ (новые создают версии)."""
    stem = _clean_stem(stem)
    d = OUT / f"{stem}.d"
    variants = []
    if d.is_dir():
        for p in sorted(d.glob("dsp-*.flac")):
            item = {"file": f"{stem}.d/{p.name}", "url": f"/outputs/{stem}.d/{p.name}"}
            variants.append(item)
    return {"variants": variants}


@app.post("/api/gallery/{stem}/dsp")
def dsp_apply(stem: str, req: DspRequest):
    """preview=True — фрагмент с 20-й секунды (временный аудиофайл в <stem>.d/);
    иначе цепочка создаёт полноценную версию трека (в «Версиях песни»)."""
    stem = _clean_stem(stem)
    row = _need_track(stem)
    flac = OUT / row["file"]
    import dsp_chains

    d = OUT / f"{stem}.d"
    d.mkdir(exist_ok=True)
    try:
        if req.preview:
            name = f"dsp-preview-{req.chain}.flac"
            dst = d / name
            merged = dsp_chains.run_chain(
                flac, dst, req.chain, req.params,
                span=(dsp_chains.PREVIEW_START, dsp_chains.PREVIEW_DUR))
            return {"file": f"{stem}.d/{name}", "url": f"/outputs/{stem}.d/{name}",
                    "params": merged, "metrics": None}
        tmp = d / f"dsp-{uuid.uuid4().hex[:8]}.flac"
        merged = dsp_chains.run_chain(flac, tmp, req.chain, req.params)
    except KeyError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(500, f"ffmpeg: {e}")
    chain = dsp_chains.CHAINS_BY_ID[req.chain]
    ver = _create_version(stem, tmp, chain.name.lower())
    return {"stem": ver["stem"], "file": ver["file"], "url": ver["audio_url"],
            "label": ver.get("version_label"), "params": merged}


# ---------- версии песни ----------

def _root_of(row: dict) -> str:
    return row.get("version_root") or row["stem"]


def _create_version(src_stem: str, new_flac: Path, label: str,
                    latent_bytes: bytes | None = None,
                    timeline: "_Timeline | None" = None) -> dict:
    """Новая версия песни: flac переименовывается из tmp, строка в БД
    копируется с родителя; латенты (опц. трансформированные) и таймлайн
    адаптируются под трансформации времени."""
    import soundfile as sf

    src = _need_track(src_stem)
    prow = store.get_track(src_stem)
    root = _root_of(prow)
    # накопительная метка цепочки правок: «кассета» → «кассета + перегруз»
    plabel = src.get("version_label")
    if plabel:
        label = f"{plabel} + {label}"
    n = store.next_version_n(root)
    while (OUT / f"{root}.v{n}.flac").exists():
        n += 1
    stem = f"{root}.v{n}"
    dst = OUT / f"{stem}.flac"
    new_flac.rename(dst)
    info = sf.info(str(dst))
    now = datetime.now().isoformat(timespec="seconds")
    row = {
        "stem": stem, "file": dst.name,
        "title": src.get("title") or "",
        "duration_s": round(info.frames / info.samplerate, 2),
        "sample_rate": info.samplerate,
        "style": src.get("style") or "", "style_base": src.get("style_base") or "",
        "voice": src.get("voice") or "", "lyrics": src.get("lyrics") or "",
        "cot": src.get("cot") or "full", "cfg_scale": src.get("cfg_scale"),
        "seed": src.get("seed"), "arc": src.get("arc") or "",
        "draft": 1 if src.get("draft") else 0,
        "ts": now, "created": now,
        "abc_text": src.get("abc_text"), "abc_source": src.get("abc_source"),
        "score_abc": src.get("score_abc"),
        "latent": latent_bytes if latent_bytes is not None else src.get("latent"),
        "version_of": src_stem, "version_root": root, "version_n": n,
        "version_label": label, "group_main": 0,
    }
    store.upsert_track(row)
    try:
        import audio_metrics

        store.put_metrics(stem, audio_metrics.analyze_file(dst))
    except Exception:  # noqa: BLE001 — версия валидна и без метрик
        pass
    try:
        _adapt_score(src, stem, dst, timeline or _Timeline())
    except Exception:  # noqa: BLE001
        pass
    return store.track_to_api(store.get_track(stem))


class _Timeline:
    """Композиция трансформаций времени «запись → рабочий файл»: вырезы,
    вставки, дубли, обмены. Каждый op добавляет шаг; регионы последующих
    операций и таймлайн партитуры финальной версии прогоняются через fold."""
    def __init__(self):
        self.steps: list = []

    def map(self, t: float) -> float:
        for s in self.steps:
            t = s(t)
        return max(0.0, t)

    def add(self, step):
        self.steps.append(step)


def _adapt_score(src_row: dict, new_stem: str, new_flac: Path, timeline: "_Timeline"):
    """Таймлайн версии: партитура та же, такты прогоняются через трансформации
    времени (вырезы/вставки), rms пересчитывается по новому flac."""
    abc_text = src_row.get("abc_text") or src_row.get("score_abc")
    if not abc_text:
        return
    import abcparse

    tl = abcparse.parse_abc(abc_text)
    if timeline.steps:
        bars = []
        for b in tl["bars"]:
            s = timeline.map(b["start_sec"])
            e = timeline.map(b["end_sec"])
            if e - s >= 0.05:
                bars.append({**b, "start_sec": round(s, 2), "end_sec": round(e, 2)})
        tl["bars"] = bars
        tl["duration_sec"] = round(bars[-1]["end_sec"], 2) if bars else 0.0
    tl["rms_sections"] = _rms_sections(new_flac, tl["bars"])
    store.put_score_cache(new_stem, _abc_hash(abc_text), tl)


def _lat_slice_frames(lat: np.ndarray | None, wa: float, wb: float) -> np.ndarray | None:
    """Срез латента в рабочих секундах [wa, wb]."""
    if lat is None:
        return None
    a = max(0, min(int(wa * LATENT_HZ), len(lat)))
    b = max(a, min(int(wb * LATENT_HZ), len(lat)))
    return lat[a:b].copy()


class Region(BaseModel):
    from_sec: float = Field(..., ge=0)
    to_sec: float = Field(..., gt=0)


class EditOp(BaseModel):
    op: str = Field(..., pattern="^(trim-keep|trim-cut|fade|fade-region|gain|chain|"
                                "silence|duplicate|insert-from|swap)$")
    params: dict = Field(default_factory=dict)
    region: Optional[Region] = None
    region2: Optional[Region] = None   # второй фрагмент для swap


class EditRequest(BaseModel):
    """Набор изменений одной кнопкой: ops применяются по порядку, результат —
    ОДНА версия (preview=true — временный файл без версии). Регионы задаются
    в координатах записи и после вырезов/вставок пересчитываются автоматически."""
    ops: list[EditOp] = Field(default_factory=list, max_length=16)
    op: Optional[str] = None
    params: dict = Field(default_factory=dict)
    region: Optional[Region] = None
    label: str = Field("", max_length=120)
    preview: bool = Field(False, description="Рендер во временный файл, без версии")


def _clampf(v, lo: float, hi: float, dflt: float) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return dflt
    return min(hi, max(lo, v))


def _microfade(seg: np.ndarray, sr: int, ms: int = 10):
    n = min(len(seg), int(sr * ms / 1000))
    if n:
        ramp = np.linspace(0.0, 1.0, n)[:, None]
        seg[:n] *= ramp
        seg[-n:] *= ramp[::-1]


def _frames(lat: np.ndarray | None, t_w: float) -> int:
    if lat is None:
        return 0
    return max(0, min(int(round(t_w * LATENT_HZ)), len(lat)))


@app.post("/api/gallery/{stem}/edit")
def track_edit(stem: str, req: EditRequest):
    """Набор правок одним синхронным рендером: обрезка, фейды, гейн, цепочки,
    тишина, дубли, вставки, обмены. Результат — одна версия (или preview).
    Латенты трансформируются зеркально аудио, партитура — через fold времени."""
    stem = _clean_stem(stem)
    row = _need_track(stem)
    flac = OUT / row["file"]
    dur = row.get("duration_s") or 0.0
    ops = list(req.ops)
    if not ops and req.op:
        ops = [EditOp(op=req.op, params=req.params, region=req.region)]
    if not ops:
        raise HTTPException(422, "пустой набор изменений")
    import soundfile as sf

    d = OUT / f"{stem}.d"
    d.mkdir(exist_ok=True)

    labels: list[str] = []
    merged = None
    tl = _Timeline()                      # время записи → рабочий файл
    lat = None
    if row.get("latent"):
        try:
            lat = np.load(io.BytesIO(row["latent"])).copy()
        except Exception:  # noqa: BLE001 — правки аудио работают и без латентов
            lat = None

    def lat_bytes() -> bytes | None:
        if lat is None:
            return None
        buf = io.BytesIO()
        np.save(buf, np.asarray(lat, dtype=np.float16))
        return buf.getvalue()

    temps: list[Path] = []
    cur = flac
    data = None
    sr = None

    def _freeze() -> "callable":
        """Снимок текущего fold'а: шаги обязаны ссылаться на состояние ДО себя
        (иначе tl.map внутри шага — бесконечная рекурсия)."""
        steps = list(tl.steps)

        def f(t: float) -> float:
            for s in steps:
                t = s(t)
            return max(0.0, t)
        return f

    for eo in ops:
        nxt = d / f"edit-{uuid.uuid4().hex[:8]}.flac"
        temps.append(nxt)
        prev = _freeze()
        p = eo.params
        reg = None
        if eo.region:
            a = max(0.0, min(eo.region.from_sec, dur - 0.1))
            b = min(max(eo.region.to_sec, a + 0.1), dur)
            reg = (a, b)

        if eo.op in ("trim-keep", "trim-cut", "silence", "duplicate",
                     "fade-region", "gain", "fade"):
            data, sr = sf.read(str(cur), dtype="float32", always_2d=True)

        if eo.op in ("trim-keep", "trim-cut"):
            if reg is None:
                raise HTTPException(422, f"«{eo.op}»: нужен выделенный диапазон")
            wa, wb = prev(reg[0]), prev(reg[1])
            a_i = max(0, min(int(wa * sr), len(data) - 1))
            b_i = max(a_i + 1, min(int(wb * sr), len(data)))
            if eo.op == "trim-keep":
                out = data[a_i:b_i].copy()
                _microfade(out, sr)
                lat = _lat_slice_frames(lat, wa, wb)
                fa = wa
                tl.add(lambda t, pr=prev, fa=fa, a=a, b=b: pr(min(max(t, a), b)) - fa)
                labels.append(f"фрагмент {reg[0]:.0f}–{reg[1]:.0f} с")
            else:
                # кроссфейд 10 мс на стыке склейки — без щелчка
                xf = min(int(0.01 * sr), a_i, len(data) - b_i)
                if xf > 0:
                    wgt = np.linspace(0.0, 1.0, xf)[:, None]
                    joint = data[a_i - xf:a_i] * (1 - wgt) + data[b_i:b_i + xf] * wgt
                    out = np.concatenate([data[:a_i - xf], joint, data[b_i + xf:]])
                else:
                    out = np.concatenate([data[:a_i], data[b_i:]])
                la, lb = _frames(lat, wa), _frames(lat, wb)
                if lat is not None:
                    lat = np.concatenate([lat[:la], lat[lb:]])
                fa, fb = wa, wb
                tl.add(lambda t, pr=prev, fa=fa, fb=fb, a=a, b=b:
                       pr(t) if t < a else (fa if t < b else pr(t) - (fb - fa)))
                labels.append(f"вырезка {reg[0]:.0f}–{reg[1]:.1f} с")
            sf.write(str(nxt), out, sr, subtype="PCM_24")
        elif eo.op == "silence":
            if reg is None:
                raise HTTPException(422, "«тишина»: нужен выделенный диапазон")
            wa, wb = prev(reg[0]), prev(reg[1])
            a_i = max(0, min(int(wa * sr), len(data)))
            b_i = max(a_i, min(int(wb * sr), len(data)))
            data[a_i:b_i] = 0.0
            xr = min(int(0.005 * sr), a_i, len(data) - b_i)
            if xr > 0:
                data[a_i - xr:a_i] *= np.linspace(1.0, 0.0, xr)[:, None]
                data[b_i:b_i + xr] *= np.linspace(0.0, 1.0, xr)[:, None]
            sf.write(str(nxt), data, sr, subtype="PCM_24")
            labels.append(f"тишина {reg[0]:.0f}–{reg[1]:.0f} с")
        elif eo.op == "duplicate":
            if reg is None:
                raise HTTPException(422, "«дубликат»: нужен выделенный диапазон")
            n_times = max(1, min(8, int(_clampf(p.get("times"), 1, 8, 1))))
            wa, wb = prev(reg[0]), prev(reg[1])
            a_i = max(0, min(int(wa * sr), len(data) - 1))
            b_i = max(a_i + 1, min(int(wb * sr), len(data)))
            seg = data[a_i:b_i]
            out = np.concatenate([data[:b_i], np.tile(seg, (n_times, 1)), data[b_i:]])
            la, lb = _frames(lat, wa), _frames(lat, wb)
            if lat is not None and lb > la:
                lseg = lat[la:lb]
                lat = np.concatenate([lat[:lb], np.tile(lseg, (n_times, 1)), lat[lb:]])
            fb = wb
            add_len = (wb - wa) * n_times
            tl.add(lambda t, pr=prev, fb=fb, add_len=add_len, b=b:
                   pr(t) if t <= b else pr(t) + add_len)
            sf.write(str(nxt), out, sr, subtype="PCM_24")
            labels.append(f"дубликат {reg[0]:.0f}–{reg[1]:.0f} с ×{n_times}")
        elif eo.op == "insert-from":
            try:
                s1 = max(0.0, _clampf(p.get("src_from"), 0, dur, 0.0))
                s2 = min(dur, max(s1 + 0.1, _clampf(p.get("src_to"), 0, dur, s1 + 1.0)))
                at = max(0.0, min(_clampf(p.get("at"), 0, dur, 0.0), dur))
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(422, "«вставка»: нужны src_from/src_to/at")
            data, sr = sf.read(str(cur), dtype="float32", always_2d=True)
            ws1, ws2, wat = prev(s1), prev(s2), prev(at)
            i1 = max(0, min(int(ws1 * sr), len(data) - 1))
            i2 = max(i1 + 1, min(int(ws2 * sr), len(data)))
            j = max(0, min(int(wat * sr), len(data)))
            seg = data[i1:i2]
            out = np.concatenate([data[:j], seg, data[j:]])
            f1, f2, fj = _frames(lat, ws1), _frames(lat, ws2), _frames(lat, wat)
            if lat is not None and f2 > f1:
                lat = np.concatenate([lat[:fj], lat[f1:f2], lat[fj:]])
            ins_len = ws2 - ws1
            tl.add(lambda t, pr=prev, at=at, ins_len=ins_len:
                   pr(t) if t <= at else pr(t) + ins_len)
            sf.write(str(nxt), out, sr, subtype="PCM_24")
            if abs(s1 - at) < 0.05 and abs(s2 - s1) > 0:
                labels.append(f"дубликат {s1:.0f}–{s2:.0f} с")
            else:
                labels.append(f"вставка {s1:.0f}–{s2:.0f} с → {at:.0f} с")
        elif eo.op == "swap":
            r1, r2 = eo.region, eo.region2
            if not r1 or not r2:
                raise HTTPException(422, "«поменять местами»: нужны два фрагмента")
            a1 = max(0.0, min(r1.from_sec, r2.from_sec, dur - 0.2))
            b1 = min(max(r1.to_sec if r1.from_sec <= r2.from_sec else r2.to_sec,
                         a1 + 0.1), dur)
            a2 = max(b1, min(max(r1.from_sec, r2.from_sec), dur - 0.1))
            b2 = min(max(r1.to_sec if r1.from_sec > r2.from_sec else r2.to_sec,
                         a2 + 0.1), dur)
            data, sr = sf.read(str(cur), dtype="float32", always_2d=True)
            m1a, m1b, m2a, m2b = prev(a1), prev(b1), prev(a2), prev(b2)
            i1a, i1b = int(m1a * sr), int(m1b * sr)
            i2a, i2b = int(m2a * sr), int(m2b * sr)
            i1b = max(i1a + 1, min(i1b, len(data)))
            i2b = max(i2a + 1, min(i2b, len(data)))
            A, B = data[i1a:i1b].copy(), data[i2a:i2b].copy()
            out = np.concatenate([data[:i1a], B, data[i1b:i2a], A, data[i2b:]])
            if lat is not None:
                la1, lb1 = _frames(lat, m1a), _frames(lat, m1b)
                la2, lb2 = _frames(lat, m2a), _frames(lat, m2b)
                LA, LB = lat[la1:lb1].copy(), lat[la2:lb2].copy()
                lat = np.concatenate([lat[:la1], LB, lat[lb1:la2], LA, lat[lb2:]])
            tl.add(lambda t, pr=prev, a1=a1, b1=b1, a2=a2,
                   f1a=m1a, f1b=m1b, f2a=m2a, f2b=m2b: (
                       pr(t) if t < a1 else
                       (f2b - f1b + pr(t)) if t < b1 else
                       (f1a + (f2b - f2a) + (pr(t) - f1b)) if t < a2 else
                       (f1a + (pr(t) - f2a)) if t < b2 else pr(t)))
            sf.write(str(nxt), out, sr, subtype="PCM_24")
            labels.append(f"поменять {a1:.0f}–{b1:.0f} ↔ {a2:.0f}–{b2:.0f} с")
        elif eo.op == "fade":
            fi = _clampf(p.get("fade_in"), 0, 30, 1.0)
            fo = _clampf(p.get("fade_out"), 0, 30, 2.0)
            out = data.copy()
            ni, no = int(fi * sr), int(fo * sr)
            if ni > 0:
                out[:ni] *= np.linspace(0.0, 1.0, ni)[:, None]
            if no > 0:
                out[-no:] *= np.linspace(1.0, 0.0, no)[:, None]
            sf.write(str(nxt), out, sr, subtype="PCM_24")
            labels.append("фейды")
        elif eo.op == "fade-region":
            if reg is None:
                raise HTTPException(422, "«фейды на выделении»: нужен диапазон")
            fi = _clampf(p.get("fade_in"), 0, 30, 0.5)
            fo = _clampf(p.get("fade_out"), 0, 30, 0.5)
            wa, wb = prev(reg[0]), prev(reg[1])
            a_i = max(0, min(int(wa * sr), len(data)))
            b_i = max(a_i, min(int(wb * sr), len(data)))
            ni = min(int(fi * sr), (b_i - a_i) // 2 or 1)
            no = min(int(fo * sr), (b_i - a_i) // 2 or 1)
            if ni > 0:
                data[a_i:a_i + ni] *= np.linspace(0.0, 1.0, ni)[:, None]
            if no > 0:
                data[b_i - no:b_i] *= np.linspace(1.0, 0.0, no)[:, None]
            sf.write(str(nxt), data, sr, subtype="PCM_24")
            labels.append(f"фейды на {reg[0]:.0f}–{reg[1]:.0f} с")
        elif eo.op == "gain":
            db = _clampf(p.get("db"), -24, 24, 0.0)
            g = 10 ** (db / 20)
            out = data.copy()
            if reg is None:
                out *= g
                labels.append(f"гейн {db:+.1f} дБ")
            else:
                # плавный контур гейна на краях региона — без щелчков на стыках
                env = np.full(len(out), 1.0)
                a_i = max(0, min(int(prev(reg[0]) * sr), len(out) - 1))
                b_i = max(a_i + 1, min(int(prev(reg[1]) * sr), len(out)))
                xf = min(int(0.01 * sr), (b_i - a_i) // 2 or 1)
                env[a_i:b_i] = g
                env[a_i:a_i + xf] = np.linspace(1.0, g, xf)
                env[b_i - xf:b_i] = np.linspace(g, 1.0, xf)
                out *= env[:, None]
                labels.append(f"гейн {db:+.1f} дБ ({reg[0]:.0f}–{reg[1]:.0f} с)")
            sf.write(str(nxt), out, sr, subtype="PCM_24")
        elif eo.op == "chain":
            import dsp_chains

            chain_id = str(p.get("chain") or "tape")
            if chain_id not in dsp_chains.CHAINS_BY_ID:
                raise HTTPException(422, f"неизвестная цепочка: {chain_id}")
            cparams = {k: v for k, v in p.items() if k != "chain"}
            try:
                if reg is None:
                    merged = dsp_chains.run_chain(cur, nxt, chain_id, cparams)
                else:
                    a_w = max(0.0, prev(reg[0]))
                    b_w = prev(reg[1])
                    frag = d / f"edit-frag-{uuid.uuid4().hex[:8]}.flac"
                    merged = dsp_chains.run_chain(cur, frag, chain_id, cparams,
                                                  span=(a_w, b_w - a_w))
                    base, csr = sf.read(str(cur), dtype="float32", always_2d=True)
                    proc, psr = sf.read(str(frag), dtype="float32", always_2d=True)
                    if psr != csr:
                        raise RuntimeError("частота обработанного фрагмента не совпала")
                    sr = csr
                    a_i = max(0, min(int(a_w * sr), len(base) - 1))
                    b_i = max(a_i + 1, min(int(b_w * sr), len(base)))
                    n = min(len(proc), b_i - a_i)
                    xf = min(int(0.01 * sr), n // 2 or 1)
                    wgt = np.ones(n)
                    wgt[:xf] = np.linspace(0.0, 1.0, xf)
                    wgt[n - xf:] = np.linspace(1.0, 0.0, xf)
                    base[a_i:a_i + n] = proc[:n] * wgt[:, None] + base[a_i:a_i + n] * (1 - wgt[:, None])
                    sf.write(str(nxt), base, sr, subtype="PCM_24")
                    frag.unlink(missing_ok=True)
            except KeyError as e:
                raise HTTPException(422, str(e))
            except RuntimeError as e:
                raise HTTPException(500, f"ffmpeg: {e}")
            name = dsp_chains.CHAINS_BY_ID[chain_id].name.lower()
            labels.append(name if reg is None else f"{name} ({reg[0]:.0f}–{reg[1]:.0f} с)")
        cur = nxt

    if req.preview:
        # «прослушать, что получится» — без создания версии
        for t in temps[:-1]:
            t.unlink(missing_ok=True)
        prev = d / "edit-preview.flac"
        cur.rename(prev)
        info = sf.info(str(prev))
        return {"preview": True, "url": f"/outputs/{stem}.d/{prev.name}",
                "duration_s": round(info.frames / info.samplerate, 2),
                "label": (req.label.strip() or " + ".join(labels))[:120]}

    for t in temps[:-1]:
        t.unlink(missing_ok=True)
    label = req.label.strip() or " + ".join(labels)
    ver = _create_version(stem, cur, label[:120], latent_bytes=lat_bytes(), timeline=tl)
    return {"stem": ver["stem"], "file": ver["file"], "url": ver["audio_url"],
            "label": ver.get("version_label"), "version_n": ver.get("version_n"),
            "params": merged}


@app.get("/api/gallery/{stem}/peaks")
def track_peaks(stem: str, n: int = 2400,
                from_sec: float = 0.0, to_sec: Optional[float] = None):
    """Пики волны: min/max/rms по n бакетам (для канваса студии).
    Полный трек кешируется в БД по n; зум-окна считаются на лету."""
    stem = _clean_stem(stem)
    row = _need_track(stem)
    n = min(8000, max(32, int(n)))
    full = from_sec <= 0 and to_sec is None
    if full:
        cached = store.get_peaks(stem, n)
        if cached:
            return cached
    import soundfile as sf

    flac = OUT / row["file"]
    with sf.SoundFile(str(flac)) as f:
        sr, frames = f.samplerate, len(f)
        a = int(from_sec * sr)
        b = frames if to_sec is None else min(frames, int(to_sec * sr))
        a = max(0, min(a, frames - 1))
        b = max(a + 1, min(b, frames))
        f.seek(a)
        data = f.read(b - a, dtype="float32", always_2d=True)
    bucket = max(1, data.shape[0] // n)
    m = (data.shape[0] // bucket) * bucket
    view = data[:m].reshape(-1, bucket, data.shape[1])
    payload = {
        "duration_s": row.get("duration_s") or round(frames / sr, 2),
        "from_sec": round(a / sr, 3),
        "to_sec": round(b / sr, 3),
        "min": view.min(axis=(1, 2)).round(4).tolist(),
        "max": view.max(axis=(1, 2)).round(4).tolist(),
        "rms": np.sqrt((view.mean(axis=2) ** 2).mean(axis=1)).round(4).tolist(),
    }
    if full:
        store.put_peaks(stem, n, payload)
    return payload


@app.post("/api/gallery/{stem}/set-main")
def track_set_main(stem: str):
    """Флаг «основная» версии песни (внутри группы корень + версии)."""
    stem = _clean_stem(stem)
    row = _need_track(stem)
    root = _root_of(row)
    store.set_main(root, stem)
    return {"ok": True, "main": stem}


# ---------- ABC готовой записи: посмотреть/править/перегенерировать ----------

@app.get("/api/gallery/{stem}/abc")
def track_abc(stem: str):
    stem = _clean_stem(stem)
    row = _need_track(stem)
    text = row.get("abc_text") or row.get("score_abc")
    if not text:
        raise HTTPException(404, "у записи нет партитуры")
    import abcparse

    tl = abcparse.parse_abc(text)
    return {
        "abc": text,
        "source": "edited" if row.get("abc_text") else "model",
        "abc_source": row.get("abc_source"),
        "duration_sec": tl.get("duration_sec"),
        "audio_duration": row.get("duration_s"),
        "tempo_bpm": tl.get("tempo_bpm"),
    }


class AbcRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=400_000)


@app.post("/api/gallery/{stem}/abc")
def track_abc_save(stem: str, req: AbcRequest):
    """Сохранить правленую партитуру (план модели в БД не трогается),
    сбросить кеш таймлайна, вернуть свежий таймлайн с rms."""
    stem = _clean_stem(stem)
    row = _need_track(stem)
    import abcparse

    text = req.text.strip()
    try:
        tl = abcparse.parse_abc(text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"ABC не разобран: {type(e).__name__}: {e}")
    if not tl.get("bars"):
        raise HTTPException(422, "в партитуре нет тактов — проверьте нотацию")
    store.update_track(stem, abc_text=text, abc_source="edit")
    store.del_score_cache(stem)
    tl["rms_sections"] = _rms_sections(OUT / row["file"], tl["bars"])
    store.put_score_cache(stem, _abc_hash(text), tl)
    return tl


@app.post("/api/gallery/{stem}/abc/reset")
def track_abc_reset(stem: str):
    """Вернуть план модели (правка стирается)."""
    stem = _clean_stem(stem)
    row = _need_track(stem)
    if not row.get("score_abc"):
        raise HTTPException(404, "плана модели нет — сбрасывать нечего")
    store.update_track(stem, abc_text=None, abc_source=None)
    store.del_score_cache(stem)
    return track_score(stem)


@app.get("/api/gallery/{stem}/abc/download")
def track_abc_download(stem: str):
    stem = _clean_stem(stem)
    row = _need_track(stem)
    text = row.get("abc_text") or row.get("score_abc")
    if not text:
        raise HTTPException(404, "у записи нет партитуры")
    name = (slugify(row.get("title") or "") or stem)[:60] + ".abc"
    return Response(
        text, media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


class RerenderRequest(BaseModel):
    abc: str = Field("", max_length=400_000, description="Пусто — взять сохранённую партитуру")
    draft: bool = Field(False, description="Черновик ~15–20 с")


@app.post("/api/gallery/{stem}/rerender")
def track_rerender(stem: str, req: RerenderRequest):
    """Ре-рендер песни по (правленой) партитуре: стиль/голос/лирика/seed/cfg
    берутся из записи, cot=melody. Результат — новая версия песни."""
    stem = _clean_stem(stem)
    row = _need_track(stem)
    score = (req.abc or "").strip() or row.get("abc_text") or row.get("score_abc")
    if not score:
        raise HTTPException(422, "у записи нет партитуры — ре-рендер невозможен")
    lyrics = (row.get("lyrics") or "").strip()
    if len(lyrics) < 2:
        raise HTTPException(422, "у записи нет лирики")
    if ENGINE.error:
        raise HTTPException(503, f"модель не загрузилась: {ENGINE.error}")
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[jid] = {
            "id": jid, "status": "queued", "stage": "в очереди",
            "style": row.get("style") or "", "style_base": row.get("style_base") or "",
            "voice": row.get("voice") or "", "title": row.get("title") or "",
            "lyrics": lyrics, "cot": "melody", "cfg_scale": row.get("cfg_scale"),
            "seed": row.get("seed") if row.get("seed") is not None else -1,
            "abc": score, "arc": "", "draft": req.draft,
            "version_parent": stem,
            "version_label": "рендер по ABC" + (" (черновик)" if req.draft else ""),
            "elapsed_s": 0.0, "tokens": 0, "tok_per_s": None, "pct": None,
            "cancel": False, "created": time.time(),
        }
    store.job_create(jid, {"kind": "rerender", "parent": stem,
                           "title": row.get("title"), "draft": req.draft})
    QUEUE.put(jid)
    return {"id": jid, "parent": stem}


# ---------- микшер слоёв: овердабы + импортированное аудио → версия-микс ----------

@app.get("/api/gallery/{stem}/lanes")
def track_lanes(stem: str):
    """Дорожки-слои: овердаб-партии и импортированное аудио этой песни."""
    stem = _clean_stem(stem)
    _need_track(stem)
    lanes = [{"stem": c["stem"], "title": c.get("title") or c["stem"],
              "duration_s": c.get("duration_s"),
              "kind": "overdub" if c.get("overdub_of") else "import",
              "url": f"/outputs/{c['file']}"}
             for c in store.lanes_rows(stem)]
    return {"lanes": lanes}


@app.post("/api/gallery/{stem}/import-lane")
async def track_import_lane(stem: str, request: Request):
    """Импорт своего аудио (mp3/wav/flac/…) как дорожки-слоя песни.
    Тело — сырые байты файла; ?title= — подпись дорожки."""
    from fastapi.concurrency import run_in_threadpool

    stem = _clean_stem(stem)
    _need_track(stem)
    data = await request.body()
    if not data:
        raise HTTPException(422, "пустое тело — приложите аудиофайл")
    if len(data) > 100 * 2**20:
        raise HTTPException(422, "файл больше 100 МБ")
    title = (request.query_params.get("title") or "").replace("\n", " ").strip()[:80]

    def _import() -> dict:
        import subprocess
        import tempfile

        now = datetime.now()
        new_stem = (f"{now.strftime('%Y%m%d-%H%M%S')}-"
                    f"{slugify(title) or 'import'}-{uuid.uuid4().hex[:6]}")
        dst = OUT / f"{new_stem}.flac"
        with tempfile.TemporaryDirectory(prefix="imp-") as tmp:
            src = Path(tmp) / "in"
            src.write_bytes(data)
            proc = subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(src), "-ar", "48000", "-ac", "2",
                 "-codec:a", "flac", "-sample_fmt", "s32",
                 "-bits_per_raw_sample", "24", str(dst)],
                capture_output=True, timeout=300)
        if proc.returncode != 0 or not dst.is_file():
            err = proc.stderr.decode(errors="replace").strip().splitlines()[-1:]
            raise HTTPException(422, "аудио не разобрано: " +
                                (err[0] if err else f"exit {proc.returncode}"))
        import soundfile as sf

        info = sf.info(str(dst))
        now_iso = now.isoformat(timespec="seconds")
        store.upsert_track({
            "stem": new_stem, "file": dst.name,
            "title": title or "импорт",
            "duration_s": round(info.frames / info.samplerate, 2),
            "sample_rate": info.samplerate,
            "ts": now_iso, "created": now_iso, "lane_of": stem,
        })
        return {"stem": new_stem, "title": title or "импорт",
                "duration_s": round(info.frames / info.samplerate, 2),
                "url": f"/outputs/{dst.name}"}

    return await run_in_threadpool(_import)


class CompPart(BaseModel):
    stem: str
    from_sec: float = Field(..., ge=0)
    to_sec: float = Field(..., gt=0)


class CompRequest(BaseModel):
    parts: list[CompPart] = Field(..., min_length=1, max_length=16)
    label: str = Field("", max_length=120)


@app.post("/api/gallery/{stem}/comp")
def track_comp(stem: str, req: CompRequest):
    """Компинг: склейка фрагментов разных версий одной песни в новую версию.
    Латент собирается из латентов частей, таймлайн секций — best-effort."""
    import soundfile as sf

    stem = _clean_stem(stem)
    row = _need_track(stem)
    root = _root_of(row)
    d = OUT / f"{stem}.d"
    d.mkdir(exist_ok=True)
    tmp = d / f"comp-{uuid.uuid4().hex[:8]}.flac"

    audio_parts, lat_parts, bar_parts = [], [], []
    offset = 0.0
    names = []
    import abcparse

    for part in req.parts:
        pstem = _clean_stem(part.stem)
        prow = store.get_track(pstem)
        if prow is None or _root_of(prow) != root:
            raise HTTPException(422, f"{part.stem} — не из этой песни")
        pdur = prow.get("duration_s") or 0.0
        a = max(0.0, min(part.from_sec, pdur - 0.1))
        b = min(max(part.to_sec, a + 0.1), pdur)
        seg, sr = sf.read(str(OUT / prow["file"]), dtype="float32",
                          always_2d=True, start=int(a * 48000), stop=int(b * 48000))
        audio_parts.append(seg)
        if prow.get("latent"):
            try:
                z = np.load(io.BytesIO(prow["latent"]))
                la, lb = int(a * LATENT_HZ), int(b * LATENT_HZ)
                lat_parts.append(z[max(0, la):max(la + 1, min(lb, len(z)))])
            except Exception:  # noqa: BLE001
                pass
        # таймлайн части: такты, попавшие в диапазон, со сдвигом на offset
        pabc = prow.get("abc_text") or prow.get("score_abc")
        if pabc:
            try:
                ptl = abcparse.parse_abc(pabc)
                for bar in ptl["bars"]:
                    if bar["end_sec"] <= a or bar["start_sec"] >= b:
                        continue
                    s = max(a, bar["start_sec"]) - a + offset
                    e = min(b, bar["end_sec"]) - a + offset
                    if e - s >= 0.05:
                        bar_parts.append({**bar, "start_sec": round(s, 2),
                                          "end_sec": round(e, 2)})
            except Exception:  # noqa: BLE001
                pass
        names.append(("ориг" if pstem == root else
                      f"v{prow.get('version_n') or '?'}") + f"[{a:.0f}–{b:.0f}]")
        offset += b - a
    if not audio_parts:
        raise HTTPException(422, "пустой комп")
    out = np.concatenate(audio_parts)
    peak = float(np.abs(out).max())
    if peak > 1.0:
        out /= peak
    sf.write(str(tmp), out, sr, subtype="PCM_24")
    lat = None
    if lat_parts:
        lat = np.concatenate(lat_parts)
        buf = io.BytesIO()
        np.save(buf, np.asarray(lat, dtype=np.float16))
        lat = buf.getvalue()
    label = req.label.strip() or "комп: " + " + ".join(names)
    ver = _create_version(stem, tmp, label[:120], latent_bytes=lat)
    if bar_parts:  # свой таймлайн вместо унаследованного
        bar_parts.sort(key=lambda x: x["start_sec"])
        tlc = {"tempo_bpm": 0, "key": "", "meter": "4/4", "unit": 1,
               "voices": {}, "voice_order": [], "bars": bar_parts,
               "duration_sec": round(offset, 2),
               "rms_sections": _rms_sections(OUT / f"{ver['stem']}.flac", bar_parts)}
        store.put_score_cache(ver["stem"], "comp:" + _abc_hash(label), tlc)
    return {"stem": ver["stem"], "file": ver["file"], "url": ver["audio_url"],
            "label": ver.get("version_label"), "version_n": ver.get("version_n")}


@app.get("/api/gallery/{stem}/export")
def track_export(stem: str, fmt: str = "flac",
                 from_sec: float = 0.0, to_sec: Optional[float] = None):
    """Экспорт диапазона в файл (по умолчанию вся песня), без создания версии."""
    import subprocess

    if fmt not in ("flac", "mp3"):
        raise HTTPException(422, "fmt: flac или mp3")
    stem = _clean_stem(stem)
    row = _need_track(stem)
    dur = row.get("duration_s") or 0.0
    a = max(0.0, min(from_sec, dur - 0.5))
    b = min(dur if to_sec is None else max(a + 0.5, min(to_sec, dur)), dur)
    d = OUT / f"{stem}.d"
    d.mkdir(exist_ok=True)
    name = f"export-{a:.0f}-{b:.1f}.{fmt}"
    out = d / name
    if not out.is_file():
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", "-i", str(OUT / row["file"])]
        if fmt == "mp3":
            cmd += ["-codec:a", "libmp3lame", "-b:a", "320k"]
        cmd.append(str(out))
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise HTTPException(500, "ffmpeg: экспорт не удался")
    base = slugify(row.get("title") or "") or stem
    fname = f"{base} [{a:.0f}-{b:.0f}c].{fmt}"
    return FileResponse(out, filename=fname,
                        media_type="audio/flac" if fmt == "flac" else "audio/mpeg")


@app.get("/api/gallery/{stem}/export-multitrack")
def track_export_multitrack(stem: str):
    """Мультитрек-экспорт: zip с основным треком и всеми слоями (flac)."""
    import zipfile

    stem = _clean_stem(stem)
    row = _need_track(stem)
    lanes = store.lanes_rows(stem)
    files = [(row.get("title") or "main", OUT / row["file"])]
    for l in lanes:
        f = OUT / l["file"]
        if f.is_file():
            files.append((l.get("title") or l["stem"], f))
    d = OUT / f"{stem}.d"
    d.mkdir(exist_ok=True)
    zpath = d / "multitrack.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        used = set()
        for name, f in files:
            arc = (slugify(name) or "track") + ".flac"
            k = 2
            while arc in used:
                arc = f"{slugify(name)}-{k}.flac"
                k += 1
            used.add(arc)
            z.write(f, arcname=arc)
    base = slugify(row.get("title") or "") or stem
    return FileResponse(zpath, filename=f"{base} (multitrack).zip",
                        media_type="application/zip")


class Lane(BaseModel):
    stem: str
    gain: float = Field(1.0, ge=0.0, le=2.0)
    mute: bool = False


class MixdownRequest(BaseModel):
    lanes: list[Lane] = Field(..., min_length=1,
                              description="Первая дорожка — сам трек")
    label: str = Field("", max_length=120)


@app.post("/api/gallery/{stem}/mixdown")
def track_mixdown(stem: str, req: MixdownRequest):
    """Собрать микс из слоёв (трек + овердабы с гейнами) → новая версия.
    Обобщение _mix_overdub: микс до самой длинной дорожки, фейд 50 мс
    в конце, пик-нормализация."""
    stem = _clean_stem(stem)
    main = _need_track(stem)
    if req.lanes[0].stem != stem:
        raise HTTPException(422, "первая дорожка микса — сам трек")
    import soundfile as sf

    parts: list[tuple[Lane, Path, str]] = []   # (lane, flac, подпись)
    for lane in req.lanes:
        if lane.stem == stem:
            parts.append((lane, OUT / main["file"], main.get("title") or stem))
            continue
        child = store.get_track(lane.stem)
        if child is None or (child.get("overdub_of") != stem and
                             child.get("lane_of") != stem):
            raise HTTPException(422, f"{lane.stem} — не слой этого трека")
        f = OUT / child["file"]
        if not f.is_file():
            raise HTTPException(404, f"аудио {lane.stem} не найдено")
        parts.append((lane, f, child.get("title") or lane.stem))
    loaded = []
    sr = None
    for lane, f, _t in parts:
        if lane.mute:
            continue
        a, asr = sf.read(str(f), dtype="float32", always_2d=True)
        if sr is None:
            sr = asr
        elif asr != sr:
            raise HTTPException(422, "частоты дорожек не совпали")
        loaded.append((a, lane.gain))
    if not loaded:
        raise HTTPException(422, "все дорожки выключены — миксовать нечего")
    n = max(len(a) for a, _ in loaded)
    mix = np.zeros((n, loaded[0][0].shape[1]), dtype=np.float32)
    for a, g in loaded:
        mix[:len(a)] += a * g
    fade = min(int(0.05 * sr), n)
    if fade:
        mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
    peak = float(np.abs(mix).max())
    if peak > 1.0:
        mix /= peak
    d = OUT / f"{stem}.d"
    d.mkdir(exist_ok=True)
    tmp = d / f"mix-{uuid.uuid4().hex[:8]}.flac"
    sf.write(str(tmp), mix, sr, subtype="PCM_24")
    names = " + ".join(t for lane, _f, t in parts if not lane.mute and lane.stem != stem)
    label = req.label.strip() or ("микс: " + names if names else "микс (соло)")
    ver = _create_version(stem, tmp, label[:120])
    return {"stem": ver["stem"], "file": ver["file"], "url": ver["audio_url"],
            "label": ver.get("version_label"), "version_n": ver.get("version_n")}


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
    instrumental: bool = Field(False, description="Без вокала: лирика заменяется [Instrumental]-секциями по партитуре")
    gain: float = Field(0.5, ge=0.05, le=1.0)
    seed: int = Field(-1, ge=-1)
    title: str = Field("")


@app.post("/api/gallery/{stem}/overdub")
def track_overdub(stem: str, req: OverdubRequest):
    """Овердаб: джоба-потомок по партитуре записи (из БД); после рендера
    автоматически смешивается с родителем (gain)."""
    stem = _clean_stem(stem)
    row = _need_track(stem)
    abc_text = row.get("abc_text") or row.get("score_abc")
    if not abc_text:
        raise HTTPException(404, "у записи нет партитуры — овердаб невозможен")
    if ENGINE.error:
        raise HTTPException(503, f"модель не загрузилась: {ENGINE.error}")
    if req.instrumental:
        # инструментал: теги [Instrumental] по числу секций партитуры —
        # партия без вокала, идущая по той же структуре, что и трек
        import abcparse

        try:
            tl = abcparse.parse_abc(abc_text)
            seen, n = set(), 0
            for b in tl["bars"]:
                if b["section"] not in seen:
                    seen.add(b["section"])
                    n += 1
            n = n or 4
        except Exception:  # noqa: BLE001 — партитура не разобралась, возьмём 4
            n = 4
        lyrics = "\n".join(["[Instrumental]"] * max(1, min(16, n)))
    else:
        lyrics = req.lyrics.strip()
        if not lyrics:
            lyrics = (row.get("lyrics") or "").strip()
        if len(lyrics) < 2:
            raise HTTPException(422, "нет лирики ни в запросе, ни у родителя")
    jid = uuid.uuid4().hex[:12]
    style_base = req.style.strip()
    title = (req.title.strip() or
             f"овердаб{' (инструментал)' if req.instrumental else ''} · {stem}")[:80]
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
    store.job_create(jid, {"kind": "overdub", "parent": stem, "title": title,
                           "gain": req.gain, "seed": req.seed})
    QUEUE.put(jid)
    return {"id": jid, "parent": stem}


# ---------- переименование (только мета в БД) ----------

class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1)


@app.post("/api/gallery/{stem}/rename")
def rename_track(stem: str, req: RenameRequest):
    stem = _clean_stem(stem)
    if store.get_track(stem) is None:
        raise HTTPException(404, "запись не найдена")
    title = req.title.strip()[:80]
    store.update_track(stem, title=title)
    return {"title": title}


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
    """Удаление: аудиофайлы (flac + кеши конверсий + производное аудио в
    <stem>.d/) и строки БД. Удаление корня каскадно удаляет все версии песни."""
    import re

    if not re.fullmatch(r"[\w.-]+", stem):
        raise HTTPException(422, "недопустимое имя")
    stem = stem[:-5] if stem.endswith(".flac") else stem
    row = store.get_track(stem)
    if row is None:
        raise HTTPException(404, "запись не найдена")
    root = _root_of(row)
    if stem == root:
        stems = [r["stem"] for r in store.group_rows(root)]
    else:
        stems = [stem]
    removed = []
    for s in stems:
        for suf in (".flac", ".wav", ".mp3", ".48k.mp3"):
            p = OUT / f"{s}{suf}"
            if p.is_file():
                p.unlink()
                removed.append(p.name)
        d = OUT / f"{s}.d"
        if d.is_dir():
            shutil.rmtree(d)
            removed.append(f"{d.name}/")
    store.delete_tracks(stems)
    return {"ok": True, "removed": removed, "stems": stems}


@app.post("/api/gallery/{stem}/like")
def like_track(stem: str):
    """Отметка «понравилось»: liked в БД (галерея его уже отдаёт)."""
    import re

    if not re.fullmatch(r"[\w.-]+", stem):
        raise HTTPException(422, "недопустимое имя")
    stem = stem[:-5] if stem.endswith(".flac") else stem
    row = store.get_track(stem)
    if row is None:
        raise HTTPException(404, "запись не найдена")
    liked = not bool(row.get("liked"))
    store.update_track(stem, liked=1 if liked else 0)
    return {"liked": liked}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    with JOBS_LOCK:
        job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404, "задача не найдена")
    return job


@app.get("/api/gallery")
def gallery():
    return store.list_tracks()


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/studio")
def studio():
    """Студия-редактор: волна, правки звука, ABC, версии песни."""
    return FileResponse(BASE / "static" / "studio.html")


@app.get("/abc")
def abc_editor():
    """Автономный ABC-редактор: текст, ноты, проигрывание — без привязки к песне."""
    return FileResponse(BASE / "static" / "abc.html")


app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
