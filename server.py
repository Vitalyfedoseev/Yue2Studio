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
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
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

    counters = {"phase": "план", "tokens": 0}

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
            _set(
                job["id"], stage=counters["phase"], tokens=counters["tokens"],
                elapsed_s=round(elapsed, 1), tok_per_s=round(tps, 1) if tps else None,
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

        t1 = time.time()
        plan = pipe.plan(**kw, on_token=on_token)
        counters["phase"] = "семантика"
        t2 = time.time()
        _set(job["id"], stage="семантика", plan_s=round(t2 - t1, 1))

        from yue2.protocol import CONTEXT, Sampling

        # контекст 24576 общий: префикс (стиль+лирика+ABC-план) + песня ≤ CONTEXT;
        # пайплайн не усекает молча — подгоняем бюджет под фактический префикс
        budget = max(200, min(MAX_SEM_TOKENS, CONTEXT - len(plan.prefix) - 4))
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
        }
        if job.get("abc"):
            (OUT / f"{stem}.abc").write_text(job["abc"], encoding="utf-8")
            meta["abc_file"] = f"{stem}.abc"   # сама партитура не в мете — бывает >100 КБ
            meta["abc_chars"] = len(job["abc"])
        (OUT / f"{stem}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        _set(job["id"], status="done", stage="done", **{"result": meta})
    except InterruptedError:
        _set(job["id"], status="error", error="остановлено пользователем")
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


@app.get("/api/health")
def health():
    vram_free_gb = None
    if torch.cuda.is_available():
        free_b, _ = torch.cuda.mem_get_info()
        vram_free_gb = round(free_b / 2**30, 1)
    node = find_node()
    return {
        "ready": ENGINE.pipe is not None and ENGINE.error is None,
        "detail": ENGINE.error or ENGINE.detail,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_free_gb": vram_free_gb,
        "queue": QUEUE.qsize(),
        "sampling_rate": SAMPLE_RATE,
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
            "abc": abc or None,
            "elapsed_s": 0.0, "tokens": 0, "tok_per_s": None,
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
    job["cancel"] = True
    return {"ok": True}


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
    """Удаление записи: flac + json + кеш mp3. stem — имя без расширения или .flac."""
    import re

    if not re.fullmatch(r"[\w.-]+", stem):
        raise HTTPException(422, "недопустимое имя")
    stem = stem[:-5] if stem.endswith(".flac") else stem
    removed = []
    for suf in (".flac", ".json", ".mp3", ".48k.mp3", ".abc"):
        p = OUT / f"{stem}{suf}"
        if p.is_file():
            p.unlink()
            removed.append(p.name)
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
