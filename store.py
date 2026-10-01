"""SQLite-хранилище библиотеки: мета, партитуры, латенты, кеши, джобы.

На диске остаются только аудиофайлы: outputs/<stem>.flac, кеши конверсий
(.wav/.mp3) и производное аудио (превью/миксы/dsp-варианты в <stem>.d/).
Всё остальное — мета, ABC-партитуры, латенты float16 (.npy-сериализация),
метрики, пики волны, таймлайны — живёт в outputs/library.db, единственном
не-аудио файле. Legacy-записи (sidecar-файлы прежнего формата) импортирует
migrate_legacy() при старте, после чего их файлы удаляются.
"""
import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
# YUE2_DB — для тестов (отдельная пустая БД); прод: outputs/library.db
DB_PATH = Path(os.environ.get("YUE2_DB") or (OUT / "library.db"))

_local = threading.local()


def db() -> sqlite3.Connection:
    """Соединение на поток: sync-эндпоинты FastAPI работают в threadpool."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks(
  stem TEXT PRIMARY KEY,
  file TEXT NOT NULL,
  title TEXT DEFAULT '',
  duration_s REAL,
  sample_rate INTEGER,
  style TEXT DEFAULT '',
  style_base TEXT DEFAULT '',
  voice TEXT DEFAULT '',
  lyrics TEXT DEFAULT '',
  cot TEXT DEFAULT 'full',
  cfg_scale REAL,
  seed INTEGER,
  arc TEXT DEFAULT '',
  draft INTEGER DEFAULT 0,
  tokens INTEGER,
  total_s REAL,
  peak_vram_gb REAL,
  ts TEXT,
  liked INTEGER DEFAULT 0,
  abc_text TEXT,          -- внешняя/правленая партитура (бывш. <stem>.abc)
  abc_source TEXT,        -- 'cover' | 'edit'
  score_abc TEXT,         -- план модели (бывш. <stem>.score.abc)
  latent BLOB,            -- латенты float16, .npy-сериализация
  overdub_of TEXT,
  overdub_file TEXT,      -- микс в <parent>.d/ (аудио, остаётся на диске)
  overdub_error TEXT,
  lane_of TEXT,           -- импортированный аудио-слой этой песни (студия)
  version_of TEXT,        -- непосредственный родитель версии
  version_root TEXT,      -- корень песни (стем исходной генерации)
  version_n INTEGER,
  version_label TEXT,     -- накопительная: «кассета», «кассета + перегруз»
  group_main INTEGER DEFAULT 0,
  created TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_root ON tracks(version_root);
CREATE INDEX IF NOT EXISTS idx_tracks_overdub ON tracks(overdub_of);

CREATE TABLE IF NOT EXISTS metrics(
  stem TEXT PRIMARY KEY, data TEXT NOT NULL, ts TEXT
);
CREATE TABLE IF NOT EXISTS peaks(
  stem TEXT, n INTEGER, data TEXT NOT NULL, PRIMARY KEY(stem, n)
);
CREATE TABLE IF NOT EXISTS score_cache(
  stem TEXT PRIMARY KEY, abc_hash TEXT, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, status TEXT, stage TEXT, params TEXT,
  result_stem TEXT, error TEXT, created REAL, updated REAL
);
"""

_COLS = ("stem file title duration_s sample_rate style style_base voice lyrics cot "
         "cfg_scale seed arc draft tokens total_s peak_vram_gb ts liked abc_text "
         "abc_source score_abc latent overdub_of overdub_file overdub_error lane_of "
         "version_of version_root version_n version_label group_main created").split()

_BOOL_COLS = {"liked", "draft", "group_main"}

# Колонки, добавленные после первого релиза схемы (идемпотентный ALTER)
_MIGRATE_COLS = {"tracks": ["lane_of TEXT"]}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_db():
    db().executescript(SCHEMA)
    # идемпотентная миграция колонок для БД, созданных прежней версией схемы
    for tbl, cols in _MIGRATE_COLS.items():
        for coldef in cols:
            try:
                db().execute(f"ALTER TABLE {tbl} ADD COLUMN {coldef}")
            except sqlite3.OperationalError:
                pass  # колонка уже есть
    db().commit()


# ---------- tracks ----------

def upsert_track(row: dict):
    """Полная строка трека (отсутствующие колонки — NULL/дефолты)."""
    vals = {k: row.get(k) for k in _COLS if k in row}
    cols = list(vals)
    q = (f"INSERT OR REPLACE INTO tracks({','.join(cols)}) "
         f"VALUES({','.join('?' * len(cols))})")
    db().execute(q, [vals[c] for c in cols])
    db().commit()


def update_track(stem: str, **fields):
    vals = {k: v for k, v in fields.items() if k in _COLS}
    if not vals:
        return
    q = f"UPDATE tracks SET {','.join(f'{c}=?' for c in vals)} WHERE stem=?"
    db().execute(q, [*vals.values(), stem])
    db().commit()


def get_track(stem: str) -> dict | None:
    r = db().execute("SELECT * FROM tracks WHERE stem=?", (stem,)).fetchone()
    return dict(r) if r else None


def _bools(r: dict) -> dict:
    return {k: bool(r[k]) for k in _BOOL_COLS if r.get(k) is not None}


def track_to_api(r: dict) -> dict:
    """Строка БД → dict в формате старой меты (совместимость фронтенда)."""
    stem = r["stem"]
    m = {k: r[k] for k in _COLS if k not in ("latent",) and k in r}
    m.update(_bools(r))
    if r.get("abc_text"):
        m["abc_file"] = f"{stem}.abc"
        m["abc_chars"] = len(r["abc_text"])
    if r.get("score_abc"):
        m["score_abc_file"] = f"{stem}.score.abc"
    if r.get("latent"):
        m["latent_file"] = f"{stem}.latent.npy"
    m["audio_url"] = f"/outputs/{r['file']}"
    return m


def list_tracks(limit: int = 100) -> list[dict]:
    rows = db().execute(
        "SELECT * FROM tracks "
        "ORDER BY created DESC, COALESCE(version_n, 0) ASC, stem ASC LIMIT ?",
        (limit,),
    ).fetchall()
    return [track_to_api(dict(r)) for r in rows]


def group_rows(root: str) -> list[dict]:
    """Песня целиком: корень + все его версии."""
    rows = db().execute(
        "SELECT * FROM tracks WHERE stem=? OR version_root=? "
        "ORDER BY COALESCE(version_n, 0) ASC, created ASC",
        (root, root),
    ).fetchall()
    return [dict(r) for r in rows]


def children_rows(stem: str) -> list[dict]:
    rows = db().execute(
        "SELECT * FROM tracks WHERE overdub_of=? ORDER BY created ASC", (stem,)
    ).fetchall()
    return [dict(r) for r in rows]


def lanes_rows(stem: str) -> list[dict]:
    """Дорожки-слои песни: овердабы и импортированные аудио."""
    rows = db().execute(
        "SELECT * FROM tracks WHERE overdub_of=? OR lane_of=? ORDER BY created ASC",
        (stem, stem),
    ).fetchall()
    return [dict(r) for r in rows]


def next_version_n(root: str) -> int:
    r = db().execute(
        "SELECT MAX(version_n) FROM tracks WHERE version_root=?", (root,)
    ).fetchone()
    return int(r[0] or 0) + 1


def set_main(root: str, stem: str):
    db().execute(
        "UPDATE tracks SET group_main=CASE WHEN stem=? THEN 1 ELSE 0 END "
        "WHERE stem=? OR version_root=?",
        (stem, root, root),
    )
    db().commit()


def delete_tracks(stems: list[str]):
    qmarks = ",".join("?" * len(stems))
    for tbl in ("tracks", "metrics", "peaks", "score_cache"):
        db().execute(f"DELETE FROM {tbl} WHERE stem IN ({qmarks})", stems)
    db().commit()


# ---------- metrics / peaks / score cache ----------

def get_metrics(stem: str) -> dict | None:
    r = db().execute("SELECT data FROM metrics WHERE stem=?", (stem,)).fetchone()
    return json.loads(r[0]) if r else None


def put_metrics(stem: str, data: dict):
    db().execute(
        "INSERT OR REPLACE INTO metrics(stem, data, ts) VALUES(?,?,?)",
        (stem, json.dumps(data, ensure_ascii=False), _now()),
    )
    db().commit()


def get_peaks(stem: str, n: int) -> dict | None:
    r = db().execute(
        "SELECT data FROM peaks WHERE stem=? AND n=?", (stem, n)
    ).fetchone()
    return json.loads(r[0]) if r else None


def put_peaks(stem: str, n: int, data: dict):
    db().execute(
        "INSERT OR REPLACE INTO peaks(stem, n, data) VALUES(?,?,?)",
        (stem, n, json.dumps(data, ensure_ascii=False)),
    )
    db().commit()


def get_score_cache(stem: str, abc_hash: str) -> dict | None:
    r = db().execute(
        "SELECT data FROM score_cache WHERE stem=? AND abc_hash=?",
        (stem, abc_hash),
    ).fetchone()
    return json.loads(r[0]) if r else None


def put_score_cache(stem: str, abc_hash: str, data: dict):
    db().execute(
        "INSERT OR REPLACE INTO score_cache(stem, abc_hash, data) VALUES(?,?,?)",
        (stem, abc_hash, json.dumps(data, ensure_ascii=False)),
    )
    db().commit()


def del_score_cache(stem: str):
    db().execute("DELETE FROM score_cache WHERE stem=?", (stem,))
    db().commit()


# ---------- jobs (история очереди; live-прогресс остаётся в памяти) ----------

def job_create(jid: str, params: dict):
    db().execute(
        "INSERT OR REPLACE INTO jobs(id, status, stage, params, created, updated) "
        "VALUES(?,?,?,?,?,?)",
        (jid, "queued", "в очереди", json.dumps(params, ensure_ascii=False),
         datetime.now().timestamp(), datetime.now().timestamp()),
    )
    db().commit()


def job_update(jid: str, **fields):
    vals = {k: v for k, v in fields.items()
            if k in ("status", "stage", "error", "result_stem") and v is not None}
    if not vals:
        return
    vals["updated"] = datetime.now().timestamp()
    q = f"UPDATE jobs SET {','.join(f'{c}=?' for c in vals)} WHERE id=?"
    db().execute(q, [*vals.values(), jid])
    db().commit()


def mark_interrupted():
    """Незавершённые джобы прежнего запуска — в ошибку (без авто-перезапуска)."""
    db().execute(
        "UPDATE jobs SET status='error', error='прервано рестартом сервера', "
        "updated=? WHERE status IN ('queued','running')",
        (datetime.now().timestamp(),),
    )
    db().commit()


# ---------- миграция legacy-записей ----------

def _read_text(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8") if p.is_file() else None
    except OSError:
        return None


def migrate_legacy() -> int:
    """Импорт записей старого формата в БД; после успешного импорта stem'а
    его sidecar-файлы удаляются — в файловой остаются только аудио.
    Аудио (.flac, кеши конверсий, производное аудио в <stem>.d/) не трогаем."""
    imported = 0
    for flac in sorted(OUT.glob("*.flac")):
        stem = flac.name[:-len(".flac")]
        if get_track(stem):
            continue
        try:
            meta = {}
            mp = OUT / f"{stem}.json"
            if mp.is_file():
                meta = json.loads(mp.read_text(encoding="utf-8"))
            abc_text = _read_text(OUT / f"{stem}.abc")
            score_abc = _read_text(OUT / f"{stem}.score.abc")
            npy = OUT / f"{stem}.latent.npy"
            latent = npy.read_bytes() if npy.is_file() else None
            row = {
                "stem": stem,
                "file": flac.name,
                "title": meta.get("title") or "",
                "duration_s": meta.get("duration_s"),
                "sample_rate": meta.get("sample_rate"),
                "style": meta.get("style") or "",
                "style_base": meta.get("style_base") or meta.get("style") or "",
                "voice": meta.get("voice") or "",
                "lyrics": meta.get("lyrics") or "",
                "cot": meta.get("cot") or "full",
                "cfg_scale": meta.get("cfg_scale"),
                "seed": meta.get("seed"),
                "arc": meta.get("arc") or "",
                "draft": 1 if meta.get("draft") else 0,
                "tokens": meta.get("tokens"),
                "total_s": meta.get("total_s"),
                "peak_vram_gb": meta.get("peak_vram_gb"),
                "ts": meta.get("ts"),
                "liked": 1 if meta.get("liked") else 0,
                "abc_text": abc_text,
                "abc_source": "cover" if (abc_text and meta.get("abc_file")) else None,
                "score_abc": score_abc,
                "latent": latent,
                "overdub_of": meta.get("overdub_of"),
                "overdub_file": meta.get("overdub_file"),
                "overdub_error": meta.get("overdub_error"),
                "created": meta.get("ts") or _now(),
            }
            upsert_track(row)
            # импорт состоялся — сверяем строку с исходными файлами байт-в-байт,
            # и только потом чистим sidecar'ы (не-аудио)
            chk = get_track(stem)
            ok = bool(chk) and chk["file"] == flac.name
            ok = ok and (abc_text is None or chk["abc_text"] == abc_text)
            ok = ok and (score_abc is None or chk["score_abc"] == score_abc)
            ok = ok and (latent is None or chk["latent"] == latent)
            if ok:
                for suf in (".json", ".abc", ".score.abc", ".latent.npy"):
                    p = OUT / f"{stem}{suf}"
                    if p.is_file():
                        p.unlink()
                d = OUT / f"{stem}.d"
                if d.is_dir():
                    for p in d.iterdir():
                        if p.suffix == ".json" or p.name.endswith(".metrics.json"):
                            p.unlink()
                imported += 1
        except Exception as e:  # noqa: BLE001 — одна битая запись не рушит миграцию
            print(f"[store] миграция {stem}: пропущена ({type(e).__name__}: {e})")
    return imported


init_db()
