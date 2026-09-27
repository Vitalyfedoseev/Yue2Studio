"""API-тесты без GPU: TestClient без lifespan (модель не грузится).

Записи-фикстуры создаются в outputs/ с уникальным префиксом zztest- и
удаляются целиком (включая outputs/<stem>.d/).
"""
import json
import os
import shutil
import time
import uuid

os.environ.setdefault("YUE2_NO_MODEL", "1")

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import server as srv

client = TestClient(srv.app)  # без with → lifespan не запускается

ABC = """X:1
T:Test
M:4/4
L:1/8
Q:1/4=120
K:C
V: Vocal
% intro
C2 D2 E2 G2 | A4 G2 | z4 z4 |
% verse
V: Ins
"Am"c2 e2 g2 e2 | "G"d4 B4 | z8 |"""


@pytest.fixture(scope="module")
def stem():
    s = "zztest-" + uuid.uuid4().hex[:8]
    sr = 48000
    click = np.zeros((3 * sr, 2), dtype="float32")
    for k in range(0, 3 * sr, sr // 2):   # клик каждые 0.5 с — темп различим
        click[k:k + 200, :] = 0.8
    sf.write(str(srv.OUT / f"{s}.flac"), click, sr, subtype="PCM_24")
    (srv.OUT / f"{s}.score.abc").write_text(ABC, encoding="utf-8")
    meta = {
        "file": f"{s}.flac", "title": "тест-запись", "duration_s": 3.0,
        "sample_rate": sr, "style": "test style", "style_base": "test style",
        "voice": "", "lyrics": "[Instrumental]", "cot": "full", "cfg_scale": None,
        "seed": 42, "tokens": 100, "total_s": 1.0, "peak_vram_gb": 1.0,
        "ts": "2026-01-01T00:00:00", "score_abc_file": f"{s}.score.abc",
        "arc": "", "draft": False,
    }
    (srv.OUT / f"{s}.json").write_text(json.dumps(meta), encoding="utf-8")
    yield s
    for suf in (".flac", ".json", ".score.abc", ".wav", ".mp3", ".48k.mp3",
                ".latent.npy", ".abc"):
        p = srv.OUT / f"{s}{suf}"
        if p.is_file():
            p.unlink()
    d = srv.OUT / f"{s}.d"
    if d.is_dir():
        shutil.rmtree(d)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    for k in ("ready", "detail", "queue", "sampling_rate", "metrics_ok", "midi2abc"):
        assert k in j
    assert j["metrics_ok"] is True


def test_gallery_contains_record(stem):
    r = client.get("/api/gallery")
    assert r.status_code == 200
    mine = [i for i in r.json() if i["file"] == f"{stem}.flac"]
    assert mine and mine[0]["title"] == "тест-запись"


def test_rename(stem):
    r = client.post(f"/api/gallery/{stem}/rename", json={"title": "новое имя"})
    assert r.status_code == 200 and r.json()["title"] == "новое имя"
    meta = json.loads((srv.OUT / f"{stem}.json").read_text(encoding="utf-8"))
    assert meta["title"] == "новое имя"


def test_score_timeline(stem):
    r = client.get(f"/api/gallery/{stem}/score")
    assert r.status_code == 200
    j = r.json()
    assert j["tempo_bpm"] == 120.0
    assert len(j["bars"]) == 6
    assert j["duration_sec"] == 11.5
    assert isinstance(j["rms_sections"], list) and j["rms_sections"]
    # кеш записан в <stem>.d/
    assert (srv.OUT / f"{stem}.d" / "score.json").is_file()


def test_analyze(stem):
    r = client.post(f"/api/gallery/{stem}/analyze")
    assert r.status_code == 200
    m = r.json()
    for k in ("tempo_bpm", "key", "rms_p95_db", "dyn_range_db", "bands", "crest_db"):
        assert k in m
    assert isinstance(m["key"], str)


def test_dsp_chains_meta():
    r = client.get("/api/dsp/chains")
    assert r.status_code == 200
    chains = r.json()
    assert [c["id"] for c in chains] == ["wall", "wall-lite", "tape"]
    assert all(len(c["params"]) == 3 for c in chains)


def test_dsp_apply_and_variants(stem):
    r = client.post(f"/api/gallery/{stem}/dsp",
                    json={"chain": "wall", "params": {"exciter": 2.5}, "preview": False})
    assert r.status_code == 200
    j = r.json()
    assert j["file"] == f"{stem}.d/dsp-wall.flac"
    assert (srv.OUT / f"{stem}.d" / "dsp-wall.flac").is_file()
    assert j["params"]["wall"] == 0.5          # дефолт подставился
    lst = client.get(f"/api/gallery/{stem}/dsp").json()["variants"]
    assert any(v["file"].endswith("dsp-wall.flac") for v in lst)


def test_wav_and_mp3_conversion(stem):
    r = client.get(f"/api/outputs/{stem}.flac/wav")
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/wav")
    r2 = client.get(f"/api/outputs/{stem}.flac/mp3")
    assert r2.status_code == 200


def test_cancel_queued_job():
    jid = uuid.uuid4().hex[:12]
    with srv.JOBS_LOCK:
        srv.JOBS[jid] = {"id": jid, "status": "queued", "stage": "в очереди",
                         "cancel": False, "created": time.time()}
    r = client.post(f"/api/jobs/{jid}/cancel")
    assert r.status_code == 200
    assert srv.JOBS[jid]["status"] == "canceled"


def test_generate_validation():
    # arc + внешний abc несовместимы
    r = client.post("/api/generate", json={
        "style": "rock", "lyrics": "[Verse]\ntest",
        "abc": "X:1\nQ:1/4=120", "arc": "build"})
    assert r.status_code == 422
    # arc + cot=off несовместимы
    r2 = client.post("/api/generate", json={
        "style": "rock", "lyrics": "[Verse]\ntest", "cot": "off", "arc": "burst"})
    assert r2.status_code == 422


def test_generate_and_plan_accept_empty_arc():
    """Регрессия: фронт всегда шлёт arc:"" — пустая строка должна проходить
    паттерн (раньше pydantic отдавал 422 на оба эндпоинта)."""
    r = client.post("/api/generate", json={
        "style": "rock", "lyrics": "[Verse]\ntest", "arc": "", "draft": True})
    assert r.status_code == 200
    jid = r.json()["id"]
    assert srv.JOBS[jid]["draft"] is True
    del srv.JOBS[jid]  # worker в тестах не запущен — убираем за собой
    r2 = client.post("/api/plan", json={
        "style": "rock", "lyrics": "[Verse]\ntest", "arc": ""})
    assert r2.status_code == 503  # модель не грузится (YUE2_NO_MODEL), но не 422


def test_plan_without_model_503():
    r = client.post("/api/plan", json={"style": "rock", "lyrics": "[Verse]\ntest"})
    assert r.status_code == 503


def test_delete_cleans_everything(stem):
    r = client.delete(f"/api/gallery/{stem}")
    assert r.status_code == 200
    removed = r.json()["removed"]
    assert f"{stem}.flac" in removed
    assert f"{stem}.score.abc" in removed
    assert f"{stem}.d/" in removed
    assert not (srv.OUT / f"{stem}.json").is_file()
