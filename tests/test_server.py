"""API-тесты без GPU: TestClient без lifespan (модель не грузится).

БД — отдельная во tmp (YUE2_DB до импорта server). Записи-фикстуры создаются
в outputs/ с уникальным префиксом zztest- и удаляются целиком (файлы + строки
БД). Миграция legacy-записей тестируется в изолированном каталоге.
"""
import io
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

os.environ.setdefault("YUE2_NO_MODEL", "1")
os.environ.setdefault("YUE2_DB",
                      os.path.join(tempfile.mkdtemp(prefix="yue2-testdb-"), "library.db"))

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import server as srv
import store

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


def _latent_bytes(seconds: float) -> bytes:
    z = np.zeros((int(seconds * 25), 64), dtype=np.float16)  # 25 кадров/с
    z[:, 0] = 0.1
    buf = io.BytesIO()
    np.save(buf, z)
    return buf.getvalue()


def _make_track(s: str, with_abc: bool = True):
    sr = 48000
    click = np.zeros((3 * sr, 2), dtype="float32")
    for k in range(0, 3 * sr, sr // 2):   # клик каждые 0.5 с — темп различим
        click[k:k + 200, :] = 0.8
    sf.write(str(srv.OUT / f"{s}.flac"), click, sr, subtype="PCM_24")
    store.upsert_track({
        "stem": s, "file": f"{s}.flac", "title": "тест-запись", "duration_s": 3.0,
        "sample_rate": sr, "style": "test style", "style_base": "test style",
        "voice": "", "lyrics": "[Instrumental]", "cot": "full", "cfg_scale": None,
        "seed": 42, "tokens": 100, "total_s": 1.0, "peak_vram_gb": 1.0,
        "ts": "2026-01-01T00:00:00", "created": "2026-01-01T00:00:00",
        "score_abc": ABC if with_abc else None,
        "latent": _latent_bytes(3.0), "arc": "", "draft": 0,
    })


def _cleanup(s: str):
    stems = [r["stem"] for r in store.group_rows(s)]
    stems += [c["stem"] for c in store.children_rows(s)]
    for st in set(stems):
        for suf in (".flac", ".wav", ".mp3", ".48k.mp3"):
            p = srv.OUT / f"{st}{suf}"
            if p.is_file():
                p.unlink()
        d = srv.OUT / f"{st}.d"
        if d.is_dir():
            shutil.rmtree(d)
    store.delete_tracks(list(set(stems)) + [s])


@pytest.fixture(scope="module")
def stem():
    s = "zztest-" + uuid.uuid4().hex[:8]
    _make_track(s)
    yield s
    _cleanup(s)


@pytest.fixture()
def trk():
    s = "zztest-" + uuid.uuid4().hex[:8]
    _make_track(s)
    yield s
    _cleanup(s)


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
    # алиасы совместимости со старой метой
    assert mine[0]["score_abc_file"] == f"{stem}.score.abc"
    assert mine[0]["audio_url"] == f"/outputs/{stem}.flac"


def test_rename(stem):
    r = client.post(f"/api/gallery/{stem}/rename", json={"title": "новое имя"})
    assert r.status_code == 200 and r.json()["title"] == "новое имя"
    assert store.get_track(stem)["title"] == "новое имя"


def test_score_timeline(stem):
    r = client.get(f"/api/gallery/{stem}/score")
    assert r.status_code == 200
    j = r.json()
    assert j["tempo_bpm"] == 120.0
    assert len(j["bars"]) == 6
    assert j["duration_sec"] == 11.5
    assert isinstance(j["rms_sections"], list) and j["rms_sections"]
    # кеш в БД: второй вызов отдаёт то же самое
    r2 = client.get(f"/api/gallery/{stem}/score")
    assert r2.json()["duration_sec"] == j["duration_sec"]


def test_analyze(stem):
    r = client.post(f"/api/gallery/{stem}/analyze")
    assert r.status_code == 200
    m = r.json()
    for k in ("tempo_bpm", "key", "rms_p95_db", "dyn_range_db", "bands", "crest_db"):
        assert k in m
    assert isinstance(m["key"], str)
    assert store.get_metrics(stem) is not None


def test_abc_get_save_reset(stem):
    r = client.get(f"/api/gallery/{stem}/abc")
    assert r.status_code == 200
    j = r.json()
    assert j["source"] == "model"
    assert j["abc"] == ABC
    assert j["audio_duration"] == 3.0

    # правка: темп 120→90, длительность партитуры растёт
    edited = ABC.replace("Q:1/4=120", "Q:1/4=90")
    r2 = client.post(f"/api/gallery/{stem}/abc", json={"text": edited})
    assert r2.status_code == 200
    assert r2.json()["duration_sec"] > 11.5

    # /score теперь отражает правку
    r3 = client.get(f"/api/gallery/{stem}/score")
    assert r3.json()["duration_sec"] == r2.json()["duration_sec"]

    r4 = client.get(f"/api/gallery/{stem}/abc")
    assert r4.json()["source"] == "edited"

    # сброс к плану модели
    r5 = client.post(f"/api/gallery/{stem}/abc/reset")
    assert r5.status_code == 200
    assert r5.json()["duration_sec"] == 11.5
    r6 = client.get(f"/api/gallery/{stem}/abc")
    assert r6.json()["source"] == "model"

    # мусор отклоняется
    r7 = client.post(f"/api/gallery/{stem}/abc", json={"text": "коротко"})
    assert r7.status_code == 422


def test_peaks(stem):
    r = client.get(f"/api/gallery/{stem}/peaks", params={"n": 100})
    assert r.status_code == 200
    j = r.json()
    assert j["duration_s"] == 3.0
    assert len(j["min"]) == len(j["max"]) == len(j["rms"]) == 100
    assert max(j["max"]) > 0.5                       # клики 0.8 видны
    assert store.get_peaks(stem, 100) is not None    # кеш в БД
    # зум-окно не кешируется, но считается
    r2 = client.get(f"/api/gallery/{stem}/peaks", params={"n": 50, "from_sec": 1, "to_sec": 2})
    assert r2.status_code == 200
    j2 = r2.json()
    assert abs(j2["from_sec"] - 1.0) < 0.01
    assert 0 < len(j2["max"]) <= 50


def test_versions_edit_and_numbering(trk):
    # trim-keep: оставить 1–2 с
    r = client.post(f"/api/gallery/{trk}/edit", json={
        "op": "trim-keep", "region": {"from_sec": 1.0, "to_sec": 2.0}})
    assert r.status_code == 200
    v1 = r.json()["stem"]
    assert v1 == f"{trk}.v1"
    assert (srv.OUT / f"{v1}.flac").is_file()
    row = store.get_track(v1)
    assert row["version_of"] == trk and row["version_root"] == trk
    assert row["version_n"] == 1 and row["version_label"] == "фрагмент 1–2 с"
    assert abs(row["duration_s"] - 1.0) < 0.05
    # латент обрезан пропорционально: 1 с × 25 кадров
    z = np.load(io.BytesIO(row["latent"]))
    assert abs(len(z) - 25) <= 2
    # версия видна в галерее с бейдж-полем
    items = {i["file"]: i for i in client.get("/api/gallery").json()}
    assert items[f"{v1}.flac"]["version_of"] == trk
    # таймлайн версии отдаётся (адаптированный под вырезы)
    r_s = client.get(f"/api/gallery/{v1}/score")
    assert r_s.status_code == 200
    for b in r_s.json()["bars"]:
        assert b["end_sec"] > b["start_sec"]

    # нумерация растёт; метка накопительная
    r2 = client.post(f"/api/gallery/{trk}/edit", json={"op": "gain", "params": {"db": -3}})
    v2 = r2.json()["stem"]
    assert v2 == f"{trk}.v2"
    assert store.get_track(v2)["version_label"] == "гейн -3.0 дБ"

    # trim-cut: вырезать 0.5–1.0 с
    r3 = client.post(f"/api/gallery/{trk}/edit", json={
        "op": "trim-cut", "region": {"from_sec": 0.5, "to_sec": 1.0}})
    v3 = r3.json()["stem"]
    assert abs(store.get_track(v3)["duration_s"] - 2.5) < 0.05

    # версия версии: корень группы прежний
    r4 = client.post(f"/api/gallery/{v1}/edit", json={"op": "gain", "params": {"db": 2}})
    v4 = r4.json()["stem"]
    assert v4 == f"{trk}.v4"
    row4 = store.get_track(v4)
    assert row4["version_of"] == v1 and row4["version_root"] == trk
    assert row4["version_label"] == "фрагмент 1–2 с + гейн +2.0 дБ"


def test_edit_batch_one_version(trk):
    """Набор изменений → одна версия: суммарная длительность, метка, латент."""
    r = client.post(f"/api/gallery/{trk}/edit", json={"ops": [
        {"op": "trim-cut", "region": {"from_sec": 0.5, "to_sec": 1.0}},
        {"op": "gain", "params": {"db": -3}},
        {"op": "chain", "params": {"chain": "tape"}},
    ]})
    assert r.status_code == 200
    j = r.json()
    assert j["stem"] == f"{trk}.v1"          # один набор = одна версия
    assert j["label"].count(" + ") == 2      # «вырезка … + гейн … + кассета»
    row = store.get_track(j["stem"])
    assert abs(row["duration_s"] - 2.5) < 0.05
    # латент: 3 c − 0.5 c = 2.5 c × 25 кадров ≈ 62–63
    z = np.load(io.BytesIO(row["latent"]))
    assert abs(len(z) - 62) <= 3


def test_edit_batch_region_remapped_after_cut(trk):
    """Регион операции задан в координатах записи; после вырезки он
    пересчитывается (гейн падает на тот же материал, а не со сдвигом)."""
    r = client.post(f"/api/gallery/{trk}/edit", json={"ops": [
        {"op": "trim-cut", "region": {"from_sec": 0.0, "to_sec": 1.0}},
        {"op": "gain", "params": {"db": 6}, "region": {"from_sec": 1.5, "to_sec": 3.0}},
    ]})
    assert r.status_code == 200
    row = store.get_track(r.json()["stem"])
    assert abs(row["duration_s"] - 2.0) < 0.05


def test_edit_batch_empty_and_limit(trk):
    r = client.post(f"/api/gallery/{trk}/edit", json={})
    assert r.status_code == 422
    # trim без выделения внутри набора
    r2 = client.post(f"/api/gallery/{trk}/edit", json={"ops": [{"op": "trim-keep"}]})
    assert r2.status_code == 422


def test_edit_silence(trk):
    r = client.post(f"/api/gallery/{trk}/edit", json={
        "ops": [{"op": "silence", "region": {"from_sec": 0.5, "to_sec": 1.5}}]})
    assert r.status_code == 200
    j = r.json()
    assert abs(store.get_track(j["stem"])["duration_s"] - 3.0) < 0.05
    import soundfile as _sf
    data, _sr = _sf.read(str(srv.OUT / f"{j['stem']}.flac"), always_2d=True)
    # внутри заглушенного диапазона тишина, на клике в 2 c — сигнал
    assert float(np.abs(data[int(0.7 * 48000):int(1.4 * 48000)]).max()) == 0.0
    assert float(np.abs(data[int(2.0 * 48000):int(2.1 * 48000)]).max()) > 0.5


def test_edit_duplicate_and_insert(trk):
    # дубликат секунды дважды → 3 + 2 = 5 c
    r = client.post(f"/api/gallery/{trk}/edit", json={"ops": [
        {"op": "duplicate", "region": {"from_sec": 1.0, "to_sec": 2.0},
         "params": {"times": 2}}]})
    assert r.status_code == 200
    j = r.json()
    assert abs(store.get_track(j["stem"])["duration_s"] - 5.0) < 0.05
    z = np.load(io.BytesIO(store.get_track(j["stem"])["latent"]))
    assert abs(len(z) - 125) <= 3          # 5 c × 25 кадров
    # вставка фрагмента [0..1] в позицию 3 → 4 c
    r2 = client.post(f"/api/gallery/{trk}/edit", json={"ops": [
        {"op": "insert-from", "params": {"src_from": 0.0, "src_to": 1.0, "at": 3.0}}]})
    assert r2.status_code == 200
    assert abs(store.get_track(r2.json()["stem"])["duration_s"] - 4.0) < 0.05


def test_edit_region_remapped_after_insert(trk):
    """Гейн-регион задан в координатах записи: вставка 1 c в 0.5 с сдвигает
    его вправо — набор проходит без ошибок пересчёта."""
    r = client.post(f"/api/gallery/{trk}/edit", json={"ops": [
        {"op": "insert-from", "params": {"src_from": 1.0, "src_to": 2.0, "at": 0.5}},
        {"op": "gain", "params": {"db": 6}, "region": {"from_sec": 2.0, "to_sec": 3.0}},
    ]})
    assert r.status_code == 200
    assert abs(store.get_track(r.json()["stem"])["duration_s"] - 4.0) < 0.05


def test_edit_swap(trk):
    # поменять [0.5..1.0] и [2.0..2.5]: длина та же, материал обменялся
    r = client.post(f"/api/gallery/{trk}/edit", json={"ops": [
        {"op": "silence", "region": {"from_sec": 0.5, "to_sec": 1.0}},
        {"op": "swap", "region": {"from_sec": 0.5, "to_sec": 1.0},
         "region2": {"from_sec": 2.0, "to_sec": 2.5}}]})
    assert r.status_code == 200
    j = r.json()
    assert abs(store.get_track(j["stem"])["duration_s"] - 3.0) < 0.05
    import soundfile as _sf
    data, _ = _sf.read(str(srv.OUT / f"{j['stem']}.flac"), always_2d=True)
    # на месте A теперь материал из B (клик на 2.0 → встал на 0.5), на месте B — тишина
    assert float(np.abs(data[int(0.5 * 48000):int(0.55 * 48000)]).max()) > 0.5
    assert float(np.abs(data[int(2.1 * 48000):int(2.4 * 48000)]).max()) == 0.0


def test_edit_fade_region(trk):
    r = client.post(f"/api/gallery/{trk}/edit", json={
        "ops": [{"op": "fade-region", "region": {"from_sec": 0.0, "to_sec": 2.0},
                 "params": {"fade_in": 1.0, "fade_out": 0.0}}]})
    assert r.status_code == 200
    j = r.json()
    assert abs(store.get_track(j["stem"])["duration_s"] - 3.0) < 0.05
    import soundfile as _sf
    data, _ = _sf.read(str(srv.OUT / f"{j['stem']}.flac"), always_2d=True)
    # начало затухает: клик в 0 с ослаблен фейдом 1 с
    assert float(np.abs(data[:int(0.05 * 48000)]).max()) < 0.2


def test_edit_preview_no_version(trk):
    r = client.post(f"/api/gallery/{trk}/edit", json={
        "preview": True,
        "ops": [{"op": "trim-cut", "region": {"from_sec": 0.0, "to_sec": 1.0}}]})
    assert r.status_code == 200
    j = r.json()
    assert j.get("preview") is True and "url" in j
    assert abs(j["duration_s"] - 2.0) < 0.05
    assert (srv.OUT / j["url"].replace("/outputs/", "")).is_file()
    # версии не появилось
    assert store.next_version_n(trk) == 1


def test_import_lane_and_mixdown(trk):
    import soundfile as _sf

    buf = io.BytesIO()
    _sf.write(buf, np.zeros((44100, 2), dtype="float32"), 44100, format="WAV",
              subtype="PCM_16")
    lane_stem = None
    try:
        r = client.post(f"/api/gallery/{trk}/import-lane?title=мой слой",
                        content=buf.getvalue())
        assert r.status_code == 200
        j = r.json()
        lane_stem = j["stem"]
        row = store.get_track(lane_stem)
        assert row["lane_of"] == trk and abs(row["duration_s"] - 1.0) < 0.05
        # виден в /lanes с kind=import
        lanes = client.get(f"/api/gallery/{trk}/lanes").json()["lanes"]
        assert any(l["stem"] == lane_stem and l["kind"] == "import" for l in lanes)
        # микс с импортированным слоем
        r2 = client.post(f"/api/gallery/{trk}/mixdown", json={
            "lanes": [{"stem": trk, "gain": 1.0}, {"stem": lane_stem, "gain": 0.5}]})
        assert r2.status_code == 200
        assert store.get_track(r2.json()["stem"])["version_of"] == trk
        # чужой слой отвергается
        r3 = client.post(f"/api/gallery/{trk}/mixdown", json={
            "lanes": [{"stem": trk}, {"stem": "zztest-alien"}]})
        assert r3.status_code == 422
    finally:
        if lane_stem:
            store.delete_tracks([lane_stem])
            (srv.OUT / f"{lane_stem}.flac").unlink(missing_ok=True)


def test_comp_from_versions(trk):
    # v1: фрагмент 0–2 c
    r = client.post(f"/api/gallery/{trk}/edit", json={
        "ops": [{"op": "trim-keep", "region": {"from_sec": 0.0, "to_sec": 2.0}}]})
    v1 = r.json()["stem"]
    # комп: первые 0.5 c оригинала + 1 c из v1
    r2 = client.post(f"/api/gallery/{trk}/comp", json={"parts": [
        {"stem": trk, "from_sec": 0.0, "to_sec": 0.5},
        {"stem": v1, "from_sec": 0.5, "to_sec": 1.5}]})
    assert r2.status_code == 200
    j = r2.json()
    assert j["label"].startswith("комп:")
    row = store.get_track(j["stem"])
    assert abs(row["duration_s"] - 1.5) < 0.05
    z = np.load(io.BytesIO(row["latent"]))
    assert abs(len(z) - 38) <= 4          # 1.5 c × 25
    # чужая версия в частях отвергается
    r3 = client.post(f"/api/gallery/{trk}/comp", json={"parts": [
        {"stem": "zztest-alien", "from_sec": 0, "to_sec": 1}]})
    assert r3.status_code == 422


def test_export_selection(trk):
    r = client.get(f"/api/gallery/{trk}/export",
                   params={"from_sec": 0.5, "to_sec": 2.0, "fmt": "flac"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/flac")
    assert len(r.content) > 1000
    assert "attachment" in r.headers["content-disposition"]
    r2 = client.get(f"/api/gallery/{trk}/export",
                    params={"from_sec": 0.5, "to_sec": 2.0, "fmt": "mp3"})
    assert r2.status_code == 200
    assert r2.headers["content-type"].startswith("audio/mpeg")
    r3 = client.get(f"/api/gallery/{trk}/export", params={"fmt": "ogg"})
    assert r3.status_code == 422


def test_export_multitrack(trk):
    import zipfile as _zf

    import soundfile as _sf
    buf = io.BytesIO()
    _sf.write(buf, np.zeros((24000, 2), dtype="float32"), 48000, format="WAV")
    lane = None
    try:
        lane = client.post(f"/api/gallery/{trk}/import-lane?title=слой",
                           content=buf.getvalue()).json()
        r = client.get(f"/api/gallery/{trk}/export-multitrack")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/zip")
        zf = _zf.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()
        assert len(names) == 2 and all(n.endswith(".flac") for n in names)
    finally:
        if lane and lane.get("stem"):
            store.delete_tracks([lane["stem"]])
            (srv.OUT / f"{lane['stem']}.flac").unlink(missing_ok=True)


def test_edit_fade_and_chain(trk):
    r = client.post(f"/api/gallery/{trk}/edit",
                    json={"op": "fade", "params": {"fade_in": 0.5, "fade_out": 1.0}})
    assert r.status_code == 200
    assert r.json()["label"] == "фейды"
    assert abs(store.get_track(r.json()["stem"])["duration_s"] - 3.0) < 0.05

    r2 = client.post(f"/api/gallery/{trk}/edit", json={
        "op": "chain", "params": {"chain": "tape", "wow": 0.2}})
    assert r2.status_code == 200
    j = r2.json()
    assert j["label"] == "кассета"
    assert j["params"]["wow"] == 0.2                  # параметр прошёл клэмп/подстановку
    assert abs(store.get_track(j["stem"])["duration_s"] - 3.0) < 0.05

    # chain по выделению — длительность не меняется
    r3 = client.post(f"/api/gallery/{trk}/edit", json={
        "op": "chain", "params": {"chain": "overdrive", "drive": 12},
        "region": {"from_sec": 0.5, "to_sec": 2.5}})
    assert r3.status_code == 200
    assert abs(store.get_track(r3.json()["stem"])["duration_s"] - 3.0) < 0.05


def test_edit_validation(trk):
    # trim без выделения
    r = client.post(f"/api/gallery/{trk}/edit", json={"op": "trim-keep"})
    assert r.status_code == 422
    # неизвестная цепочка
    r2 = client.post(f"/api/gallery/{trk}/edit", json={
        "op": "chain", "params": {"chain": "nope"}})
    assert r2.status_code == 422
    # неизвестная запись
    r3 = client.post("/api/gallery/zztest-nothing/edit", json={"op": "gain"})
    assert r3.status_code == 404


def test_dsp_chains_meta():
    r = client.get("/api/dsp/chains")
    assert r.status_code == 200
    chains = r.json()
    assert [c["id"] for c in chains] == ["wall", "wall-lite", "tape", "overdrive", "master"]
    assert all(len(c["params"]) >= 3 for c in chains)


def test_dsp_apply_creates_version(trk):
    r = client.post(f"/api/gallery/{trk}/dsp", json={"chain": "tape", "params": {}})
    assert r.status_code == 200
    j = r.json()
    assert j["stem"] == f"{trk}.v1"
    assert j["label"] == "кассета"
    assert j["params"]["wow"] == 0.1                  # дефолт подставился
    assert (srv.OUT / f"{j['stem']}.flac").is_file()
    assert store.get_track(j["stem"])["version_of"] == trk


def test_dsp_preview_no_version(trk):
    r = client.post(f"/api/gallery/{trk}/dsp",
                    json={"chain": "wall", "params": {"exciter": 2.5}, "preview": True})
    assert r.status_code == 200
    j = r.json()
    assert j["file"].endswith("dsp-preview-wall.flac")
    assert (srv.OUT / j["file"]).is_file()
    # версии не появилось
    assert store.next_version_n(trk) == 1


def test_set_main_and_delete_cascade(trk):
    for _ in range(2):
        client.post(f"/api/gallery/{trk}/edit", json={"op": "gain", "params": {"db": -1}})
    v1, v2 = f"{trk}.v1", f"{trk}.v2"

    r = client.post(f"/api/gallery/{v1}/set-main")
    assert r.status_code == 200 and r.json()["main"] == v1
    assert store.get_track(v1)["group_main"] == 1
    assert store.get_track(trk)["group_main"] == 0
    assert store.get_track(v2)["group_main"] == 0

    # удаление отдельной версии не трогает остальные
    r2 = client.delete(f"/api/gallery/{v2}")
    assert r2.status_code == 200
    assert store.get_track(v2) is None
    assert store.get_track(v1) is not None and store.get_track(trk) is not None
    assert not (srv.OUT / f"{v2}.flac").is_file()
    assert (srv.OUT / f"{v1}.flac").is_file()

    # удаление корня — каскад всей группы
    r3 = client.delete(f"/api/gallery/{trk}")
    assert r3.status_code == 200
    assert store.get_track(trk) is None and store.get_track(v1) is None
    assert not (srv.OUT / f"{trk}.flac").is_file()
    assert not (srv.OUT / f"{v1}.flac").is_file()


def test_lanes_and_mixdown(trk):
    child = trk + "-od"
    _make_track(child)
    store.update_track(child, overdub_of=trk, title="партия гитары")
    try:
        r = client.get(f"/api/gallery/{trk}/lanes")
        assert r.status_code == 200
        lanes = r.json()["lanes"]
        assert len(lanes) == 1 and lanes[0]["stem"] == child

        r2 = client.post(f"/api/gallery/{trk}/mixdown", json={
            "lanes": [{"stem": trk, "gain": 1.0},
                      {"stem": child, "gain": 0.5, "mute": False}]})
        assert r2.status_code == 200
        j = r2.json()
        assert j["stem"] == f"{trk}.v1"
        assert j["label"].startswith("микс:")
        assert abs(store.get_track(j["stem"])["duration_s"] - 3.0) < 0.05

        # чужая дорожка отвергается
        r3 = client.post(f"/api/gallery/{trk}/mixdown", json={
            "lanes": [{"stem": trk}, {"stem": "zztest-alien", "gain": 0.5}]})
        assert r3.status_code == 422
    finally:
        _cleanup(child)


def test_overdub_instrumental(trk):
    """«Без вокала»: лирика заменяется тегами [Instrumental] по числу секций
    партитуры (по умолчанию включено на фронте)."""
    r = client.post(f"/api/gallery/{trk}/overdub", json={
        "style": "acoustic guitar", "instrumental": True})
    assert r.status_code == 200
    jid = r.json()["id"]
    job = srv.JOBS[jid]
    assert job["cot"] == "melody"
    lines = job["lyrics"].splitlines()
    assert lines and all(l == "[Instrumental]" for l in lines)
    assert 1 <= len(lines) <= 16        # по секциям партитуры (у тестовой ABC — 2)
    assert job["title"].startswith("овердаб (инструментал)")
    del srv.JOBS[jid]
    # без лирики и без инструментала — по-прежнему 422 (временно пустим лирику)
    store.update_track(trk, lyrics="")
    try:
        r2 = client.post(f"/api/gallery/{trk}/overdub", json={"style": "acoustic guitar"})
        assert r2.status_code == 422
    finally:
        store.update_track(trk, lyrics="[Instrumental]")


def test_rerender_validation(trk):
    # без партитуры — 422
    bare = trk + "-bare"
    _make_track(bare, with_abc=False)
    try:
        r = client.post(f"/api/gallery/{bare}/rerender", json={})
        assert r.status_code == 422
    finally:
        _cleanup(bare)
    # с партитурой — джоба в очереди с version_parent
    r2 = client.post(f"/api/gallery/{trk}/rerender", json={"draft": True})
    assert r2.status_code == 200
    jid = r2.json()["id"]
    job = srv.JOBS[jid]
    assert job["version_parent"] == trk
    assert job["cot"] == "melody" and job["draft"] is True
    assert job["seed"] == 42                      # сид родителя
    assert job["abc"] == ABC                      # партитура из записи
    del srv.JOBS[jid]
    # неизвестная запись
    assert client.post("/api/gallery/zztest-nothing/rerender", json={}).status_code == 404


def test_migration_imports_legacy(monkeypatch):
    tmp = Path(tempfile.mkdtemp(prefix="yue2-migr-"))
    monkeypatch.setattr(store, "OUT", tmp)
    s = "zztest-" + uuid.uuid4().hex[:8]
    sr = 48000
    data = np.zeros((sr, 2), dtype="float32")
    data[:, 0] = 0.5
    sf.write(str(store.OUT / f"{s}.flac"), data, sr, subtype="PCM_24")
    meta = {"file": f"{s}.flac", "title": "легаси", "duration_s": 1.0,
            "sample_rate": sr, "style": "s", "style_base": "s",
            "lyrics": "[Instrumental]", "seed": 7, "liked": True,
            "ts": "2026-01-02T00:00:00"}
    (store.OUT / f"{s}.json").write_text(json.dumps(meta), encoding="utf-8")
    (store.OUT / f"{s}.score.abc").write_text(ABC, encoding="utf-8")
    (store.OUT / f"{s}.abc").write_text(ABC, encoding="utf-8")
    np.save(str(store.OUT / f"{s}.latent.npy"), np.zeros((10, 64), dtype=np.float16))
    try:
        assert store.migrate_legacy() >= 1
        row = store.get_track(s)
        assert row["title"] == "легаси" and row["liked"] == 1
        assert row["score_abc"] == ABC and row["abc_text"] == ABC
        assert row["latent"] and len(row["latent"]) > 100
        # sidecar'ы удалены, аудио осталось
        assert (store.OUT / f"{s}.flac").is_file()
        for suf in (".json", ".abc", ".score.abc", ".latent.npy"):
            assert not (store.OUT / f"{s}{suf}").is_file()
        # повторный запуск — идемпотентен
        before = store.get_track(s)["stem"]
        store.migrate_legacy()
        assert store.get_track(s)["stem"] == before
    finally:
        for suf in (".flac", ".json", ".abc", ".score.abc", ".latent.npy"):
            p = store.OUT / f"{s}{suf}"
            if p.is_file():
                p.unlink()
        shutil.rmtree(tmp, ignore_errors=True)
        monkeypatch.undo()
        store.delete_tracks([s])


def test_wav_and_mp3_conversion(stem):
    r = client.get(f"/api/outputs/{stem}.flac/wav")
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/wav")
    r2 = client.get(f"/api/outputs/{stem}.flac/mp3")
    assert r2.status_code == 200


def test_abc_download(stem):
    r = client.get(f"/api/gallery/{stem}/abc/download")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "attachment" in r.headers["content-disposition"]
    assert r.text == ABC


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


def test_delete_missing_404():
    assert client.delete("/api/gallery/zztest-nothing").status_code == 404
