"""Аудио-метрики записи: порт analyze_file из yue-studio/worker/dsp.py.

librosa грузится лениво внутри функции — сервер стартует и без неё
(в health отражается как metrics_ok: false).
"""
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def analyze_file(path) -> dict:
    """FLAC/MP3 → словарь метрик: темп, тональность, динамика, полосы,
    стерео. Все значения округлены как в yue-studio (совместимые дельты)."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    rms = librosa.feature.rms(y=y)[0]
    rms_db = 20 * np.log10(rms + 1e-9)
    p95 = round(float(np.percentile(rms_db, 95)), 1)
    p50 = round(float(np.percentile(rms_db, 50)), 1)
    p10 = round(float(np.percentile(rms_db, 10)), 1)
    # пол шума — медиана RMS 5% самых тихих окон
    quiet = np.argsort(rms_db)[: max(1, len(rms_db) // 20)]
    noise_floor = round(float(np.median(rms_db[quiet])), 1)

    tempo = librosa.beat.beat_track(
        onset_envelope=librosa.onset.onset_strength(y=y, sr=sr), sr=sr)[0]
    tempo = float(np.atleast_1d(tempo)[0])
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    key = NOTES[int(np.argmax(chroma))]

    S = np.abs(librosa.stft(y, n_fft=2048))
    S_mean = S.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr)
    total = float(S_mean.sum()) + 1e-12

    def band(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return round(100 * float(S_mean[m].sum()) / total, 1)

    flat = librosa.feature.spectral_flatness(y=y)[0]
    n = min(len(flat), len(rms_db))
    flat_loud = flat[:n][rms_db[:n] > np.percentile(rms_db[:n], 75)]

    stereo_corr = mid_db = side_db = None
    try:
        y2, _sr2 = librosa.load(str(path), sr=44100, mono=False)
        if y2.ndim == 2 and y2.shape[0] == 2:
            left, right = y2[0], y2[1]
            stereo_corr = round(float(np.corrcoef(left, right)[0, 1]), 2)
            mid_db = round(20 * np.log10(float(np.std(left + right)) + 1e-9), 1)
            side_db = round(20 * np.log10(float(np.std(left - right)) + 1e-9), 1)
    except Exception:  # noqa: BLE001 — моно/битый файл — просто без стерео-полей
        pass

    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean())
    cum = np.cumsum(S_mean) / total
    f95 = float(freqs[min(int(np.searchsorted(cum, 0.95)), len(freqs) - 1)])
    f99 = float(freqs[min(int(np.searchsorted(cum, 0.99)), len(freqs) - 1)])

    return {
        "duration_sec": round(len(y) / sr, 1),
        "tempo_bpm": round(tempo, 1),
        "key": key,
        "rms_p95_db": p95,
        "rms_median_db": p50,
        "rms_p10_db": p10,
        "dyn_range_db": round(p95 - p10, 1),
        "crest_db": round(p95 - p50, 1),
        "peak": round(float(np.abs(y).max()), 3),
        "clip_pct": round(float(np.mean(np.abs(y) > 0.98)) * 100, 3),
        "noise_floor_db": noise_floor,
        "signal_noise_db": round(p95 - noise_floor, 1),
        "bands": {
            "bass": band(0, 150),
            "low_mid": band(150, 500),
            "mid": band(500, 2000),
            "high": band(2000, 8000),
            "air": band(8000, 11025),
        },
        "centroid_hz": round(centroid),
        "f95_hz": round(f95),
        "f99_hz": round(f99),
        "flatness_median": round(float(np.median(flat)), 3),
        "flatness_loud": round(float(np.median(flat_loud)) if flat_loud.size else float(np.median(flat)), 3),
        "stereo_corr": stereo_corr,
        "mid_db": mid_db,
        "side_db": side_db,
    }
