# audio2midi (Spotify basic-pitch)

Транскрипция аудио → MIDI: [basic-pitch](https://github.com/spotify/basic-pitch)
(Apache-2.0), модель ICASSP 2022 (`nmp.onnx`) в комплекте колеса. ONNX-инференс
на CPU, отдельный venv `venv/` — окружение проекта (.venv с жёсткими пинами
torch/numpy под yue2_infer) не трогается.

Скрипт `a2m.py <вход.аудио> <выход.mid>`: ffmpeg декодирует в wav 22.05 кГц моно
(сетка, на которой училась модель), basic-pitch извлекает ноты, stdout —
JSON `{notes, duration_s}`. Ограничения: ≤15 мин, ≥0.5 с. Пороги: onset 0.5,
frame 0.3, минимум 30 Гц. Вызывается из `server.py` (`/api/midi2abc`, аудио-вход).

## Установка (уже выполнена, для воспроизведения)

`PIP_CONFIG_FILE=/dev/null` обязателен — системный pip.conf с NGC-индексом
ломает установку (см. README проекта). Обход двух подводных камней:

1. У basic-pitch 0.4.0 базовый pin `tensorflow<2.15.1` (Linux, py≥3.11) не имеет
   колёс под py3.12 → резолвер откатывается к basic-pitch 0.3.0 и падает на
   сборке. Лечится установкой стека без него + basic-pitch `--no-deps`
   (для ONNX-инференса TF не нужен).
2. resampy 0.4.2 импортирует `pkg_resources`, удалённый из setuptools ≥82
   → пин `setuptools<82`.

```bash
python3 -m venv venv
PIP_CONFIG_FILE=/dev/null venv/bin/pip install 'numpy==1.26.4' scipy librosa \
  'resampy==0.4.2' pretty_midi mir_eval scikit-learn onnxruntime 'setuptools<82'
PIP_CONFIG_FILE=/dev/null venv/bin/pip install --no-deps 'basic-pitch==0.4.0'
```

numpy 1.26, а не 2.x — код basic-pitch 0.4.0 (нач. 2024) писался до numpy 2.

## Качество

Чистая моно-мелодия распознаётся точно (синтетика E4…A5 — все ноты верно).
Полномиксовая запись (вокал+аккомпанемент) даёт полифоническую «кашу» —
для каверов лучше вокальная/мелодическая дорожка по отдельности, лишнее
вычищается в редактируемом поле ABC в UI.
