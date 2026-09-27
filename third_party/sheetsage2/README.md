# SheetSage2 (m-a-p) — аудио → нативный ABC YuE2

[SheetSage2](https://huggingface.co/m-a-p/SheetSage2) — транскрипция «music
audio to editable scores» от той же команды, что YuE2 (MERT2-энкодер + BART-декодер,
CC-BY-NC 4.0). Основной движок `/api/midi2abc` для аудио-входа: выдаёт СРАЗУ
ABC в родном диалекте YuE2 — голоса `V: Vocal`/`V: Ins`, секции `% verse`/
`% chorus`, многотактные паузы `Z4|`, честные темп/тональность (lead sheet,
не каша нот). Спецрежим `melody_only=True` — обе мелодии без аккордов,
«для передачи в YuE2» (формулировка карточки модели).

- Снапшот модели: `SheetSage2/` (139 файлов; базовый MERT-v2-FullSong
  подтягивается в HF-кеш при первом запуске ~3 мин, дальше из кеша).
- Обёртка `ss2.py <вход.аудио> <выход.abc> <выход.mid>`: ffmpeg-CLI → wav
  24 кГц моно → `transcribe(melody_only=True)`; stdout — JSON
  `{seconds, device, notes, duration_s, abc_chars, midi_bytes}`. CUDA, при
  OOM — пересадка на CPU. Вызывается из ОСНОВНОГО .venv сервера подпроцессом.
- Требуется вход в HF с доступом к m-a-p/SheetSage2 и m-a-p/MERT-v2-FullSong
  (токен уже в `~/.cache/huggingface/token`).

## Установка (уже выполнена)

Зависимости добавлены в основной `.venv` (PIP_CONFIG_FILE=/dev/null, без
апгрейдов пинов): `torchaudio==2.10.0+cu128` (совпадает с torch), scipy,
`pretty_midi/mido/mir_eval/six/importlib_resources/decorator` (--no-deps).

## Грабли, уже закрытые в коде

1. **Потерянный файл в кеше remote-кода.** Transformers копирует кастомные
   модули в `~/.cache/huggingface/modules/transformers_modules/<имя-папки>/`
   и при рекурсивном разборе импортов теряет `chord_spelling_sheetsage2.py`.
   Папка снапшота переименована из `model` в `SheetSage2` (уникальный ключ
   кеша), а `heal_module_cache()` в ss2.py докладывает недостающие .py при
   каждом запуске.
2. Статический ffmpeg (без shared-библиотек) не годится для внутреннего
   декодера torchaudio — поэтому ss2.py декодирует сам через ffmpeg-CLI.

## Замеры (RTX 5060 Ti)

«Лёд под ногами майора» (2:07): 8.7 с GPU → 459 нот, ABC 1708 симв. /
1133 токена, секции intro/verse/chorus/interlude/outro, Q:1/4=103, K:Em.
Basic-pitch на той же песне: 4344 симв. без структуры.
