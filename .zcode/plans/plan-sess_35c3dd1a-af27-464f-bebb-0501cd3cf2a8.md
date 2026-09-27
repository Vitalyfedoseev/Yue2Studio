# План: подтянуть yue2-web до ядра функциональности yue-studio (фазы 0–2)

## Принятые решения
- **Объём:** фазы 0–2 — быстрые UI-победы, план/ABC-артефакты, пост-обработка на текущем стеке.
- **Без тяжёлых зависимостей:** никакого demucs (стемов/минуса), Whisper, Ollama. Перевод стиля рус→eng — только оффлайн-словарём.
- **Форма стиля — облегчённая:** библиотека стилей/пресеты как кликабельные теги к текущей строке + словарь подсказок ru→en. Слоты и стойку не портируем.

## Не-цели
Стемы/минус, распознавание текста из аудио, LLM-копайтер, слотовая форма + стойка, студия с пиано-роллом, импорт своих треков, корпус-профили, SQLite-очередь, MCP, i18n.

## Этап 0 — .gitignore (по вашему запросу, перед всем остальным)
Проект пока не под git, но готовим его: создать `/home/vit-ai/Pictures/tttt/yue2-web/.gitignore`:
- Python: `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.ruff_cache/`, `.venv/`;
- генерация: `outputs/` (FLAC/MP3/меты + будущие производные `outputs/<stem>.d/`);
- кеши: `.cache/`;
- сторонние венвы и тяжёлое: `third_party/*/venv/`, `third_party/midi2abc/node_modules/`, `third_party/sheetsage2/SheetSage2/` (веса модели);
- архивы: `*.zip`, `*.tar.gz` (сюда попадает бэкап `уюе2.zip`, 421 МБ);
- системный мусор и логи: `.DS_Store`, `Thumbs.db`, `*.log`.
С Coment: wheel `yue2_infer-*.whl` (66 КБ) сознательно версионируется — vendored-зависимость.

## Архитектурные решения
1. **Сервер** остаётся монолитом `server.py`; новое — отдельными модулями в корне, портируются из yue-studio почти дословно:
   - `arc.py` ← `yue-studio/worker/arc.py` (драматургия build/wave/burst);
   - `abcparse.py` ← `yue-studio/worker/abcparse.py` (чистый re, таймлайн ABC);
   - `audio_metrics.py` ← `analyze_file` из `yue-studio/worker/dsp.py` (librosa-метрики);
   - `dsp_chains.py` ← цепочки из `yue-studio/internal/dsp/dsp.go` на subprocess ffmpeg (wall, wall-lite, tape + параметры).
2. **Артефакты записи** (сейчас `outputs/<stem>.{flac,json,mp3,abc}`): добавляем `<stem>.score.abc` (модельный план) и `<stem>.latent.npy` (float16, [T,64], ~2 МБ/мин, LATENT_HZ=25). Производные — в подпапку `outputs/<stem>.d/`: `score.json` (кеш таймлайна), `metrics.json`, `preview-<f>-<t>.flac`, `dsp-<chain>.flac`(+`.metrics.json`), `overdub-<child>.flac`. Статика `/outputs` уже смонтирована — подпапка раздаётся автоматически; `DELETE /api/gallery/{stem}` чистит всё.
3. **GPU-конкуренция:** глобальный `PIPE_LOCK`; `_run_job` держит его всю генерацию; `/api/plan` и `/preview` берут с `timeout=1` → 409 «GPU занят генерацией».
4. **Фронтенд:** инлайн-JS `index.html` (760 строк) переносится без изменений в `static/app.js`, данные стилей — в `static/stylelib.js` (классические `<script>`, без сборки). Новый UI: модалка плана, сворачиваемая библиотека стилей, панели метрик/DSP/овердаба, лента секций с превью.

## Этапы

### Этап 1 — рефакторинг фронта (база, без смены поведения)
Вынести инлайн-скрипт `static/index.html` → `static/app.js` как есть. Проверка: страница работает идентично.

### Этап 2 — сервер: отмена, прогресс, черновик, драматургия
- `cancel_job` (server.py:551): для `queued` → сразу `status="canceled"` (worker скипает не-queued); для running — флаг `cancel`. Кнопка «Остановить» отменяет все активные задачи.
- `GenRequest`: `draft: bool = False`, `arc: str = ""` (паттерн `^(build|wave|burst)$`, несовместим с внешним `abc`).
- Черновик: в `_run_job` (server.py:201) `budget = min(budget, 450)`; мета `draft: true` + чип; кнопка «Черновик» в UI.
- Драматургия: порт `arc.py` дословно; при `arc` — первый `pipe.plan()` → `apply_arc(plan.abc, arc)` → повторный план с `abc=` (стадия «arc-plan»); стиль дополняется `style_with_arc`. Селектор в ряду режимов + чип в мете.
- Честный прогресс: budget → `counters`; watcher пишет `pct = min(99, tokens*100//budget)` на фазе семантики; фронт берёт `pct`, фолбэк — текущая эвристика.

### Этап 3 — артефакты генерации
В `_run_job` (server.py:222–256): писать `<stem>.score.abc` = `plan.abc` (план есть и при внешнем `abc=`), `np.save(...latent.npy, latents.astype(np.float16))`; мета: `score_abc_file`, `latent_file`. `delete_track` (server.py:585): + `.score.abc`, `.latent.npy`, + `shutil.rmtree(outputs/<stem>.d)`.

### Этап 4 — plan-only и редактор ABC
- `POST /api/plan` `{style, voice, lyrics, cot, seed}` → `{abc, seconds, tokens, truncated}`; 422 при `cot=off`; под `PIPE_LOCK` (409 при занятой GPU).
- UI: кнопка «План (ABC)» → модалка: правка ABC, статистика (сек/токены/усечение), «Рендер по этому ABC» (заполняет поле `abc` — механика кавера уже есть), краткая шпаргалка по нотации.

### Этап 5 — таймлайн и превью фрагментов
- `GET /api/gallery/{stem}/score` → `abcparse.parse_abc(score.abc)` + `rms_sections` (soundfile+numpy, без librosa); кеш `<stem>.d/score.json`; 404 если нет score.abc.
- `POST /api/gallery/{stem}/preview` `{from_sec, to_sec}`: валидации yue-studio (≥1 c, клэмпы по длительности звука), срез латентов `z[int(f*25):int(t*25)].astype(float32)` → `pipe.decode` под `PIPE_LOCK` → `<stem>.d/preview-<f>-<t>.flac`.
- UI: лента секций на карточке (имя + диапазон сек), клик → превью куска (клэмп по `duration_s`), мини-`<audio>`.

### Этап 6 — метрики
- Установка: `.venv/bin/pip install --dry-run librosa` — если резолвер не трогает запиненные torch/numpy/transformers, ставить в основной .venv; иначе отдельный `third_party/metrics` venv (numpy+librosa+soundfile) субпроцессом по образцу SS2. Решение по выводу dry-run.
- `audio_metrics.py` = порт `analyze_file` (темп, тональность, rms p95/p50/p10, dyn, crest, полосы, centroid, f95/f99, flatness ×2, stereo corr, mid/side).
- `POST /api/gallery/{stem}/analyze?fresh=` → кеш `metrics.json`; в health — `metrics_ok`.
- UI: свёрнутый блок «Метрики» + селект «сравнить с…» → дельты на клиенте.

### Этап 7 — DSP-цепочки (ffmpeg)
- Предусловие: `ffmpeg -filters` проверить aexciter/acrusher/alimiter/vibrato/anoisesrc (ffmpeg статический); отсутствующие — деградация графа с пометкой.
- `dsp_chains.py`: wall/wall-lite/tape, по 3 параметра с клампом; запуск `ffmpeg -y -hide_banner -loglevel error [-ss 20 -t 15] -i in -filter_complex … -map [out] out`.
- `POST /api/gallery/{stem}/dsp` `{chain, params, preview}` → файл + метрики; `GET` — список вариантов.
- UI: панель на карточке: цепочка, 3 крутилки, «Применить» / «Превью 15с» (с 20-й сек, автоплей), варианты с инлайн-дельтами, скачивание.

### Этап 8 — овердаб
- `POST /api/gallery/{stem}/overdub` `{style, lyrics="", gain=0.5, seed=-1}` → джоба generate с `abc=` из score.abc родителя (или `.abc` кавера), поля `overdub_parent/overdub_gain` → мета ребёнка `overdub_of`.
- После done ребёнка: numpy-микс `_mix_overdub` (порт из yue-studio: gain, кроссфейд 50 мс, пик-нормализация) → `<parent>.d/overdub-<child>.flac` + метрики.
- UI: панель на карточке: стиль + чипы партий (odPartyChips, 14 пар ru/en — константа в app.js), gain 0.1–1, лирика (кнопка «♪ текст этого трека»), «Сгенерировать овердаб».

### Этап 9 — экспорт и облегчённая библиотека стилей
- `GET /api/outputs/{name}/wav` по образцу mp3 (ffmpeg PCM_16) + кнопка WAV.
- `static/stylelib.js`: данные `groups.js` (29 групп) + `presets.js` (10 пресетов с сидами) + плоский словарь `slotDict` ru→en — снять `export`.
- UI: сворачиваемый блок «Библиотека стилей» под полем стиля: группы → стили; клик — применить строкой 1:1 (пресет подставляет и seed); «сохранить текущий стиль» → свои группы в localStorage.
- Словарь ru→en: при отправке автоматом заменять известные русские слова/фразы из `slotDict`; непереведённую кириллицу подсветить предупреждением. Без Ollama — только словарь.

### Этап 10 — UI-победы
- «↺ Повторить» на карточке: заполнить форму из меты (style_base/voice/lyrics/cot/cfg/seed уже в json).
- Веер «×N» (2..8): базовый seed (случайный при -1) + i.
- Инструментал: тумблер «без слов» + длина (auto/3/8/16/36 × `[Instrumental]`).
- Плеер: prev/next по галерее, автопереход по окончании, громкость в localStorage.
- Галерея: поиск (title/lyrics/style), переименование — `POST /api/gallery/{stem}/rename {title}` (только мета, файлы не трогаем).
- Тема: светлая/тёмная через CSS-переменные + `data-theme`, выбор в localStorage.

## Порядок и зависимости
0 → 1 → 2 → 3 (фундамент артефактов) → 4, 5 (нужны score/latent) → 6 → 7, 8 → 9, 10 (независимы).

## Тестирование и приёмка
- Юнит (`tests/`, pytest, без GPU): `arc.apply_arc`, `abcparse.parse_abc` на фикстурах, клампинг dsp_chains, rename/delete/gallery через TestClient.
- Сервер на тестовом порту **8221** (`YUE2_PORT=8221`), прод на 8220 не трогать до приёмки.
- GPU-проверки вручную: draft-генерация → score.abc/latent.npy; /api/plan → правка → рендер; превью секции; analyze; dsp apply+preview; овердаб; веер ×3; отмена queued.
- `test_smoke.py` не трогаем; README дополнить (эндпоинты, артефакты, ограничения).

## Риски
- **librosa vs запиненный .venv** — dry-run до установки, фолбэк third_party/metrics venv.
- **Фильтры статического ffmpeg** — проверка `-filters` до написания dsp_chains, деградация графа.
- **latent.npy на 10–14 мин** — float16 ≈ 20–40 МБ, чистится удалением записи.
- **409 на /plan и /preview при генерации** — осознанно (одна GPU), понятное сообщение в UI.
- **Двойной план при arc** — как в yue-studio, +10–60 c.