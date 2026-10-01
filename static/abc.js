// ABC-редактор: текст → ноты (abcjs SVG) → звук (Web Audio + локальные саундфонты).
// Курсор: играющая нота подсвечивается, следует автоскролл и выделение в тексте.
const $ = id => document.getElementById(id);

// Саундфонты инструментов качаются скриптом scripts/fetch_soundfonts.sh
const SF_URL = '/static/vendor/abcjs/soundfonts/';
const SF_MULT = 0.4;   // уровень FluidR3_GM, как в дефолте abcjs

const EXAMPLES = {
simple:
`X:1
T:Мелодия с аккордами
M:4/4
L:1/8
Q:1/4=100
K:G
"Am"A2 AB c2 ed|"G"B2 d2 "D"e3 f|"Em"gf ed "C"e2 c2|"D"A4 z4|
"Am"A2 AB c2 ed|"G"B2 d2 "D"e3 f|"Em"gf ed "C"B2 A2|"Am"A8|]`,
voices:
`X:1
T:Два голоса (дуэт)
M:3/4
L:1/4
Q:1/4=92
K:Am
V:1 program=0 0 name="Фортепиано"
e2 f|g2 a2|b2 a2|g3|a2 b|c'3|b2 g|e3||
V:2 program=0 24 name="Гитара"
A,2 C,|E,2 A,2|G,2 E,|A,3|A,2 B,|C,3|E,2 C,|A,3||`,
yue:
`X:1
T:Лирика и секции
M:4/4
L:1/8
Q:1/4=110
K:Em
% verse
E2 E2 G2 A2|G2 F2 E2 D2|E2 E2 G2 G2|A4 z4|
w: Во по-ле берё-за сто-я-ла
% chorus
B2 B2 c2 B2|A2 G2 A2 B2|c2 B2 G4|E8|
w: зе-лё-на-я, куд-ря-ва-я`,
};

const S = {
  text: '', visualObj: null, renderId: 0,
  timer: null, synth: null, synthReady: false, synthRenderId: -1,
  playing: false, pausedAt: 0, durationSec: 0,
  loop: false, hl: null, fg: '#e8eaf2', tempoFactor: 1,
  rafOn: false, saveTimer: null, renderTimer: null,
};

function fmtT(s){ s = Math.max(0, s || 0); const m = Math.floor(s/60);
  const x = (s - m*60).toFixed(1); return m + ':' + (x < 10 ? '0' : '') + x; }
const errHtml = e => `<span class="err">${(e && e.message || e)}</span>`;

// ==================== рендер нот ====================
function themeFg() {
  return getComputedStyle(document.documentElement).getPropertyValue('--text').trim() || '#e8eaf2';
}
function renderOpts() {
  return {
    responsive: 'resize',
    add_classes: true,
    foregroundColor: S.fg,
    wrap: { minSpacing: 1.4, maxSpacing: 2.4 },
    paddingtop: 4, paddingbottom: 4, paddingleft: 2, paddingright: 2,
  };
}

function renderSheet() {
  const text = $('abcText').value;
  S.fg = themeFg();
  unhighlight();
  let objs = null;
  try { objs = ABCJS.renderAbc('sheet', text, renderOpts()); }
  catch (e) { renderFail(e); return; }
  if (!objs || !objs.length) {
    if (!text.trim()) { $('sheet').innerHTML = ''; $('sheetEmpty').hidden = false; }
    else renderFail(new Error('Партитура пуста — проверьте заголовок X: и K:'));
    return;
  }
  S.visualObj = objs[0];
  S.renderId++;
  $('sheetEmpty').hidden = true;
  $('sheetErr').hidden = true;
  showWarnings(objs[0].warnings);
  buildTimer();
  updateChips();
  queueSave();
}

function renderFail(e) {
  $('sheetErr').textContent = String(e.message || e);
  $('sheetErr').hidden = false;
  $('sheetEmpty').hidden = true;
  $('chParse').hidden = true; $('chBpm').hidden = true;
  S.visualObj = null; S.durationSec = 0;
  if (S.playing) stop();
  updateTime(0);
}

function showWarnings(list) {
  const box = $('edStat');
  if (!list || !list.length) { box.textContent = ''; return; }
  const msgs = list.map(w => w.message || w.text || String(w)).join('\n');
  box.innerHTML = `<span class="err">${msgs.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}</span>`;
}

function updateChips() {
  const v = S.visualObj;
  if (!v) return;
  const title = (v.metaText && v.metaText.title) || '';
  const measures = maxMeasure(v);
  const dur = fmtT(S.durationSec);
  $('chParse').hidden = false;
  $('chParse').className = 'chip';
  $('chParse').textContent = (title ? title + ' · ' : '') + measures + ' тактов · ' + dur;
  const bpm = Math.round(baseBpm());
  $('chBpm').hidden = false;
  $('chBpm').textContent = bpm + ' BPM';
}

function maxMeasure(v) {
  let m = 0;
  for (const t of (S.timer ? S.timer.noteTimings : []))
    if (typeof t.measureNumber === 'number' && t.measureNumber > m) m = t.measureNumber;
  return m + 1;
}

function baseBpm() {
  try {
    const t = S.visualObj && S.visualObj.metaText && S.visualObj.metaText.tempo;
    const b = S.visualObj.getBpm(t);
    if (b && isFinite(b) && b > 0) return b;
  } catch {}
  return 120;
}

// ==================== тайминг (курсор) ====================
function effectiveQpm() {
  const f = parseFloat($('tempoSel').value) || 1;
  if (f === 1) return null;               // темп из Q: — таймер и синт резолвят одинаково
  return Math.max(10, Math.round(baseBpm() * f));
}

function buildTimer() {
  const pos = playingPosSec();
  if (S.timer) { S.timer.pause(); clearTimeout(S.timer.joggerTimer); }
  S.timer = new ABCJS.TimingCallbacks(S.visualObj, {
    qpm: effectiveQpm(),
    eventCallback: onNoteEvent,
  });
  S.durationSec = S.timer.noteTimings.length
    ? S.timer.noteTimings[S.timer.noteTimings.length - 1].milliseconds / 1000 : 0;
  if (S.playing && S.durationSec > pos + 0.05) {
    S.timer.start(pos, 'seconds');
    S.timer.pausedPercent = null;
  } else if (S.playing) {
    stop();   // правка укоротила партитуру ниже позиции игры
  } else if (pos > 0) {
    // курсор на паузе — подсветить ноту в этой позиции
    S.timer.setProgress(Math.min(pos, S.durationSec), 'seconds');
  }
  updateTime(playingPosSec());
}

function playingPosSec() {
  if (S.playing && S.timer) return Math.min(S.timer.currentMillisecond(), S.durationSec * 1000) / 1000;
  return S.pausedAt;
}

function onNoteEvent(ev) {
  if (!ev) { onTuneEnd(); return; }
  unhighlight();
  if (ev.elements) {
    setGroupColor(ev.elements, 'var(--accent2)');
    S.hl = ev.elements;
  }
  autoscroll(ev);
  // курсор в тексте — только когда пользователь не правит прямо сейчас
  const ta = $('abcText');
  if (document.activeElement !== ta && ev.startChar >= 0 && ta.setSelectionRange)
    ta.setSelectionRange(ev.startChar, Math.max(ev.endChar, ev.startChar + 1));
  updateTime(ev.milliseconds / 1000);
}

function onTuneEnd() {
  if (S.loop && S.playing) {
    S.timer.setProgress(0, 'seconds');
    S.timer.startTime = performance.now();     // таймер пойдёт по второму кругу
    if (S.synthReady) { try { S.synth.seek(0, 'seconds'); } catch {} }
    return 'continue';
  }
  finishStop();
  return true;
}

function finishStop() {
  S.playing = false; S.pausedAt = 0;
  unhighlight();
  if (S.timer) { S.timer.pause(); S.timer.reset(); clearTimeout(S.timer.joggerTimer); }
  // хвост последней ноты (фейд 200 мс) не обрываем — синт доиграет сам
  $('btnPlay').textContent = '▶';
  updateTime(0); updateProgress(0);
}

function setGroupColor(elements, color) {
  if (!elements) return;
  for (const arr of elements)
    for (const el of arr) {
      if (!el || !el.setAttribute) continue;
      el.setAttribute('fill', color);
      el.setAttribute('stroke', color);
    }
}
function unhighlight() {
  if (S.hl) setGroupColor(S.hl, S.fg);
  S.hl = null;
}

function autoscroll(ev) {
  if (!ev.elements || !ev.elements.length || !ev.elements[0].length) return;
  const box = $('sheetBox'), el = ev.elements[0][0];
  const r = el.getBoundingClientRect(), b = box.getBoundingClientRect();
  if (r.top < b.top + 8 || r.bottom > b.bottom - 24)
    box.scrollTop += r.top - b.top - Math.max(0, (b.height - r.height) / 2);
}

function updateTime(sec) {
  $('tTime').textContent = fmtT(sec) + ' / ' + fmtT(S.durationSec);
  updateProgress(sec);
}
function updateProgress(sec) {
  $('pfill').style.width = S.durationSec ? (100 * sec / S.durationSec).toFixed(2) + '%' : '0%';
}

// ==================== синт ====================
function ensureSynth() {
  if (!ABCJS.synth.supportsAudio()) {
    $('edStat').innerHTML = '<span class="err">Браузер не поддерживает Web Audio</span>';
    return false;
  }
  return true;
}

async function rebuildSynth() {
  if (S.synth) { try { S.synth.stop(); } catch {} }
  S.synthReady = false;
  S.synth = new ABCJS.synth.CreateSynth();
  const opts = {
    soundFontUrl: SF_URL,
    soundFontVolumeMultiplier: SF_MULT,
    onEnded: () => { if (!S.playing && !S.pausedAt) return; },   // хвост — тихо
  };
  const q = effectiveQpm();
  if (q) opts.qpm = q;
  await S.synth.init({ visualObj: S.visualObj, options: opts });
  await S.synth.prime();
  S.synthReady = true;
  S.synthRenderId = S.renderId;
}

async function play() {
  if (!S.visualObj) return;
  if (!ensureSynth()) return;
  const from = Math.min(S.pausedAt || 0, Math.max(0, S.durationSec - 0.05));
  $('btnPlay').textContent = '…';
  try {
    if (!S.synthReady || S.synthRenderId !== S.renderId) await rebuildSynth();
    if (!S.synthReady) return;
    S.synth.seek(from, 'seconds');   // не running → просто ставит позицию
    S.synth.resume();                // = start() от сохранённой позиции
    S.timer.start(from, 'seconds');
    S.timer.pausedPercent = null;
    S.playing = true; S.pausedAt = 0;
    $('btnPlay').textContent = '⏸';
    uiTick();
  } catch (e) {
    S.playing = false;
    $('btnPlay').textContent = '▶';
    showSynthError(e);
  }
}

function pause() {
  if (!S.playing) return;
  S.pausedAt = S.synth ? S.synth.pause() : playingPosSec();
  if (S.timer) { S.timer.pause(); clearTimeout(S.timer.joggerTimer); }
  S.playing = false;
  unhighlight();
  $('btnPlay').textContent = '▶';
}

function stop() {
  if (S.synthReady && S.synth) { try { S.synth.stop(); } catch {} }
  if (S.timer) { S.timer.stop(); clearTimeout(S.timer.joggerTimer); }
  S.playing = false; S.pausedAt = 0;
  unhighlight();
  $('btnPlay').textContent = '▶';
  updateTime(0);
}

function showSynthError(e) {
  const msg = String((e && e.message) || e);
  const m = msg.match(/\/([^/]+)-mp3\//);
  if (m) {
    $('edStat').innerHTML = `<span class="err">Саундфонта инструмента «${m[1]}» не скачана — этого голоса не будет. ` +
      `Допишите инструмент в scripts/fetch_soundfonts.sh и запустите его, или укажите программу из списка в шпаргалке.</span>`;
  } else {
    $('edStat').innerHTML = errHtml(new Error('Синт: ' + msg));
  }
}

// ==================== переходы ====================
async function seekTo(sec) {
  sec = Math.max(0, Math.min(S.durationSec - 0.02, sec));
  if (!S.timer || !S.durationSec) return;
  if (S.playing) {
    S.timer.pause(); clearTimeout(S.timer.joggerTimer);
    if (S.synthReady) { try { S.synth.seek(sec, 'seconds'); } catch {} }
    S.timer.start(sec, 'seconds');
    S.timer.pausedPercent = null;
  } else {
    S.pausedAt = sec;
    S.timer.setProgress(sec, 'seconds');   // подсветит ноту в новой позиции
    updateTime(sec);
  }
}

// клик по ноте → ближайшее событие партитуры (по координатам SVG) → переход
$('sheet').addEventListener('click', e => {
  if (!S.timer || !S.visualObj) return;
  const svg = $('sheet').querySelector('svg');
  if (!svg || !svg.getScreenCTM) return;
  const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse());
  let best = null, bestD = Infinity;
  for (const t of S.timer.noteTimings) {
    if (t.type !== 'event' || typeof t.left !== 'number') continue;
    const dx = Math.max(t.left - pt.x, 0, pt.x - (t.left + (t.width || 0)));
    const dy = Math.max(t.top - pt.y, 0, pt.y - (t.top + (t.height || 20)));
    const d = dx + dy * 2;   // своя строка важнее, чем позиция в ней
    if (d < bestD) { bestD = d; best = t; }
  }
  if (best && best.milliseconds >= 0) seekTo(best.milliseconds / 1000);
});
$('pbar').addEventListener('click', e => {
  const r = $('pbar').getBoundingClientRect();
  seekTo((e.clientX - r.left) / r.width * S.durationSec);
});

// ==================== транспорт и клавиши ====================
$('btnPlay').onclick = () => { S.playing ? pause() : play(); };
$('btnStop').onclick = stop;
$('btnLoop').onclick = () => { S.loop = !S.loop; $('btnLoop').classList.toggle('on', S.loop); };
$('tempoSel').onchange = async () => {
  if (!S.visualObj) return;
  const wasPlaying = S.playing;
  if (wasPlaying) {
    // та же музыкальная позиция в новом темпе: секунды × (старый/новый темп)
    const nf = parseFloat($('tempoSel').value) || 1;
    const oldPos = playingPosSec();
    pause();   // S.pausedAt = позиция синтезатора (ещё старый темп)
    S.pausedAt = Math.max(0, Math.min(oldPos * (S.tempoFactor / nf), S.durationSec - 0.05));
    S.tempoFactor = nf;
  }
  buildTimer();
  if (!wasPlaying) return;
  $('btnPlay').textContent = '…';
  try {
    await rebuildSynth();
    S.synth.seek(S.pausedAt, 'seconds');
    S.synth.resume();
    S.timer.start(S.pausedAt, 'seconds');
    S.timer.pausedPercent = null;
    S.playing = true; S.pausedAt = 0;
    $('btnPlay').textContent = '⏸';
    uiTick();
  } catch (e) { $('btnPlay').textContent = '▶'; showSynthError(e); }
};

function uiTick() {
  if (S.rafOn) return;
  S.rafOn = true;
  (function tick() {
    if (!S.playing || !S.timer) { S.rafOn = false; return; }
    updateTime(Math.min(S.timer.currentMillisecond(), S.durationSec * 1000) / 1000);
    requestAnimationFrame(tick);
  })();
}

document.addEventListener('keydown', e => {
  const t = e.target;
  const inField = t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.isContentEditable);
  // --- миди-клавиатура: при записи клавиатура принадлежит ей целиком ---
  if (KB.armed) {
    if (e.code === 'Space' || e.code === 'Escape') { e.preventDefault(); kbStopRecord(); return; }
    if (e.code === 'KeyZ') { e.preventDefault(); kbShift(-1); return; }
    if (e.code === 'KeyX') { e.preventDefault(); kbShift(1); return; }
    if (KB_KEYMAP[e.code] !== undefined) {
      e.preventDefault();
      if (!e.repeat) kbNoteOn(kbBaseMidi() + KB_KEYMAP[e.code], e.code);
      return;
    }
  } else if (!inField && t !== ta && $('kbPanel').open) {
    // панель открыта, фокус вне полей — клавиши просто играют
    if (KB_KEYMAP[e.code] !== undefined) {
      e.preventDefault();
      if (!e.repeat) kbNoteOn(kbBaseMidi() + KB_KEYMAP[e.code], e.code);
      return;
    }
    if (e.code === 'KeyZ') { e.preventDefault(); kbShift(-1); return; }
    if (e.code === 'KeyX') { e.preventDefault(); kbShift(1); return; }
  }
  // --- транспорт ---
  if (inField) return;
  if (t === ta) { if (e.code === 'Escape') ta.blur(); return; }   // в тексте — только Esc
  if (e.code === 'Space') { e.preventDefault(); S.playing ? pause() : play(); }
  else if (e.code === 'Escape') stop();
});

// ==================== редактор ====================
let ta = $('abcText');
ta.addEventListener('input', () => {
  clearTimeout(S.renderTimer);
  S.renderTimer = setTimeout(renderSheet, 250);
});

function queueSave() {
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(() => {
    try {
      localStorage.setItem('yue_abc_editor_text', ta.value);
      const c = $('chSaved');
      c.hidden = false; c.className = 'chip ok'; c.textContent = 'сохранено локально';
      clearTimeout(queueSave.flash);
      queueSave.flash = setTimeout(() => { c.hidden = true; }, 1500);
    } catch {}
  }, 500);
}

$('btnCopy').onclick = async () => {
  try { await navigator.clipboard.writeText(ta.value); $('edStat').textContent = 'скопировано в буфер'; }
  catch { ta.select(); document.execCommand('copy'); $('edStat').textContent = 'скопировано'; }
};

$('btnDl').onclick = () => {
  const title = (S.visualObj && S.visualObj.metaText && S.visualObj.metaText.title || 'tune')
    .replace(/[^\wа-яёА-ЯЁ -]/g, '').trim() || 'tune';
  const blob = new Blob([ta.value], { type: 'text/plain;charset=utf-8' });
  const a = $('btnDl');
  a.href = URL.createObjectURL(blob);
  a.download = title + '.abc';
};

$('fileIn').onchange = async () => {
  const f = $('fileIn').files[0];
  if (!f) return;
  ta.value = await f.text();
  renderSheet();
  $('fileIn').value = '';
};

$('btnClear').onclick = () => { ta.value = ''; renderSheet(); ta.focus(); };

(function initExMenu() {
  const menu = $('exMenu');
  menu.querySelector('.iconbtn').onclick = e => { e.stopPropagation(); menu.classList.toggle('open'); };
  document.addEventListener('click', e => { if (!menu.contains(e.target)) menu.classList.remove('open'); });
  for (const a of menu.querySelectorAll('a[data-ex]'))
    a.onclick = () => { ta.value = EXAMPLES[a.dataset.ex]; renderSheet(); menu.classList.remove('open'); };
})();

// ==================== тема и запуск ====================
(function () {
  const saved = localStorage.getItem('yue_theme');
  if (saved) document.documentElement.dataset.theme = saved;
  $('themeToggle').onclick = () => {
    const nxt = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = nxt;
    localStorage.setItem('yue_theme', nxt);
    renderSheet();   // перерисовать ноты в цветах темы
  };
})();

$('instHint').textContent =
  'Скачанные инструменты (GM-программа в V: program=0 N): фортепиано 0, эл.пиано 4, ' +
  'колокольчики 9, вибрафон 11, муз.шкатулка 10, орган 16, церковный орган 19, аккордеон 21, ' +
  'гармоника 22, гитара нейлон 24 / сталь 25 / джаз 26 / чистая 27 / глушёная 28 / овердрайв 29 / дисторшн 30, ' +
  'контрабас 32, бас-гитара 33/34, скрипка 40, альт 41, виолончель 42, контрабас-смычок 43, ' +
  'арфа 46, тимпаны 47, струнные 48/49, пиццикато 45, труба 56, приглушённая труба 59, ' +
  'тромбон 57, валторна 60, альт-сакс 65, тенор-сакс 66, кларнет 71, гобой 68, флейта 73, ' +
  'флейта-пан 75, хор 52. Остальные — без звука.';

(async function init() {
  if (typeof ABCJS === 'undefined') {
    $('sheetErr').textContent = 'Не загрузилась библиотека abcjs (static/vendor/abcjs/abcjs-basic-min.js)';
    $('sheetErr').hidden = false;
    return;
  }
  ta.value = localStorage.getItem('yue_abc_editor_text') || EXAMPLES.simple;
  renderSheet();
})();

// ==================== миди-клавиатура ====================
// Раскладка как в DAW: A S D F G H J K L ; ' — белые, W E T Y U O P — чёрные.
// Звук — те же локальные саундфонты (по ноте), запись → ABC с квантизацией в сетку.
const KB_KEYMAP = { KeyA:0, KeyW:1, KeyS:2, KeyE:3, KeyD:4, KeyF:5, KeyT:6, KeyG:7,
  KeyY:8, KeyH:9, KeyU:10, KeyJ:11, KeyK:12, KeyO:13, KeyL:14, KeyP:15, Semicolon:16, Quote:17 };

// [группа, имя саундфонта, GM-программа, подпись] — порядок GM
const KB_INSTRUMENTS = [
  ['Клавишные', 'acoustic_grand_piano', 0, 'фортепиано'],
  ['Клавишные', 'electric_piano_1', 4, 'электропиано'],
  ['Клавишные', 'harpsichord', 6, 'клавесин'],
  ['Молоточковые', 'glockenspiel', 9, 'колокольчики'],
  ['Молоточковые', 'vibraphone', 11, 'вибрафон'],
  ['Молоточковые', 'music_box', 10, 'муз. шкатулка'],
  ['Органы', 'drawbar_organ', 16, 'орган'],
  ['Органы', 'church_organ', 19, 'церковный орган'],
  ['Язычковые', 'accordion', 21, 'аккордеон'],
  ['Язычковые', 'harmonica', 22, 'гармоника'],
  ['Гитары', 'acoustic_guitar_nylon', 24, 'гитара (нейлон)'],
  ['Гитары', 'acoustic_guitar_steel', 25, 'гитара (сталь)'],
  ['Гитары', 'electric_guitar_jazz', 26, 'электрогитара (джаз)'],
  ['Гитары', 'electric_guitar_clean', 27, 'электрогитара (чистая)'],
  ['Гитары', 'electric_guitar_muted', 28, 'электрогитара (глушёная)'],
  ['Гитары', 'overdriven_guitar', 29, 'овердрайв'],
  ['Гитары', 'distortion_guitar', 30, 'дисторшн'],
  ['Басы', 'acoustic_bass', 32, 'контрабас'],
  ['Басы', 'electric_bass_finger', 33, 'бас-гитара (палец)'],
  ['Басы', 'electric_bass_pick', 34, 'бас-гитара (медиатор)'],
  ['Струнные', 'violin', 40, 'скрипка'],
  ['Струнные', 'viola', 41, 'альт'],
  ['Струнные', 'cello', 42, 'виолончель'],
  ['Струнные', 'contrabass', 43, 'контрабас (смычок)'],
  ['Струнные', 'pizzicato_strings', 45, 'пиццикато'],
  ['Струнные', 'orchestral_harp', 46, 'арфа'],
  ['Струнные', 'timpani', 47, 'тимпаны'],
  ['Струнные', 'string_ensemble_1', 48, 'струнные'],
  ['Струнные', 'string_ensemble_2', 49, 'струнные 2'],
  ['Голос', 'choir_aahs', 52, 'хор'],
  ['Духи медные', 'trumpet', 56, 'труба'],
  ['Духи медные', 'trombone', 57, 'тромбон'],
  ['Духи медные', 'muted_trumpet', 59, 'труба (сурдина)'],
  ['Духи медные', 'french_horn', 60, 'валторна'],
  ['Духи деревянные', 'alto_sax', 65, 'альт-сакс'],
  ['Духи деревянные', 'tenor_sax', 66, 'тенор-сакс'],
  ['Духи деревянные', 'oboe', 68, 'гобой'],
  ['Духи деревянные', 'clarinet', 71, 'кларнет'],
  ['Духи деревянные', 'flute', 73, 'флейта'],
  ['Духи деревянные', 'pan_flute', 75, 'флейта-пан'],
];

const KB = {
  armed: false, notes: [], undoText: null,
  instr: 'acoustic_grand_piano', prog: 0, instrRu: 'фортепиано',
  octave: 3, meter: '4/4', bpm: 100, metro: false,
  metroTimer: null, nextBeat: 0, beatN: 0,
  ctx: null, buffers: new Map(), missing: new Set(),
  active: new Map(),   // midi → {code, t0, src, gain, buf}
};

// имена нот в саундфонтах — бемольные (Gb3, Bb0), см. fetch_soundfonts.sh
const KB_SF_NAMES = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B'];
const KB_ABC = ['C','D','E','F','G','A','B'];
const KB_SHARP_OF = { 1:0, 3:1, 6:3, 8:4, 10:5 };   // диез → номер белой ноты

function kbSfName(midi) { return KB_SF_NAMES[midi % 12] + (Math.floor(midi / 12) - 1); }
function kbBaseMidi() { return (KB.octave + 1) * 12; }   // C3 = 48

function kbCtx() {
  if (KB.ctx) return KB.ctx;
  let c = (ABCJS.synth.activeAudioContext && ABCJS.synth.activeAudioContext()) || null;
  if (!c) {
    c = new (window.AudioContext || window.webkitAudioContext)();
    try { ABCJS.synth.registerAudioContext(c); } catch {}
  }
  KB.ctx = c;
  return c;
}

async function kbBuffer(midi) {
  const ctx = kbCtx();
  const key = KB.instr + '|' + midi;
  if (KB.buffers.has(key)) return KB.buffers.get(key);
  let buf = null;
  for (const ins of [KB.instr, 'acoustic_grand_piano']) {   // вне диапазона — фортепиано
    const fk = ins + '|' + midi;
    if (KB.buffers.has(fk)) { buf = KB.buffers.get(fk); break; }
    if (KB.missing.has(fk)) continue;
    try {
      const r = await fetch(`${SF_URL}${ins}-mp3/${kbSfName(midi)}.mp3`);
      if (!r.ok) throw new Error(r.status);
      buf = await ctx.decodeAudioData(await r.arrayBuffer());
      KB.buffers.set(fk, buf);
      break;
    } catch { KB.missing.add(fk); }
  }
  if (buf) KB.buffers.set(key, buf);
  return buf;
}

async function kbNoteOn(midi, code) {
  if (KB.active.has(midi)) return;
  const ctx = kbCtx();
  if (ctx.state === 'suspended') ctx.resume();
  const rec = { code: code || null, t0: ctx.currentTime, src: null, gain: null, buf: null };
  KB.active.set(midi, rec);
  kbKeyLight(midi, true);
  const buf = await kbBuffer(midi);
  if (!buf) { kbStrip(midi, true); return; }               // нет саундфонта — тихо
  if (!KB.active.has(midi)) return;                        // успели отпустить
  rec.buf = buf;
  const src = ctx.createBufferSource(); src.buffer = buf;
  const g = ctx.createGain();
  g.gain.setValueAtTime(0, ctx.currentTime);
  g.gain.linearRampToValueAtTime(0.9, ctx.currentTime + 0.008);
  src.connect(g); g.connect(ctx.destination);
  src.start();
  rec.src = src; rec.gain = g;
  kbStrip(midi, false);
}

function kbNoteOff(midi) {
  const rec = KB.active.get(midi);
  if (!rec) return;
  KB.active.delete(midi);
  kbKeyLight(midi, false);
  const ctx = kbCtx(), now = ctx.currentTime;
  if (rec.gain) {
    rec.gain.gain.cancelScheduledValues(now);
    rec.gain.gain.setValueAtTime(rec.gain.gain.value, now);
    rec.gain.gain.setTargetAtTime(0, now, 0.04);
    try { rec.src.stop(now + 0.3); } catch {}
  }
  if (KB.armed && rec.buf)
    KB.notes.push({ midi, t0: rec.t0, t1: Math.min(now, rec.t0 + rec.buf.duration) });
}

function kbAllNotesOff() { for (const m of [...KB.active.keys()]) kbNoteOff(m); }

function kbShift(dir) {
  KB.octave = Math.max(1, Math.min(6, KB.octave + dir));
  kbAllNotesOff();
  kbOctLabel();
}

function kbOctLabel() {
  $('kbOct').textContent = `C${KB.octave}–F${KB.octave + 1}`;
}

// ---- визуальная клавиатура (2 октавы), играет и мышью ----
const KB_PTR = new Map();   // pointerId → midi
function kbBuildKeys() {
  const box = $('kbKeys');
  const whites = [];
  for (let o = 0; o < 2; o++) for (const s of [0, 2, 4, 5, 7, 9, 11]) whites.push(o * 12 + s);
  whites.push(24);
  const wpc = 100 / whites.length;
  whites.forEach((s, i) => {
    const el = document.createElement('div');
    el.className = 'wk';
    el.style.left = (i * wpc) + '%'; el.style.width = wpc + '%';
    el.dataset.semi = s;
    box.appendChild(el);
  });
  const blackPos = { 1: 0.66, 3: 1.76, 6: 3.62, 8: 4.68, 10: 5.74 };
  for (let o = 0; o < 2; o++)
    for (const [s, pos] of Object.entries(blackPos)) {
      const el = document.createElement('div');
      el.className = 'bk';
      el.style.left = ((o * 7 + pos) * wpc) + '%'; el.style.width = (wpc * 0.62) + '%';
      el.dataset.semi = o * 12 + (+s);
      box.appendChild(el);
    }
  box.addEventListener('pointerdown', e => {
    const t = e.target.closest('[data-semi]');
    if (!t) return;
    e.preventDefault();
    const midi = kbBaseMidi() + (+t.dataset.semi);
    KB_PTR.set(e.pointerId, midi);
    kbNoteOn(midi);
  });
  box.addEventListener('contextmenu', e => e.preventDefault());
  const up = e => {
    if (!KB_PTR.has(e.pointerId)) return;
    kbNoteOff(KB_PTR.get(e.pointerId));
    KB_PTR.delete(e.pointerId);
  };
  document.addEventListener('pointerup', up);
  document.addEventListener('pointercancel', up);
}
function kbKeyLight(midi, on) {
  const el = $('kbKeys').querySelector(`[data-semi="${midi - kbBaseMidi()}"]`);
  if (el) el.classList.toggle('down', on);
}

// ---- лента набранных нот и статус ----
function kbStrip(midi, missing) {
  const box = $('kbStrip');
  const span = document.createElement('span');
  span.textContent = kbName(midi) + (missing ? '!' : ' ');
  if (missing) span.style.color = 'var(--warn)';
  box.appendChild(span);
  while (box.childNodes.length > 24) box.removeChild(box.firstChild);
}
function kbStatUpdate() {
  const el = $('kbStat');
  const dur = KB.notes.length ? KB.notes[KB.notes.length - 1].t1 - KB.notes[0].t0 : 0;
  if (KB.armed)
    el.innerHTML = `<span class="rec">● запись</span> · нот: ${KB.notes.length} · ${fmtT(dur)} · пробел — стоп`;
  else if (KB.undoText)
    el.textContent = `вставлено нот: ${KB.undoText.len} · ${fmtT(dur)} — «↺ отменить» вернёт прежний текст`;
  else
    el.textContent = 'Клавиши играют при открытой панели — жмите «запись» и набирайте мелодию';
}

// ---- метроном ----
function kbMetroStart() {
  if (!KB.metro) return;
  const ctx = kbCtx();
  KB.beatN = 0;
  KB.nextBeat = ctx.currentTime + 0.1;
  KB.metroTimer = setInterval(() => {
    const beats = parseInt(KB.meter) || 4;
    while (KB.nextBeat < ctx.currentTime + 0.15) {
      kbClick(KB.nextBeat, KB.beatN % beats === 0);
      KB.beatN++;
      KB.nextBeat += 60 / KB.bpm;
    }
  }, 40);
}
function kbClick(t, accent) {
  const ctx = kbCtx();
  const o = ctx.createOscillator(), g = ctx.createGain();
  o.frequency.value = accent ? 1600 : 1050;
  g.gain.setValueAtTime(accent ? 0.5 : 0.28, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + 0.05);
  o.connect(g); g.connect(ctx.destination);
  o.start(t); o.stop(t + 0.06);
}
function kbMetroStop() {
  if (KB.metroTimer) { clearInterval(KB.metroTimer); KB.metroTimer = null; }
}

// ---- запись → ABC ----
function kbArm() {
  if (S.playing) stop();
  KB.notes = []; KB.undoText = null;
  $('btnKbUndo').hidden = true;
  KB.armed = true;
  $('kb').classList.add('recOn');
  $('btnKbRec').textContent = '⏹ стоп записи';
  const ctx = kbCtx();
  if (ctx.state === 'suspended') ctx.resume();
  kbMetroStart();
  kbStatUpdate();
}
function kbStopRecord() {
  KB.armed = false;
  kbMetroStop();
  $('kb').classList.remove('recOn');
  $('btnKbRec').textContent = '⏺ запись';
  kbAllNotesOff();
  if (!KB.notes.length) { kbStatUpdate(); return; }
  kbInsert();
}

// midi → ABC-нота: C4(60)=C, C3=C,, c5=c; диезы явно (^F)
function kbName(midi) {
  const oct = Math.floor(midi / 12) - 1;
  const pc = midi % 12;
  const wi = [0, 2, 4, 5, 7, 9, 11].indexOf(pc);
  if (oct <= 4)
    return (wi >= 0 ? KB_ABC[wi] : '^' + KB_ABC[KB_SHARP_OF[pc]]) + ','.repeat(4 - oct);
  return (wi >= 0 ? KB_ABC[wi].toLowerCase() : '^' + KB_ABC[KB_SHARP_OF[pc]].toLowerCase())
    + "'".repeat(oct - 5);
}
function kbDurSuffix(u) {
  u = Math.round(u * 2) / 2;                 // точность до 1/16 (половинки единицы)
  if (u <= 0.5) return '/2';
  if (u === 1) return '';
  if (Number.isInteger(u)) return String(u);
  return Math.floor(u) + '/2';               // напр. 3/2
}
function kbToAbc() {
  if (!KB.notes.length) return '';
  const notes = [...KB.notes].sort((a, b) => a.t0 - b.t0 || a.midi - b.midi);
  const spq = 60 / KB.bpm;                   // секунд на четверть
  const gRaw = parseFloat($('kbGrid').value);
  const g = gRaw > 0 ? gRaw : 0.5;           // «выкл» — округляем до 1/32
  const barU = { '4/4': 8, '3/4': 6, '2/4': 4, '6/8': 6 }[KB.meter] || 8;
  const q = u => Math.round(u / g) * g;
  const t0 = notes[0].t0;
  const ev = notes.map(n => ({
    midi: n.midi, raw: n.t0,
    on: Math.max(0, q((n.t0 - t0) / spq * 2)),
    dur: Math.max(g, q((n.t1 - n.t0) / spq * 2)),
  }));
  // аккорд — только реально одновременные нажатия (±40 мс), иначе джиттер таймеров
  // склеит подряд идущие ноты в фальшивый аккорд
  const items = [];
  for (let i = 0; i < ev.length; i++) {
    const grp = [ev[i]];
    while (i + 1 < ev.length && Math.abs(ev[i + 1].raw - grp[0].raw) < 0.04) grp.push(ev[++i]);
    items.push({
      midi: grp.map(x => x.midi),
      on: Math.min(...grp.map(x => x.on)),
      dur: Math.min(...grp.map(x => x.dur)),
    });
  }
  items.sort((a, b) => a.on - b.on);
  // монодорожка: столкновение квантованной позиции — легато; нота не налезает на следующую
  for (let i = 0; i < items.length; i++) {
    if (i > 0 && items[i].on <= items[i - 1].on + 1e-6)
      items[i].on = items[i - 1].on + items[i - 1].dur;
    if (items[i].midi.length === 1 && i + 1 < items.length) {
      const room = Math.floor((items[i + 1].on - items[i].on) / g) * g;
      if (room > 0 && items[i].dur > room) items[i].dur = Math.max(g, room);
    }
  }
  const tokens = [];
  let inBar = 0, pos = 0;
  const emit = (body, units) => {
    tokens.push(body + kbDurSuffix(units));
    inBar += units;
    while (inBar >= barU - 1e-6) { tokens.push('|'); inBar -= barU; }
  };
  for (const it of items) {
    if (it.on - pos > 1e-6) emit('z', q(it.on - pos));   // пауза из пропуска
    emit(it.midi.length > 1 ? '[' + it.midi.map(kbName).join('') + ']' : kbName(it.midi[0]), it.dur);
    pos = it.on + it.dur;
  }
  if (inBar > 1e-6) tokens.push('|');
  return tokens.join(' ');
}

function kbInsert() {
  const frag = kbToAbc();
  if (!frag) return;
  const prev = ta.value, prevSel = ta.selectionStart;
  if (!prev.trim()) {
    const d = new Date();
    const stamp = d.toLocaleDateString('ru-RU') + ' ' +
      d.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });
    const text = `X:1\nT:Запись ${stamp}\nM:${KB.meter}\nL:1/8\nQ:1/4=${KB.bpm}\nK:C\n` +
      `V:1 program=0 ${KB.prog} name="${KB.instrRu}"\n${frag}\n`;
    ta.value = text;
    ta.selectionStart = ta.selectionEnd = text.length;
  } else {
    ta.setRangeText(frag + ' ', prevSel, prevSel, 'end');
  }
  KB.undoText = { text: prev, sel: prevSel, len: KB.notes.length };
  $('btnKbUndo').hidden = false;
  ta.dispatchEvent(new Event('input'));
  kbStatUpdate();
}

// ---- панель ----
(function kbInit() {
  const sel = $('kbInstr');
  const groups = new Map();
  for (const [gr, name, prog, ru] of KB_INSTRUMENTS) {
    if (!groups.has(gr)) groups.set(gr, []);
    groups.get(gr).push([name, prog, ru]);
  }
  for (const [gr, items] of groups) {
    const og = document.createElement('optgroup');
    og.label = gr;
    for (const [name, prog, ru] of items) {
      const o = document.createElement('option');
      o.value = name; o.dataset.prog = prog; o.dataset.ru = ru;
      o.textContent = `${ru} (${prog})`;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  sel.onchange = () => {
    const o = sel.selectedOptions[0];
    KB.instr = sel.value; KB.prog = +o.dataset.prog; KB.instrRu = o.dataset.ru;
  };
  $('kbMeter').onchange = () => { KB.meter = $('kbMeter').value; };
  $('kbBpm').oninput = () => {
    KB.bpm = Math.max(30, Math.min(300, parseInt($('kbBpm').value) || 100));
  };
  $('btnKbMetro').onclick = () => {
    KB.metro = !KB.metro;
    $('btnKbMetro').classList.toggle('on', KB.metro);
  };
  $('btnKbRec').onclick = () => { KB.armed ? kbStopRecord() : kbArm(); };
  $('btnKbUndo').onclick = () => {
    if (!KB.undoText) return;
    ta.value = KB.undoText.text;
    ta.selectionStart = ta.selectionEnd = KB.undoText.sel;
    KB.undoText = null;
    $('btnKbUndo').hidden = true;
    ta.dispatchEvent(new Event('input'));
    kbStatUpdate();
  };
  $('btnKbOctDn').onclick = () => kbShift(-1);
  $('btnKbOctUp').onclick = () => kbShift(1);
  kbBuildKeys();
  kbOctLabel();
  kbStatUpdate();
})();

document.addEventListener('keyup', e => {
  if (KB_KEYMAP[e.code] === undefined) return;
  for (const [midi, rec] of KB.active)
    if (rec.code === e.code) { kbNoteOff(midi); break; }
});
