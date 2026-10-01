// Студия v2: многодорожечный таймлайн (Suno-стиль) + Audacity-операции.
// Vanilla JS без зависимостей: волна на canvas с серверными пиками,
// микс в реальном времени через Web Audio, правки — набором изменений.
const $ = id => document.getElementById(id);

const S = {
  stem: null, row: null, items: [],
  dur: 0, axisDur: 0,             // длительность версии и общей оси (с слоями)
  view: { a: 0, b: 0 },
  sel: null, clipboard: null, swapA: null,
  loop: false, snap: true, score: null,
  lanes: [],                       // [{stem,title,kind,url,dur,peaks,buffer,gain,mute,solo,canvas}]
  laneState: new Map(),            // stem -> {gain,mute,solo} (переживает перерисовки)
  chains: [],
  pending: [], undoStack: [], redoStack: [],
  compParts: [], compRoot: null,
};

function fmtDur(s){ s = Math.round(s || 0); return Math.floor(s/60) + ':' + String(s % 60).padStart(2, '0'); }
function fmtT(s){ s = Math.max(0, s || 0); const m = Math.floor(s/60); const x = (s - m*60).toFixed(1);
  return m + ':' + (x < 10 ? '0' : '') + x; }

async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === 'string' ? j.detail : (r.status + ' ' + JSON.stringify(j.detail || '')));
  return j;
}
const stemOf = f => (f || '').replace(/\.flac$/, '');
const rootOf = m => m.version_root || stemOf(m.file);
const errHtml = e => `<span class="err">${e.message || e}</span>`;
const esc = s => String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

// ==================== Web Audio транспорт ====================
const TA = {
  ctx: null, startedAt: 0, offset: 0, playing: false, nodes: [],
  ensure() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === 'suspended') this.ctx.resume();
    return this.ctx;
  },
  pos() { return this.playing ? this.ctx.currentTime - this.startedAt + this.offset : this.offset; },
  audible(lane) {
    const anySolo = S.lanes.some(l => l.solo);
    return anySolo ? lane.solo : !lane.mute;
  },
  async play(from) {
    const ctx = this.ensure();
    this.stopSources();
    if (from == null) from = this.offset;
    for (const lane of S.lanes) {
      lane.gainNode = null;
      if (!this.audible(lane)) continue;
      try {
        if (!lane.buffer) lane.buffer = await this.decode(lane.url);
      } catch (e) { lane.decodeError = true; renderLaneErr(lane); continue; }
      const src = ctx.createBufferSource();
      src.buffer = lane.buffer;
      const g = ctx.createGain();
      g.gain.value = lane.gain || 1;
      lane.gainNode = g;
      src.connect(g); g.connect(ctx.destination);
      src.start(0, Math.min(from, lane.buffer.duration - 0.01));
      lane.srcNode = src;
    }
    this.startedAt = ctx.currentTime;
    this.offset = from;
    this.playing = true;
    rafLoop();
  },
  async decode(url) {
    const buf = await (await fetch(url)).arrayBuffer();
    return await this.ensure().decodeAudioData(buf);
  },
  stopSources() {
    for (const lane of S.lanes) {
      if (lane.srcNode) { try { lane.srcNode.stop(); } catch {} lane.srcNode = null; }
      lane.gainNode = null;
    }
  },
  pause() {
    if (!this.playing) return;
    const p = this.pos();
    this.stopSources();
    this.playing = false;
    this.offset = p;
    drawAll();
  },
  seek(t) {
    const was = this.playing;
    this.stopSources();
    this.playing = false;
    this.offset = Math.max(0, t);
    if (was) this.play(this.offset);
    else drawAll();
  },
  applyGains() {
    for (const lane of S.lanes) {
      if (lane.gainNode) lane.gainNode.gain.value = this.audible(lane) ? (lane.gain || 1) : 0;
    }
  },
};

// ==================== галерея и версии ====================
async function loadGallery() {
  try { S.items = await (await fetch('/api/gallery')).json(); }
  catch { S.items = []; }
}
function findItem(stem) { return S.items.find(i => stemOf(i.file) === stem); }
function groupItems(stem) {
  const it = findItem(stem);
  if (!it) return [];
  const root = rootOf(it);
  return S.items.filter(i => rootOf(i) === root)
    .sort((a, b) => (a.version_n || 0) - (b.version_n || 0));
}
function shortName(stem) {
  const it = findItem(stem);
  if (!it) return stem.slice(-6);
  return it.version_of ? `v${it.version_n || '?'}` : 'ориг';
}
async function adoptVersion(stem, statusEl) {
  await loadGallery();
  if (statusEl) statusEl.innerHTML = `готово — переключился на <b>${shortName(stem)}</b>`;
  await switchStem(stem);
}

// ==================== переключение записи ====================
async function switchStem(stem, keepView) {
  const it = findItem(stem);
  if (!it) return;
  TA.pause();
  S.stem = stem; S.row = it; S.dur = it.duration_s || 0;
  S.loop = false;
  S.pending = []; S.undoStack = []; S.redoStack = []; renderPending();
  const root = rootOf(it);
  if (S.compRoot !== root) { S.compParts = []; S.compRoot = root; renderComp(); }
  if (!keepView || S.view.b > S.axisDur || S.view.b - S.view.a <= 0) S.view = { a: 0, b: S.dur };
  clearSel();
  $('btnLoop').classList.remove('on');
  history.replaceState(null, '', '/studio?stem=' + encodeURIComponent(stem));

  $('tTitle').textContent = it.title || (it.style_base || it.style || '').split(/[.\n]/)[0] || stem;
  const lab = $('chLabel');
  lab.hidden = false;
  lab.textContent = it.version_of ? (it.version_n ? `v${it.version_n} · ${it.version_label || ''}` : (it.version_label || 'вариант')) : 'оригинал';
  lab.className = 'chip' + (it.group_main ? ' ok' : '');
  if (it.group_main) lab.textContent += ' · основная';
  $('chDur').hidden = false; $('chDur').textContent = fmtDur(S.dur);
  $('chSeed').hidden = false; $('chSeed').textContent = 'seed ' + (it.seed ?? '—');

  await buildLanes();
  loadScore();
  loadAbc();
  updateExport();
  renderVersions();
  drawAll();
}

// ---- дорожки: основной трек + слои ----
async function deleteLane(lane) {
  const n = groupItems(lane.stem).length - 1;   // версии самого слоя, если есть
  const msg = `Удалить слой «${lane.title}»${n > 0 ? ` и все его ${n} версий` : ''}? Файл будет стёрт безвозвратно.`;
  if (!confirm(msg)) return;
  TA.stopSources();
  try {
    const r = await fetch(`/api/gallery/${encodeURIComponent(lane.stem)}`, { method: 'DELETE' });
    if (!r.ok) { const e = await r.json().catch(() => ({})); alert('Не удалось удалить: ' + (e.detail || r.status)); return; }
  } catch { alert('Сервер недоступен'); return; }
  await loadGallery();
  await buildLanes();
  drawAll();
  $('mixStat').textContent = `слой «${lane.title}» удалён`;
}

async function buildLanes() {
  let lanes = [];
  try { lanes = (await api(`/api/gallery/${encodeURIComponent(S.stem)}/lanes`)).lanes || []; }
  catch {}
  const mk = (o) => {
    const st = S.laneState.get(o.stem) || { gain: o.gain0, mute: false, solo: false };
    S.laneState.set(o.stem, st);
    return {
      stem: o.stem, title: o.title, kind: o.kind, url: o.url, dur: o.dur || 0,
      peaks: null, buffer: null, decodeError: false, srcNode: null,
      gain: st.gain, mute: st.mute, solo: st.solo, canvas: null,
    };
  };
  const list = [mk({ stem: S.stem, title: (S.row.title || 'основной трек'),
                     kind: 'main', url: S.row.audio_url, dur: S.dur, gain0: 1.0 })];
  for (const l of lanes)
    list.push(mk({ stem: l.stem, title: l.title, kind: l.kind, url: l.url,
                   dur: l.duration_s, gain0: 0.5 }));
  S.lanes = list;
  S.axisDur = Math.max(S.dur, ...list.map(l => l.dur || 0));
  renderLaneStack();
  for (const lane of S.lanes) fetchLanePeaks(lane);
  renderOdlanes();
}

function renderLaneStack() {
  const box = $('laneStack');
  box.innerHTML = '';
  for (const lane of S.lanes) {
    const row = document.createElement('div');
    row.className = 'laneTrack' + (lane.kind === 'main' ? ' mainLane' : '');
    const head = document.createElement('div');
    head.className = 'laneHead';
    const kindTxt = lane.kind === 'main' ? 'трек · версия' :
      (lane.kind === 'overdub' ? 'овердаб' : 'импорт');
    head.innerHTML = `<div class="lname"></div>
      <div class="lkind">${kindTxt} · ${fmtDur(lane.dur)}</div>
      <div class="lctl">
        <span title="гейн">Vol</span>
        <input type="range" min="0" max="1.5" step="0.05" value="${lane.gain}">
        <label title="выключить слой"><input type="checkbox" ${lane.mute ? 'checked' : ''}>M</label>
        <label title="соло — слышна только эта"><input type="checkbox" ${lane.solo ? 'checked' : ''}>S</label>
        ${lane.kind !== 'main' ? '<span class="ldel" title="удалить слой">✕</span>' : ''}
      </div>`;
    head.querySelector('.lname').textContent = lane.title;
    const rng = head.querySelector('input[type=range]');
    const mute = head.querySelector('.lctl label input');      // первый чекбокс = M
    const solo = head.querySelectorAll('.lctl label input')[1];
    rng.oninput = () => { lane.gain = parseFloat(rng.value);
      S.laneState.get(lane.stem).gain = lane.gain; TA.applyGains(); };
    mute.onchange = () => { lane.mute = mute.checked;
      S.laneState.get(lane.stem).mute = lane.mute; TA.applyGains(); };
    solo.onchange = () => { lane.solo = solo.checked;
      S.laneState.get(lane.stem).solo = lane.solo; TA.applyGains(); };
    const delBtn = head.querySelector('.ldel');
    if (delBtn) delBtn.onclick = () => deleteLane(lane);
    const cv = document.createElement('canvas');
    cv.className = 'laneWave';
    row.append(head, cv);
    lane.canvas = cv;
    bindTimelineCanvas(cv);
    box.appendChild(row);
  }
}
function renderLaneErr(lane) {
  if (lane.kind !== 'main' && lane.canvas) {
    const ctx = lane.canvas.getContext('2d');
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--panel2');
    ctx.fillRect(0, 0, lane.canvas.width, lane.canvas.height);
  }
}

async function fetchLanePeaks(lane) {
  try {
    const { a, b } = S.view;
    const full = a <= 0.001 && b >= S.axisDur - 0.001;
    const n = Math.min(4000, Math.max(300, Math.round(lane.canvas.clientWidth * (window.devicePixelRatio || 1) / 2)));
    const q = new URLSearchParams({ n });
    if (!full) { q.set('from_sec', a.toFixed(3)); q.set('to_sec', b.toFixed(3)); }
    lane.peaks = await api(`/api/gallery/${encodeURIComponent(lane.stem)}/peaks?` + q);
    drawAll();
  } catch {}
}

// ==================== отрисовка ====================
function drawAll() { drawRuler(); for (const l of S.lanes) drawLane(l); updateClock(); }

function themeColors() {
  const css = getComputedStyle(document.documentElement);
  const g = k => css.getPropertyValue(k).trim();
  return { bg: g('--panel2'), text: g('--text'), muted: g('--muted'),
           accent: g('--accent'), accent2: g('--accent2'), border: g('--border'),
           warn: g('--warn') };
}

function drawRuler() {
  const c = $('ruler');
  if (!c.clientWidth) return;
  const dpr = window.devicePixelRatio || 1;
  const h = 30;
  if (c.width !== Math.round(c.clientWidth * dpr)) { c.width = Math.round(c.clientWidth * dpr); c.height = h * dpr; }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const C = themeColors();
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, c.clientWidth, h);
  if (!S.axisDur) return;
  const { a, b } = S.view, span = Math.max(0.01, b - a);
  const X = t => (t - a) / span * c.clientWidth;
  // секции (текущая — ярче)
  if (S.score && S.score.rms_sections) {
    const now = TA.pos();
    const cur = S.score.rms_sections.findIndex(s => now >= s.start_sec && now < s.end_sec);
    S.score.rms_sections.forEach((s, i) => {
      const x0 = Math.max(0, X(s.start_sec)), x1 = Math.min(c.clientWidth, X(s.end_sec));
      if (x1 <= 0 || x0 >= c.clientWidth) return;
      ctx.fillStyle = i === cur ? 'rgba(124,108,255,.38)' :
        (i % 2 ? 'rgba(124,108,255,.10)' : 'rgba(124,108,255,.18)');
      ctx.fillRect(x0, 0, x1 - x0, h);
      ctx.fillStyle = C.text; ctx.font = i === cur ? 'bold 10px Inter, sans-serif' : '10px Inter, sans-serif';
      ctx.textBaseline = 'middle';
      if (x1 - x0 > 40) ctx.fillText(s.section, x0 + 4, h / 2 + 1);
    });
  }
  // деления
  ctx.fillStyle = C.muted; ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'top';
  const step = span > 240 ? 60 : span > 90 ? 30 : span > 40 ? 10 : span > 15 ? 5 : span > 6 ? 2 : span > 2 ? 1 : 0.5;
  for (let t = Math.ceil(a / step) * step; t <= b + 1e-6; t += step) {
    const x = X(t);
    ctx.fillText(fmtT(t), x + 3, 3);
    ctx.fillRect(x, 0, 1, 5);
  }
  // выделение + плеяхед
  if (S.sel) {
    const x0 = X(S.sel.a), x1 = X(S.sel.b);
    ctx.fillStyle = 'rgba(124,108,255,.30)';
    ctx.fillRect(x0, 0, x1 - x0, h);
  }
  const px = X(TA.pos());
  if (px >= -1 && px <= c.clientWidth + 1) {
    ctx.fillStyle = C.warn;
    ctx.beginPath(); ctx.moveTo(px - 4, 0); ctx.lineTo(px + 4, 0); ctx.lineTo(px, 7); ctx.fill();
  }
}

function drawLane(lane) {
  const c = lane.canvas;
  if (!c || !c.clientWidth) return;
  const dpr = window.devicePixelRatio || 1;
  const hCSS = lane.kind === 'main' ? 170 : 96;
  if (c.width !== Math.round(c.clientWidth * dpr) || c.height !== hCSS * dpr) {
    c.width = Math.round(c.clientWidth * dpr); c.height = hCSS * dpr;
  }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const C = themeColors();
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, c.clientWidth, hCSS);
  if (!S.axisDur) return;
  const { a, b } = S.view, span = Math.max(0.01, b - a);
  const X = t => (t - a) / span * c.clientWidth;
  const mid = hCSS / 2, amp = hCSS / 2 - 6;
  // фон заглохшей дорожки
  if (!TA.audible(lane)) { ctx.fillStyle = 'rgba(0,0,0,.25)'; ctx.fillRect(0, 0, c.clientWidth, hCSS); }
  if (lane.decodeError) {
    ctx.fillStyle = C.warn; ctx.font = '11px Inter, sans-serif';
    ctx.fillText('не удалось декодировать — файл будет в миксе на сервере', 10, mid);
    return;
  }
  if (lane.peaks && lane.peaks.min && lane.peaks.min.length) {
    const p = lane.peaks, n = p.min.length;
    const pa = p.from_sec, pspan = Math.max(1e-6, p.to_sec - p.from_sec);
    const Y = v => mid - Math.max(-1, Math.min(1, v)) * amp;
    for (let px = 0; px < c.clientWidth; px++) {
      const t0 = a + px / c.clientWidth * span, t1 = a + (px + 1) / c.clientWidth * span;
      let i0 = Math.floor((t0 - pa) / pspan * n), i1 = Math.ceil((t1 - pa) / pspan * n);
      i0 = Math.max(0, Math.min(n - 1, i0)); i1 = Math.max(i0 + 1, Math.min(n, i1));
      let mn = 1, mx = -1, rsum = 0, rc = 0;
      for (let i = i0; i < i1; i++) {
        if (p.min[i] < mn) mn = p.min[i];
        if (p.max[i] > mx) mx = p.max[i];
        rsum += p.rms[i] * p.rms[i]; rc++;
      }
      if (mn > mx) continue;
      const rms = Math.sqrt(rsum / Math.max(1, rc));
      ctx.fillStyle = lane.kind === 'main' ? C.accent2 + 'aa' : C.accent + '77';
      ctx.fillRect(px, Y(mx), 1, Math.max(1, Y(mn) - Y(mx)));
      ctx.fillStyle = lane.kind === 'main' ? C.accent : C.muted;
      ctx.fillRect(px, Y(rms), 1, Math.max(1, Y(-rms) - Y(rms)));
    }
    ctx.strokeStyle = C.border; ctx.globalAlpha = .5;
    ctx.beginPath(); ctx.moveTo(0, mid + .5); ctx.lineTo(c.clientWidth, mid + .5); ctx.stroke();
    ctx.globalAlpha = 1;
  } else if (lane.kind !== 'main') {
    ctx.fillStyle = C.muted; ctx.font = '11px Inter, sans-serif';
    ctx.fillText('загружаю…', 10, mid);
  }
  // свап-режим: подсветка первого фрагмента
  if (S.swapA) {
    const x0 = X(S.swapA.a), x1 = X(S.swapA.b);
    ctx.fillStyle = 'rgba(255,204,102,.18)';
    ctx.fillRect(x0, 0, x1 - x0, hCSS);
    ctx.strokeStyle = C.warn;
    ctx.strokeRect(x0 + .5, .5, x1 - x0 - 1, hCSS - 1);
  }
  // выделение
  if (S.sel) {
    const x0 = X(S.sel.a), x1 = X(S.sel.b);
    ctx.fillStyle = 'rgba(124,108,255,.16)';
    ctx.fillRect(x0, 0, x1 - x0, hCSS);
    ctx.strokeStyle = C.accent;
    ctx.beginPath(); ctx.moveTo(x0 + .5, 0); ctx.lineTo(x0 + .5, hCSS);
    ctx.moveTo(x1 + .5, 0); ctx.lineTo(x1 + .5, hCSS); ctx.stroke();
  }
  // плеяхед
  const t = TA.pos();
  if (t >= a && t <= b) {
    const x = X(t);
    ctx.fillStyle = C.warn;
    ctx.fillRect(x, 0, 1, hCSS);
  }
}

function updateClock() {
  $('tTime').textContent = fmtT(TA.pos()) + ' / ' + fmtT(S.axisDur);
}

let rafOn = false;
function rafLoop() {
  if (rafOn) return;
  rafOn = true;
  (function tick() {
    if (!TA.playing) { rafOn = false; return; }
    // конец оси
    const p = TA.pos();
    if (p >= S.axisDur - 0.02) { TA.pause(); TA.offset = S.axisDur; drawAll(); return; }
    // луп
    if (S.loop && S.sel && p > S.sel.b + 0.02) { TA.seek(S.sel.a); }
    // авто-следование: перелистываем окно, когда плеяхед ушёл за край
    const { a, b } = S.view, span = b - a;
    if (span < S.axisDur - 0.01 && (p < a + span * 0.02 || p > b - span * 0.05)) {
      const na = Math.max(0, Math.min(S.axisDur - span, p - span * 0.15));
      setView(na, na + span);
    } else drawAll();
    requestAnimationFrame(tick);
  })();
}

// ==================== взаимодействие с таймлайном ====================
function xToT(canvas, x) {
  const { a, b } = S.view, span = b - a;
  return a + x / canvas.clientWidth * span;
}
function snapT(t) {
  if (!S.snap) return t;
  const targets = [0, S.axisDur];
  if (S.score && S.score.rms_sections)
    for (const s of S.score.rms_sections) { targets.push(s.start_sec, s.end_sec); }
  let best = t, bd = 0.6;
  for (const g of targets) {
    const d = Math.abs(g - t);
    if (d < bd) { bd = d; best = g; }
  }
  return best;
}
let drag = null;
function bindTimelineCanvas(cv) {
  cv.addEventListener('mousedown', e => {
    const r = cv.getBoundingClientRect();
    drag = { cv, x0: e.clientX - r.left, t0: xToT(cv, e.clientX - r.left),
             moved: false, pan: e.shiftKey || e.button === 1 };
    if (drag.pan) { drag.view0 = { ...S.view }; e.preventDefault(); }
  });
  cv.addEventListener('mousemove', e => {
    if (!drag || drag.cv !== cv) return;
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left;
    if (drag.pan) {
      const dx = (x - drag.x0) / cv.clientWidth * (S.view.b - S.view.a);
      setView(drag.view0.a - dx, drag.view0.b - dx);
      return;
    }
    if (Math.abs(x - drag.x0) > 4) drag.moved = true;
    if (drag.moved) {
      const t1 = xToT(cv, x);
      S.sel = { a: Math.max(0, Math.min(drag.t0, t1)), b: Math.min(S.axisDur, Math.max(drag.t0, t1)) };
      updateSelInfo(); drawAll();
    }
  });
  cv.addEventListener('dblclick', () => { if (S.sel) setView(S.sel.a, S.sel.b); });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    const r = cv.getBoundingClientRect();
    zoomAt(Math.pow(1.0022, e.deltaY), xToT(cv, e.clientX - r.left));
  }, { passive: false });
}
window.addEventListener('mouseup', e => {
  if (!drag) return;
  const { pan, moved, t0, cv } = drag;
  drag = null;
  if (pan || moved) {
    if (moved && S.sel) {
      // прилипание краёв выделения к секциям/краям
      S.sel = { a: snapT(S.sel.a), b: Math.max(snapT(S.sel.b), snapT(S.sel.a) + 0.1) };
      updateSelInfo();
    }
    if (moved && S.sel && S.swapA) completeSwap();
    drawAll();
    return;
  }
  // клик — переход
  TA.seek(Math.max(0, Math.min(S.axisDur - 0.05, t0)));
  drawAll();
});
$('ruler').addEventListener('mousedown', e => {
  const r = $('ruler').getBoundingClientRect();
  const t = xToT($('ruler'), e.clientX - r.left);
  TA.seek(Math.max(0, Math.min(S.axisDur - 0.05, t)));
});
new ResizeObserver(() => { drawAll(); }).observe($('laneStack'));

function updateSelInfo() {
  const el = $('selInfo');
  if (!S.sel) { el.textContent = 'тяните мышью по волне — выделение диапазона'; return; }
  el.textContent = `выделение: ${S.sel.a.toFixed(1)}–${S.sel.b.toFixed(1)} с (${(S.sel.b - S.sel.a).toFixed(1)})`;
  const has = !!S.sel;
  for (const id of ['btnKeep', 'btnCut', 'btnSilence', 'btnDup', 'btnFadeReg', 'btnSwap', 'btnCopy', 'btnCompAdd'])
    $(id).disabled = !has;
  $('fxScopeSel').disabled = !has;
  $('gainScope').textContent = has ? '(выделение)' : '(весь трек)';
  updateExportSel();
}
function clearSel() {
  S.sel = null;
  for (const id of ['btnKeep', 'btnCut', 'btnSilence', 'btnDup', 'btnFadeReg', 'btnSwap', 'btnCopy', 'btnCompAdd'])
    $(id).disabled = true;
  $('fxScopeSel').disabled = true;
  const all = document.querySelector('input[name=fxScope][value=all]');
  if (all) all.checked = true;
  $('gainScope').textContent = '(весь трек)';
  updateSelInfo(); drawAll();
}
function completeSwap() {
  // S.swapA + текущее S.sel → операция
  const a = S.swapA, b = S.sel;
  S.swapA = null;
  $('btnSwap').classList.remove('on');
  if (!b) return;
  if (a.a < b.b && b.a < a.b) {
    $('editStat').textContent = 'фрагменты пересекаются — выделите непересекающийся второй';
    return;
  }
  addPending('swap', `поменять ${a.a.toFixed(0)}–${a.b.toFixed(0)} ↔ ${b.a.toFixed(0)}–${b.b.toFixed(0)} с`,
             {}, a, b);
}

// зум/панорама
let zoomTimer = null;
function setView(a, b) {
  a = Math.max(0, a); b = Math.min(S.axisDur, Math.max(a + 0.5, b));
  S.view = { a, b };
  drawAll();
  clearTimeout(zoomTimer);
  zoomTimer = setTimeout(refetchPeaks, 180);
}
function refetchPeaks() { for (const l of S.lanes) fetchLanePeaks(l); }
function zoomAt(factor, anchorT) {
  const { a, b } = S.view, span = b - a;
  const ns = Math.min(S.axisDur, Math.max(0.5, span * factor));
  const frac = (anchorT - a) / span;
  setView(anchorT - frac * ns, anchorT - frac * ns + ns);
}
$('btnZoomIn').onclick = () => zoomAt(0.6, (S.view.a + S.view.b) / 2);
$('btnZoomOut').onclick = () => zoomAt(1 / 0.6, (S.view.a + S.view.b) / 2);
$('btnFit').onclick = () => setView(0, S.axisDur);
$('btnSelAll').onclick = () => { S.sel = { a: 0, b: S.axisDur }; updateSelInfo(); drawAll(); };
$('btnClearSel').onclick = clearSel;
$('btnSnap').onclick = () => { S.snap = !S.snap; $('btnSnap').classList.toggle('on', S.snap); };
$('btnSnap').classList.add('on');
$('btnLoop').onclick = () => {
  S.loop = !S.loop;
  $('btnLoop').classList.toggle('on', S.loop);
  if (S.loop && !S.sel) { S.sel = { a: 0, b: S.axisDur }; updateSelInfo(); }
  drawAll();
};

// ==================== транспорт ====================
$('btnPlay').onclick = () => { if (TA.playing) TA.pause(); else TA.play(); };
function syncPlayBtn() {
  setInterval(() => { $('btnPlay').textContent = TA.playing ? '⏸' : '▶'; }, 250);
}
syncPlayBtn();

document.addEventListener('keydown', e => {
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
  if (e.code === 'Space') { e.preventDefault(); if (TA.playing) TA.pause(); else TA.play(); }
  else if (e.key === 'Home') { TA.seek(0); }
  else if (e.key === 'End') { TA.seek(S.axisDur - 0.05); }
  else if (e.key === 'ArrowLeft') { TA.seek(TA.pos() - (e.shiftKey ? 5 : 1)); }
  else if (e.key === 'ArrowRight') { TA.seek(TA.pos() + (e.shiftKey ? 5 : 1)); }
  else if (e.key === 'Delete' || e.key === 'Backspace') { if (S.sel) addPending('trim-cut', `вырезать ${S.sel.a.toFixed(0)}–${S.sel.b.toFixed(0)} с`, {}, S.sel); }
  else if (e.key === 'Escape') { if (S.swapA) { S.swapA = null; $('btnSwap').classList.remove('on'); } else clearSel(); }
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? redoSet() : undoSet(); }
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); redoSet(); }
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'c') { if (S.sel) { S.clipboard = { ...S.sel }; $('btnPaste').disabled = false; $('editStat').textContent = 'скопировано в буфер'; } }
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'v') { pasteClip(); }
});

// ==================== набор изменений ====================
function addPending(op, label, params, region, region2) {
  S.pending.push({
    op, label, params: params || {},
    region: region ? { from_sec: +region.a.toFixed(3), to_sec: +region.b.toFixed(3) } : null,
    region2: region2 ? { from_sec: +region2.a.toFixed(3), to_sec: +region2.b.toFixed(3) } : null,
  });
  S.undoStack = []; S.redoStack = [];
  renderPending();
}
function renderPending() {
  const box = $('pendingList');
  box.innerHTML = '';
  S.pending.forEach((o, i) => {
    const d = document.createElement('div');
    d.className = 'pend';
    const n = document.createElement('span');
    n.className = 'pnum'; n.textContent = i + 1;
    const t = document.createElement('span');
    t.className = 'ptext'; t.textContent = o.label;
    const x = document.createElement('span');
    x.className = 'pdel'; x.textContent = '✕'; x.title = 'убрать из набора';
    x.onclick = () => { S.pending.splice(i, 1); renderPending(); };
    d.append(n, t, x);
    box.appendChild(d);
  });
  const b = $('btnCreateVer');
  b.disabled = !S.pending.length;
  b.textContent = S.pending.length
    ? `создать версию · изменений: ${S.pending.length}` : 'создать версию';
  $('btnUndoSet').disabled = !S.pending.length;
  $('btnRedoSet').disabled = !S.redoStack.length;
  $('btnListen').style.opacity = S.pending.length ? 1 : .4;
}
function undoSet() {
  if (!S.pending.length) return;
  S.redoStack.push(S.pending.pop());
  renderPending();
}
function redoSet() {
  if (!S.redoStack.length) return;
  S.pending.push(S.redoStack.pop());
  renderPending();
}
$('btnUndoSet').onclick = undoSet;
$('btnRedoSet').onclick = redoSet;

function pasteClip() {
  if (!S.clipboard) { $('editStat').textContent = 'буфер пуст — сначала скопируйте'; return; }
  const at = TA.pos();
  addPending('insert-from',
    `вставка ${S.clipboard.a.toFixed(0)}–${S.clipboard.b.toFixed(0)} с → ${at.toFixed(0)} с`,
    { src_from: S.clipboard.a, src_to: S.clipboard.b, at });
  $('editStat').textContent = '';
}
$('btnCopy').onclick = () => { S.clipboard = { ...S.sel }; $('btnPaste').disabled = false;
  $('editStat').textContent = 'скопировано — «вставить →» положит в позицию плеяхеда'; };
$('btnPaste').onclick = pasteClip;
$('btnDup').onclick = () => {
  addPending('duplicate', `дубликат ${S.sel.a.toFixed(0)}–${S.sel.b.toFixed(0)} с`, {}, S.sel);
  $('editStat').textContent = '';
};
$('btnKeep').onclick = () => {
  addPending('trim-keep', `оставить ${S.sel.a.toFixed(0)}–${S.sel.b.toFixed(0)} с`, {}, S.sel);
  $('editStat').textContent = '';
};
$('btnCut').onclick = () => {
  addPending('trim-cut', `вырезать ${S.sel.a.toFixed(0)}–${S.sel.b.toFixed(0)} с`, {}, S.sel);
  $('editStat').textContent = '';
};
$('btnSilence').onclick = () => {
  addPending('silence', `тишина ${S.sel.a.toFixed(0)}–${S.sel.b.toFixed(0)} с`, {}, S.sel);
  $('editStat').textContent = '';
};
$('btnSwap').onclick = () => {
  if (S.swapA) { completeSwap(); return; }
  S.swapA = { ...S.sel };
  $('btnSwap').classList.add('on');
  $('editStat').innerHTML = '<span style="color:var(--warn)">первый фрагмент запомнен — выделите второй</span>';
};
$('btnFadeReg').onclick = () => {
  addPending('fade-region', `фейды на ${S.sel.a.toFixed(0)}–${S.sel.b.toFixed(0)} с`,
             { fade_in: 0.5, fade_out: 0.5 }, S.sel);
  $('editStat').textContent = '';
};
$('gainDb').oninput = () => { $('gainVal').textContent = $('gainDb').value; };
$('btnGain').onclick = () => {
  const db = parseFloat($('gainDb').value);
  const scope = S.sel ? ` (${S.sel.a.toFixed(0)}–${S.sel.b.toFixed(0)} с)` : '';
  addPending('gain', `гейн ${db > 0 ? '+' : ''}${db} дБ${scope}`, { db }, S.sel || null);
  $('editStat').textContent = '';
};
$('btnFade').onclick = () => {
  const fi = parseFloat($('fadeIn').value) || 0, fo = parseFloat($('fadeOut').value) || 0;
  addPending('fade', `фейды ${fi} / ${fo} с`, { fade_in: fi, fade_out: fo }, null);
  $('editStat').textContent = '';
};
$('btnListen').onclick = async () => {
  if (!S.pending.length) return;
  $('editStat').innerHTML = '<span style="color:var(--muted)">рендерю набор…</span>';
  try {
    const j = await api(`/api/gallery/${encodeURIComponent(S.stem)}/edit`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ops: S.pending, preview: true }) });
    $('editStat').textContent = `превью набора: ${j.label}`;
    playOneShot(j.url);
  } catch (e) { $('editStat').innerHTML = errHtml(e); }
};
$('btnCreateVer').onclick = async () => {
  if (!S.pending.length) return;
  $('editStat').innerHTML = '<span style="color:var(--muted)">рендерю набор…</span>';
  try {
    const j = await api(`/api/gallery/${encodeURIComponent(S.stem)}/edit`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ops: S.pending }) });
    S.pending = []; S.undoStack = []; S.redoStack = []; renderPending();
    await adoptVersion(j.stem, $('editStat'));
  } catch (e) { $('editStat').innerHTML = errHtml(e); }
};

function playOneShot(url) {
  const a = $('oneShot');
  a.src = url + (url.includes('?') ? '&' : '?') + 't=' + Date.now();
  a.play().catch(() => {});
}

// ==================== секции/партитура ====================
async function loadScore() {
  S.score = null;
  try { S.score = await api(`/api/gallery/${encodeURIComponent(S.stem)}/score`); }
  catch {}
  $('chBpm').hidden = !(S.score && S.score.tempo_bpm);
  $('chBpm').textContent = S.score ? Math.round(S.score.tempo_bpm) + ' BPM' : '';
  drawAll();
}

// ==================== эффекты ====================
async function initFx() {
  try { S.chains = await (await fetch('/api/dsp/chains')).json(); } catch { S.chains = []; }
  const sel = $('fxChain');
  sel.innerHTML = '';
  for (const c of S.chains) {
    const o = document.createElement('option');
    o.value = c.id; o.textContent = `${c.name} — ${c.note}`;
    sel.appendChild(o);
  }
  sel.onchange = renderFxParams;
  renderFxParams();
}
function renderFxParams() {
  const c = S.chains.find(x => x.id === $('fxChain').value);
  const box = $('fxParams'); box.innerHTML = '';
  if (!c) return;
  for (const p of c.params) {
    const d = document.createElement('div');
    d.className = 'param'; d.dataset.pid = p.id;
    d.innerHTML = `<label><span>${p.label}</span><b>${p.default}</b></label>
      <input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.default}">`;
    const rng = d.querySelector('input'), val = d.querySelector('b');
    rng.oninput = () => { val.textContent = rng.value; };
    box.appendChild(d);
  }
}
function fxParams() {
  const params = { chain: $('fxChain').value };
  for (const d of document.querySelectorAll('#fxParams .param'))
    params[d.dataset.pid] = parseFloat(d.querySelector('input').value);
  return params;
}
$('btnFxPreview').onclick = async () => {
  $('fxStat').innerHTML = '<span style="color:var(--muted)">рендерю фрагмент…</span>';
  try {
    const j = await api(`/api/gallery/${encodeURIComponent(S.stem)}/dsp`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chain: $('fxChain').value, params: fxParams(), preview: true }) });
    $('fxStat').textContent = 'превью готово (фрагмент 20–35 с оригинала)';
    playOneShot(j.url);
  } catch (e) { $('fxStat').innerHTML = errHtml(e); }
};
$('btnFxApply').onclick = () => {
  const c = S.chains.find(x => x.id === $('fxChain').value);
  const scope = document.querySelector('input[name=fxScope]:checked').value;
  const region = (scope === 'sel' && S.sel) ? S.sel : null;
  const scopeTxt = region ? ` (${region.a.toFixed(0)}–${region.b.toFixed(0)} с)` : '';
  addPending('chain', (c ? c.name.toLowerCase() : 'эффект') + scopeTxt, fxParams(), region);
  $('fxStat').textContent = 'добавлено в набор';
};

// ==================== компинг ====================
function renderComp() {
  const box = $('compList');
  box.innerHTML = '';
  S.compParts.forEach((p, i) => {
    const d = document.createElement('div');
    d.className = 'cpart';
    const t = document.createElement('span');
    t.className = 'ctext';
    t.textContent = `${i + 1}. ${p.name}: ${p.from.toFixed(0)}–${p.to.toFixed(0)} с (${(p.to - p.from).toFixed(1)})`;
    const x = document.createElement('span');
    x.className = 'cdel'; x.textContent = '✕';
    x.onclick = () => { S.compParts.splice(i, 1); renderComp(); };
    d.append(t, x);
    box.appendChild(d);
  });
  $('btnCompGo').disabled = !S.compParts.length;
}
$('btnCompAdd').onclick = () => {
  if (!S.sel) return;
  S.compParts.push({ stem: S.stem, from: S.sel.a, to: S.sel.b, name: shortName(S.stem) });
  renderComp();
  $('compStat').textContent = `кусок из ${shortName(S.stem)} — переключите версию внизу, чтобы взять следующий`;
};
$('btnCompGo').onclick = async () => {
  if (!S.compParts.length) return;
  $('compStat').innerHTML = '<span style="color:var(--muted)">собираю комп…</span>';
  try {
    const j = await api(`/api/gallery/${encodeURIComponent(S.stem)}/comp`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parts: S.compParts.map(p => ({
        stem: p.stem, from_sec: +p.from.toFixed(3), to_sec: +p.to.toFixed(3) })) }) });
    S.compParts = []; renderComp();
    await adoptVersion(j.stem, $('compStat'));
  } catch (e) { $('compStat').innerHTML = errHtml(e); }
};

// ==================== ABC ====================
async function loadAbc() {
  $('rerenderStat').textContent = '';
  try {
    const j = await api(`/api/gallery/${encodeURIComponent(S.stem)}/abc`);
    $('abcText').value = j.abc;
    updateAbcStat(j);
    $('abcDl').href = `/api/gallery/${encodeURIComponent(S.stem)}/abc/download`;
    $('abcDl').hidden = false;
  } catch (e) {
    $('abcText').value = '';
    $('abcStat').innerHTML = errHtml(e);
    $('abcDl').hidden = true;
  }
}
function updateAbcStat(j) {
  const d = j.duration_sec, ad = j.audio_duration;
  let s = `источник: ${j.source === 'edited' ? 'правленая' : 'план модели'} · ABC ≈ ${fmtDur(d)}`;
  if (ad && d && Math.abs(d - ad) > 3) {
    s += ` · ⚠ расходится с аудио (${fmtDur(ad)})`;
    $('abcStat').style.color = 'var(--warn)';
  } else {
    if (ad) s += ` · аудио ${fmtDur(ad)}`;
    $('abcStat').style.color = '';
  }
  $('abcStat').textContent = s;
  $('btnAbcReset').hidden = j.source !== 'edited';
}
$('btnAbcSave').onclick = async () => {
  const text = $('abcText').value.trim();
  if (text.length < 10) { $('abcStat').innerHTML = '<span class="err">Партитура пуста</span>'; return; }
  $('abcStat').innerHTML = '<span style="color:var(--muted)">проверяю и сохраняю…</span>';
  try {
    const tl = await api(`/api/gallery/${encodeURIComponent(S.stem)}/abc`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }) });
    S.score = tl;
    $('abcStat').style.color = '';
    $('abcStat').textContent = `партитура сохранена · ≈ ${fmtDur(tl.duration_sec)} · ${tl.bars.length} тактов`;
    $('btnAbcReset').hidden = false;
    drawAll();
  } catch (e) { $('abcStat').innerHTML = errHtml(e); }
};
$('btnAbcReset').onclick = async () => {
  if (!confirm('Вернуть план модели? Правка партитуры сотрётся.')) return;
  try {
    S.score = await api(`/api/gallery/${encodeURIComponent(S.stem)}/abc/reset`, { method: 'POST' });
    drawAll();
    await loadAbc();
  } catch (e) { $('abcStat').innerHTML = errHtml(e); }
};

let rrTimer = null;
$('btnRerender').onclick = async () => {
  const text = $('abcText').value.trim();
  if (text.length < 20) { $('rerenderStat').innerHTML = '<span class="err">Партитура пуста</span>'; return; }
  $('rerenderStat').innerHTML = '<span style="color:var(--muted)">ставлю в очередь…</span>';
  try {
    const j = await api(`/api/gallery/${encodeURIComponent(S.stem)}/rerender`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ abc: text, draft: $('rrDraft').checked }) });
    $('rerenderStat').textContent = 'в очереди — генерация 1–3 мин';
    if (rrTimer) clearInterval(rrTimer);
    rrTimer = setInterval(async () => {
      let st;
      try { st = await (await fetch('/api/jobs/' + j.id)).json(); } catch { return; }
      if (st.status === 'done') {
        clearInterval(rrTimer); rrTimer = null;
        await adoptVersion(st.result.stem, $('rerenderStat'));
      } else if (st.status === 'error' || st.status === 'canceled') {
        clearInterval(rrTimer); rrTimer = null;
        $('rerenderStat').innerHTML = errHtml(new Error(st.error || 'отменена'));
      } else {
        $('rerenderStat').textContent =
          (st.stage || '') + (st.tokens ? ` · токенов ${st.tokens}` : '') +
          (st.elapsed_s ? ` · ${Math.round(st.elapsed_s)} с` : '');
      }
    }, 2000);
  } catch (e) { $('rerenderStat').innerHTML = errHtml(e); }
};

// ==================== овердаб, слои, микс, импорт ====================
function renderOdlanes() {
  const box = $('laneList');
  box.innerHTML = '';
  const lanes = S.lanes.filter(l => l.kind !== 'main');
  $('btnMix').hidden = !lanes.length;
  for (const lane of lanes) {
    const d = document.createElement('div');
    d.className = 'lane-mix';
    d.innerHTML = `<span class="iconbtn" title="послушать партию">▶</span>
      <span class="lname"></span>
      <span class="hint">${fmtDur(lane.dur)}</span>
      <span class="iconbtn" title="удалить слой">✕</span>`;
    d.querySelector('.lname').textContent = lane.title;
    const btns = d.querySelectorAll('.iconbtn');
    btns[0].onclick = () => playOneShot(lane.url);
    btns[1].onclick = () => deleteLane(lane);
    box.appendChild(d);
  }
}
const OD_CHIPS = [
  ['акуст. гитара', 'acoustic guitar'], ['эл. гитара', 'clean electric guitar'],
  ['фузз', 'fuzz guitar'], ['пиано', 'grand piano'], ['орган', 'hammond organ'],
  ['синт', 'analog synthesizer'], ['струнные', 'string section'], ['флейта', 'flute'],
  ['сакс', 'saxophone'], ['труба', 'trumpet'], ['колокольчики', 'glockenspiel'],
  ['перкуссия', 'hand percussion'], ['хор', 'choir'], ['скрипка', 'violin'],
];
(function () {
  const box = $('odChips');
  for (const [ru, en] of OD_CHIPS) {
    const t = document.createElement('span');
    t.className = 'tag'; t.textContent = ru;
    t.onclick = () => {
      const v = $('odStyle').value.trim();
      $('odStyle').value = v ? v + ', ' + en : en;
    };
    box.appendChild(t);
  }
})();
$('odGain').oninput = () => { $('odGainVal').textContent = $('odGain').value; };
// «без вокала» — режим по умолчанию: текст партии не нужен
function odInstSync() {
  const inst = $('odInst').checked;
  $('odLyrics').disabled = inst;
  $('odLyricsLabel').hidden = inst;
  $('odLyricsFill').classList.toggle('off', inst);
  $('odInstLabel').textContent = inst
    ? 'без вокала (инструментал) — по умолчанию'
    : 'без вокала (инструментал)';
}
$('odInst').onchange = odInstSync;
odInstSync();
$('odLyricsFill').onclick = () => { if (S.row && !$('odInst').checked) $('odLyrics').value = S.row.lyrics || ''; };
let odTimer = null;
$('odGo').onclick = async () => {
  const style = $('odStyle').value.trim();
  if (style.length < 3) { $('odStat').innerHTML = '<span class="err">Опишите стиль партии (минимум 3 символа)</span>'; return; }
  $('odStat').innerHTML = '<span style="color:var(--muted)">ставлю в очередь…</span>';
  try {
    const j = await api(`/api/gallery/${encodeURIComponent(S.stem)}/overdub`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ style, lyrics: $('odLyrics').value,
                             instrumental: $('odInst').checked,
                             gain: parseFloat($('odGain').value) }) });
    const odStem = S.stem;   // слой привяжется к этой записи, даже если уйти
    $('odStat').textContent = 'в очереди — придёт слоем в этот список';
    if (odTimer) clearInterval(odTimer);
    odTimer = setInterval(async () => {
      let st;
      try { st = await (await fetch('/api/jobs/' + j.id)).json(); } catch { return; }
      if (st.status === 'done') {
        clearInterval(odTimer); odTimer = null;
        $('odStat').textContent = 'овердаб готов — слой добавлен';
        await loadGallery();
        if (S.stem === odStem) { await buildLanes(); drawAll(); }
      } else if (st.status === 'error' || st.status === 'canceled') {
        clearInterval(odTimer); odTimer = null;
        $('odStat').innerHTML = errHtml(new Error(st.error || 'отменена'));
      } else {
        $('odStat').textContent =
          (st.stage || '') + (st.tokens ? ` · токенов ${st.tokens}` : '') +
          (st.elapsed_s ? ` · ${Math.round(st.elapsed_s)} с` : '');
      }
    }, 2000);
  } catch (e) { $('odStat').innerHTML = errHtml(e); }
};
$('odImport').onchange = async () => {
  const f = $('odImport').files[0];
  if (!f) return;
  $('mixStat').innerHTML = `<span style="color:var(--muted)">импортирую ${esc(f.name)}…</span>`;
  try {
    const title = f.name.replace(/\.[^.]+$/, '');
    const j = await api(`/api/gallery/${encodeURIComponent(S.stem)}/import-lane?title=` +
                        encodeURIComponent(title), { method: 'POST', body: await f.arrayBuffer() });
    $('mixStat').textContent = `слой «${j.title}» добавлен (${fmtDur(j.duration_s)})`;
    await buildLanes();
    drawAll();
  } catch (e) { $('mixStat').innerHTML = errHtml(e); }
  $('odImport').value = '';
};
$('btnMix').onclick = async () => {
  $('mixStat').innerHTML = '<span style="color:var(--muted)">собираю микс…</span>';
  try {
    const lanes = S.lanes.map(l => ({ stem: l.stem,
      gain: l.gain, mute: !TA.audible(l) }));
    const j = await api(`/api/gallery/${encodeURIComponent(S.stem)}/mixdown`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lanes }) });
    await adoptVersion(j.stem, $('mixStat'));
  } catch (e) { $('mixStat').innerHTML = errHtml(e); }
};

// ==================== экспорт ====================
$('exportBtn').onclick = () => $('exportMenu').classList.toggle('open');
document.addEventListener('click', e => {
  if (!$('exportMenu').contains(e.target)) $('exportMenu').classList.remove('open');
});
function updateExport() {
  const f = S.row ? S.row.file : '';
  $('exFlac').href = S.row ? S.row.audio_url : '#';
  $('exFlac').setAttribute('download', (S.row ? (S.row.title || S.stem) : 'song') + '.flac');
  $('exMp3').href = `/api/outputs/${encodeURIComponent(f)}/mp3`;
  $('exMp3').setAttribute('download', (S.row ? (S.row.title || S.stem) : 'song') + '.mp3');
  $('exMulti').href = `/api/gallery/${encodeURIComponent(S.stem)}/export-multitrack`;
  updateExportSel();
}
function updateExportSel() {
  const ok = !!S.sel;
  const mk = (el, fmt) => {
    el.classList.toggle('dis', !ok);
    if (ok) el.href = `/api/gallery/${encodeURIComponent(S.stem)}/export?fmt=${fmt}` +
      `&from_sec=${S.sel.a.toFixed(2)}&to_sec=${S.sel.b.toFixed(2)}`;
  };
  mk($('exSelFlac'), 'flac');
  mk($('exSelMp3'), 'mp3');
}

// ==================== версии ====================
function renderVersions() {
  const box = $('verList');
  box.innerHTML = '';
  const group = groupItems(S.stem);
  if (!group.length) { box.innerHTML = '<div class="hint">запись не найдена</div>'; return; }
  for (const it of group) {
    const st = stemOf(it.file);
    const d = document.createElement('div');
    d.className = 'vrow' + (st === S.stem ? ' cur' : '');
    const star = document.createElement('span');
    star.className = 'vstar' + (it.group_main ? ' main' : '');
    star.textContent = it.group_main ? '★' : '☆';
    star.title = it.group_main ? 'основная версия' : 'сделать основной';
    star.onclick = async e => {
      e.stopPropagation();
      try { await api(`/api/gallery/${encodeURIComponent(st)}/set-main`, { method: 'POST' }); }
      catch (err) { alert(err.message); return; }
      await loadGallery();
      S.row = findItem(S.stem) || S.row;
      renderVersions();
      const cur = findItem(S.stem);
      const lab = $('chLabel');
      lab.className = 'chip' + (cur && cur.group_main ? ' ok' : '');
      if (cur && cur.group_main && !lab.textContent.includes('основная')) lab.textContent += ' · основная';
    };
    const name = document.createElement('div');
    name.className = 'vname';
    name.innerHTML = it.version_of
      ? `<small>v${it.version_n} ·</small> ${esc(it.version_label || 'вариант')}`
      : 'оригинал';
    if (it.group_main) name.innerHTML += ' <span class="chip mini ok">основная</span>';
    if (it.version_of) name.innerHTML += ' <span class="chip mini">вариант-трек</span>';
    const dur = document.createElement('span');
    dur.className = 'vdur'; dur.textContent = fmtDur(it.duration_s);
    const play = document.createElement('span');
    play.className = 'iconbtn'; play.textContent = '▶'; play.title = 'играть';
    play.onclick = e => { e.stopPropagation(); switchStem(st).then(() => TA.play()); };
    const del = document.createElement('span');
    del.className = 'vdel'; del.textContent = '✕'; del.title = 'удалить';
    del.onclick = async e => {
      e.stopPropagation();
      const isRoot = !it.version_of;
      const n = group.length;
      const msg = isRoot
        ? `Удалить песню целиком — оригинал и все ${n - 1} версий? Файлы будут стёрты безвозвратно.`
        : `Удалить версию «${it.version_label || st}»?`;
      if (!confirm(msg)) return;
      try { await fetch(`/api/gallery/${encodeURIComponent(st)}`, { method: 'DELETE' }); }
      catch { return; }
      await loadGallery();
      if (st === S.stem) {
        const rest = groupItems(rootOf(it)).find(g => stemOf(g.file) !== st);
        if (rest) await switchStem(stemOf(rest.file));
        else { S.stem = null; S.dur = 0; renderVersions(); }
      } else renderVersions();
    };
    d.append(star, name, dur, play, del);
    d.onclick = () => switchStem(st);
    box.appendChild(d);
  }
}

// ==================== тема и запуск ====================
(function () {
  const saved = localStorage.getItem('yue_theme');
  if (saved) document.documentElement.dataset.theme = saved;
  $('themeToggle').onclick = () => {
    const nxt = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = nxt;
    localStorage.setItem('yue_theme', nxt);
    drawAll();
  };
})();

(async function init() {
  await loadGallery();
  initFx();
  if (!S.items.length) {
    $('tTitle').textContent = 'Студия — библиотека пуста';
    $('verList').innerHTML = '<div class="hint">Сгенерируйте песню на главной — она появится здесь.</div>';
    return;
  }
  const q = new URLSearchParams(location.search).get('stem');
  const target = (q && findItem(q)) ? q : stemOf(S.items[0].file);
  await switchStem(target);
})();
