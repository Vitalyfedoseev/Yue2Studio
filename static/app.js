const $ = id => document.getElementById(id);
const STAGE_RU = {
  queued: 'в очереди', 'план ABC': 'план: модель пишет ABC-нотацию песни',
  'arc-plan': 'драматургия: правка плана под дугу',
  'семантика': 'семантика: LM генерирует песню токен за токеном',
  'акустические латенты': 'flow-matching: латенты акустики',
  'VAE-декод': 'VAE: латенты → звук 48 кГц', done: 'готово',
};

let pollTimer = null, currentListId = null;

function fmtDur(s){ s = Math.round(s); return Math.floor(s/60)+':'+String(s%60).padStart(2,'0'); }
function fmtSec(s){ return s>=90 ? fmtDur(s) : Math.round(s)+' с'; }

// ---------- health ----------
async function pollHealth(){
  try {
    const h = await (await fetch('/api/health')).json();
    const m = $('chipModel'), ok = h.ready;
    m.textContent = `модель: ${ok ? 'готова (bf16)' : (h.detail||'грузится')}`;
    m.className = 'chip ' + (ok ? 'ok' : (h.detail||'').startsWith('ошибка') ? 'err' : '');
    $('chipGpu').textContent = h.gpu ? `${h.gpu.split('NVIDIA ').pop()} · своб. ${h.vram_free_gb} ГБ` : 'gpu: —';
    $('chipQueue').textContent = `очередь: ${h.queue}`;
    if (!ok && !activeJobs.length) $('stLine').innerHTML = `<b>${h.detail||'загрузка…'}</b>`;
    return h;
  } catch { $('chipModel').textContent = 'сервер недоступен'; return null; }
}

// ---------- пресеты и голоса ----------
const PRESETS = {
  'фолк-метал': 'Russian folk metal, symphonic death metal. 180 BPM, dark battle energy. Arrangement: tremolo guitars and blast beats under accordion and zhaleika, choir stabs in the chorus, bass-drop before each refrain, folk melody intro on gusli.',
  'симфо-готик': 'Gothic metal, symphonic metal, doom. 90 BPM, E minor, epic and tragic. Arrangement: heavy rhythm guitars, orchestral strings and choir pads, piano intro, dramatic build into a towering chorus, cathedral reverb.',
  'нью-метал + электроника': 'Nu metal fused with dark metal and high-BPM electronic layers. 140 BPM. Arrangement: hypnotic one-note synth ostinato, down-tuned riffing, industrial percussion, glitch fills, whisper-to-scream dynamics.',
  'индастриал-симфо': 'Symphonic industrial metal, cinematic orchestral. 130 BPM. Arrangement: martial drum machines, brass swells, distorted bass synth, orchestral breakdowns, cold futuristic menace, spoken-word intro through a vinyl filter.',
  'панк 77': 'Raw 1977 punk rock. 170 BPM, fast and sneering. Arrangement: three buzzsaw chords, racing bare drums, shouted gang chorus, no solos, two minutes of pure spit.',
  'хардкор/панк': 'Modern hardcore punk. 190 BPM, aggressive and desperate. Arrangement: d-beat drumming, metallic power-chord breaks, half-time mosh part in the middle, gang shouts, raw basement production.',
  'детройт-техно': 'Detroit techno. 128 BPM, stark and hypnotic. Arrangement: rolling 909 kick, icy string stabs, acid bassline slowly opening the filter, minimal, machine soul, long DJ-style intro and outro.',
  'мелодик-техно': 'Melodic techno, dark and driving. 122 BPM. Arrangement: pulsing bass offbeat, melancholic arpeggio growing over 8 bars, atmosphere pads, tight kick, festival drop in the middle, dreamy breakdown.',
  'дарк-техно / EBM': 'Dark techno, EBM, warehouse. 135 BPM, cold and relentless. Arrangement: distorted analog kick, menacing bass sequence, metallic percussion, dystopian vocal stabs, no sunlight, hypnotic loop evolution.',
  'фанк / ню-диско': 'Funk / nu-disco, 108 BPM, groovy and infectious. Confident female lead vocal with ad-libs. Slap bass, wah guitar, four-on-the-floor drums, brass stabs in the chorus, filter sweeps on the outro.',
  'синти-поп 80-х': 'Synth-pop, 80s, 110 BPM, A minor, energetic neon-night mood. Soft female lead with light vocoder doubles in chorus. Analog synth bass, arpeggiated lead, gated drums, punchy chorus.',
  'акустическая баллада': 'Acoustic pop ballad, 72 BPM, C major, warm and intimate. Soft female lead, close and breathy. Fingerpicked guitar, soft piano, brushed drums and upright bass enter in the chorus.',
  'рок': 'Alternative rock, 128 BPM, E minor, driving and defiant. Raspy male lead, powerful belts in the chorus. Distorted rhythm guitars, driving bass, hard-hitting drums, guitar solo in the bridge.',
  'рэп': 'Boom-bap hip-hop, 88 BPM, G minor, confident and gritty. Low male rap lead, tight flow, ad-libs in gaps; soulful female hook. Dusty drum breaks, upright piano loop, sub bass, vinyl crackle.',
  'шансон по-русски': 'Russian chanson / city romance, 84 BPM, D minor, melancholic and nostalgic. Expressive male baritone with slight rasp. Acoustic guitar, accordion, upright bass, brushed snare, violin in the bridge.',
};
// Голоса — категории + подпись + краткое пояснение (тултип) + полный блок
// Vocal Details по паттерну промпт-гайда (тембр, манера по секциям, гармонии, FX).
const VOICES = {
  'Женские': [
    { label: 'пронзительный ваил', tip: 'Пронзительное драматическое сопрано: режущие верха, дикие выдержанные ваилы — хэви/пауэр-метал.',
      text: 'Vocal Details: Singer A (Female), a piercing, dramatic rock soprano with a glass-cutting upper register and a wild, sustained wail. Verses ride a tense mid-range; the chorus explodes into stratospheric belts and screaming high notes held to the last bar. Doubles an octave up in the final chorus. Aggressive compression, bright EQ, long delay on the wails.' },
    { label: 'рок-меццо с хрипотцой', tip: 'Мощное рок-меццо с виски-хрипотцой, ломающейся на верхах, — классика женского хард-рока.',
      text: 'Vocal Details: Singer A (Female), a powerful rock mezzo with a whiskey rasp that breaks beautifully on the top notes. Sultry, defiant verses; a full-throated, chesty belt in the chorus. Raspy ad-libs between lines. Close and dry with a touch of analog saturation.' },
    { label: 'готик-сопрано', tip: 'Тёмное симфоническое сопрано: холодный кристалл, оперные верха поверх стены гитар — готик-метал.',
      text: 'Vocal Details: Singer A (Female), a dark symphonic metal soprano with classical training and a cold, crystalline tone. Chant-like, restrained verses; operatic high notes soaring over the wall of guitars in the chorus. Choir doubling and ghostly whispers underneath. Cathedral reverb, orchestral arrangement around the voice.' },
    { label: 'фолк-метал сопрано', tip: 'Светлое скандинавское фолк-сопрано с кулинг-зовами в верхнем регистре — фолк/викинг-метал.',
      text: 'Vocal Details: Singer A (Female), a clear Scandinavian-tinged folk-metal soprano with a bright, open ring and kulning-like call notes. Lullaby-soft verses over an acoustic intro; fierce, celebration-high choruses against blast beats. Folk harmonies in fifths in the chorus. Natural voice, live-room ambience.' },
    { label: 'метал-скрим', tip: 'Женский экстрим-вокал: чёрные скримы в припевах, гроул на серединах, редкие чистые эхо-вставки.',
      text: 'Vocal Details: Singer A (Female), a fierce extreme-metal vocalist: blackened high screams tearing through choruses, mid-range growls in the verses, sudden clean echoes as contrast. Layered screams panned wide. Heavily compressed, front-of-mix aggression.' },
    { label: 'гранж-альт', tip: 'Сырой гранж-альт 90-х: надтреснутая эмоция, ленивый куплет, хриплый отчаянный припев.',
      text: 'Vocal Details: Singer A (Female), a raw 90s grunge alto with a cracked, emotional edge and slurred, behind-the-beat phrasing. Apathetic mumbling verses detonating into a hoarse, desperate chorus. Doubles loose and slightly detuned. Lo-fi, dry, ugly-beautiful.' },
    { label: 'хриплый рок-контральто', tip: 'Низкий контральто с гравием и дерзкой подачей — рок-н-ролльный свинг, виски и сигареты.',
      text: 'Vocal Details: Singer A (Female), a low, whiskey-soaked rock contralto with gravel in the chest register and a sneering attitude. Swaggering verses, throaty sustained lows in the chorus, occasional fierce climbs. Tight dry mix, dirty tube saturation.' },
    { label: 'тёплое меццо', tip: 'Нежное низкое женское, дыхтовое, близкий микрофон — идеал для баллад и инди.',
      text: 'Vocal Details: Singer A (Female), a warm mezzo-soprano with a breathy low register. Intimate, close-miked delivery in the verses with clear diction; a gentle lift into the chorus. Light stacked harmonies in the chorus only. Restrained plate reverb, subtle saturation on sustained notes.' },
    { label: 'яркое сопрано', tip: 'Молодое звонкое сопрано, режет микстейк; мощный бельтый припев — большой поп.',
      text: 'Vocal Details: Singer A (Female), a bright, youthful soprano that cuts through the mix. Melodic and precise in the verses; powerful belted chorus with sustained notes. Multi-layered self-harmonies stacked in thirds in the choruses. Moderate reverb, delay throws, subtle pitch correction and compression in the chorus.' },
    { label: 'глубокий альт', tip: 'Тёмный контральто на грудном регистре, смоковые низы — соул, лаунж, ночной вайб.',
      text: 'Vocal Details: Singer A (Female), a dark, rich contralto with a heavy chest register and smoky lows. Confident, laid-back phrasing; soulful melisma at phrase ends. A single unison double in the chorus. Warm tape saturation, minimal reverb, intimate and dry.' },
    { label: 'джазовый дым', tip: 'Хрипловатый джазовый вокал чуть позади доли, лёгкое вибрато — кабаре, нуар.',
      text: 'Vocal Details: Singer A (Female), a smoky, husky jazz voice with behind-the-beat phrasing and loose vibrato. Conversational verses, gently soaring bridge. Late-night double in the final chorus, occasional low harmonies. Vintage ribbon-mic tone, tape hiss, plate reverb.' },
    { label: 'эфирное сопрано', tip: 'Воздушное головное сопрано-шёпот, «оохи» и реверберационная дымка — дрим-поп.',
      text: 'Vocal Details: Singer A (Female), an airy, ethereal soprano floating in head voice, whispery and weightless. Half-voice verses dissolving into wordless oohs; a shimmering high register in the chorus. Ghostly octave doubles throughout. Long cathedral reverb, subtle detune chorus effect.' },
    { label: 'оперное сопрано', tip: 'Академическое лирическое сопрано: легато, контролируемое вибрато, высокие ноты — кроссовер.',
      text: 'Vocal Details: Singer A (Female), a lyric operatic soprano with full classical technique, even legato lines and a controlled spin vibrato. Declamatory verses, soaring sustained high notes in the chorus. Choir-like harmonies in the finale. Hall acoustics, natural voice, minimal processing.' },
    { label: 'R&B альт', tip: 'Шёлковый альт с мелизмами, фальцет-подъёмами и госпел-рунами — современный R&B.',
      text: 'Vocal Details: Singer A (Female), a silky R&B alto with fluid melisma, breathy falsetto flips and gospel-tinged runs. Groove-locked verses, ad-libbed flourishes in the outro. Stacked harmonies answering the lead in the chorus. Warm compression, doubled lead on hooks, tasteful delay throws.' },
    { label: 'панк-сорванец', tip: 'Дерзкий «пацанский» вокал с рычинкой, гэнг-крики в припеве — панк/гэрэдж.',
      text: 'Vocal Details: Singer A (Female), a bratty, snarly punk delivery with a razor edge and clipped phrasing. Shouted gang-style choruses, spoken asides in the breakdown. Unison shouting doubles in the chorus. Raw and dry, light distortion on the mic, no pitch correction.' },
    { label: 'фолк-распев', tip: 'Открытый народный распев с подголосками и мелизмами — русская фолк-традиция.',
      text: 'Vocal Details: Singer A (Female), an open-throated Russian folk soprano with ornamental melisma and a bright, nasal-forward ring. Free-rhythm intro, steady rubato verses, celebratory extended notes in the chorus. Parallel fifths folk harmonies in the chorus. Natural voice, room ambience, no FX.' },
    { label: 'рэп', tip: 'Острый женский рэп: чёткие согласные, синкопы, мелодичный хук на припеве.',
      text: 'Vocal Details: Singer A (Female), a sharp, agile rap vocal with crisp consonants and playful cadence. Syncopated flow riding the beat in the verses; a melodic sung hook if a chorus is tagged. Call-and-response doubles in the hook. Upfront and dry.' },
  ],
  'Мужские': [
    { label: 'пауэр-метал тенор', tip: 'Парящий пауэр-метал тенор: героические верха, бесконечное дыхание, финальная длинная нота.',
      text: 'Vocal Details: Singer A (Male), a soaring power-metal tenor with heroic top notes, crystal diction and endless breath. Galloping rhythmic verses; triumphant high belts and a fist-raising final long note in the last chorus. Doubled in unison plus a high harmony. Bright, big, anthem reverb.' },
    { label: 'бархатный баритон', tip: 'Мягкий низкий баритон с лёгкой хрипотцой на выдержках — соул, лаунж, шансон.',
      text: 'Vocal Details: Singer A (Male), a velvety low baritone with a soft rasp on sustained notes. Calm, conversational phrasing in the verses; fuller chest voice in the chorus. A low octave double in the chorus. Short slap delay, warm tube saturation, minimal reverb.' },
    { label: 'надрывный тенор', tip: 'Рок-тенор с надрывом: полушёпот в куплете, белтый припев — русский рок.',
      text: 'Vocal Details: Singer A (Male), a raw, impassioned tenor with a gravelly edge under strain. Restrained half-voice verses breaking into a belted, sustained chorus. Unison doubles in the chorus, rough ad-libs in the gaps. Compressed and fairly dry with a touch of room reverb.' },
    { label: 'глубокий бас', tip: 'Бас-профундо, могучие низы, неторопливые фразы — эпика, дарк-фолк, сказ.',
      text: 'Vocal Details: Singer A (Male), a resonant basso profondo with grave, measured phrasing and rolling low notes. Narrating verses, solemn sustained lows under the chorus. A sepulchral octave-below double in the outro. Minimal processing, dark room reverb, chest-resonant proximity.' },
    { label: 'чистый поп-тенор', tip: 'Гладкий «радио»-тенор с фирменной высокой нотой в припеве — мейнстрим-поп.',
      text: 'Vocal Details: Singer A (Male), a clean, polished pop tenor with boyish warmth and agile phrasing. Light rhythmic verses, a bright open chorus with a signature high note. Tight two-part harmonies in the chorus. Radio-ready polish, mild autotune glide, wide reverb on hooks.' },
    { label: 'инди-фальцет', tip: 'Интимный вокал с ломкой в хрупкий фальцет, лоу-фай подкладка — бедрум-поп.',
      text: 'Vocal Details: Singer A (Male), an intimate indie voice flipping between a soft chest register and a fragile airy falsetto. Bedroom-close verses, falsetto soaring over the chorus. Lo-fi double, occasional whisper layer. Dry and close, tape wobble, no pitch correction.' },
    { label: 'фолк-баритон', tip: 'Тёплый рассказчик с лёгкой хрипотцой — авторская песня, акустика, фолк.',
      text: 'Vocal Details: Singer A (Male), a warm folk baritone with a storytelling lilt, light rasp and open vowels. Plaintive picked-guitar verses, a hearty raised chorus. Bare-bones delivery, one low harmony in the final chorus. Natural, dry, single-take feel.' },
    { label: 'гроулинг + клин', tip: 'Экстрим-метал: чистые куплеты, гроул в припеве, скрим на бридже.',
      text: 'Vocal Details: Singer A (Male), a versatile extreme-metal vocalist: clean, brooding verses; guttural growls driving the choruses; high screams accenting the bridge and outro. Layered growls in the chorus. Heavily compressed, mid-scooped, tight and dry.' },
    { label: 'рэп', tip: 'Низкий уверенный рэп: плотный флоу, эд-либы в паузах — бум-бэп, трэп.',
      text: 'Vocal Details: Singer A (Male), a low, confident rap vocal with tight flow and crisp articulation. Laid-back verse cadences; a sung hook if a chorus is tagged. Ad-libs in the gaps, occasional doubles on line ends. Upfront and dry, light saturation.' },
  ],
  'Ансамбли': [
    { label: 'дуэт м+ж', tip: 'Он поёт куплет, она — припев; гармония вдвоём в финале.',
      text: 'Vocal Details: a male-female duet. Singer A (Male), a warm baritone, leads the verses; Singer B (Female), a bright mezzo, answers and leads the chorus. Tight two-part harmony in the final chorus, trading lines in the bridge. Both close-miked, light plate reverb.' },
    { label: 'госпел-хор', tip: 'Сольный вокал + хор в антифоне, хлопки в аутро — госпел, соул.',
      text: 'Vocal Details: a lead vocal joined by a gospel choir. Singer A leads with impassioned, melismatic phrasing; the choir answers in call-and-response, swelling to full stacked harmonies in the chorus and hand-clapping backing in the outro. Room-filling choir ambience, natural dynamics.' },
  ],
  'Без вокала': [
    { label: 'инструментал', tip: 'Вокала нет вообще: ведущую мелодию играет главный инструмент аранжировки.',
      text: 'Instrumental: no vocals at all. The lead melodic role is carried by the main instrument of the arrangement, with instrumental solo lines where sung sections would be.' },
  ],
};
const SECTIONS = ['[intro]','[verse]','[pre-chorus]','[chorus]','[verse 2]','[bridge]','[instrumental]','[solo]','[outro]'];

for (const [name, text] of Object.entries(PRESETS)) {
  const t = document.createElement('span'); t.className = 'tag'; t.textContent = name;
  t.onclick = () => { $('style').value = text; };
  $('stylePresets').appendChild(t);
}

// ---------- библиотека стилей + словарь ru→en (stylelib.js) ----------
const MY_STYLES_KEY = 'yue_custom_styles';
function myStyles() {
  try {
    const r = JSON.parse(localStorage.getItem(MY_STYLES_KEY) || '[]');
    return Array.isArray(r) ? r : [];
  } catch { return []; }
}
function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }
function translateStyleDict(s) {
  const dict = (window.STYLELIB || {}).dict || {};
  let out = s;
  for (const k of Object.keys(dict).sort((a, b) => b.length - a.length)) {
    const re = new RegExp('(^|[^\\p{L}])' + escapeRe(k) + '(?=$|[^\\p{L}])', 'giu');
    out = out.replace(re, (m, p) => p + dict[k]);
  }
  return out;
}
(function initStyleLib() {
  if (!window.STYLELIB || !$('libGroup')) return;
  const gs = $('libGroup');
  const renderGroups = () => {
    gs.innerHTML = '';
    const my = myStyles();
    if (my.length) {
      const o = document.createElement('option');
      o.value = '__my__'; o.textContent = '★ Мои стили';
      gs.appendChild(o);
    }
    for (const g of window.STYLELIB.groups) {
      const o = document.createElement('option');
      o.value = g.id; o.textContent = g.name;
      gs.appendChild(o);
    }
    renderItems();
  };
  const renderItems = () => {
    const box = $('libItems'); box.innerHTML = '';
    let items;
    if (gs.value === '__my__') {
      items = myStyles().map((s, idx) => ({ ...s, idx, mine: true }));
    } else {
      items = ((window.STYLELIB.groups.find(g => g.id === gs.value) || {}).items) || [];
    }
    for (const it of items) {
      const t = document.createElement('span');
      t.className = 'tag';
      t.textContent = it.name + (it.seed ? ' ⚡' : '');
      t.title = it.style;
      t.onclick = () => {
        $('style').value = it.style;
        if (it.seed) $('seed').value = it.seed;
        if (it.lyrics && !$('lyrics').value.trim()) $('lyrics').value = it.lyrics;
      };
      if (it.mine) {
        const x = document.createElement('i');
        x.textContent = ' ✕';
        x.style.cssText = 'font-style:normal; color:var(--err); margin-left:4px';
        x.title = 'удалить из «Мои»';
        x.onclick = e => {
          e.stopPropagation();
          const arr = myStyles();
          arr.splice(it.idx, 1);
          localStorage.setItem(MY_STYLES_KEY, JSON.stringify(arr));
          renderGroups();
        };
        t.appendChild(x);
      }
      box.appendChild(t);
    }
  };
  gs.onchange = renderItems;
  $('libSave').onclick = () => {
    const style = $('style').value.trim();
    if (style.length < 3) { alert('Поле стиля пусто — нечего сохранять'); return; }
    const name = prompt('Название стиля для «Мои»:', style.slice(0, 32));
    if (name === null) return;
    const arr = myStyles();
    arr.push({ name: name.trim() || style.slice(0, 32), style });
    localStorage.setItem(MY_STYLES_KEY, JSON.stringify(arr));
    renderGroups();
    gs.value = '__my__';
    renderItems();
  };
  renderGroups();
})();

// селектор голоса: табы категорий + скроллящийся список с тултипами
let voiceTab = 'Женские', voiceSel = null;
function renderVoiceTabs() {
  const tabs = $('voiceTabs'); tabs.innerHTML = '';
  for (const cat of Object.keys(VOICES)) {
    const t = document.createElement('span');
    t.className = 'vtab' + (cat === voiceTab ? ' active' : '');
    t.textContent = cat;
    t.onclick = () => { voiceTab = cat; renderVoiceTabs(); renderVoiceList(); };
    tabs.appendChild(t);
  }
}
function renderVoiceList() {
  const list = $('voiceList'); list.innerHTML = '';
  for (const v of VOICES[voiceTab]) {
    const d = document.createElement('div');
    d.className = 'vopt' + (voiceSel === v ? ' sel' : '');
    d.dataset.tip = v.tip;
    const label = document.createElement('span');
    label.className = 'vlabel'; label.textContent = v.label;
    const q = document.createElement('span');
    q.className = 'q'; q.textContent = '?'; q.title = '';
    d.appendChild(label); d.appendChild(q);
    d.onclick = () => {
      voiceSel = v;
      $('voice').value = v.text;
      renderVoiceList();
    };
    list.appendChild(d);
  }
}
renderVoiceTabs();
renderVoiceList();

// тултип голосов: один элемент вне списка, position:fixed + высокий z-index;
// триггер — значок «?» в строке
const voiceTip = document.createElement('div');
voiceTip.id = 'voiceTip';
document.body.appendChild(voiceTip);
$('voiceList').addEventListener('mouseover', e => {
  const q = e.target.closest('.q');
  if (!q) return;
  const opt = q.closest('.vopt');
  if (!opt || !opt.dataset.tip) return;
  voiceTip.textContent = opt.dataset.tip;
  voiceTip.style.display = 'block';
  const r = q.getBoundingClientRect(), t = voiceTip.getBoundingClientRect();
  let x = Math.min(r.left + r.width / 2 - 14, window.innerWidth - t.width - 12);
  let y = r.top - t.height - 8;               // над строкой
  if (y < 8) y = r.bottom + 8;                // не влезает сверху — снизу
  voiceTip.style.left = x + 'px';
  voiceTip.style.top = y + 'px';
});
$('voiceList').addEventListener('mouseout', e => {
  if (!(e.relatedTarget && e.relatedTarget.closest && e.relatedTarget.closest('.q')))
    voiceTip.style.display = 'none';
});
for (const s of SECTIONS) {
  const t = document.createElement('span'); t.className = 'tag'; t.textContent = s;
  t.onclick = () => {
    const ta = $('lyrics');
    const add = (ta.value && !ta.value.endsWith('\n') ? '\n' : '') + s + '\n';
    ta.value += add; ta.focus();
    ta.setSelectionRange(ta.value.length, ta.value.length);
  };
  $('secTags').appendChild(t);
}
// ---------- кавер по MIDI/аудио: сервер извлекает ноты (midi2abc, basic-pitch), ABC редактируем ----------
const MIDI_HINT_DEFAULT = 'Загрузите .mid или аудио (mp3/wav/flac/…) — аудио разбирает SheetSage2 (родная транскрипция YuE2: голоса Vocal/Ins, секции, темп; при сбое — basic-pitch), ноты показываются в ABC, можно править. Режим плана станет melody.';
let midiLoaded = false, midiBlobUrl = null;
function resetMidi() {
  midiLoaded = false;
  $('midiFile').value = '';
  $('midiResult').hidden = true;
  $('midiHint').textContent = MIDI_HINT_DEFAULT;
  if (midiBlobUrl) { URL.revokeObjectURL(midiBlobUrl); midiBlobUrl = null; }
  $('midiDl').hidden = true;
}
$('midiClear').onclick = resetMidi;
$('midiFile').onchange = async () => {
  const f = $('midiFile').files[0];
  if (!f) return;
  const isAudio = !/\.(mid|midi)$/i.test(f.name);
  $('midiHint').textContent = isAudio
    ? `слушаю ${f.name} — извлекаю ноты (SheetSage2, до минуты)…`
    : `конвертирую ${f.name}…`;
  try {
    const title = f.name.replace(/\.[^.]+$/, '');
    const r = await fetch('/api/midi2abc?title=' + encodeURIComponent(title),
                          { method: 'POST', body: await f.arrayBuffer() });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { resetMidi(); $('midiHint').innerHTML = `<span class="err">${j.detail || r.status}</span>`; return; }
    midiLoaded = true;
    $('abc').value = j.abc;
    $('midiResult').hidden = false;
    $('abcStat').textContent = j.transcribed
      ? `${f.name} → ${j.engine === 'sheetsage2' ? 'SheetSage2' : 'basic-pitch'}: ${j.notes ?? '?'} нот → ABC: ${j.chars} символов`
      : `${f.name} → ABC: ${j.chars} символов`;
    if (j.thinned) $('abcStat').textContent += ' · партитура была слишком плотной, оставлена мелодия';
    if (j.transcribed && j.midi_b64) {
      const bytes = Uint8Array.from(atob(j.midi_b64), c => c.charCodeAt(0));
      midiBlobUrl = URL.createObjectURL(new Blob([bytes], { type: 'audio/midi' }));
      $('midiDl').href = midiBlobUrl;
      $('midiDl').setAttribute('download', (title || 'cover') + '.mid');
      $('midiDl').hidden = false;
    }
    $('midiHint').textContent = 'Модель споёт эти ноты (план не пишет). Полномиксовое аудио даёт густую партитуру — лучше вокал/мелодия по отдельности; лишнее можно вычистить в поле выше.';
    if ($('cot').value === 'off') $('cot').value = 'melody';
  } catch (e) { resetMidi(); $('midiHint').innerHTML = `<span class="err">сервер недоступен: ${e}</span>`; }
};

// ---------- generate: очередь задач (кнопка всегда активна) ----------
let activeJobs = [];      // id в порядке отправки; первая — обрабатывается/показывается
let jobStates = {};        // id -> последнее состояние
let lastQueueError = null;

async function submit(draft = false) {
  unlockNotify(); // жест пользователя — разблокировать звук уведомления
  let style = $('style').value.trim();
  // словарь ru→en: известные слова заменяем молча, неизвестные — предупреждаем
  let styleWarnText = '';
  if (/[а-яё]/i.test(style) && window.STYLELIB) {
    const t = translateStyleDict(style);
    if (t !== style) style = t;
    if (/[а-яё]/i.test(style))
      styleWarnText = 'часть русского текста не переведена словарём — осталась как есть (модель лучше понимает английские теги)';
  }
  if ($('styleWarn')) {
    $('styleWarn').hidden = !styleWarnText;
    $('styleWarn').textContent = styleWarnText;
  }
  const voice = $('voice').value.trim();
  let lyrics = $('lyrics').value.trim();
  if ($('noVocals') && $('noVocals').checked) {
    // инструментал заданной длины: N секций [Instrumental]
    const n = parseInt($('instLen').value) || 1;
    lyrics = Array(n).fill('[Instrumental]').join('\n');
  }
  if (style.length < 3) { $('stLine').innerHTML = '<span class="err">Опишите стиль (обязательно)</span>'; return; }
  if (lyrics.length < 2) { $('stLine').innerHTML = '<span class="err">Вставьте лирику — по её объёму строится песня</span>'; return; }
  const cfg = $('cfg').value === '' ? null : parseFloat($('cfg').value);
  const abc = midiLoaded ? $('abc').value.trim() : null;
  let cot = $('cot').value;
  if (abc && cot === 'off') { cot = 'melody'; $('cot').value = 'melody'; } // сервер отвергнет off+abc
  let arcSel = $('arcSel') ? $('arcSel').value : '';
  if (abc && arcSel) { arcSel = ''; $('arcSel').value = ''; } // драматургия не работает с внешней партитурой
  // seed=-1/некорректный отправляем как есть: случайный выберет сервер
  const raw = parseInt($('seed').value);
  const seed = (Number.isNaN(raw) || raw < 0) ? -1 : raw;
  const body = { style, voice, lyrics, title: $('title').value.trim(), cot, cfg_scale: cfg,
                 seed, abc, draft, arc: arcSel };
  try {
    const r = await fetch('/api/generate', { method: 'POST',
      headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    if (!r.ok) { const j = await r.json().catch(() => ({})); $('stLine').innerHTML = `<span class="err">${errText(j, r.status)}</span>`; return; }
    const { id } = await r.json();
    activeJobs.push(id);
    jobStates[id] = { id, status: 'queued', stage: 'в очереди', tokens: 0, elapsed_s: 0 };
  } catch (e) { $('stLine').innerHTML = `<span class="err">сервер недоступен: ${e}</span>`; return; }
  finalMsg = null; lastQueueError = null;
  renderJobs();
  if (!pollTimer) { pollTimer = setInterval(pollAll, 2000); }
}

$('go').onclick = () => submit(false);
$('goDraft').onclick = () => submit(true);

$('stop').onclick = async () => {
  if (!activeJobs.length) return;
  $('stLine').textContent = 'останавливаю…';
  for (const id of activeJobs) {
    try { await fetch(`/api/jobs/${id}/cancel`, { method: 'POST' }); } catch {}
  }
};

function setBar(p){ $('barFill').style.width = Math.max(2, Math.min(100, p)) + '%'; }

async function pollAll() {
  const shownId = activeJobs[0];
  const results = await Promise.all(activeJobs.map(id =>
    fetch('/api/jobs/' + id).then(r => r.json()).catch(() => null)));
  const stillActive = [];
  activeJobs.forEach((id, i) => {
    const j = results[i];
    if (!j) { stillActive.push(id); return; }     // сети моргнули — оставим
    jobStates[id] = j;
    if (j.status === 'done') {
      notifyDone();
      play(`/outputs/${encodeURIComponent(j.result.file)}`, j.result, false);
      loadGallery();
      if (id === shownId)
        finalMsg = `<b>Готово</b> — ${j.result.duration_s} c аудио за ${fmtSec(j.result.total_s)}`;
    }
    else if (j.status === 'error' || j.status === 'canceled') {
      lastQueueError = `задача ${id.slice(0,6)}: ${j.error || 'отменена'}`;
      if (id === shownId) finalMsg = null;
    }
    else stillActive.push(id);
  });
  activeJobs = stillActive;
  renderJobs();
  if (!activeJobs.length && pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

let finalMsg = null;

function renderJobs() {
  $('stop').style.display = activeJobs.length ? 'block' : 'none';
  const ql = $('queueLine');
  const waiting = activeJobs.length - 1;
  let extra = waiting > 0 ? `в очереди ещё: ${waiting}` : '';
  if (lastQueueError && activeJobs.length) extra += (extra ? ' · ' : '') + `⚠ ${lastQueueError}`;
  ql.hidden = !extra; ql.textContent = extra;

  const id = activeJobs[0];
  if (!id) {
    if (lastQueueError) { $('stLine').innerHTML = `<span class="err">${lastQueueError}</span>`; setBar(0); }
    else if (finalMsg) { $('stLine').innerHTML = finalMsg; setBar(100); }
    return;
  }
  const j = jobStates[id] || { status: 'queued', stage: 'в очереди' };
  if (j.status === 'queued') { setBar(4); $('stLine').textContent = 'в очереди…'; return; }
  const st = STAGE_RU[j.stage] || j.stage || '';
  if (j.status === 'error') { setBar(0); $('stLine').innerHTML = `<span class="err">${j.error}</span>`; return; }
  // бегущий прогресс: честный % с сервера (семантика), иначе доля стадий
  if (j.pct != null && j.stage === 'семантика') {
    setBar(j.pct);
  } else {
    const order = ['план ABC','arc-plan','семантика','акустические латенты','VAE-декод'];
    const idx = order.indexOf(j.stage);
    setBar(idx >= 0 ? 8 + idx*22 + Math.min(14, (j.tokens||0)/3000*14) : 5);
  }
  let line = `<b>${st}</b>`;
  if (j.stage === 'семантика') {
    line += `<br>токенов: ${j.tokens||0}`;
    if (j.tok_per_s) line += ` · ${j.tok_per_s} ток/с`;
  } else if (j.stage === 'план ABC' && j.tokens) line += `<br>символов нот: ${j.tokens}`;
  if (j.elapsed_s) line += ` · прошло ${fmtSec(j.elapsed_s)}`;
  $('stLine').innerHTML = line;
}

// ---------- player / track card / gallery ----------
// короткое звуковое уведомление о готовности (Web Audio, без файлов);
// AudioContext разблокируется кликом по «Сгенерировать»
let notifyCtx = null;
function unlockNotify() {
  try {
    notifyCtx = notifyCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (notifyCtx.state === 'suspended') notifyCtx.resume();
  } catch {}
}
function notifyDone() {
  try {
    unlockNotify();
    if (!notifyCtx || notifyCtx.state !== 'running') return;
    const t = notifyCtx.currentTime;
    const g = notifyCtx.createGain();
    g.connect(notifyCtx.destination);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.22, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.75);
    [880, 1318.5].forEach((f, i) => {  // A5 → E6, «динь-динь» ~0.7 c
      const o = notifyCtx.createOscillator();
      o.type = 'sine';
      o.frequency.value = f;
      o.connect(g);
      o.start(t + i * 0.16);
      o.stop(t + i * 0.16 + 0.45);
    });
  } catch {}
}

let currentMeta = null;   // выбранная запись
let onlyLiked = false;    // фильтр истории: только ★
function fmtChips(m) {
  const c = [fmtDur(m.duration_s), m.sample_rate + ' Гц'];
  if (m.liked) c.push('★ нравится');
  if (m.version_of) c.push('вариант-трек');
  if (m.group_main) c.push('основная');
  if (m.abc_file) c.push('кавер по MIDI');
  if (m.draft) c.push('черновик');
  if (m.arc) c.push('дуга: ' + m.arc);
  if (m.overdub_of) c.push('овердаб');
  if (m.cot && m.cot !== 'full') c.push('режим ' + m.cot);
  if (m.cfg_scale) c.push('cfg ' + m.cfg_scale);
  c.push('seed ' + m.seed);
  if (m.peak_vram_gb) c.push('VRAM ' + m.peak_vram_gb + ' ГБ');
  if (m.total_s) c.push('ген. ' + fmtSec(m.total_s));
  if (m.ts) c.push(m.ts.slice(0, 16).replace('T', ' '));
  return c;
}

function play(url, meta, autoplay) {
  currentMeta = meta;
  currentListId = url;
  $('playerEmpty').style.display = 'none';
  $('playerWrap').classList.add('ready');
  $('trackInfo').hidden = false;

  $('tTitle').textContent = meta.title || (meta.style_base || meta.style || '').split(/[.\n]/)[0];
  $('tChips').innerHTML = fmtChips(meta).map(t => `<span class="chip">${t}</span>`).join('');
  $('tStyle').textContent = meta.style_base || meta.style || '';
  const vb = $('tVoiceBlock');
  if (meta.voice) { vb.hidden = false; $('tVoice').textContent = meta.voice; } else vb.hidden = true;
  $('tLyrics').textContent = meta.lyrics || '';

  const stem = meta.file.replace(/\.flac$/, '');
  // имя при скачивании: название песни, если задано (иначе серверный stem);
  // вычищаем символы, недопустимые в именах файлов Windows
  const dlBase = ((meta.title || '').replace(/[\\/:*?"<>|]+/g, ' ').trim() || stem);
  const mp320 = `/api/outputs/${encodeURIComponent(meta.file)}/mp3`;
  $('dlMp320').href = mp320;  $('dlMp320').setAttribute('download', dlBase + '.mp3');
  $('dlMp48').href = mp320 + '?q=48'; $('dlMp48').setAttribute('download', dlBase + '.48k.mp3');
  $('dlFlac').href = url;     $('dlFlac').setAttribute('download', dlBase + '.flac');
  $('dlWav').href = `/api/outputs/${encodeURIComponent(meta.file)}/wav`;
  $('dlWav').setAttribute('download', dlBase + '.wav');
  if (meta.abc_file) {
    $('dlAbc').hidden = false;
    $('dlAbc').href = `/api/gallery/${encodeURIComponent(stem)}/abc/download`;
    $('dlAbc').setAttribute('download', dlBase + '.abc');
  } else $('dlAbc').hidden = true;
  if (meta.overdub_file) {
    $('dlOdMix').hidden = false;
    $('dlOdMix').href = '/outputs/' + meta.overdub_file.split('/').map(encodeURIComponent).join('/');
    $('dlOdMix').setAttribute('download', dlBase + ' (овердаб-микс).flac');
  } else $('dlOdMix').hidden = true;

  const t = [];
  t.push(`файл: ${meta.file}`);
  if (meta.abc_chars) t.push(`партитура MIDI: ${meta.abc_chars} симв. ABC`);
  if (meta.score_abc_file) t.push(`партитура модели: ${meta.score_abc_file}`);
  if (meta.overdub_file) t.push(`смешанный овердаб: ${meta.overdub_file}`);
  if (meta.overdub_error) t.push(`овердаб-микс не удался: ${meta.overdub_error}`);
  if (meta.tokens) t.push(`токенов: ${meta.tokens}`);
  if (meta.total_s) t.push(`время генерации: ${fmtSec(meta.total_s)}`);
  if (meta.peak_vram_gb) t.push(`пик VRAM: ${meta.peak_vram_gb} ГБ`);
  t.push(`длительность: ${meta.duration_s} c · ${meta.sample_rate} Гц FLAC`);
  $('tTech').textContent = t.join('\n');

  // мгновенный старт: плеер играет MP3-версию (10 МБ вместо 55 МБ FLAC —
  // начинает звучать сразу даже по Wi-Fi); FLAC остаётся в скачиваниях
  $('player').src = (meta && meta.file)
    ? `/api/outputs/${encodeURIComponent(meta.file)}/mp3` : url;
  if (autoplay) $('player').play().catch(()=>{});
  markGalleryItem();
}

function markGalleryItem() {
  for (const el of document.querySelectorAll('.gitem')) {
    el.classList.toggle('playing', el.dataset.file === (currentMeta && currentMeta.file));
  }
}

// сброс карточки трека (после удаления проигрываемой записи)
function resetPlayer() {
  currentMeta = null; currentListId = null;
  $('player').removeAttribute('src'); $('player').load();
  $('trackInfo').hidden = true;
  $('playerWrap').classList.remove('ready');
  $('playerEmpty').style.display = 'block';
}

$('tDelete').onclick = async () => {
  if (!currentMeta) return;
  const name = currentMeta.title || currentMeta.file;
  if (!confirm(`Удалить «${name}»? Файлы будут стёрты безвозвратно.`)) return;
  const stem = currentMeta.file.replace(/\.flac$/, '');
  try {
    const r = await fetch(`/api/gallery/${encodeURIComponent(stem)}`, { method: 'DELETE' });
    if (!r.ok) { const e = await r.json(); alert('Не удалось удалить: ' + (e.detail || r.status)); return; }
  } catch (e) { alert('Сервер недоступен: ' + e); return; }
  resetPlayer();
  loadGallery();
};

let searchQ = '';            // фильтр истории
let galleryItemsCache = [];  // полный список для селекта сравнения метрик
let playList = [];           // текущий видимый список — очередь плеера

$('gSearch').addEventListener('input', () => {
  searchQ = $('gSearch').value.trim().toLowerCase();
  loadGallery();
});

async function loadGallery(){
  let items = [];
  try { items = await (await fetch('/api/gallery')).json(); } catch { return; }
  galleryItemsCache = items;
  const likedCount = items.filter(i => i.liked).length;
  if (onlyLiked) items = items.filter(i => i.liked);
  if (searchQ) items = items.filter(i =>
    ((i.title || '') + ' ' + (i.lyrics || '') + ' ' + (i.style || '')).toLowerCase().includes(searchQ));
  playList = items;
  $('gSearch').hidden = false;
  const g = $('gallery'); g.innerHTML = '';
  if (!items.length && !likedCount) return;
  const h = document.createElement('div'); h.className = 'label ghdr';
  h.textContent = onlyLiked ? 'История · понравилось' : 'История';
  const f = document.createElement('span');
  f.className = 'tag' + (onlyLiked ? ' on' : '');
  f.textContent = `★ ${likedCount}`;
  f.title = 'только понравившиеся';
  f.onclick = () => { onlyLiked = !onlyLiked; loadGallery(); };
  h.appendChild(f);
  g.appendChild(h);
  if (!items.length) {
    const e = document.createElement('div'); e.className = 'hint';
    e.textContent = 'Пока ничего не отмечено — нажмите ★ у трека в списке';
    g.appendChild(e);
    return;
  }
  for (const it of items) {
    const d = document.createElement('div');
    d.className = 'gitem';
    d.dataset.file = it.file;
    d.innerHTML = `<span class="gstar${it.liked ? ' liked' : ''}" title="понравилось">${it.liked ? '★' : '☆'}</span>` +
      `<div class="gname"><div></div><div></div></div>` +
      `<div class="dur">${it.duration_s} c</div>` +
      `<span class="gdel" title="удалить запись">✕</span>`;
    d.querySelector('.gname div').textContent = it.title || (it.style_base || it.style || '').split(/[.\n]/)[0];
    d.querySelectorAll('.gname div')[1].textContent =
      (it.lyrics||'').replace(/\[.*?\]/g,' ').trim().slice(0,80) || it.file;

    d.querySelector('.gstar').onclick = async e => {
      e.stopPropagation();
      const stem = it.file.replace(/\.flac$/, '');
      try {
        const r = await fetch(`/api/gallery/${encodeURIComponent(stem)}/like`, { method: 'POST' });
        if (!r.ok) return;
        const j = await r.json();
        it.liked = j.liked;
        if (currentMeta && currentMeta.file === it.file) {
          currentMeta.liked = j.liked;
          $('tChips').innerHTML = fmtChips(currentMeta).map(t => `<span class="chip">${t}</span>`).join('');
        }
        loadGallery();
      } catch {}
    };

    d.querySelector('.gdel').onclick = async e => {
      e.stopPropagation();
      const name = it.title || it.file;
      if (!confirm(`Удалить «${name}»? Файлы будут стёрты безвозвратно.`)) return;
      const stem = it.file.replace(/\.flac$/, '');
      try {
        const r = await fetch(`/api/gallery/${encodeURIComponent(stem)}`, { method: 'DELETE' });
        if (!r.ok) { const e2 = await r.json(); alert('Не удалось удалить: ' + (e2.detail || r.status)); return; }
      } catch (e2) { alert('Сервер недоступен: ' + e2); return; }
      if (currentMeta && currentMeta.file === it.file) resetPlayer();
      loadGallery();
    };

    d.onclick = () => play(it.audio_url || `/outputs/${encodeURIComponent(it.file)}`, it, true);
    g.appendChild(d);
  }
  markGalleryItem();
}

pollHealth(); setInterval(pollHealth, 5000);
loadGallery();

// ==================== план ABC: модалка (plan-only без рендера) ====================
// детали ошибок бывают списком (422 pydantic) — приводим к строке
function errText(j, code) {
  const d = j && j.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) return d.map(e => e.msg || JSON.stringify(e)).join('; ');
  return d ? JSON.stringify(d) : code;
}

$('goPlan').onclick = async () => {
  const style = $('style').value.trim();
  const lyrics = $('lyrics').value.trim();
  if (style.length < 3 || lyrics.length < 2) {
    $('stLine').innerHTML = '<span class="err">Нужны стиль и лирика — по ним строится план</span>'; return;
  }
  let cot = $('cot').value;
  if (cot === 'off') { $('cot').value = 'melody'; cot = 'melody'; }
  $('planAbc').value = '';
  $('planStat').textContent = 'модель пишет план… (~10–60 с, при идущей генерации — отказ)';
  $('planOverlay').classList.add('open');
  try {
    const r = await fetch('/api/plan', { method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ style, voice: $('voice').value.trim(), lyrics, cot,
        seed: parseInt($('seed').value) || -1,
        arc: $('arcSel') ? $('arcSel').value : '' }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { $('planStat').innerHTML = `<span class="err">${errText(j, r.status)}</span>`; return; }
    $('planAbc').value = j.abc;
    $('planStat').textContent = `≈ ${fmtSec(j.seconds || 0)} · ${j.tokens} токен. ABC · seed ${j.seed} · план можно править`;
  } catch (e) { $('planStat').innerHTML = `<span class="err">сервер недоступен: ${e}</span>`; }
};
$('planClose').onclick = () => $('planOverlay').classList.remove('open');
$('planOverlay').addEventListener('click', e => {
  if (e.target === $('planOverlay')) $('planOverlay').classList.remove('open');
});
$('planRender').onclick = () => {
  const abc = $('planAbc').value.trim();
  if (abc.length < 20) { $('planStat').innerHTML = '<span class="err">План пуст — нечего рендерить</span>'; return; }
  // план идёт тем же путём, что кавер: поле ABC, план-генерация пропускается
  midiLoaded = true;
  $('abc').value = abc;
  $('midiResult').hidden = false;
  $('abcStat').textContent = 'свой план (как построила модель / после правок)';
  $('midiDl').hidden = true;
  $('midiHint').textContent = 'Рендер пойдёт по этому ABC — авторский план строиться не будет.';
  $('planOverlay').classList.remove('open');
  submit(false, 1);
};

// ==================== повтор параметров, переименование ====================
$('tRepeat').onclick = () => {
  if (!currentMeta) return;
  const m = currentMeta;
  $('title').value = m.title || '';
  $('style').value = m.style_base || m.style || '';
  $('voice').value = m.voice || '';
  $('lyrics').value = m.lyrics || '';
  $('cot').value = m.cot || 'full';
  $('cfg').value = (m.cfg_scale != null && m.cfg_scale !== '') ? m.cfg_scale : '';
  $('seed').value = (m.seed != null) ? m.seed : -1;
  resetMidi();
  if (m.abc_file) {  // был кавер/правка — вернуть партитуру в поле ABC
    fetch(`/api/gallery/${encodeURIComponent(stem)}/abc`)
      .then(r => r.ok ? r.json() : null)
      .then(j => {
        if (!j || !j.abc) return;
        midiLoaded = true;
        $('abc').value = j.abc;
        $('midiResult').hidden = false;
        $('abcStat').textContent = `повтор кавера: ${m.abc_chars || j.abc.length} симв. ABC`;
      }).catch(() => {});
  }
  window.scrollTo({ top: 0, behavior: 'smooth' });
};
$('tStudio').onclick = () => {
  if (!currentMeta) return;
  const stem = currentMeta.file.replace(/\.flac$/, '');
  window.open('/studio?stem=' + encodeURIComponent(stem), '_blank');
};
$('tRename').onclick = async () => {
  if (!currentMeta) return;
  const name = prompt('Новое название (метка в истории — файлы не переименовываются):',
                      currentMeta.title || '');
  if (name === null) return;
  const stem = currentMeta.file.replace(/\.flac$/, '');
  try {
    const r = await fetch(`/api/gallery/${encodeURIComponent(stem)}/rename`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ title: name }) });
    if (!r.ok) { const e = await r.json(); alert('Не удалось: ' + (e.detail || r.status)); return; }
    currentMeta.title = name.trim();
    $('tTitle').textContent = currentMeta.title ||
      (currentMeta.style_base || currentMeta.style || '').split(/[.\n]/)[0];
    loadGallery();
  } catch (e) { alert('Сервер недоступен: ' + e); }
};

// ==================== очередь плеера и тема ====================
$('pNext').onclick = () => stepPlay(1);
$('pPrev').onclick = () => stepPlay(-1);
function stepPlay(dir) {
  if (!playList.length || !currentMeta) return;
  let i = playList.findIndex(m => m.file === currentMeta.file);
  if (i < 0) i = 0;
  const nxt = playList[(i + dir + playList.length) % playList.length];
  play(nxt.audio_url || `/outputs/${encodeURIComponent(nxt.file)}`, nxt, true);
}
$('player').addEventListener('ended', () => stepPlay(1));
(function () {
  const p = $('player');
  const v = parseFloat(localStorage.getItem('yue_volume') || '1');
  if (!Number.isNaN(v)) p.volume = Math.min(1, Math.max(0, v));
  p.addEventListener('volumechange', () => localStorage.setItem('yue_volume', p.volume));
})();
(function () {
  const saved = localStorage.getItem('yue_theme');
  if (saved) document.documentElement.dataset.theme = saved;
  $('themeToggle').onclick = () => {
    const nxt = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = nxt;
    localStorage.setItem('yue_theme', nxt);
  };
})();
