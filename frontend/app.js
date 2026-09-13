/* app.js — Deaf/HoH Accessibility Assistant frontend logic */

const CONFIG = {
  API_BASE: '', // same origin
  TICK_INTERVAL_MS: 2800,       // faster cadence so captions/headlines appear sooner
  AUDIO_CHUNK_MS: 2800,         // shorter chunk = less wait before processing (was 4000)
  RECORDING_DURATION_MS: 3000,  // personal-sound enrol clip length
  SIGN_RECORDING_FRAME_INTERVAL_MS: 100,
  MAX_CAPTIONS_KEPT: 200,       // continuous mode: keep long history
  MAX_SOUNDS_KEPT: 200,
  MAX_HEADLINES_KEPT: 50,
  SKIP_SILENCE_THRESHOLD: 0.005, // skip sending audio when RMS is below this
};

// --- State Management ---
let currentState = {
  activeTab: 'live',
  isListening: false,
  stream: null,
  signPreviewStream: null,
  signRecording: false,
  signCaptureTimer: null,
  signCapturedFrames: [],
  tickTimer: null,
  
  enrolClips: [], // array of base64 strings
  enrolRecording: false,

  // ── Session log (kept across the entire listening session,
  //    never cleared until the user explicitly resets) ───────────────
  sessionLog: {
    captions: [],     // [{ts, text, speaker, emotion}]
    sounds: [],       // [{ts, label, priority, confidence}]
    hazards: [],      // [{ts, label, position}]
    signs: [],        // [{ts, label}]
    emotions: [],     // [{ts, label, confidence, mixed}]
    sessionStart: null,
  },

  stats: {
    eventCount: 0,
    criticalCount: 0
  }
};

// --- DOM Elements ---
const el = {
  tabs: document.querySelectorAll('.tab'),
  pages: document.querySelectorAll('.page'),
  
  // Live
  startBtn: document.getElementById('start-btn'),
  stopBtn: document.getElementById('stop-btn'),
  status: document.getElementById('status'),
  headlines: document.getElementById('headlines'),
  captions: document.getElementById('captions'),
  sounds: document.getElementById('sounds'),
  scene: document.getElementById('scene'),
  lipRel: document.getElementById('lip-rel'),
  speaker: document.getElementById('speaker'),
  knownPerson: document.getElementById('known-person'),
  latency: document.getElementById('latency'),
  eventCount: document.getElementById('event-count'),
  criticalCount: document.getElementById('critical-count'),
  webcamPreview: document.getElementById('webcam-preview'),
  
  // Personal
  enrolLabel: document.getElementById('enrol-label'),
  enrolPriority: document.getElementById('enrol-priority'),
  enrolRecordBtn: document.getElementById('enrol-record'),
  enrolCount: document.getElementById('enrol-count'),
  enrolSubmitBtn: document.getElementById('enrol-submit'),
  enrolClearBtn: document.getElementById('enrol-clear'),
  enrolledList: document.getElementById('enrolled-list'),
  
  // Summary
  summaryBtn: document.getElementById('generate-summary'),
  summaryText: document.getElementById('summary-text'),
  summaryStats: document.getElementById('summary-stats'),
  
  // Diary
  diaryWindow: document.getElementById('diary-window'),
  diaryMinPriority: document.getElementById('diary-min-priority'),
  diaryRefreshBtn: document.getElementById('diary-refresh'),
  diaryHistogram: document.getElementById('diary-histogram'),
  diaryEvents: document.getElementById('diary-events'),
};

// --- Critical-event haptic + flash alert ---
// Per-session dedupe of fired alerts (so the same alarm doesn't strobe).
const _firedCritical = new Set();
// User preference, persisted in localStorage. Default ON.
let alertsEnabled = (() => {
  const v = localStorage.getItem('accessibility.alerts.enabled');
  return v === null ? true : v === 'true';
})();

function triggerCriticalAlert(ev) {
  if (!alertsEnabled) return;
  // Dedupe key: same source+label inside a one-second bucket only fires once.
  const bucket = Math.floor((ev.ts != null ? ev.ts : Date.now() / 1000));
  const key = `${ev.source || 'unknown'}|${ev.label || 'unknown'}|${bucket}`;
  if (_firedCritical.has(key)) return;
  _firedCritical.add(key);
  // Trim the dedupe set so it doesn't grow forever (keep last ~500).
  if (_firedCritical.size > 500) {
    const it = _firedCritical.values();
    for (let i = 0; i < 100; i++) _firedCritical.delete(it.next().value);
  }

  // 1. Screen flash — a sustained ~4-second pulsing red so an emergency
  //    (smoke alarm, your name) is impossible to miss, not a single blink.
  document.body.classList.add('critical-flash');
  setTimeout(() => document.body.classList.remove('critical-flash'), 4000);

  // 2. Haptic — a long, insistent pulse train (~3.5s), feature-detect first.
  if ('vibrate' in navigator) {
    try {
      navigator.vibrate([500, 200, 500, 200, 500, 200, 500, 200, 500]);
    } catch (e) { /* ignore */ }
  }
}

// --- Feedback wiring for personal-sound matches ---
// Each personal-sound alert carries a match_id. Clicking ✓/✗ POSTs
// /enrolled/feedback so the backend can incrementally refine the
// embedding (positive) or bump the threshold (negative).
function wireFeedbackButtons(li, matchId) {
  // Defer the listener attachment to next tick so innerHTML has parsed.
  setTimeout(() => {
    const wrap = li.querySelector(`[data-mid="${matchId}"]`);
    if (!wrap) return;
    const yes = wrap.querySelector('.feedback-yes');
    const no  = wrap.querySelector('.feedback-no');
    const post = async (is_positive) => {
      yes.disabled = true; no.disabled = true;
      try {
        const r = await fetch(`${CONFIG.API_BASE}/enrolled/feedback`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({match_id: matchId, is_positive})
        });
        const data = await r.json();
        if (data.ok) {
          wrap.innerHTML = `<small class="muted">${is_positive ? '✓ learned' : '✗ noted'}</small>`;
        } else {
          wrap.innerHTML = `<small class="muted">${data.error || 'failed'}</small>`;
        }
      } catch (e) {
        wrap.innerHTML = `<small class="muted">network error</small>`;
      }
    };
    if (yes) yes.addEventListener('click', () => post(true));
    if (no)  no.addEventListener('click', () => post(false));
  }, 0);
}

// --- Name & Keyword Alerts panel (replaces the old Sound Direction panel) ---
function setupNameAlertControls() {
  const nameSave = document.getElementById('alert-self-save');
  const nameInput = document.getElementById('alert-self-name');
  const kwSave = document.getElementById('alert-keywords-save');
  const kwInput = document.getElementById('alert-keywords');

  if (nameSave && nameInput) {
    const save = async () => {
      const name = (nameInput.value || '').trim();
      if (!name) return;
      try {
        await fetch(`${CONFIG.API_BASE}/me/name`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({name})
        });
        nameSave.textContent = 'Saved ✓';
        setTimeout(() => { nameSave.textContent = 'Save'; }, 1500);
      } catch (e) { console.error(e); }
    };
    nameSave.addEventListener('click', save);
    nameInput.addEventListener('keydown', e => { if (e.key === 'Enter') save(); });
  }

  if (kwSave && kwInput) {
    const save = async () => {
      const words = (kwInput.value || '').split(',').map(s => s.trim()).filter(Boolean);
      try {
        await fetch(`${CONFIG.API_BASE}/keywords`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({keywords: words})
        });
        kwSave.textContent = 'Saved ✓';
        setTimeout(() => { kwSave.textContent = 'Save'; }, 1500);
      } catch (e) { console.error(e); }
    };
    kwSave.addEventListener('click', save);
    kwInput.addEventListener('keydown', e => { if (e.key === 'Enter') save(); });
  }
}

async function loadNameAlertConfig() {
  try {
    const [pp, kw] = await Promise.all([
      fetch(`${CONFIG.API_BASE}/people`).then(r => r.json()).catch(() => ({})),
      fetch(`${CONFIG.API_BASE}/keywords`).then(r => r.json()).catch(() => ({})),
    ]);
    const nameInput = document.getElementById('alert-self-name');
    const kwInput = document.getElementById('alert-keywords');
    if (nameInput && pp.self_name) nameInput.value = pp.self_name;
    if (kwInput && Array.isArray(kw.keywords)) kwInput.value = kw.keywords.join(', ');
  } catch (e) { /* ignore */ }
}

function updateNameAlerts(nameCall) {
  // nameCall = { self_called, keywords_heard: [], mentioned: [] }
  const list = document.getElementById('name-alerts');
  const empty = document.getElementById('name-alerts-empty');
  if (!list) return;

  const entries = [];
  if (nameCall.self_called) entries.push({ text: '🚨 Your name was called!', critical: true });
  (nameCall.keywords_heard || []).forEach(kw =>
    entries.push({ text: `🚨 Keyword heard: "${kw}"`, critical: true }));
  (nameCall.mentioned || []).forEach(n =>
    entries.push({ text: `💬 ${n} was mentioned`, critical: false }));

  if (entries.length === 0) return;
  if (empty) empty.style.display = 'none';

  const time = new Date().toLocaleTimeString();
  entries.forEach(e => {
    const li = document.createElement('li');
    li.className = e.critical ? 'critical' : 'inform';
    li.innerHTML = `<span>${escapeHtml(e.text)}</span> <small>${time}</small>`;
    list.prepend(li);
  });
  while (list.children.length > 30) list.removeChild(list.lastChild);
}

// ─── Streaming / partial captions ───────────────────────────────────
// A Web Audio VAD loop captures each utterance as it is spoken, shows a
// live grey "interim" caption that updates ~every second, and finalises
// it (solid) when you stop speaking. Interim buffers use a fast tiny
// model; the final buffer uses the accurate model and also runs name /
// keyword detection. While active, /process skips its own STT.
let streamingActive = (() => {
  const v = localStorage.getItem('accessibility.stream.enabled');
  return v === null ? true : v === 'true';
})();
// Most recent enrolled person seen on camera. The tick updates this; the
// streaming caption path reads it, since the two run independently.
let _lastKnownPerson = null;      // {name, similarity, ts}
const _stream = { ctx: null, src: null, proc: null,
  speaking: false, utter: [], silenceStart: 0, speechStart: 0, lastInterim: 0,
  interimEl: null };

const STREAM_RMS = 0.014;        // speech energy threshold
const STREAM_SILENCE_MS = 700;   // trailing silence that ends an utterance
const STREAM_INTERIM_MS = 1000;  // how often to refresh the interim caption
const STREAM_MAX_MS = 12000;     // force-finalise a very long utterance

function _concatF32(chunks) {
  let n = 0; for (const c of chunks) n += c.length;
  const out = new Float32Array(n);
  let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}
function _encodeWav(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8, 'WAVE');
  w(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, 'data'); v.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true); off += 2;
  }
  return buf;
}
function _abToB64(buf) {
  let bin = ''; const bytes = new Uint8Array(buf); const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

async function _sendStreamChunk(pcm, sr, isFinal) {
  try {
    const b64 = _abToB64(_encodeWav(pcm, sr));
    const r = await fetch(`${CONFIG.API_BASE}/stt/stream`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio_b64: b64, is_final: isFinal }),
    });
    const data = await r.json();
    if (isFinal) renderFinalCaption(data.text, data.name_call);
    else renderInterimCaption(data.text);
  } catch (e) { /* transient — ignore */ }
}

function renderInterimCaption(text) {
  if (!text) return;
  if (!_stream.interimEl) {
    _stream.interimEl = document.createElement('p');
    _stream.interimEl.className = 'caption-interim';
    el.captions.prepend(_stream.interimEl);
  }
  _stream.interimEl.innerHTML = `<em>${escapeHtml(text)}…</em>`;
}

function renderFinalCaption(text, nameCall) {
  if (_stream.interimEl) { _stream.interimEl.remove(); _stream.interimEl = null; }
  if (text) {
    const p = document.createElement('p');
    const time = new Date().toLocaleTimeString();
    // Attribute the caption to the enrolled person currently on camera, if
    // one was recognised recently (within the last 10 s).
    const known = (_lastKnownPerson && (Date.now()/1000 - _lastKnownPerson.ts) < 10)
                ? _lastKnownPerson.name : null;
    const who = known ? `<span class="speaker">${escapeHtml(known)}:</span> ` : '';
    p.innerHTML = `<small class="caption-time">[${time}]</small> ${who}${escapeHtml(text)}`;
    el.captions.prepend(p);
    currentState.sessionLog.captions.push({ ts: Date.now() / 1000, text, speaker: known || 'Speaker', emotion: null });
    while (el.captions.children.length > CONFIG.MAX_CAPTIONS_KEPT) {
      el.captions.removeChild(el.captions.lastChild);
    }
  }
  // Name / keyword alerts come from the final caption now.
  if (nameCall) {
    updateNameAlerts(nameCall);
    if (nameCall.self_called || (nameCall.keywords_heard || []).length) {
      triggerCriticalAlert({ source: 'speech', label: 'name-or-keyword', ts: Date.now() / 1000 });
    }
  }
}

function _finaliseUtterance() {
  if (_stream.utter.length) {
    _sendStreamChunk(_concatF32(_stream.utter), _stream.ctx.sampleRate, true);
  }
  _stream.speaking = false; _stream.utter = []; _stream.silenceStart = 0;
}

function startCaptionStream(stream) {
  if (!streamingActive || _stream.ctx) return;
  const tracks = stream.getAudioTracks ? stream.getAudioTracks() : [];
  if (!tracks.length) return;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    _stream.ctx = new Ctx();
    if (_stream.ctx.state === 'suspended') _stream.ctx.resume();
    _stream.src = _stream.ctx.createMediaStreamSource(new MediaStream(tracks));
    _stream.proc = _stream.ctx.createScriptProcessor(4096, 1, 1);
    const zero = _stream.ctx.createGain(); zero.gain.value = 0;
    _stream.src.connect(_stream.proc);
    _stream.proc.connect(zero); zero.connect(_stream.ctx.destination);
    _stream.proc.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      let sum = 0; for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
      const rms = Math.sqrt(sum / input.length);
      const now = performance.now();
      if (rms >= STREAM_RMS) {
        if (!_stream.speaking) { _stream.speaking = true; _stream.speechStart = now; _stream.utter = []; _stream.lastInterim = now; }
        _stream.utter.push(new Float32Array(input));
        _stream.silenceStart = 0;
        if (now - _stream.lastInterim >= STREAM_INTERIM_MS) {
          _stream.lastInterim = now;
          _sendStreamChunk(_concatF32(_stream.utter), _stream.ctx.sampleRate, false);
        }
        if (now - _stream.speechStart >= STREAM_MAX_MS) _finaliseUtterance();
      } else if (_stream.speaking) {
        _stream.utter.push(new Float32Array(input));
        if (_stream.silenceStart === 0) _stream.silenceStart = now;
        else if (now - _stream.silenceStart >= STREAM_SILENCE_MS) _finaliseUtterance();
      }
    };
  } catch (e) {
    console.warn('[stream] caption stream unavailable, using /process captions:', e);
    streamingActive = false;
  }
}

function stopCaptionStream() {
  try {
    if (_stream.proc) { _stream.proc.disconnect(); _stream.proc.onaudioprocess = null; }
    if (_stream.src) _stream.src.disconnect();
    if (_stream.ctx) _stream.ctx.close();
  } catch (e) { /* ignore */ }
  _stream.ctx = null; _stream.src = null; _stream.proc = null;
  _stream.speaking = false; _stream.utter = [];
  if (_stream.interimEl) { _stream.interimEl.remove(); _stream.interimEl = null; }
}

// --- "Name this speaker" prompt ---
// When the diarizer reports a never-before-seen SPEAKER_XX, show an
// inline prompt above the Captions panel so the user can map it to a
// friendly name (Sarah / Mike / Mum). Only shows ONCE per raw id per
// session so it doesn't nag.
const _promptedSpeakers = new Set();

function offerSpeakerNamePrompt(rawId) {
  if (!rawId || _promptedSpeakers.has(rawId)) return;
  _promptedSpeakers.add(rawId);

  // Idempotent: if a prompt for this id is already in the DOM, skip.
  if (document.getElementById(`speaker-prompt-${rawId}`)) return;

  const captionsCard = el.captions?.parentElement;
  if (!captionsCard) return;

  const box = document.createElement('div');
  box.id = `speaker-prompt-${rawId}`;
  box.className = 'speaker-prompt';
  box.innerHTML = `
    <span>Unknown speaker <strong>${escapeHtml(rawId)}</strong> — name them?</span>
    <input type="text" placeholder="e.g. Sarah" maxlength="40" />
    <button class="primary">Save</button>
    <button class="ghost dismiss">×</button>
  `;
  const input = box.querySelector('input');
  const save = box.querySelector('button.primary');
  const dismiss = box.querySelector('button.dismiss');

  save.addEventListener('click', async () => {
    const name = (input.value || '').trim();
    if (!name) return;
    try {
      await fetch(`${CONFIG.API_BASE}/speakers/name`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({raw_id: rawId, name})
      });
    } catch (e) { /* fail silent; user can retry */ }
    box.remove();
  });
  dismiss.addEventListener('click', () => box.remove());
  input.addEventListener('keydown', e => { if (e.key === 'Enter') save.click(); });

  // Insert just above the captions panel.
  captionsCard.insertBefore(box, el.captions);
}

// --- Initialization ---
function init() {
  setupNavigation();
  setupLiveControls();
  setupPersonalControls();
  setupPeopleControls();
  setupNameAlertControls();
  setupSummaryControls();
  setupDiaryControls();

  // Wire the critical-alerts toggle in the header.
  const toggle = document.getElementById('alerts-enabled');
  if (toggle) {
    toggle.checked = alertsEnabled;
    toggle.addEventListener('change', () => {
      alertsEnabled = toggle.checked;
      localStorage.setItem('accessibility.alerts.enabled', String(alertsEnabled));
    });
  }

  // Initial data load
  fetchEnrolledSounds();
  loadNameAlertConfig();   // populate the name + keyword fields from the backend
  // Verify high-quality TTS is reachable; warn if not
  checkTtsHealth();
}

// --- Navigation ---
function setupNavigation() {
  el.tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.id.replace('tab-', '');
      switchTab(target);
    });
  });
}

function switchTab(target) {
  currentState.activeTab = target;
  
  el.tabs.forEach(t => {
    t.classList.toggle('active', t.id === `tab-${target}`);
  });
  
  el.pages.forEach(p => {
    p.classList.toggle('active', p.id === `page-${target}`);
  });
  
  if (target === 'diary') refreshDiary();
  if (target === 'personal') fetchEnrolledSounds();
  if (target === 'people') fetchPeople();
}

// --- Live Mode ---
function setupLiveControls() {
  el.startBtn.addEventListener('click', startListening);
  el.stopBtn.addEventListener('click', stopListening);
}

async function startListening() {
  if (currentState.isListening && currentState.stream) {
    showWebcamPreview(currentState.stream);
    return currentState.stream;
  }

  try {
    releaseSignPreviewStream();

    // Requesting both audio and video for multimodal fusion
    currentState.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
      video: { width: 640, height: 480 }
    });
    
    showWebcamPreview(currentState.stream);
    
    currentState.isListening = true;
    // Mark session start; sessionLog accumulates from here until Stop.
    currentState.sessionLog.sessionStart = Date.now() / 1000;
    el.startBtn.classList.add('hidden');
    el.stopBtn.classList.remove('hidden');
    el.status.textContent = '🔴 recording — listening continuously…';

    startTickLoop();
    startCaptionStream(currentState.stream);   // live partial captions
    return currentState.stream;
  } catch (err) {
    console.error('Error accessing media devices:', err);
    alert('Could not access microphone or camera. Please ensure permissions are granted.');
    return null;
  }
}

async function stopListening() {
  currentState.isListening = false;
  if (currentState.tickTimer) clearTimeout(currentState.tickTimer);
  stopCaptionStream();
  if (currentState.signRecording) {
    cancelSignRecording('Sign recording stopped.');
  }
  
  // Stop the main media stream (audio + video tracks)
  if (currentState.stream) {
    currentState.stream.getTracks().forEach(track => {
      try { track.stop(); } catch (e) {}
    });
    currentState.stream = null;
  }
  // Also release any sign-preview-only stream so the camera LED turns off
  releaseSignPreviewStream();

  hideWebcamPreviewIfIdle();
  el.startBtn.classList.remove('hidden');
  el.stopBtn.classList.add('hidden');

  // Wait for any in-flight tick to finish before building the report.
  // Without this, events detected in the last tick appear in the panel
  // but show as "0" in the session summary (race condition).
  if (_pendingTick) {
    el.status.textContent = 'finishing…';
    try {
      // Cap at 10 s so a hung tick never blocks the Stop flow
      await Promise.race([
        _pendingTick,
        new Promise(r => setTimeout(r, 10000)),
      ]);
    } catch (_) {}
  }

  // Build & display the final session report
  const report = buildSessionReport();
  showSessionReport(report);
  el.status.textContent = `✓ session ended — ${report.duration_min} min, ${report.totals.events} events`;
}

// ── Build the end-of-session report from the accumulated log ──────
function buildSessionReport() {
  const L = currentState.sessionLog;
  const end = Date.now() / 1000;
  const start = L.sessionStart || end;
  const durSec = end - start;

  // Tally emotion histogram
  const emoCounts = {};
  for (const e of L.emotions) {
    emoCounts[e.label] = (emoCounts[e.label] || 0) + 1;
  }

  // Tally sound histogram
  const soundCounts = {};
  for (const s of L.sounds) {
    soundCounts[s.label] = (soundCounts[s.label] || 0) + 1;
  }

  const criticalEvents = L.sounds.filter(s => s.priority === 'CRITICAL');

  return {
    start, end,
    duration_min: (durSec / 60).toFixed(1),
    duration_sec: Math.round(durSec),
    totals: {
      events: L.captions.length + L.sounds.length + L.hazards.length + L.signs.length,
      captions: L.captions.length,
      sounds: L.sounds.length,
      hazards: L.hazards.length,
      signs: L.signs.length,
      critical: criticalEvents.length,
    },
    emotion_histogram: emoCounts,
    sound_histogram: soundCounts,
    transcript: L.captions.map(c => `[${new Date(c.ts*1000).toLocaleTimeString()}] ${c.speaker}: ${c.text}`).join('\n'),
    critical_events: criticalEvents,
    log: L,  // full log embedded for download
  };
}

// ── Render the final session report panel + offer downloads ───────
function showSessionReport(report) {
  let panel = document.getElementById('session-report');
  if (!panel) {
    // Create panel under the headlines section if it doesn't exist yet
    panel = document.createElement('div');
    panel.id = 'session-report';
    panel.className = 'card session-report';
    const liveTab = document.getElementById('page-live') || document.body;
    liveTab.appendChild(panel);
  }
  const r = report;

  const emoLines = Object.entries(r.emotion_histogram)
    .sort((a,b) => b[1]-a[1])
    .map(([k,v]) => `<li><strong>${escapeHtml(k)}</strong> — ${v} tick${v>1?'s':''}</li>`)
    .join('');

  const soundLines = Object.entries(r.sound_histogram)
    .sort((a,b) => b[1]-a[1]).slice(0, 12)
    .map(([k,v]) => `<li>${escapeHtml(k)} <span class="muted">×${v}</span></li>`)
    .join('');

  const critLines = r.critical_events.map(c =>
    `<li class="critical">🔴 ${escapeHtml(c.label)} at ${new Date(c.ts*1000).toLocaleTimeString()}</li>`
  ).join('') || '<li class="muted">No critical events. ✓</li>';

  panel.innerHTML = `
    <h2>📜 Session Report</h2>
    <p class="muted">Duration: <strong>${r.duration_min} min</strong> ·
       Events: <strong>${r.totals.events}</strong>
       (captions ${r.totals.captions} · sounds ${r.totals.sounds} ·
        hazards ${r.totals.hazards} · signs ${r.totals.signs} ·
        <span style="color:#D32F2F;">critical ${r.totals.critical}</span>)
    </p>

    <h3>🚨 Critical events</h3>
    <ul class="event-list">${critLines}</ul>

    <h3>😊 Emotion timeline (per-tick label counts)</h3>
    <ul class="event-list">${emoLines || '<li class="muted">No emotion data.</li>'}</ul>

    <h3>🔊 Most frequent sounds</h3>
    <ul class="event-list">${soundLines || '<li class="muted">No sounds detected.</li>'}</ul>

    <h3>📝 Full transcript</h3>
    <pre class="transcript">${escapeHtml(r.transcript) || '(no speech captured)'}</pre>

    <div class="controls">
      <button id="dl-transcript" class="primary">⬇ Download transcript (.txt)</button>
      <button id="dl-report" class="secondary">⬇ Download full JSON report</button>
      <button id="speak-transcript" class="secondary">🔊 Speak the last 5 captions</button>
      <button id="new-session" class="ghost">↻ New session</button>
    </div>
  `;

  // Wire downloads
  document.getElementById('dl-transcript').onclick = () => downloadFile(
    `deaf-accessibility-transcript-${Date.now()}.txt`, r.transcript, 'text/plain');
  document.getElementById('dl-report').onclick = () => downloadFile(
    `deaf-accessibility-session-${Date.now()}.json`,
    JSON.stringify(r, null, 2), 'application/json');
  document.getElementById('speak-transcript').onclick = () => speakLastCaptions(r.log.captions.slice(-5));
  document.getElementById('new-session').onclick = resetSessionLog;
}

function downloadFile(filename, content, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 200);
}

async function speakLastCaptions(captions) {
  if (!captions || !captions.length) return;
  const joined = captions.map(c => c.text).join('. ');
  try {
    const res = await fetch(`${CONFIG.API_BASE}/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: joined }),
    });
    const data = await res.json();
    if (data.audio_b64) {
      const audio = new Audio('data:audio/mpeg;base64,' + data.audio_b64);
      audio.play();
      return;
    }
  } catch (e) {
    console.warn('Server TTS failed, falling back to browser TTS:', e);
  }
  // Fallback to browser TTS
  if ('speechSynthesis' in window) {
    const u = new SpeechSynthesisUtterance(joined);
    window.speechSynthesis.speak(u);
  }
}

function resetSessionLog() {
  currentState.sessionLog = {
    captions: [], sounds: [], hazards: [], signs: [], emotions: [],
    sessionStart: null,
  };
  currentState.stats = { eventCount: 0, criticalCount: 0 };
  ['headlines', 'sounds', 'captions'].forEach(id => {
    const e = document.getElementById(id);
    if (e) e.innerHTML = '';
  });
  const r = document.getElementById('session-report');
  if (r) r.remove();
  el.status.textContent = 'idle';
}

function startTickLoop() {
  if (!currentState.isListening) return;

  // Track the in-flight promise so stopListening() can await it before
  // building the session report — prevents a race where the tick completes
  // after the report is already generated, causing "sounds: 0" even though
  // sounds appeared in the panel.
  _pendingTick = processTick();
  _pendingTick.finally(() => {
    _pendingTick = null;
    if (currentState.isListening) {
      currentState.tickTimer = setTimeout(startTickLoop, CONFIG.TICK_INTERVAL_MS);
    }
  });
}

async function processTick() {
  el.status.textContent = '🔴 listening…';

  // 1. Capture a longer audio chunk (default 4s) so sentence boundaries
  //    aren't sliced mid-word. We sample video frames in parallel.
  const audioPromise = captureAudio(CONFIG.AUDIO_CHUNK_MS);
  const frames = [];

  // Capture 4 frames (reduced from 8) over the audio window for motion + emotion smoothing.
  // This reduces client-side processing time and payload size.
  for (let i = 0; i < 3; i++) {
    const f = captureVideoFrame();
    if (f) frames.push(f.split(',')[1]);
    await new Promise(r => setTimeout(r, CONFIG.AUDIO_CHUNK_MS / 3));
  }

  const audioBlob = await audioPromise;
  const audioB64 = await blobToBase64(audioBlob);

  // 2. Client-side silence detection — avoid a 6-second server round-trip
  //    when the audio is just background hum. Compute RMS from the blob.
  let audioRms = 1.0;
  try {
    audioRms = await computeBlobRms(audioBlob);
  } catch (e) { /* ignore — fall back to sending */ }

  if (audioRms < CONFIG.SKIP_SILENCE_THRESHOLD && frames.length === 0) {
    // Nothing to process — skip this tick entirely.
    el.status.textContent = '🔴 listening… (quiet)';
    return;
  }

  try {
    const response = await fetch(`${CONFIG.API_BASE}/process`, {
      method: 'POST',
      body: JSON.stringify({
        audio_b64: audioB64,
        frames_b64: frames,
        audio_rms_hint: audioRms,
        stream_captions: streamingActive,   // when true, /process skips STT
      }),
      headers: { 'Content-Type': 'application/json' }
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);

    const data = await response.json();
    console.debug('[tick response]', JSON.stringify(data, null, 2));
    updateLiveUI(data);

    el.status.textContent = 'listening...';
  } catch (err) {
    console.error('Processing error:', err);
    el.status.textContent = 'error connecting...';
  }
}

function updateLiveUI(data) {
  const nowTs = Date.now() / 1000;
  const tickTime = new Date().toLocaleTimeString();
  const log = currentState.sessionLog;

  // ── Headlines (continuous, capped at MAX_HEADLINES_KEPT) ─────────
  if (data.headlines && data.headlines.length > 0) {
    data.headlines.forEach(h => {
      const li = document.createElement('li');
      const pName = (h.priority_name || 'ambient').toLowerCase();
      li.className = pName;
      li.innerHTML = `<span>${h.label}</span> <small>${tickTime}</small>`;
      el.headlines.prepend(li);

      currentState.stats.eventCount++;
      if (h.priority_name === 'CRITICAL') {
        currentState.stats.criticalCount++;
        triggerCriticalAlert(h);
      }
    });
    while (el.headlines.children.length > CONFIG.MAX_HEADLINES_KEPT) {
      el.headlines.removeChild(el.headlines.lastChild);
    }
  }

  // ── Captions (CONTINUOUS — log every tick into sessionLog) ───────
  // Skip when the streaming-caption loop is active — it owns the panel.
  const sttText = data.caption_with_emotion || (data.stt && data.stt.text);
  if (sttText && !streamingActive) {
    // Priority order for speaker label:
    //   1. data.stt.speaker_name — friendly name mapped from pyannote
    //      diarization (e.g. "Sarah") — set via /speakers/name endpoint.
    //   2. data.stt.speaker — raw pyannote id (SPEAKER_00) — render that
    //      so the user knows there's an un-named speaker to label.
    //   3. data.face.speaker_attribution.label — fall back to the face
    //      tracker's heuristic position-based attribution.
    //   4. "Speaker" — generic last resort.
    const friendlyName = data.stt?.speaker_name;
    const rawSpeakerId = data.stt?.speaker;
    // If the face tracker identified an enrolled person, prefer that name
    // over a raw SPEAKER_XX label (face id is more user-meaningful).
    const identifiedFaceName = data.identified_person?.name;
    const faceLabel = data.face?.speaker_attribution?.label;
    const positionLabel = data.face_attribution?.speaker_attribution?.position
        ? `speaker_${data.face_attribution.speaker_attribution.position}` : null;
    const speaker = friendlyName
                 || identifiedFaceName
                 || rawSpeakerId
                 || faceLabel
                 || positionLabel
                 || 'Speaker';

    // Surface unknown speakers so the user can name them.
    if (rawSpeakerId && !friendlyName) {
      offerSpeakerNamePrompt(rawSpeakerId);
    }

    const p = document.createElement('p');
    const time = new Date().toLocaleTimeString();
    p.innerHTML = `<small class="caption-time">[${time}]</small> <span class="speaker">${escapeHtml(speaker)}:</span> ${escapeHtml(sttText)}`;
    el.captions.prepend(p);

    // Record in session log (immutable history)
    log.captions.push({
      ts: nowTs, text: sttText, speaker,
      emotion: data.emotion?.label || null,
    });

    // Cap DOM at MAX_CAPTIONS_KEPT (session log is uncapped — full record)
    while (el.captions.children.length > CONFIG.MAX_CAPTIONS_KEPT) {
      el.captions.removeChild(el.captions.lastChild);
    }
  }

  // ── Sounds (continuous, with direction) ──────────────────────────
  const sounds = data.sounds || (data.events && data.events.filter(e => e.source === 'sound'));
  if (sounds && sounds.length > 0) {
    sounds.forEach(s => {
      const li = document.createElement('li');
      const pName = (s.priority_name || 'ambient').toLowerCase();
      li.className = pName;
      const dir = s.direction ? ` [${s.direction}]` : '';
      const conf = s.confidence != null ? ` (${Math.round(s.confidence * 100)}%)` : '';

      // Personal-sound matches carry a match_id in extra — render ✓/✗
      // feedback buttons next to them so the user can confirm or reject.
      // This is the continuous-learning loop.
      const matchId = s.extra?.match_id;
      const feedbackHtml = matchId
        ? `<span class="feedback-buttons" data-mid="${matchId}">
             <button class="feedback-yes" title="Correct — refine this recogniser">✓</button>
             <button class="feedback-no"  title="Wrong — raise threshold">✗</button>
           </span>`
        : '';
      li.innerHTML = `<span>${escapeHtml(s.label)}${dir}${conf}</span> ${feedbackHtml}<small>${tickTime}</small>`;
      if (matchId) wireFeedbackButtons(li, matchId);

      el.sounds.prepend(li);
      log.sounds.push({
        ts: nowTs, label: s.label, priority: s.priority_name,
        confidence: s.confidence, direction: s.direction,
      });
    });
    while (el.sounds.children.length > CONFIG.MAX_SOUNDS_KEPT) {
      el.sounds.removeChild(el.sounds.lastChild);
    }
  }

  // ── Hazards logged for the report ───────────────────────────────
  if (data.hazards && data.hazards.length > 0) {
    data.hazards.forEach(h => log.hazards.push({
      ts: nowTs, label: h.label, position: h.radar_position,
      priority: h.priority_name, approaching: h.approaching,
    }));
  }

  // ── Sign events logged ──────────────────────────────────────────
  if (data.sign && data.sign.label) {
    log.signs.push({ ts: nowTs, label: data.sign.label, confidence: data.sign.confidence });
  }

  // ── Emotion (logged every tick — even when neutral, for trend chart) ──
  if (data.emotion) {
    log.emotions.push({
      ts: nowTs,
      label: data.emotion.label,
      confidence: data.emotion.confidence,
      mixed: data.emotion.mixed_signal,
    });
  }

  // ── Scene & Reliability (latest only — these are state, not events) ──
  const sceneText = data.scene_caption || data.scene || '';
  console.debug('[scene]', sceneText ? sceneText : '(empty in response)', 'el.scene?', !!el.scene);
  if (sceneText && el.scene) {
    el.scene.textContent = sceneText;
  }
  // ── Recognised person (independent of whether anyone is speaking) ──
  if (data.identified_person && data.identified_person.name) {
    _lastKnownPerson = {
      name: data.identified_person.name,
      similarity: data.identified_person.similarity,
      ts: Date.now() / 1000,
    };
    if (el.knownPerson) {
      const pct = Math.round((data.identified_person.similarity || 0) * 100);
      el.knownPerson.innerHTML =
        `<strong style="color:#15803D">\u2713 ${escapeHtml(data.identified_person.name)}</strong> ` +
        `<small>(${pct}% match)</small>`;
    }
  } else if (el.knownPerson) {
    // Keep the last recognition visible briefly so it does not flicker when a
    // single frame fails to detect a face.
    const stale = !_lastKnownPerson || (Date.now()/1000 - _lastKnownPerson.ts) > 10;
    if (stale) {
      _lastKnownPerson = null;
      el.knownPerson.textContent = '— nobody enrolled recognised —';
    }
  }

  const rel = data.lip_reliability_obj?.reliability ?? data.lip_reliability;
  el.lipRel.textContent = typeof rel === 'number' ? rel.toFixed(2) : '—';
  el.speaker.textContent = data.face?.speaker_attribution?.label
                        || (data.face_attribution?.speaker_attribution?.position || 'none');

  // ── Global Stats ────────────────────────────────────────────────
  el.latency.textContent = data.latency_ms ? `${data.latency_ms}ms` : '—';
  el.eventCount.textContent = currentState.stats.eventCount;
  el.criticalCount.textContent = currentState.stats.criticalCount;
}

// Small helper used in continuous captions
function escapeHtml(s) {
  if (typeof s !== 'string') return s;
  return s.replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// --- Utilities ---
// Quick client-side RMS computation on an audio Blob.
// Used to skip server round-trips when the user is silent.
async function computeBlobRms(blob) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await blob.arrayBuffer();
    const decoded = await ctx.decodeAudioData(buf.slice(0));
    const data = decoded.getChannelData(0);
    let sum = 0;
    // Sample every 100th frame to keep this fast on long clips
    const step = Math.max(1, Math.floor(data.length / 4000));
    let n = 0;
    for (let i = 0; i < data.length; i += step) {
      sum += data[i] * data[i];
      n++;
    }
    ctx.close();
    return Math.sqrt(sum / Math.max(1, n));
  } catch (e) {
    return 1.0; // assume non-silent on failure
  }
}

function captureAudio(durationMs) {
  return new Promise(resolve => {
    // Choose what to record. Prefer an audio-only stream (small, fast to
    // convert server-side). If the stream has no audio track (mic blocked)
    // or building the audio-only stream fails, fall back to the full stream
    // so we still capture *something* rather than silently sending nothing.
    let recordStream = currentState.stream;
    try {
      const aTracks = currentState.stream.getAudioTracks();
      if (aTracks.length === 0) {
        console.warn('[audio] no audio track in stream — microphone may be blocked or not selected');
      } else {
        // Make sure the track is enabled.
        aTracks.forEach(t => { t.enabled = true; });
        recordStream = new MediaStream(aTracks);
      }
    } catch (e) {
      console.warn('[audio] could not isolate audio tracks, using full stream:', e);
    }

    // Pick a mimeType the browser supports (Chrome: webm/opus, Safari: mp4).
    let mime = '';
    const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'];
    if (window.MediaRecorder && MediaRecorder.isTypeSupported) {
      for (const m of candidates) {
        if (MediaRecorder.isTypeSupported(m)) { mime = m; break; }
      }
    }

    let recorder;
    const recOpts = { audioBitsPerSecond: 128000 };  // higher bitrate = clearer speech for Whisper
    if (mime) recOpts.mimeType = mime;
    try {
      recorder = new MediaRecorder(recordStream, recOpts);
    } catch (e) {
      console.error('[audio] MediaRecorder init failed, retrying on full stream:', e);
      try { recorder = new MediaRecorder(currentState.stream); }
      catch (e2) { console.error('[audio] MediaRecorder unavailable:', e2); return resolve(new Blob([])); }
    }

    const chunks = [];
    recorder.ondataavailable = e => { if (e.data && e.data.size > 0) chunks.push(e.data); };
    recorder.onerror = ev => { console.error('[audio] recorder error:', ev); };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: recorder.mimeType || mime || 'audio/webm' });
      // Visible in the browser console so capture problems are diagnosable.
      console.debug(`[audio] captured ${blob.size} bytes (${recorder.mimeType || mime})`);
      resolve(blob);
    };

    // Use a timeslice so dataavailable fires periodically; some browsers
    // (notably Safari) deliver nothing if you only flush on stop().
    try { recorder.start(500); }
    catch (e) { console.error('[audio] recorder.start failed:', e); return resolve(new Blob([])); }
    setTimeout(() => {
      try { if (recorder.state === 'recording') recorder.stop(); } catch (e) { /* ignore */ }
    }, durationMs);
  });
}

function captureVideoFrame() {
  const video = el.webcamPreview;
  if (!video || video.videoWidth === 0) return null;
  
  const canvas = document.createElement('canvas');
  // Scale down for speed/bandwidth
  canvas.width = 320; 
  canvas.height = 240;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.6);
}

function showWebcamPreview(stream) {
  if (!stream) return;
  if (el.webcamPreview.srcObject !== stream) {
    el.webcamPreview.srcObject = stream;
  }
  el.webcamPreview.classList.remove('hidden');
}

function hideWebcamPreviewIfIdle() {
  if (currentState.isListening || currentState.signPreviewStream) return;
  // Fully release the <video> element so the OS camera indicator turns off.
  // Pause first, then null both srcObject AND src, then call .load() to
  // force the element to forget the stream.
  try {
    el.webcamPreview.pause();
  } catch (e) {}
  el.webcamPreview.srcObject = null;
  el.webcamPreview.removeAttribute('src');
  try { el.webcamPreview.load(); } catch (e) {}
  el.webcamPreview.classList.add('hidden');
}

function releaseSignPreviewStream() {
  if (!currentState.signPreviewStream) return;
  currentState.signPreviewStream.getTracks().forEach(track => track.stop());
  currentState.signPreviewStream = null;
  hideWebcamPreviewIfIdle();
}

async function ensureSignCaptureStream() {
  if (currentState.stream && currentState.stream.getVideoTracks().length > 0) {
    showWebcamPreview(currentState.stream);
    return currentState.stream;
  }
  if (currentState.signPreviewStream && currentState.signPreviewStream.getVideoTracks().length > 0) {
    showWebcamPreview(currentState.signPreviewStream);
    return currentState.signPreviewStream;
  }

  currentState.signPreviewStream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 }
  });
  showWebcamPreview(currentState.signPreviewStream);
  return currentState.signPreviewStream;
}

async function waitForVideoReady() {
  const video = el.webcamPreview;
  if (!video) throw new Error('Camera preview not available.');

  try {
    await video.play();
  } catch (_err) {
    // autoplay can fail before metadata is ready; we wait below.
  }

  if (video.videoWidth > 0 && video.videoHeight > 0) return;

  await new Promise((resolve, reject) => {
    const onReady = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error('Camera preview failed to start.'));
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error('Camera preview timed out.'));
    }, 4000);
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener('loadedmetadata', onReady);
      video.removeEventListener('canplay', onReady);
      video.removeEventListener('error', onError);
    };

    video.addEventListener('loadedmetadata', onReady);
    video.addEventListener('canplay', onReady);
    video.addEventListener('error', onError);
  });
}

/**
 * Speak `text` using the server's ElevenLabs voice when possible, falling
 * back to browser speechSynthesis (robotic) ONLY as a true last resort.
 * The dedicated /sign/recognize and /sign/speak endpoints already return
 * audio_b64 from ElevenLabs, so this function is only a safety net.
 */
async function speakText(text) {
  if (!text) return;
  // Prefer server-side TTS (ElevenLabs → OpenAI → macOS → gTTS)
  try {
    const res = await fetch(`${CONFIG.API_BASE}/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (data.audio_b64) {
      const audio = new Audio('data:audio/mpeg;base64,' + data.audio_b64);
      await audio.play();
      return;
    }
  } catch (e) {
    console.warn('Server TTS failed, falling back to browser TTS:', e);
  }
  // Last-resort browser speechSynthesis (sounds robotic — only fires if
  // ALL server backends failed including ElevenLabs/OpenAI/macOS/gTTS).
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  }
}

/**
 * On page load, ping /tts/status. If ElevenLabs is not configured or has
 * failed (free tier exhausted), show a non-blocking banner so the user
 * understands why the voice may sound robotic.
 */
async function checkTtsHealth() {
  try {
    const r = await fetch(`${CONFIG.API_BASE}/tts/status`);
    const s = await r.json();
    const headerBar = document.querySelector('header');
    if (!s.elevenlabs_configured || s.elevenlabs_disabled) {
      if (document.getElementById('tts-warning')) return;
      const warn = document.createElement('div');
      warn.id = 'tts-warning';
      warn.className = 'tts-warning';
      const reason = !s.elevenlabs_configured
        ? 'No ELEVENLABS_API_KEY set'
        : `ElevenLabs disabled: ${s.elevenlabs_last_error || 'free tier exhausted'}`;
      warn.innerHTML = `⚠️ <strong>Voice may sound robotic</strong> — ${reason}. Add a key in <code>.env</code> for natural neural TTS.`;
      headerBar.insertAdjacentElement('afterend', warn);
    }
  } catch (e) {
    console.debug('TTS status check failed (offline?)', e);
  }
}

// Returns BARE base64 (no "data:...;base64," prefix). Every caller sends this
// value straight to the backend, which base64-decodes it directly. Defined
// once here; do NOT redefine it elsewhere (a second definition previously
// shadowed this one and broke the audio upload).
function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result || '';
      const comma = result.indexOf(',');
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// --- Personal Sounds ---
function setupPersonalControls() {
  el.enrolRecordBtn.addEventListener('click', recordEnrolExample);
  el.enrolSubmitBtn.addEventListener('click', submitEnrolment);
  el.enrolClearBtn.addEventListener('click', clearEnrolment);
}

async function recordEnrolExample() {
  if (currentState.enrolRecording) return;
  
  try {
    // We use the existing stream if possible, or request a new one
    const stream = currentState.stream || await navigator.mediaDevices.getUserMedia({ audio: true });
    
    currentState.enrolRecording = true;
    el.enrolRecordBtn.textContent = '● Recording...';
    el.enrolRecordBtn.classList.add('pulse');
    
    // Record audio-only, with a timeslice (Safari needs it) and a supported
    // mimeType, mirroring the live-capture path.
    const audioStream = stream.getAudioTracks
      ? new MediaStream(stream.getAudioTracks()) : stream;
    let mime = '';
    for (const m of ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(m)) {
        mime = m; break;
      }
    }
    const recorder = mime
      ? new MediaRecorder(audioStream, { mimeType: mime })
      : new MediaRecorder(audioStream);
    const chunks = [];
    recorder.ondataavailable = e => { if (e.data && e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = async () => {
      const blob = new Blob(chunks, { type: recorder.mimeType || mime || 'audio/webm' });
      const b64 = await blobToBase64(blob);
      // blobToBase64 returns bare base64 already; tolerate a data-URL too.
      const bare = (b64 && b64.includes(',')) ? b64.split(',')[1] : b64;
      if (bare) currentState.enrolClips.push(bare);

      currentState.enrolRecording = false;
      el.enrolRecordBtn.textContent = '● Record example (3s)';
      el.enrolRecordBtn.classList.remove('pulse');
      el.enrolCount.textContent = `${currentState.enrolClips.length} examples recorded`;

      // YamNet personalization is much more stable with a few examples.
      if (currentState.enrolClips.length >= 3) {
        el.enrolSubmitBtn.disabled = false;
      }
    };

    recorder.start(500);
    setTimeout(() => {
      try { if (recorder.state === 'recording') recorder.stop(); } catch (e) { /* ignore */ }
    }, CONFIG.RECORDING_DURATION_MS);
  } catch (err) {
    console.error('Error recording example:', err);
    alert('Recording failed. Check microphone permissions.');
  }
}

async function submitEnrolment() {
  const label = el.enrolLabel.value.trim();
  const priority = parseInt(el.enrolPriority.value);
  
  if (!label) {
    alert('Please enter a name for this sound (e.g., "My Doorbell")');
    return;
  }
  
  el.enrolSubmitBtn.disabled = true;
  el.enrolSubmitBtn.textContent = 'Enrolling...';
  
  try {
    const response = await fetch(`${CONFIG.API_BASE}/enrol`, {
      method: 'POST',
      body: JSON.stringify({
        label,
        priority,
        clips_b64: currentState.enrolClips
      }),
      headers: { 'Content-Type': 'application/json' }
    });
    
    const data = await response.json();
    if (data.ok) {
      alert(`Successfully enrolled "${label}"`);
      clearEnrolment();
      fetchEnrolledSounds();
    } else {
      const reasons = {
        audio_conversion_failed: 'Could not read the recorded audio. Check microphone permission and try recording again.',
        no_embeddings: 'Could not extract sound features. Record 3 clear examples with the sound close to the microphone.',
        missing_label_or_clips: 'Add a sound name and record at least 3 examples.',
      };
      alert(`Failed: ${reasons[data.reason] || data.reason}`);
    }
  } catch (err) {
    console.error('Enrolment error:', err);
  } finally {
    el.enrolSubmitBtn.textContent = 'Enrol Sound';
  }
}

function clearEnrolment() {
  currentState.enrolClips = [];
  el.enrolCount.textContent = '0 examples recorded';
  el.enrolSubmitBtn.disabled = true;
  el.enrolLabel.value = '';
}

async function fetchEnrolledSounds() {
  try {
    const response = await fetch(`${CONFIG.API_BASE}/enrolled`);
    const data = await response.json();
    
    el.enrolledList.innerHTML = '';
    if (data.sounds && data.sounds.length > 0) {
      data.sounds.forEach(s => {
        const li = document.createElement('li');
        li.innerHTML = `
          <span><strong>${s.label}</strong> <small>(${getPriorityLabel(s.priority)})</small></span>
          <button class="ghost" onclick="removeSound('${s.label}')" title="Remove">🗑️</button>
        `;
        el.enrolledList.appendChild(li);
      });
    } else {
      el.enrolledList.innerHTML = '<li class="muted">No custom sounds enrolled yet.</li>';
    }
  } catch (err) {
    console.error('Error fetching enrolled sounds:', err);
  }
}

window.removeSound = async function(label) {
  if (!confirm(`Are you sure you want to stop monitoring for "${label}"?`)) return;
  
  try {
    const response = await fetch(`${CONFIG.API_BASE}/enrolled/${encodeURIComponent(label)}`, {
      method: 'DELETE'
    });
    const data = await response.json();
    if (data.ok) fetchEnrolledSounds();
  } catch (err) {
    console.error('Error removing sound:', err);
  }
};

function getPriorityLabel(p) {
  if (p >= 3) return 'Critical';
  if (p >= 2) return 'Important';
  return 'Inform';
}

// --- People (face + voice + name-call alert) ---
// Buffers for in-progress enrolment; cleared after submit.
let _peopleEnrolFaceFrames = [];      // base64 JPEG strings
let _peopleEnrolVoiceClips = [];      // base64 WAV strings

function setupPeopleControls() {
  const nameSaveBtn = document.getElementById('self-name-save');
  const nameInput = document.getElementById('self-name-input');
  if (nameSaveBtn && nameInput) {
    nameSaveBtn.addEventListener('click', saveSelfName);
    nameInput.addEventListener('keydown', e => { if (e.key === 'Enter') saveSelfName(); });
  }

  const faceBtn = document.getElementById('people-enrol-face');
  const voiceBtn = document.getElementById('people-enrol-voice');
  const submit = document.getElementById('people-enrol-submit');
  const clear = document.getElementById('people-enrol-clear');
  if (faceBtn) faceBtn.addEventListener('click', recordPersonFace);
  if (voiceBtn) voiceBtn.addEventListener('click', recordPersonVoice);
  if (submit) submit.addEventListener('click', submitPersonEnrolment);
  if (clear) clear.addEventListener('click', clearPersonEnrolment);
}

async function saveSelfName() {
  const input = document.getElementById('self-name-input');
  const name = (input.value || '').trim();
  if (!name) return;
  try {
    await fetch(`${CONFIG.API_BASE}/me/name`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name})
    });
    document.getElementById('self-name-current').textContent = `Saved: ${name}`;
  } catch (e) { console.error(e); }
}

async function recordPersonFace() {
  // Use the webcam to grab ~9 frames over 3 seconds.
  const btn = document.getElementById('people-enrol-face');
  const counter = document.getElementById('people-face-count');
  btn.disabled = true;
  btn.textContent = '● Recording…';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: true});
    const video = document.createElement('video');
    video.srcObject = stream;
    await video.play();
    await new Promise(r => setTimeout(r, 400));  // settle exposure

    const canvas = document.createElement('canvas');
    canvas.width = 320; canvas.height = 240;
    const ctx = canvas.getContext('2d');
    const frames = [];
    for (let i = 0; i < 9; i++) {
      ctx.drawImage(video, 0, 0, 320, 240);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
      frames.push(dataUrl.split(',')[1]);
      await new Promise(r => setTimeout(r, 300));
    }
    stream.getTracks().forEach(t => t.stop());
    _peopleEnrolFaceFrames = _peopleEnrolFaceFrames.concat(frames);
    counter.textContent = `${_peopleEnrolFaceFrames.length} face frames`;
    refreshPeopleSubmitState();
  } catch (e) {
    alert('Camera failed: ' + e.message);
  }
  btn.disabled = false;
  btn.textContent = '📷 Record face (3 s)';
}

async function recordPersonVoice() {
  const btn = document.getElementById('people-enrol-voice');
  const counter = document.getElementById('people-voice-count');
  btn.disabled = true;
  btn.textContent = '● Recording…';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    const recorder = new MediaRecorder(stream);
    const chunks = [];
    recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    recorder.start();
    await new Promise(r => setTimeout(r, 3000));
    recorder.stop();
    await new Promise(r => recorder.onstop = r);
    stream.getTracks().forEach(t => t.stop());
    const blob = new Blob(chunks, {type: 'audio/webm'});
    const b64 = await blobToBase64(blob);
    _peopleEnrolVoiceClips.push(b64);
    counter.textContent = `${_peopleEnrolVoiceClips.length} voice clips`;
    refreshPeopleSubmitState();
  } catch (e) {
    alert('Microphone failed: ' + e.message);
  }
  btn.disabled = false;
  btn.textContent = '🎙 Record voice (3 s)';
}

function refreshPeopleSubmitState() {
  const submit = document.getElementById('people-enrol-submit');
  const name = (document.getElementById('people-enrol-name').value || '').trim();
  const hasSample = _peopleEnrolFaceFrames.length > 0 || _peopleEnrolVoiceClips.length > 0;
  submit.disabled = !(name && hasSample);
}

document.addEventListener('input', e => {
  if (e.target && e.target.id === 'people-enrol-name') refreshPeopleSubmitState();
});

async function submitPersonEnrolment() {
  const name = (document.getElementById('people-enrol-name').value || '').trim();
  if (!name) return;
  try {
    const r = await fetch(`${CONFIG.API_BASE}/people/enrol`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        name,
        frames_b64: _peopleEnrolFaceFrames,
        audio_clips_b64: _peopleEnrolVoiceClips,
      })
    });
    const data = await r.json();
    if (data.ok) {
      clearPersonEnrolment();
      fetchPeople();
    } else {
      alert('Enrolment failed: ' + (data.error || 'unknown'));
    }
  } catch (e) { alert('Network error: ' + e.message); }
}

function clearPersonEnrolment() {
  _peopleEnrolFaceFrames = [];
  _peopleEnrolVoiceClips = [];
  document.getElementById('people-enrol-name').value = '';
  document.getElementById('people-face-count').textContent = '0 face frames';
  document.getElementById('people-voice-count').textContent = '0 voice clips';
  refreshPeopleSubmitState();
}

async function fetchPeople() {
  try {
    const r = await fetch(`${CONFIG.API_BASE}/people`);
    const data = await r.json();
    const list = document.getElementById('people-list');
    if (!list) return;
    list.innerHTML = '';
    (data.people || []).forEach(p => {
      const li = document.createElement('li');
      const face = p.n_face_examples ? `👤 ${p.n_face_examples}` : '👤 —';
      const voice = p.n_voice_examples ? `🎙 ${p.n_voice_examples}` : '🎙 —';
      li.innerHTML = `
        <span><strong>${escapeHtml(p.name)}</strong> &nbsp; ${face} &nbsp; ${voice}</span>
        <button class="ghost" data-name="${escapeHtml(p.name)}">×</button>
      `;
      li.querySelector('button').addEventListener('click', async () => {
        if (!confirm(`Remove ${p.name}?`)) return;
        await fetch(`${CONFIG.API_BASE}/people/${encodeURIComponent(p.name)}`, {method: 'DELETE'});
        fetchPeople();
      });
      list.appendChild(li);
    });
    if (data.self_name) {
      document.getElementById('self-name-current').textContent = `Saved: ${data.self_name}`;
      document.getElementById('self-name-input').value = data.self_name;
    }
  } catch (e) { console.error(e); }
}

// (blobToBase64 is defined once near the audio-capture helpers above.
//  A duplicate here previously shadowed it and broke audio upload.)

// --- Summary Mode ---
function setupSummaryControls() {
  el.summaryBtn.addEventListener('click', generateSummary);
}

async function generateSummary() {
  el.summaryBtn.disabled = true;
  el.summaryText.textContent = 'Synthesizing recent events...';
  
  try {
    const response = await fetch(`${CONFIG.API_BASE}/summary`, { method: 'POST' });
    const data = await response.json();
    
    el.summaryText.textContent = data.summary || 'No significant events found in the recent window.';
    el.summaryStats.textContent = `Based on ${data.stats?.total_events || 0} environmental events.`;
  } catch (err) {
    console.error('Summary error:', err);
    el.summaryText.textContent = 'Error contacting the AI summarizer.';
  } finally {
    el.summaryBtn.disabled = false;
  }
}

// --- Diary Mode ---
function setupDiaryControls() {
  el.diaryRefreshBtn.addEventListener('click', refreshDiary);
  el.diaryWindow.addEventListener('change', refreshDiary);
  el.diaryMinPriority.addEventListener('change', refreshDiary);
}

async function refreshDiary() {
  const since = el.diaryWindow.value;
  const minPriority = el.diaryMinPriority.value;
  
  try {
    const response = await fetch(`${CONFIG.API_BASE}/diary?since_s=${since}&min_priority=${minPriority}`);
    const data = await response.json();
    
    // 1. Histogram (Frequency)
    el.diaryHistogram.innerHTML = '';
    const histogramEntries = Object.entries(data.histogram || {}).sort((a,b) => b[1] - a[1]);
    
    if (histogramEntries.length > 0) {
      histogramEntries.forEach(([label, count]) => {
        const li = document.createElement('li');
        li.innerHTML = `<span>${label}</span> <strong>${count}</strong>`;
        el.diaryHistogram.appendChild(li);
      });
    } else {
      el.diaryHistogram.innerHTML = '<li class="muted">No data for this period.</li>';
    }
    
    // 2. Recent Events List
    el.diaryEvents.innerHTML = '';
    if (data.events && data.events.length > 0) {
      data.events.slice(0, 100).forEach(e => {
        const li = document.createElement('li');
        li.className = (e.priority_name || 'ambient').toLowerCase();
        const ts = e.ts || e.timestamp || 0;
        const timeStr = new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        li.innerHTML = `
          <div>
            <strong>${e.label}</strong><br>
            <small>${timeStr} • ${e.source}</small>
          </div>
        `;
        el.diaryEvents.appendChild(li);
      });
    } else {
      el.diaryEvents.innerHTML = '<li class="muted">No events logged.</li>';
    }
  } catch (err) {
    console.error('Diary error:', err);
  }
}

// ───────────────────────────────────────────────────────────────────
// Hazard radar / 3D compass / emotion / sign-language panels
// (extends updateLiveUI to render the new fusion outputs)
// ───────────────────────────────────────────────────────────────────

const radarCanvas = document.getElementById('radar-canvas');
const radarCtx = radarCanvas ? radarCanvas.getContext('2d') : null;
const COMPASS_TO_DEG = {
  "front": 0, "front-right": 45, "right": 90, "behind": 180,
  "left": -90, "front-left": -45, "unknown": null,
};
const COMPASS_TO_EMOJI = {
  "front": "⬆️", "front-left": "↖️", "left": "⬅️", "front-right": "↗️",
  "right": "➡️", "behind": "⬇️", "unknown": "❓",
};
const PRIORITY_COLOUR = {
  "CRITICAL": "#D32F2F",
  "IMPORTANT": "#ED7D31",
  "INFORM": "#F9A825",
  "AMBIENT": "#43A047",
};

// Sign-language gloss buffer
let signGlossBuffer = [];

// Tracks the currently-running processTick() Promise so stopListening()
// can await it before building the session report.
let _pendingTick = null;

document.getElementById('sign-speak-btn').addEventListener('click', startSignRecording);
document.getElementById('sign-stop-btn').addEventListener('click', stopSignRecording);
document.getElementById('sign-clear-btn').addEventListener('click', clearSignGloss);

function drawHazardRadar(hazards) {
  if (!radarCtx) return;
  const W = radarCanvas.width;
  const H = radarCanvas.height;
  const cx = W / 2, cy = H / 2;
  const R = Math.min(W, H) / 2 - 8;

  radarCtx.clearRect(0, 0, W, H);
  // Concentric rings
  radarCtx.strokeStyle = "rgba(31,56,100,0.2)";
  radarCtx.lineWidth = 1;
  for (const r of [R / 3, (2 * R) / 3, R]) {
    radarCtx.beginPath();
    radarCtx.arc(cx, cy, r, 0, Math.PI * 2);
    radarCtx.stroke();
  }
  // Cross-hairs
  radarCtx.strokeStyle = "rgba(31,56,100,0.15)";
  radarCtx.beginPath();
  radarCtx.moveTo(cx - R, cy); radarCtx.lineTo(cx + R, cy);
  radarCtx.moveTo(cx, cy - R); radarCtx.lineTo(cx, cy + R);
  radarCtx.stroke();
  // User dot in centre
  radarCtx.fillStyle = "#1F3864";
  radarCtx.beginPath();
  radarCtx.arc(cx, cy, 5, 0, Math.PI * 2);
  radarCtx.fill();

  // Plot hazards
  for (const h of (hazards || [])) {
    const bbox = h.bbox || [0, 0, 0, 0];
    const hcx = (bbox[0] + bbox[2]) / 2;  // 0..1 horizontal
    const hcy = (bbox[1] + bbox[3]) / 2;  // 0..1 vertical
    // Map screen-space to radar polar: x→angle, y→distance (smaller bbox = farther)
    const dx = (hcx - 0.5);
    const dy = (hcy - 0.5);
    const px = cx + dx * R * 1.7;
    const py = cy + dy * R * 1.7;
    const colour = PRIORITY_COLOUR[h.priority_name] || "#888";
    radarCtx.fillStyle = colour;
    const radius = h.approaching ? 12 : 8;
    radarCtx.beginPath();
    radarCtx.arc(px, py, radius, 0, Math.PI * 2);
    radarCtx.fill();
    if (h.approaching) {
      radarCtx.strokeStyle = colour;
      radarCtx.lineWidth = 2;
      radarCtx.beginPath();
      radarCtx.arc(px, py, radius + 5, 0, Math.PI * 2);
      radarCtx.stroke();
    }
    // Label
    radarCtx.fillStyle = "#1a1a1a";
    radarCtx.font = "11px system-ui";
    radarCtx.textAlign = "center";
    radarCtx.fillText(h.label, px, py + radius + 12);
  }
}

function updateCompass(loc) {
  const ptr = document.getElementById('compass-pointer');
  const lbl = document.getElementById('compass-label');
  if (!ptr || !lbl) return;
  const compass = (loc && loc.compass) || "unknown";
  const emoji = COMPASS_TO_EMOJI[compass] || "❓";
  ptr.textContent = emoji;
  if (loc && loc.stereo === false) {
    lbl.textContent = "— stereo audio not detected —";
  } else if (compass === "unknown") {
    lbl.textContent = "no clear direction";
  } else {
    const angle = loc.azimuth_deg != null ? ` (${Math.round(loc.azimuth_deg)}°)` : "";
    const conf = loc.confidence != null ? ` · conf ${Math.round(loc.confidence * 100)}%` : "";
    lbl.textContent = `${compass}${angle}${conf}`;
  }
}

function updateEmotion(emo) {
  if (!emo) return;
  document.getElementById('emo-emoji').textContent = emo.emoji || "😐";
  document.getElementById('emo-label').textContent =
    (emo.label || "neutral") + (emo.confidence != null
      ? ` (${Math.round(emo.confidence * 100)}%)` : "");
  document.getElementById('emo-audio').textContent =
    (emo.audio && emo.audio.label) || "—";
  document.getElementById('emo-visual').textContent =
    (emo.visual && emo.visual.label) || "—";
  document.getElementById('emo-mixed').textContent =
    emo.mixed_signal ? "⚠️ Mixed signal — voice and face disagree" : "";
  // Inline tag
  document.getElementById('emotion-tag').textContent =
    `${emo.emoji || ""} ${emo.label || ""}`;
}

// Called from the live-mode tick (one sign per tick from /process).
// Signs accumulate in signGlossBuffer across ticks.  When the user
// clicks "Start signing" → "Stop signing", all captured frames are
// analysed at once by the backend and the whole sentence is returned.
// Here we just update the running gloss so the user can see progress.
let _signBadgeTimer = null;
function appendSignEvent(sign) {
  if (!sign || !sign.label) return;
  const label = sign.label.replace(/^sign:\s*/, '');
  const conf = Math.round((sign.confidence || 0) * 100);

  // ── Real-time floating badge (UX) ─────────────────────────────────
  // Pops up over the webcam preview as soon as the per-tick endpoint
  // returns a sign — gives the user immediate confirmation.
  const badge = document.getElementById('sign-live-badge');
  if (badge) {
    badge.innerHTML =
      `🤲 ${escapeHtml(label)}<span class="sign-conf">${conf}%</span>`;
    badge.classList.remove('hidden');
    if (_signBadgeTimer) clearTimeout(_signBadgeTimer);
    _signBadgeTimer = setTimeout(() => badge.classList.add('hidden'), 2500);
  }

  // Accumulate in the gloss buffer (used by the dedicated recording path)
  signGlossBuffer.push(label);
  document.getElementById('sign-gloss').textContent =
    '🪶 Live signs: ' + signGlossBuffer.join(' → ');
  document.getElementById('sign-speak-btn').disabled = false;

  // Keep the event list updated — last 5 live-mode ticks only
  const ul = document.getElementById('sign-events');
  if (!ul) return;
  const li = document.createElement('li');
  li.className = 'important';
  li.innerHTML = `🤲 <span style="flex:1">${escapeHtml(label)}</span><small>${conf}%</small>`;
  ul.prepend(li);
  while (ul.children.length > 5) ul.removeChild(ul.lastChild);
}

async function speakSignGloss() {
  const gloss = signGlossBuffer.join(' ');
  if (!gloss) return;
  const btn = document.getElementById('sign-speak-btn');
  btn.disabled = true;
  btn.textContent = "Speaking...";
  try {
    const res = await fetch(`${CONFIG.API_BASE}/sign/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gloss, speak_audio: true }),
    });
    const data = await res.json();
    document.getElementById('sign-spoken').textContent = data.text || "";
    // Prefer ElevenLabs / OpenAI audio if the server returned it;
    // fall back to the browser's local speech synth otherwise.
    if (data.audio_b64) {
      try {
        const audio = new Audio('data:audio/mpeg;base64,' + data.audio_b64);
        await audio.play();
      } catch (e) {
        console.warn('Server TTS playback failed, using browser TTS', e);
        speakText(data.text || "");
      }
    } else {
      speakText(data.text || "");
    }
  } catch (e) {
    console.warn(e);
  } finally {
    btn.disabled = false;
    btn.textContent = "▶ Start signing";
  }
}

function setSignRecordingUi(isRecording) {
  const startBtn = document.getElementById('sign-speak-btn');
  const stopBtn = document.getElementById('sign-stop-btn');
  const clearBtn = document.getElementById('sign-clear-btn');

  startBtn.classList.toggle('hidden', isRecording);
  stopBtn.classList.toggle('hidden', !isRecording);
  clearBtn.disabled = isRecording;
}

function captureAndStoreSignFrame() {
  const frame = captureVideoFrame();
  if (frame) {
    currentState.signCapturedFrames.push(frame.split(',')[1]);
  }
}

function clearSignCaptureTimer() {
  if (!currentState.signCaptureTimer) return;
  clearInterval(currentState.signCaptureTimer);
  currentState.signCaptureTimer = null;
}

function cancelSignRecording(message = '') {
  clearSignCaptureTimer();
  currentState.signRecording = false;
  currentState.signCapturedFrames = [];
  setSignRecordingUi(false);
  if (message) {
    document.getElementById('sign-spoken').textContent = message;
  }
  if (!currentState.isListening) {
    releaseSignPreviewStream();
  }
}

async function startSignRecording() {
  if (currentState.signRecording) return;

  const spokenEl = document.getElementById('sign-spoken');
  currentState.signRecording = true;
  currentState.signCapturedFrames = [];
  setSignRecordingUi(true);
  spokenEl.textContent = 'Turning on camera and live monitoring...';
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }

  try {
    if (!currentState.isListening) {
      const liveStream = await startListening();
      if (!liveStream) {
        throw new Error('Live monitoring could not start.');
      }
    } else {
      await ensureSignCaptureStream();
    }
    await waitForVideoReady();
    spokenEl.textContent = 'Camera on. Sign now, then click Stop signing.';
    captureAndStoreSignFrame();
    currentState.signCaptureTimer = setInterval(
      captureAndStoreSignFrame,
      CONFIG.SIGN_RECORDING_FRAME_INTERVAL_MS
    );
  } catch (err) {
    console.error('Error starting sign recording:', err);
    cancelSignRecording('Could not start camera. Check camera permissions and try again.');
  }
}

async function stopSignRecording() {
  if (!currentState.signRecording) return;

  const spokenEl = document.getElementById('sign-spoken');
  clearSignCaptureTimer();
  currentState.signRecording = false;
  setSignRecordingUi(false);

  const frames = currentState.signCapturedFrames.slice();
  currentState.signCapturedFrames = [];
  if (frames.length === 0) {
    spokenEl.textContent = 'No video frames captured. Keep your hands visible and try again.';
    if (!currentState.isListening) releaseSignPreviewStream();
    return;
  }

  spokenEl.textContent = `Analysing ${frames.length} frames…`;

  // Subsample if the recording is very long — preserve full temporal
  // coverage by taking evenly-spaced frames rather than truncating.
  const MAX_SIGN_FRAMES = 200;
  let sendFrames = frames;
  if (frames.length > MAX_SIGN_FRAMES) {
    const step = frames.length / MAX_SIGN_FRAMES;
    sendFrames = Array.from(
      { length: MAX_SIGN_FRAMES },
      (_, i) => frames[Math.floor(i * step)]
    );
  }

  try {
    const res = await fetch(`${CONFIG.API_BASE}/sign/recognize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames_b64: sendFrames, speak_audio: true }),
    });
    if (!res.ok) throw new Error(`Sign recognition failed: ${res.status}`);

    const data = await res.json();
    const sequence = data.sequence || [];

    if (sequence.length > 0) {
      // ── ONE polished sentence — the whole clip as a unit ──────────
      const polished = data.text || data.gloss || sequence.map(s => s.label).join(' ');

      // Clear previous items and show the sentence as a single entry
      const ul = document.getElementById('sign-events');
      ul.innerHTML = '';
      const li = document.createElement('li');
      li.className = 'important';
      const conf = Math.round((data.confidence || 0) * 100);
      li.innerHTML = `🤲 <span style="flex:1;font-size:1.05em">${escapeHtml(polished)}</span>`
                   + `<small>${conf}%</small>`;
      ul.appendChild(li);

      // Show the raw gloss as compact secondary info
      const glossStr = data.gloss || sequence.map(s => s.label).join(' ');
      const n = sequence.length;
      document.getElementById('sign-gloss').textContent =
        `🪶 ${n} sign${n !== 1 ? 's' : ''} detected: ${glossStr}`;

      // Show sentence in the spoken area (large, no confusing notes)
      spokenEl.textContent = polished;

      // Store for session log + speak-again
      signGlossBuffer = [polished];
      document.getElementById('sign-speak-btn').disabled = false;
      currentState.sessionLog.signs.push({
        ts: Date.now() / 1000,
        label: polished,
        gloss: glossStr,
        n_signs: n,
        confidence: data.confidence,
      });

      // Play spoken audio (ElevenLabs from server, then browser fallback)
      if (data.audio_b64) {
        try {
          const audio = new Audio('data:audio/mpeg;base64,' + data.audio_b64);
          await audio.play();
        } catch (e) {
          console.warn('Server TTS failed, using browser TTS:', e);
          speakText(polished);
        }
      } else {
        speakText(polished);
      }

    } else if (data.reason === 'no_hands_visible') {
      const d = data.diagnostics || {};
      const rxd = d.frames_received || sendFrames.length;
      spokenEl.textContent =
        `Hand not detected (0/${rxd} frames). `
        + 'Keep your hand 30–60 cm from the camera, well-lit, palm facing the lens.';

    } else {
      const d = data.diagnostics || {};
      const seen = d.frames_with_hand_pretrained ?? d.frames_with_hand ?? '?';
      const rxd  = d.frames_received || sendFrames.length;
      spokenEl.textContent =
        `No clear sign in ${seen}/${rxd} frames. `
        + 'Hold each shape for ~1 second: open palm → "hello", '
        + 'thumb up → "ok", peace sign → "peace", closed fist → "yes", pointing → "you".';
    }
  } catch (err) {
    console.error('Error stopping sign recording:', err);
    spokenEl.textContent = 'Could not process the sign recording. Try again.';
  } finally {
    if (!currentState.isListening) releaseSignPreviewStream();
  }
}

function clearSignGloss() {
  if (currentState.signRecording) return;
  signGlossBuffer = [];
  document.getElementById('sign-gloss').textContent = "";
  document.getElementById('sign-spoken').textContent = "";
  document.getElementById('sign-events').innerHTML = "";
  setSignRecordingUi(false);
}

// Patch updateLiveUI to also render the new panels
const _origUpdateLiveUI = updateLiveUI;
updateLiveUI = function(data) {
  _origUpdateLiveUI(data);
  if (data.hazards) drawHazardRadar(data.hazards);
  if (data.name_call) updateNameAlerts(data.name_call);
  if (data.emotion) updateEmotion(data.emotion);
  if (data.sign) appendSignEvent(data.sign);
};

// Start everything
init();
