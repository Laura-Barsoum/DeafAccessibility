"""[Final report: Section 5.4, Table 5.3]

Fusion on scripted scenes whose real events are known in advance.

Each tick is built from ESC-50 clips and synthetic speech from macOS `say`;
nothing is recorded. Sound and speech are mixed at equal loudness, cut to the
2.8 s tick, encoded to WebM/Opus as the browser sends them, and posted with
three plain grey frames to the real /process handler with live captions off,
so speech, generic sounds, the enrolled personal sound and name and keyword
alerts all reach the fusion step. One personal sound ("Kitchen alarm") is
enrolled from three clock-alarm clips; the test scenes use other clips.

The urgency of each real event is fixed here, before running, following the
priority bands of Section 3.6:
  3  siren, crying baby, glass breaking, speech with a safety word, the user's name
  2  door knock, dog, the enrolled personal sound, ordinary speech
  0  rain, keyboard typing
Mixed scenes pair events of different urgency, so the correct first headline
is never a tie.

Every detection (before and after merge_events) and every headline is
attributed to the real event it came from: speech-source events and speech-like
AST labels to the speech, the personal label and other AST labels to the sound,
anything else to nothing. Per tick this gives duplicate headline slots, whether
the first headline is the most urgent real event, missed events, and how many
excess detections of one event merge_events removed.

Profiles and the diary go to a temporary directory; the language-model
notification is disabled because it runs after ranking. Model weights load
from the local cache, and the script refuses to score unless AST and YamNet
are the loaded backends: in a trial run a failed network request during
loading silently replaced AST with the YamNet classifier. The scene set runs
three times, because Whisper's temperature fallback makes its transcripts of
non-speech sounds vary from run to run. Writes eval_results/fusion_eval.json.
"""
from pathlib import Path
import base64, csv, json, logging, os, re, subprocess, sys, tempfile

import numpy as np
import soundfile as sf

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
MEDIA = os.path.join(SP, "media", "fusion")
os.makedirs(MEDIA, exist_ok=True)
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

TMP = tempfile.mkdtemp(prefix="fusion_eval_")
os.environ["ACCESSIBILITY_DB_PATH"] = os.path.join(TMP, "diary.db")  # set before .env, which does not override it
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # cached weights only; see the docstring
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(BACKEND, "..", ".env"))

from modules import personalizer as pm, people as pp  # noqa: E402
pm.PROFILE_DIR, pm.PROFILE_PATH = TMP, os.path.join(TMP, "personal_sounds.json")
pp.PROFILE_DIR, pp.PROFILE_PATH = TMP, os.path.join(TMP, "people.json")

import server  # noqa: E402
from modules import fusion as fusion_mod  # noqa: E402
from modules.events import Priority  # noqa: E402

SELF_NAME = "Laura"
PERSONAL = "Kitchen alarm"
SR = 16000
TICK = int(2.8 * SR)
REPEATS = 3

# ── isolation: temporary profiles and diary, no language-model call ──────
reg = pp.get_people()
reg.profile = {"people": {}, "self_name": SELF_NAME, "keywords": list(pp.DEFAULT_KEYWORDS)}
P = server.get_personal()
P.profile = {"sounds": {}}
server._diary = server.diary.Diary(os.environ["ACCESSIBILITY_DB_PATH"])


class _NoLLM:
    def compose_notification(self, *a, **k):
        return ""


server.get_llm = lambda: _NoLLM()

# ── record what fusion sees before and after merging ─────────────────────
CAPTURE = {}
_merge = fusion_mod.merge_events


def _recording_merge(events, window_s=1.5):
    CAPTURE["before"] = [(e.source, e.label, int(e.priority)) for e in events]
    out = _merge(events, window_s=window_s)
    CAPTURE["after"] = [(e.source, e.label, int(e.priority)) for e in out]
    return out


fusion_mod.merge_events = _recording_merge


class _Warnings(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


WARN = _Warnings()
logging.getLogger().addHandler(WARN)


# ── media ────────────────────────────────────────────────────────────────
def sh(*a):
    subprocess.run(a, check=True, capture_output=True)


def load16(src):
    out = os.path.join(TMP, "decode.wav")
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", str(SR), "-ac", "1", out)
    x, _ = sf.read(out, dtype="float32")
    return x


def fit(x, n):
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def loudest(x, n):
    """The n-sample window with the most energy, so short events are not cut."""
    if len(x) <= n:
        return fit(x, n)
    starts = range(0, len(x) - n + 1, SR // 10)
    i = max(starts, key=lambda s: float(np.mean(x[s:s + n] ** 2)))
    return x[i:i + n]


def level(x, target=0.05):
    """Scale to a common RMS over the non-silent samples."""
    active = x[np.abs(x) > 0.01]
    rms = float(np.sqrt(np.mean(active ** 2))) if active.size else 0.0
    return x * (target / rms) if rms > 0 else x


def webm(x, name):
    peak = float(np.max(np.abs(x))) or 1.0
    if peak > 0.98:
        x = x * (0.98 / peak)
    wav, out = os.path.join(MEDIA, name + ".wav"), os.path.join(MEDIA, name + ".webm")
    sf.write(wav, x, SR, subtype="PCM_16")
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-c:a", "libopus", "-b:a", "32k", out)
    return open(out, "rb").read()


ESC = os.path.join(BACKEND, "data", "ESC-50")
BY = {}
for row in csv.DictReader(open(os.path.join(ESC, "meta", "esc50.csv"))):
    BY.setdefault(row["category"], []).append(row["filename"])
for c in BY:
    BY[c].sort()


def clip(cat, i, seconds=2.8):
    return loudest(load16(os.path.join(ESC, "audio", BY[cat][i])), int(seconds * SR))


_speech_cache = {}


def speech(key):
    if key not in _speech_cache:
        aiff = os.path.join(MEDIA, key + ".aiff")
        sh("say", "-r", "190", "-o", aiff, SENTENCES[key][0])
        _speech_cache[key] = fit(load16(aiff), TICK)
    return _speech_cache[key]


grey = os.path.join(MEDIA, "grey.jpg")
sh("ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x7f7f7f:s=320x240", "-frames:v", "1", grey)
FRAMES = [base64.b64encode(open(grey, "rb").read()).decode()] * 3

# ── ground truth, fixed before running ───────────────────────────────────
SENTENCES = {
    "plain_a": ("Could you pass me the salt, please?", 2),
    "plain_b": ("The meeting has moved to three o'clock.", 2),
    "safety": ("Help, there is a fire in the kitchen!", 3),
    "name": (f"{SELF_NAME}, can you come here for a moment?", 3),
}
SOUNDS = {"siren": 3, "crying_baby": 3, "glass_breaking": 3, "door_wood_knock": 2, "dog": 2,
          "clock_alarm": 2, "rain": 0, "keyboard_typing": 0}

SCENES = []
for cat in ("siren", "crying_baby", "glass_breaking", "door_wood_knock", "dog"):
    SCENES += [("sound only", None, (cat, i)) for i in (0, 1, 2)]
SCENES += [("sound only", None, ("clock_alarm", i)) for i in (3, 4, 5)]  # clips 0-2 are enrolled
SCENES += [("background only", None, (cat, i)) for cat in ("rain", "keyboard_typing") for i in (0, 1)]
SCENES += [("speech only", key, None) for key in ("plain_a", "plain_b", "safety", "name")]
SCENES += [
    ("speech and sound", "plain_a", ("siren", 3)),
    ("speech and sound", "plain_a", ("crying_baby", 3)),
    ("speech and sound", "plain_b", ("glass_breaking", 3)),
    ("speech and sound", "plain_b", ("rain", 2)),
    ("speech and sound", "plain_a", ("keyboard_typing", 2)),
    ("speech and sound", "safety", ("dog", 3)),
    ("speech and sound", "safety", ("rain", 3)),
    ("speech and sound", "name", ("door_wood_knock", 3)),
    ("speech and sound", "name", ("keyboard_typing", 3)),
]

SPEECH_LIKE = re.compile(r"speech|speak|narrat|conversation|talk|whisper|\bmale\b|female", re.I)


def attribute(source, label, has_speech, sound_cat):
    """Which real event a detection came from: 'speech', 'sound' or 'nothing'."""
    if source == "speech":
        return "speech" if has_speech else "nothing"
    if source == "sound":
        if label.startswith(PERSONAL):
            return "sound" if sound_cat == "clock_alarm" else "nothing"
        if has_speech and SPEECH_LIKE.search(label):
            return "speech"
        if sound_cat:
            return "sound"
        return "speech" if has_speech else "nothing"
    return "nothing"  # vision or sign: the grey frames hold no scripted visual event


# ── enrol the personal sound through the browser codec ───────────────────
enrol_clips = [server._ffmpeg_convert(webm(level(clip("clock_alarm", i, 3.0)), f"enrol_{i}")) for i in range(3)]
ENROLMENT = P.enrol(PERSONAL, Priority.IMPORTANT, enrol_clips)
print("enrolled:", ENROLMENT, flush=True)

client = server.app.test_client()


def post(path, payload):
    return client.post(path, data=json.dumps(payload), content_type="application/json").get_json()


def tick(audio):
    CAPTURE.clear()
    return post("/process", {"audio_b64": base64.b64encode(audio).decode(), "frames_b64": FRAMES,
                             "stream_captions": False})


# warm every model so first loads do not run into the per-model timeouts
warm = webm(level(speech("plain_a")) + level(clip("dog", 4)), "warm")
post("/reset", {})
for _ in range(3):
    tick(warm)
server.process._scene_tick_counter = 0
post("/reset", {})

BACKENDS = dict(sound_classifier=getattr(server.get_audio_scene(), "_backend", None),
                personal_embedding="fallback" if P._yamnet in (None, "placeholder") else "yamnet")
print("backends:", BACKENDS, flush=True)
if BACKENDS != {"sound_classifier": "ast", "personal_embedding": "yamnet"}:
    sys.exit(f"refusing to score: the shipped models are not loaded ({BACKENDS})")


def excess(events, has_speech, cat):
    counts = {}
    for src, lab, _p in events:
        a = attribute(src, lab, has_speech, cat)
        if a != "nothing":
            counts[a] = counts.get(a, 0) + 1
    return sum(c - 1 for c in counts.values())


records = []
for rep, (k, (kind, sent, snd)) in [(r, item) for r in range(REPEATS) for item in enumerate(SCENES)]:
    parts, real, cat, clip_name = [], {}, None, None
    if sent:
        parts.append(level(speech(sent)))
        real["speech"] = SENTENCES[sent][1]
    if snd:
        cat, i = snd
        clip_name = BY[cat][i]
        parts.append(level(clip(cat, i)))
        real["sound"] = SOUNDS[cat]
    WARN.msgs.clear()
    d = tick(webm(sum(parts), f"scene_{k:02d}"))
    has_speech = sent is not None

    heads = [(h["source"], h["label"], h["priority_name"]) for h in d.get("headlines", [])]
    h_att = [attribute(s, l, has_speech, cat) for s, l, _p in heads]
    seen, dup, unwarranted = [], 0, 0
    for a in h_att:
        if real.get(a, 0) < 2:
            unwarranted += 1
        elif a in seen:
            dup += 1
        else:
            seen.append(a)
    actionable = {a: r for a, r in real.items() if r >= 2}
    top = max(actionable, key=actionable.get) if actionable else None
    before, after = CAPTURE.get("before", []), CAPTURE.get("after", [])
    first_ok = None if top is None else bool(h_att) and h_att[0] == top
    missed = sorted(a for a in actionable if a not in seen)
    rec = dict(
        repeat=rep, scene=k, kind=kind, speech=SENTENCES[sent][0] if sent else None,
        sound=f"{cat} {clip_name}" if snd else None, real=real,
        transcript=(d.get("stt") or {}).get("text", ""),
        headlines=heads, headline_events=h_att,
        duplicate_slots=dup, unwarranted_slots=unwarranted,
        unwarranted_transcript_slots=sum(1 for (s, _l, _p), a in zip(heads, h_att)
                                         if s == "speech" and real.get(a, 0) < 2),
        all_slots_one_event=len(h_att) == 3 and len(set(h_att)) == 1 and h_att[0] != "nothing",
        correct_first=first_ok,
        # a wrong first headline either outranked a headlined urgent event or the event never became a headline
        first_wrong_reason=None if first_ok in (None, True) else ("outranked" if top in h_att else "not headlined"),
        missed=missed,
        # an urgent event is crowded out only if fusion received an urgent detection of it
        missed_with_urgent_detection=[a for a in missed if any(
            attribute(s, l, has_speech, cat) == a and p >= 2 for s, l, p in before)],
        personal_matches=sum(1 for _s, l, _p in before if l.startswith(PERSONAL)),
        before_merge=before, after_merge=after,
        excess_before=excess(before, has_speech, cat), excess_after=excess(after, has_speech, cat),
        identical_label_repeats=len(before) - len({(s, l) for s, l, _p in before}),
        warnings=[m for m in WARN.msgs if "timed out" in m or "failed" in m],
        latency_ms=d.get("latency_ms"),
    )
    records.append(rec)
    print(f"r{rep} {k:02d} {kind:17s} speech={sent} sound={cat} -> {heads} first_ok={rec['correct_first']} "
          f"dup={dup} unwarranted={unwarranted} excess {rec['excess_before']}->{rec['excess_after']}", flush=True)


def frac(a, b):
    return {"n": a, "of": b, "rate": round(a / b, 3) if b else None}


slots = sum(len(r["headlines"]) for r in records)
eligible = [r for r in records if r["correct_first"] is not None]
background = [r for r in records if r["kind"] == "background only"]
summary = dict(
    ticks=len(records),
    headline_slots=slots,
    duplicate_slots=frac(sum(r["duplicate_slots"] for r in records), slots),
    unwarranted_slots=frac(sum(r["unwarranted_slots"] for r in records), slots),
    ticks_with_duplicate_headline=frac(sum(1 for r in records if r["duplicate_slots"]), len(records)),
    correct_first=frac(sum(1 for r in eligible if r["correct_first"]), len(eligible)),
    correct_first_by_kind={
        kind: frac(sum(1 for r in eligible if r["kind"] == kind and r["correct_first"]),
                   sum(1 for r in eligible if r["kind"] == kind))
        for kind in sorted({r["kind"] for r in eligible})},
    missed_events=frac(sum(len(r["missed"]) for r in records),
                       sum(sum(1 for v in r["real"].values() if v >= 2) for r in records)),
    background_ticks_with_headline=frac(sum(1 for r in background if r["headlines"]), len(background)),
    excess_detections_before_merge=sum(r["excess_before"] for r in records),
    removed_by_merge=sum(r["excess_before"] - r["excess_after"] for r in records),
    identical_label_repeats=sum(r["identical_label_repeats"] for r in records),
    ticks_with_model_warnings=sum(1 for r in records if r["warnings"]),
    unwarranted_slots_from_transcripts=sum(r["unwarranted_transcript_slots"] for r in records),
    non_speech_ticks_with_transcript=frac(sum(1 for r in records if not r["speech"] and r["transcript"].strip()),
                                          sum(1 for r in records if not r["speech"])),
    ticks_all_slots_one_event=sum(1 for r in records if r["all_slots_one_event"]),
    first_wrong_outranked=sum(1 for r in records if r["first_wrong_reason"] == "outranked"),
    first_wrong_not_headlined=sum(1 for r in records if r["first_wrong_reason"] == "not headlined"),
    missed_despite_urgent_detection=sum(len(r["missed_with_urgent_detection"]) for r in records),
    personal_matches=sum(r["personal_matches"] for r in records),
)
summary["scenes"], summary["repeats"] = len(SCENES), REPEATS
summary["per_repeat"] = []
for rep in range(REPEATS):
    rr = [r for r in records if r["repeat"] == rep]
    rslots = sum(len(r["headlines"]) for r in rr)
    summary["per_repeat"].append(dict(
        correct_first=frac(sum(1 for r in rr if r["correct_first"]), sum(1 for r in rr if r["correct_first"] is not None)),
        duplicate_slots=frac(sum(r["duplicate_slots"] for r in rr), rslots),
        unwarranted_slots=frac(sum(r["unwarranted_slots"] for r in rr), rslots),
        missed_events=sum(len(r["missed"]) for r in rr),
        removed_by_merge=sum(r["excess_before"] - r["excess_after"] for r in rr),
    ))
summary["merge_hit_rate"] = (round(summary["removed_by_merge"] / summary["excess_detections_before_merge"], 3)
                             if summary["excess_detections_before_merge"] else None)
json.dump(dict(method=__doc__, backends=BACKENDS, rules=dict(sentences=SENTENCES, sounds=SOUNDS, self_name=SELF_NAME,
                                          personal_sound=PERSONAL),
               enrolment=ENROLMENT, summary=summary, ticks=records),
          open(os.path.join(SP, "fusion_eval.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
print("DONE fusion", flush=True)
sys.stderr.flush()
os._exit(0)  # skip library teardown, which prints errors at interpreter exit after the results are written
