"""[Final report: Section 5.3, Figure 5.3]

Why the first live-caption tick of the latency benchmark took far longer than the
rest, and what the application's start-up warm-up (server._prewarm) is worth. The
benchmark imported the server without that warm-up at all.

Run first against a warm-up that sent half a second of silence and grey frames
(eval_results/first_tick_probe_silence.json): the models were loaded, but the
first real tick still paid 5.1 s, because AST met a chunk length it had not seen,
the voice-emotion model returns early on silence and had never run, and the
captioner had only described a flat grey frame. The warm-up now sends 2.8 s of
audio and a drawn scene, the same shapes a real tick brings, and this script was
run again to measure what that is worth.

This script starts the server state as the application does, runs _prewarm, then
sends five live-caption ticks built exactly as latency_on_recorded_media.py builds
them (macOS `say` speech cut to 2.8 s and Opus-coded, three WLASL frames at 320x240),
timing every stage of each tick. Profiles are the real ones, as in the benchmark;
the diary goes to a temporary file. Writes eval_results/first_tick_probe.json.
"""
from pathlib import Path
import base64, json, os, subprocess, sys, tempfile, time

BACKEND = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
TMP = tempfile.mkdtemp(prefix="first_tick_")
os.environ["ACCESSIBILITY_DB_PATH"] = os.path.join(TMP, "diary.db")


def sh(*a):
    subprocess.run(a, check=True, capture_output=True)


SENT = "The meeting has moved to three o'clock this afternoon, so please bring the report."
sh("say", "-o", f"{TMP}/speech.aiff", SENT)
sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{TMP}/speech.aiff", "-ar", "16000", "-ac", "1", "-t", "2.8", f"{TMP}/speech_2p8.wav")
sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{TMP}/speech_2p8.wav", "-c:a", "libopus", "-b:a", "32k", f"{TMP}/speech_2p8.webm")
frames = []
for i, ts in enumerate(("0.6", "1.2", "1.8")):
    out = f"{TMP}/frame_{i}.jpg"
    sh("ffmpeg", "-y", "-loglevel", "error", "-ss", ts, "-i", "data/WLASL-master/videos/69241.mp4",
       "-frames:v", "1", "-vf", "scale=320:240", "-q:v", "5", out)
    frames.append(base64.b64encode(open(out, "rb").read()).decode())
speech_webm = open(f"{TMP}/speech_2p8.webm", "rb").read()

import server  # noqa: E402
from modules.people import get_people  # noqa: E402

t = time.time()
server._prewarm()
warm_s = time.time() - t
print(f"application warm-up took {warm_s:.1f}s", flush=True)

STAGES = {}


def timed(obj, name, label):
    fn = getattr(obj, name)

    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            STAGES[label] = STAGES.get(label, 0.0) + time.perf_counter() - t0
    setattr(obj, name, wrapper)


for obj, name, label in (
        (server.get_audio_scene(), "analyse", "AST"), (server.get_personal(), "match_current", "personaliser"),
        (server.get_scene(), "describe_jpeg", "BLIP"), (server.get_face(), "analyse_frames", "face tracking"),
        (server.get_hazard(), "detect_to_events", "YOLO"), (server.get_emo_audio(), "analyse", "voice emotion"),
        (server.get_emo_visual(), "analyse_frames", "facial emotion"), (server.get_slr(), "push_frames", "sign landmarks"),
        (server.get_slr(), "classify_to_event", "sign classification"), (server.get_lip(), "reliability", "lip reliability"),
        (server.get_fuser(), "fuse", "fusion"), (get_people(), "identify_face", "face identification"),
        (server.get_diary(), "log_many", "diary"), (server.get_summer(), "add_many", "summary buffer")):
    timed(obj, name, label)
_ffmpeg = server._ffmpeg_convert


def ffmpeg_timed(raw):
    t0 = time.perf_counter()
    try:
        return _ffmpeg(raw)
    finally:
        STAGES["ffmpeg"] = STAGES.get("ffmpeg", 0.0) + time.perf_counter() - t0


server._ffmpeg_convert = ffmpeg_timed

client = server.app.test_client()
ticks = []
for i in range(5):
    STAGES.clear()
    t0 = time.perf_counter()
    r = client.post("/process", data=json.dumps({"audio_b64": base64.b64encode(speech_webm).decode(),
                                                  "frames_b64": frames, "stream_captions": True}),
                    content_type="application/json")
    wall = time.perf_counter() - t0
    body = r.get_json() or {}
    ticks.append(dict(tick=i + 1, wall_s=round(wall, 3),
                      people_enrolled=len(get_people().profile.get("people", {})),
                      stages_s={k: round(v, 3) for k, v in sorted(STAGES.items(), key=lambda kv: -kv[1])},
                      hazards=len(body.get("hazards") or []), emotion=(body.get("emotion") or {}).get("label")))
    print(ticks[-1], flush=True)

json.dump(dict(method=__doc__, warm_up_s=round(warm_s, 1), ticks=ticks),
          open(os.path.join(BACKEND, "eval_results", "first_tick_probe.json"), "w"), indent=1)
print("DONE probe", flush=True)
