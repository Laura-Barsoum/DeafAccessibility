"""[Final report: Section 5.3, Figures 3.2 and 5.3]

Per-model and end-to-end latency on realistic media. Run with no other load.

Inputs (all generated or taken from local data, nothing recorded):
  - speech: macOS `say` sentence, 16 kHz, cut to one 2.8 s tick, encoded to
    WebM/Opus exactly as the browser sends it
  - alarm:  ESC-50 clock_alarm clip, 2.8 s, WebM/Opus
  - frames: 3 JPEG frames (320x240) of a person from a local WLASL clip
Measures warm component latencies (median of 10 after 2 warm-ups), Whisper
size sweep, /stt/stream latency, and full /process ticks via the Flask test
client (the real handler, real models)."""
from pathlib import Path
import base64, csv, json, os, statistics, subprocess, sys, time

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
os.makedirs(SP, exist_ok=True)
MEDIA = os.path.join(SP, "media")
os.makedirs(MEDIA, exist_ok=True)
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(BACKEND, "..", ".env"))


def sh(*a):
    subprocess.run(a, check=True, capture_output=True)


# ── media ──────────────────────────────────────────────────────────────
SENT = "The meeting has moved to three o'clock this afternoon, so please bring the report."
sh("say", "-o", f"{MEDIA}/speech.aiff", SENT)
sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{MEDIA}/speech.aiff", "-ar", "16000", "-ac", "1", f"{MEDIA}/speech_full.wav")
sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{MEDIA}/speech_full.wav", "-t", "2.8", f"{MEDIA}/speech_2p8.wav")
sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{MEDIA}/speech_full.wav", "-t", "1.0", f"{MEDIA}/speech_1p0.wav")
sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{MEDIA}/speech_2p8.wav", "-c:a", "libopus", "-b:a", "32k", f"{MEDIA}/speech_2p8.webm")
meta = list(csv.DictReader(open("data/ESC-50/meta/esc50.csv")))
alarm = sorted(r["filename"] for r in meta if r["category"] == "clock_alarm")[0]
sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"data/ESC-50/audio/{alarm}", "-ar", "16000", "-ac", "1", "-t", "2.8", f"{MEDIA}/alarm_2p8.wav")
sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{MEDIA}/alarm_2p8.wav", "-c:a", "libopus", "-b:a", "32k", f"{MEDIA}/alarm_2p8.webm")
frames = []
for i, ts in enumerate(("0.6", "1.2", "1.8")):
    out = f"{MEDIA}/frame_{i}.jpg"
    sh("ffmpeg", "-y", "-loglevel", "error", "-ss", ts, "-i", "data/WLASL-master/videos/69241.mp4",
       "-frames:v", "1", "-vf", "scale=320:240", "-q:v", "5", out)
    frames.append(base64.b64encode(open(out, "rb").read()).decode())


def rb(p):
    return open(p, "rb").read()


speech_webm, alarm_webm = rb(f"{MEDIA}/speech_2p8.webm"), rb(f"{MEDIA}/alarm_2p8.webm")
speech_wav, speech1_wav = rb(f"{MEDIA}/speech_2p8.wav"), rb(f"{MEDIA}/speech_1p0.wav")
alarm_wav = rb(f"{MEDIA}/alarm_2p8.wav")


def timeit(fn, n=10, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort()
    return dict(median=statistics.median(ts), p90=ts[int(0.9 * (n - 1))], min=ts[0], max=ts[-1], n=n)


R = {"inputs": dict(sentence=SENT, alarm_clip=alarm, frames="WLASL 69241.mp4 @0.6/1.2/1.8s, 320x240")}

import server  # noqa: E402
from modules import stt as stt_mod  # noqa: E402

# ── Whisper size sweep (latency + transcript on the same 2.8 s chunk) ──
sweep = {}
for size in ("tiny", "base", "small", "distil-small.en"):
    s = stt_mod.SpeechToText()
    s.model_size = size
    t = time.perf_counter(); s._ensure_loaded(); load_s = time.perf_counter() - t
    stats = timeit(lambda: s.transcribe(speech_wav))
    text = s.transcribe(speech_wav).get("text", "")
    sweep[size] = dict(load_s=load_s, **stats, text=text)
    print("whisper", size, f"{stats['median']:.3f}s", repr(text), flush=True)
    del s
R["whisper_sweep"] = sweep

# ── components (the instances the server uses) ────────────────────────
C = {}
C["ffmpeg_webm_to_wav"] = timeit(lambda: server._ffmpeg_convert(speech_webm))
C["whisper_small_transcribe"] = timeit(lambda: server.get_stt().transcribe(speech_wav))
C["ast_classify_to_events"] = timeit(lambda: server.get_audio_scene().classify_to_events(alarm_wav))
C["yamnet_personal_embed"] = timeit(lambda: server.get_personal()._embed(alarm_wav))
C["yolo_detect_3frames"] = timeit(lambda: server.get_hazard().detect_to_events(frames))
def _blip_uncached():
    sd = server.get_scene()
    sd._last_caption_ts = 0.0          # bypass the 2 s internal throttle so BLIP really runs
    return sd.describe_jpeg(frames[1])


C["blip_describe_1frame"] = timeit(_blip_uncached, n=5)
R["blip_caption"] = server.get_scene()._last_caption
C["mediapipe_face_tracker_3frames"] = timeit(lambda: server.get_face().analyse_frames(frames))
R["face_tracker_backend"] = type(server.get_face()._mesh).__name__
R["face_tracker_faces"] = server.get_face().analyse_frames(frames).get("faces")
C["deepface_emotion_3frames"] = timeit(lambda: server.get_emo_visual().analyse_frames(frames))
C["wav2vec2_voice_emotion"] = timeit(lambda: server.get_emo_audio().analyse(speech_wav))


def _slr():
    server.get_slr().push_frames(frames)
    server.get_slr().classify_to_event()
    server.get_slr().reset_buffer()


C["mediapipe_tasks_sign_3frames"] = timeit(_slr)
from modules.people import get_people  # noqa: E402
if get_people().profile.get("people"):
    C["deepface_face_identify_1frame"] = timeit(lambda: get_people().identify_face(frames[1]))
C["llm_compose_notification"] = timeit(
    lambda: server.get_llm().compose_notification(
        [{"label": "Alarm clock", "priority_name": "CRITICAL", "source": "sound"}], "a person in a room"),
    n=5, warm=1)
R["components"] = C
for k, v in C.items():
    print(f"{k:34s} median {v['median']*1000:7.0f} ms  p90 {v['p90']*1000:7.0f} ms", flush=True)

# ── /stt/stream ────────────────────────────────────────────────────────
client = server.app.test_client()


def post(path, payload):
    r = client.post(path, data=json.dumps(payload), content_type="application/json")
    return r.get_json()


R["stt_stream"] = dict(
    interim_tiny_1s=timeit(lambda: post("/stt/stream", {"audio_b64": base64.b64encode(speech1_wav).decode(), "is_final": False})),
    final_small_2p8s=timeit(lambda: post("/stt/stream", {"audio_b64": base64.b64encode(speech_wav).decode(), "is_final": True})),
)
print("stt/stream", {k: round(v["median"], 3) for k, v in R["stt_stream"].items()}, flush=True)

# ── enrol one personal sound so ticks include realistic matching cost ──
from modules.events import Priority  # noqa: E402
enrol_files = sorted(r["filename"] for r in meta if r["category"] == "clock_alarm")[1:4]
enrol_clips = []
for k, fn in enumerate(enrol_files):
    outp = f"{MEDIA}/enrol_alarm_{k}.wav"
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"data/ESC-50/audio/{fn}", "-ar", "16000", "-ac", "1", "-t", "3", outp)
    enrol_clips.append(rb(outp))
R["enrolment"] = server.get_personal().enrol("Kitchen alarm", Priority.IMPORTANT, enrol_clips)
R["personal_match_prototypical"] = timeit(lambda: server.get_personal().match_prototypical(alarm_wav))
R["personal_match_example"] = [e.to_dict() for e in server.get_personal().match_prototypical(alarm_wav)]
R["personal_match_on_speech"] = [e.to_dict() for e in server.get_personal().match_prototypical(speech_wav)]
print("enrolled:", R["enrolment"], flush=True)

# ── full /process ticks ────────────────────────────────────────────────
post("/reset", {})
ticks = {}
for name, audio, streaming, n in (("streaming_captions_on", speech_webm, True, 20),
                                  ("streaming_captions_off", speech_webm, False, 20),
                                  ("critical_alarm", alarm_webm, True, 12)):
    server.process._scene_tick_counter = 0
    lat, wall, samples = [], [], []
    for i in range(n):
        payload = {"audio_b64": base64.b64encode(audio).decode(), "frames_b64": frames, "stream_captions": streaming}
        t = time.perf_counter()
        d = post("/process", payload)
        wall.append(time.perf_counter() - t)
        lat.append(d.get("latency_ms", -1) / 1000.0)
        if i in (0, n - 1):
            samples.append(dict(
                tick=i, stt=(d.get("stt") or {}).get("text", ""),
                sounds=[(e["label"], round(e["confidence"], 2), e["priority_name"]) for e in d.get("sounds", [])][:4],
                headlines=[(h["label"], h["priority_name"]) for h in d.get("headlines", [])],
                scene=d.get("scene_caption"), emotion=(d.get("emotion") or {}).get("label"),
                hazards=len(d.get("hazards") or []), notification=d.get("notification"),
            ))
    s_ = sorted(wall)
    ticks[name] = dict(n=n, wall_median=statistics.median(wall), wall_p95=s_[int(0.95 * (n - 1))],
                       wall_max=s_[-1], wall_min=s_[0], per_tick_wall=wall, per_tick_server=lat, samples=samples)
    print(f"ticks {name}: median {statistics.median(wall):.2f}s p95 {s_[int(0.95*(n-1))]:.2f}s max {s_[-1]:.2f}s", flush=True)
R["ticks"] = ticks
json.dump(R, open(os.path.join(SP, "latency_results.json"), "w"), indent=1)
print("DONE latency", flush=True)
