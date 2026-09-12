"""[Final report: Section 5.2 personal sounds]

How far does the browser's Opus codec move a clip relative to the calibrated
prototypical gate? Enrols three ESC-50 clock-alarm clips, then measures one
held-out clip's squared distance to the prototype when decoded directly and
after an Opus round trip through the server's own conversion, plus a speech
negative. Writes eval_results/codec_margin.json."""
from pathlib import Path
import csv, json, os, subprocess, sys, tempfile

import numpy as np

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
os.makedirs(SP, exist_ok=True)
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)
tmpd = tempfile.mkdtemp(prefix="codec_margin_")


def sh(*a):
    subprocess.run(a, check=True, capture_output=True)


def wav16(src, dst, seconds):
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", "16000", "-ac", "1", "-t", seconds, dst)
    return open(dst, "rb").read()


meta = list(csv.DictReader(open("data/ESC-50/meta/esc50.csv")))
alarms = sorted(r["filename"] for r in meta if r["category"] == "clock_alarm")
enrol = [wav16(f"data/ESC-50/audio/{f}", f"{tmpd}/enrol{k}.wav", "3") for k, f in enumerate(alarms[1:4])]
held = wav16(f"data/ESC-50/audio/{alarms[0]}", f"{tmpd}/held.wav", "2.8")
sh("ffmpeg", "-y", "-loglevel", "error", "-i", f"{tmpd}/held.wav", "-c:a", "libopus", "-b:a", "32k", f"{tmpd}/held.webm")
sh("say", "-o", f"{tmpd}/speech.aiff", "The meeting has moved to three o'clock this afternoon, so please bring the report.")
speech = wav16(f"{tmpd}/speech.aiff", f"{tmpd}/speech.wav", "2.8")

from server import _ffmpeg_convert  # noqa: E402  (the conversion live audio goes through)
from modules import personalizer as pm  # noqa: E402
from modules.events import Priority  # noqa: E402

held_opus = _ffmpeg_convert(open(f"{tmpd}/held.webm", "rb").read())
pm.PROFILE_DIR = tmpd
pm.PROFILE_PATH = os.path.join(tmpd, "profile.json")
P = pm.Personalizer()
P.profile = {"sounds": {}}
P.enrol("Kitchen alarm", Priority.IMPORTANT, enrol)
if P._yamnet in (None, "placeholder"):
    raise SystemExit("YamNet did not load; refusing to measure the fallback embedding")
s = P.profile["sounds"]["Kitchen alarm"]
proto, radius = np.array(s["prototype"]), float(s["radius"])


def dist(b):
    e = P._embed(b)
    e = e / np.linalg.norm(e)
    return float(np.sum((e - proto) ** 2))


out = dict(held_out_clip=alarms[0], enrolled_clips=alarms[1:4], radius=radius,
           gate_calibrated=max(radius * pm.PROTO_RADIUS_K, pm.PROTO_GATE_FLOOR), gate_original=max(radius * 3, 0.20),
           held_out_uncompressed=dist(held), held_out_opus=dist(held_opus), speech=dist(speech))
json.dump(out, open(os.path.join(SP, "codec_margin.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
