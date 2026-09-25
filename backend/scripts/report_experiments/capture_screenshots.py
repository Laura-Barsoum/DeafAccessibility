"""[Final report: Figures 4.3 to 4.6]

Capture genuine screenshots of the running application.

Chromium's fake-capture flags replace the physical camera and microphone with
recorded test media, so the real browser capture path, the real /process and
/stt/stream endpoints and the real models all run; only the devices change.
Audio: two synthesised sentences (the second calls the user's name) plus a
held-out ESC-50 clock-alarm clip. Video: a WLASL signing clip. The floating
camera preview is made invisible (it still renders, so frames are captured)
because the signer did not consent to appearing in this report (Bragg et al.,
2021), and the critical-alert flash is suppressed so panels are legible.

Requires Playwright with Chromium installed in the Python that runs this script:
    pip install playwright && python -m playwright install chromium"""
import base64, csv, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
VENV_PY = os.path.join(BACKEND, "venv", "bin", "python")
M = os.path.join(SP, "shots_media"); OUT = os.path.join(SP, "shots")
os.makedirs(M, exist_ok=True); os.makedirs(OUT, exist_ok=True)
PORT = 5051
BASE = f"http://127.0.0.1:{PORT}"


def sh(*a):
    subprocess.run(a, check=True, capture_output=True)


def pcm(src, dst, extra=()):
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", src, *extra, "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", dst)


# ── test media ──────────────────────────────────────────────────────────
for name, text in (("s1", "Hello, can you hear me? The delivery driver is at the front door."),
                   ("s2", "Laura, dinner is ready in the kitchen.")):
    sh("say", "-o", f"{M}/{name}.aiff", text)
    pcm(f"{M}/{name}.aiff", f"{M}/{name}.wav")
meta = list(csv.DictReader(open(os.path.join(BACKEND, "data/ESC-50/meta/esc50.csv"))))
alarms = sorted(r["filename"] for r in meta if r["category"] == "clock_alarm")
pcm(os.path.join(BACKEND, "data/ESC-50/audio", alarms[4]), f"{M}/alarm.wav")          # held out
sh("ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
   "-t", "1.2", "-c:a", "pcm_s16le", f"{M}/sil.wav")
with open(f"{M}/concat.txt", "w") as fh:
    for part in ("sil", "s1", "sil", "sil", "alarm", "sil", "s2", "sil", "sil"):
        fh.write(f"file '{M}/{part}.wav'\n")
sh("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f"{M}/concat.txt",
   "-c:a", "pcm_s16le", f"{M}/track.wav")
sh("ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(BACKEND, "data/WLASL-master/videos/69241.mp4"),
   "-vf", "scale=640:480,fps=15", "-pix_fmt", "yuv420p", f"{M}/cam.y4m")
enrol_b64 = []
for k, fn in enumerate(alarms[1:4]):
    out = f"{M}/enrol_{k}.wav"
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(BACKEND, "data/ESC-50/audio", fn),
       "-ar", "16000", "-ac", "1", "-t", "3", out)
    enrol_b64.append(base64.b64encode(open(out, "rb").read()).decode())


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=5).read())


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


# ── server ──────────────────────────────────────────────────────────────
log = open(os.path.join(OUT, "server.log"), "w")
proc = subprocess.Popen([VENV_PY, "server.py"], cwd=BACKEND, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, ACCESSIBILITY_PORT=str(PORT)))
try:
    for _ in range(120):
        try:
            get("/health"); break
        except Exception:
            time.sleep(1)
    t0 = time.time()
    while time.time() - t0 < 300:
        txt = open(os.path.join(OUT, "server.log")).read().lower()
        if "prewarm" in txt and any(k in txt for k in ("ready", "complete", "done", "prewarmed")):
            break
        time.sleep(2)
    print("server ready after", round(time.time() - t0), "s of prewarm", flush=True)
    post("/reset", {})
    print("self name:", post("/me/name", {"name": "Laura"}), flush=True)
    print("enrol:", post("/enrol", {"label": "Kitchen alarm", "priority": 2, "clips_b64": enrol_b64}), flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={M}/cam.y4m",
            f"--use-file-for-fake-audio-capture={M}/track.wav",
            "--autoplay-policy=no-user-gesture-required"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        ctx.grant_permissions(["camera", "microphone"], origin=BASE)
        page = ctx.new_page()
        logs = []
        page.on("console", lambda msg: logs.append(f"{msg.type}: {msg.text}"))
        page.goto(BASE)
        page.wait_for_timeout(2500)
        page.add_style_tag(content="#webcam-preview{opacity:0 !important;} #sign-live-badge{display:none !important;} "
                              "body.critical-flash::before, body.critical-flash::after{display:none !important;}")
        page.click("#start-btn")

        def card(text):
            return page.locator("div.card").filter(has=page.locator("h2, h3", has_text=text)).first

        for n, wait_ms in enumerate((30000, 25000, 25000)):
            page.wait_for_timeout(wait_ms)
            page.screenshot(path=os.path.join(OUT, f"live_full_{n}.png"), full_page=True)
            print("captured live", n, flush=True)
        for key, text in (("headlines", "Headlines"), ("captions", "Captions"), ("sounds", "Sounds"),
                          ("scene", "Scene"), ("hazard", "Hazard Radar"), ("alerts", "Keyword Alerts"),
                          ("emotion", "Emotion")):
            try:
                card(text).screenshot(path=os.path.join(OUT, f"card_{key}.png"))
            except Exception as e:
                print("card failed", key, e, flush=True)
        page.click("#sign-speak-btn")
        page.wait_for_timeout(6000)
        page.click("#sign-stop-btn")
        page.wait_for_timeout(20000)
        try:
            card("Sign Language Channel").screenshot(path=os.path.join(OUT, "card_sign.png"))
        except Exception as e:
            print("sign card failed", e, flush=True)
        page.click("#stop-btn")
        page.wait_for_timeout(3000)
        for tab, fname in (("#tab-personal", "tab_personal.png"), ("#tab-people", "tab_people.png"),
                           ("#tab-diary", "tab_diary.png"), ("#tab-summary", "tab_summary.png")):
            page.click(tab)
            page.wait_for_timeout(2500)
            if tab == "#tab-diary":
                page.click("#diary-refresh")
                page.wait_for_timeout(3000)
            if tab == "#tab-summary":
                page.click("#generate-summary")       # one language-model call on the session's events
                page.wait_for_timeout(12000)
            page.screenshot(path=os.path.join(OUT, fname), full_page=True)
        json.dump(logs, open(os.path.join(OUT, "console.json"), "w"), indent=1)
        browser.close()
    print("DONE shots", flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
