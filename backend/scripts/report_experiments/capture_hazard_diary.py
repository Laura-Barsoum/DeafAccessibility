"""[Final report: Figure 4.6]

Screenshots of the hazard channel and of the Diary and Summary tabs, taken the
same way as capture_screenshots.py: Chromium's fake camera and microphone feed
recorded test media through the real capture path to a real server and models.

Camera: bus.jpg, the test photograph that ships with Ultralytics, looped as video
so YOLO sees a bus. Only the Hazard Radar panel is captured, never the camera
image. Microphone: two synthesised sentences and a held-out ESC-50 clock-alarm clip.

The server runs with a temporary diary and temporary profiles, so no real user
data can appear on screen; the script checks the real diary was not written. The
Summary tab calls the configured language model on these test events only.

Run with a Python that has Playwright and its Chromium build installed; the
server itself runs in the project venv.
"""
import csv, json, os, shutil, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BACKEND = str(Path(__file__).resolve().parents[2])
VENV_PY = os.path.join(BACKEND, "venv", "bin", "python")
M = os.path.join(BACKEND, "eval_results", "media", "shots_extra")
OUT = M
os.makedirs(M, exist_ok=True)
PORT = 5057
BASE = f"http://127.0.0.1:{PORT}"
TMP = tempfile.mkdtemp(prefix="shots_extra_")
REAL_DIARY = os.path.join(BACKEND, "data", "diary.db")


def sh(*a):
    subprocess.run(a, check=True, capture_output=True)


def pcm(src, dst):
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", dst)


# ── test media ──────────────────────────────────────────────────────────
for name, text in (("s1", "Hello, can you hear me? The delivery driver is at the front door."),
                   ("s2", "Watch out, a bus is coming down the road.")):
    sh("say", "-o", f"{M}/{name}.aiff", text)
    pcm(f"{M}/{name}.aiff", f"{M}/{name}.wav")
meta = list(csv.DictReader(open(os.path.join(BACKEND, "data/ESC-50/meta/esc50.csv"))))
alarms = sorted(r["filename"] for r in meta if r["category"] == "clock_alarm")
pcm(os.path.join(BACKEND, "data/ESC-50/audio", alarms[4]), f"{M}/alarm.wav")
sh("ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
   "-t", "1.2", "-c:a", "pcm_s16le", f"{M}/sil.wav")
with open(f"{M}/concat.txt", "w") as fh:
    for part in ("sil", "s1", "sil", "sil", "alarm", "sil", "s2", "sil", "sil"):
        fh.write(f"file '{M}/{part}.wav'\n")
sh("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f"{M}/concat.txt",
   "-c:a", "pcm_s16le", f"{M}/track.wav")
BUS = subprocess.run([VENV_PY, "-c", "import ultralytics, os; print(os.path.join(os.path.dirname(ultralytics.__file__), 'assets', 'bus.jpg'))"],
                     capture_output=True, text=True, check=True).stdout.strip()
sh("ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", BUS, "-t", "10",
   "-vf", "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,fps=5",
   "-pix_fmt", "yuv420p", f"{M}/cam.y4m")


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=5).read())


# ── isolated server: temporary diary and profiles ────────────────────────
launcher = f"""
import os, sys, runpy
from pathlib import Path
os.environ["ACCESSIBILITY_DB_PATH"] = {os.path.join(TMP, "diary.db")!r}
sys.path.insert(0, {BACKEND!r}); os.chdir({BACKEND!r})
from modules import people, personalizer, diarized_stt
people.PROFILE_DIR, people.PROFILE_PATH = {TMP!r}, {os.path.join(TMP, "people.json")!r}
personalizer.PROFILE_DIR, personalizer.PROFILE_PATH = {TMP!r}, {os.path.join(TMP, "personal.json")!r}
diarized_stt._SPEAKERS_PATH = Path({os.path.join(TMP, "speakers.json")!r})
runpy.run_path("server.py", run_name="__main__")
"""
real_diary_mtime = os.path.getmtime(REAL_DIARY) if os.path.exists(REAL_DIARY) else None
log = open(os.path.join(OUT, "server.log"), "w")
proc = subprocess.Popen([VENV_PY, "-c", launcher], cwd=BACKEND, stdout=log, stderr=subprocess.STDOUT,
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

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={M}/cam.y4m",
            f"--use-file-for-fake-audio-capture={M}/track.wav",
            "--autoplay-policy=no-user-gesture-required"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        ctx.grant_permissions(["camera", "microphone"], origin=BASE)
        page = ctx.new_page()
        page.goto(BASE)
        page.wait_for_timeout(2500)
        page.add_style_tag(content="#webcam-preview{opacity:0 !important;} "
                              "body.critical-flash::before, body.critical-flash::after{display:none !important;}")
        page.click("#start-btn")

        def card(text):
            return page.locator("div.card").filter(has=page.locator("h2, h3", has_text=text)).first

        page.wait_for_timeout(45000)
        for key, text in (("hazard", "Hazard Radar"), ("headlines", "Headlines")):
            try:
                card(text).screenshot(path=os.path.join(OUT, f"card_{key}.png"))
            except Exception as e:
                print("card failed", key, e, flush=True)
        page.click("#stop-btn")
        page.wait_for_timeout(3000)
        page.click("#tab-diary"); page.wait_for_timeout(1500)
        page.click("#diary-refresh"); page.wait_for_timeout(3000)
        page.screenshot(path=os.path.join(OUT, "tab_diary.png"), full_page=True)
        page.click("#tab-summary"); page.wait_for_timeout(1500)
        page.click("#generate-summary"); page.wait_for_timeout(20000)
        page.screenshot(path=os.path.join(OUT, "tab_summary.png"), full_page=True)
        browser.close()
    print("DONE shots_extra", flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
    if real_diary_mtime is not None:
        assert os.path.getmtime(REAL_DIARY) == real_diary_mtime, "the real diary was written"
    print("real diary untouched; temporary data in", TMP, flush=True)
