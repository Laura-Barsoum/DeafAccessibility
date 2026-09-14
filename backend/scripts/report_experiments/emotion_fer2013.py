"""[Final report: Table 5.1, Figure 5.1d]

Facial emotion on the FER-2013 private test split (Goodfellow et al., 2013), the
per-label precision and recall benchmark planned in the preliminary report.

Images are read from backend/data/emotion_eval/fer2013/privateTest/<label>/*.png
(3,589 faces; the dataset is not committed). Each face goes through the same
DeepFace call the application makes in VisualEmotionAnalyzer._analyse_single,
and is scored twice: with the shipped per-class recalibration, and with
DeepFace's uncalibrated output, to show what the recalibration costs or gains on
a labelled set. The first 50 calibrated predictions are checked against the
shipped method itself, so the replication cannot drift from the application.

DeepFace's emotion model was trained on FER-2013's training split, so this is an
in-distribution test: a ceiling for webcam frames rather than an estimate of them.
Writes eval_results/emotion_fer2013.json.
"""
from pathlib import Path
import base64, glob, json, math, os, sys, time

import cv2
import numpy as np

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
DATA = os.path.join(BACKEND, "data", "emotion_eval", "fer2013", "privateTest")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from modules import emotion as em  # noqa: E402
from deepface import DeepFace  # noqa: E402

LABELS = em.EMOTIONS  # angry, disgust, fear, happy, sad, surprise, neutral


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def deepface_scores(img):
    """The application's DeepFace call (same arguments as _analyse_single)."""
    res = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False, detector_backend="opencv", silent=True)
    if isinstance(res, list):
        res = res[0]
    return {k.lower(): float(v) for k, v in res.get("emotion", {}).items()}


def calibrated(raw):
    """The shipped recalibration, as written in emotion.py."""
    cal = {k: raw.get(k, 0.0) * em._DEEPFACE_CALIB.get(k, 1.0) for k in (set(raw) | set(em._DEEPFACE_CALIB))}
    total = sum(cal.values()) or 1.0
    return {k: v / total for k, v in cal.items()}


def scores(pairs):
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for truth, pred in pairs:
        if pred in LABELS:
            cm[LABELS.index(truth), LABELS.index(pred)] += 1
    per = {}
    for i, lab in enumerate(LABELS):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[lab] = dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0, support=int(cm[i, :].sum()))
    correct, n = int(np.trace(cm)), len(pairs)
    return dict(accuracy=correct / n, accuracy_95=wilson(correct, n), n=n,
                macro_f1=float(np.mean([v["f1"] for v in per.values()])), per_class=per,
                predicted_share={lab: int(cm[:, i].sum()) / n for i, lab in enumerate(LABELS)},
                confusion=cm.tolist())


files = sorted((lab, f) for lab in LABELS for f in glob.glob(os.path.join(DATA, lab, "*.png")))
assert len(files) > 3000, f"expected the FER-2013 private test split in {DATA}, found {len(files)} images"
analyzer = em.VisualEmotionAnalyzer()
records, failures, t0 = [], 0, time.time()
for k, (truth, path) in enumerate(files):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    ok, jpg = cv2.imencode(".jpg", img)
    img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)           # the application receives JPEG frames
    raw = deepface_scores(img)
    if not raw:
        failures += 1
        records.append((truth, None, None)); continue
    cal = calibrated(raw)
    rp, cp = max(raw, key=raw.get), max(cal, key=cal.get)
    if k < 50:                                           # replication check against the shipped method
        shipped = analyzer._analyse_single(base64.b64encode(jpg.tobytes()).decode())
        assert max(shipped, key=shipped.get) == cp, (path, shipped, cal)
    records.append((truth, rp, cp))
    if k % 500 == 0:
        print(f"{k}/{len(files)} images, {time.time() - t0:.0f}s", flush=True)

out = dict(
    method=__doc__, dataset="FER-2013 private test split", n_images=len(files), failures=failures,
    labels=LABELS, calibration=em._DEEPFACE_CALIB,
    calibrated=scores([(t, c) for t, _r, c in records]),
    uncalibrated=scores([(t, r) for t, r, _c in records]),
    seconds=time.time() - t0,
    predictions=[[t, r, c] for t, r, c in records],
)
json.dump(out, open(os.path.join(SP, "emotion_fer2013.json"), "w"), indent=1)
for name in ("calibrated", "uncalibrated"):
    s = out[name]
    print(f"{name}: accuracy {s['accuracy']:.3f} {s['accuracy_95']} macro-F1 {s['macro_f1']:.3f}")
    print("   ", {lab: round(v["f1"], 2) for lab, v in s["per_class"].items()})
print("DONE emotion", flush=True)
