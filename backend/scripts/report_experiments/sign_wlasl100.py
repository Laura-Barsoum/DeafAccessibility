"""[Final report: Section 5.2 signing, Table 5.1]

Word-level sign recognition on the WLASL100 test split (Li et al., 2020), before
and after the TGCN tier is available, through the application's own classifier.

Clips: the 100 official WLASL100 test clips that could still be obtained (of 258),
from the Voxel51 mirror on Hugging Face, each trimmed to its annotated frame range
(backend/data/WLASL-master/videos_test100, not committed). Frames are sampled as
eval_wlasl.py samples them and passed to
SignLanguageRecognizer.classify_all_frames_combined, the method behind
/sign/recognize: once with the TGCN tier disabled (the configuration reported so
far) and once with the asl100 checkpoint from huggingface.co/sharonn18/tgcn-wlasl.
The ASL letter tier needs a model that is not cached here, so it is off in both
runs; the model is loaded from the local cache only.

The checkpoint's documentation leaves three things open: how MediaPipe's
landmarks map onto the 55 OpenPose keypoints it was trained on (the app's old
layout, OpenPose order, or OpenPose order with the hands swapped), whether
coordinates are in [0, 1] or [-1, 1], and the class order. `--select` settles
all three on the 14 older local WLASL100 clips, none of which is in the test split,
before any test clip is scored. The model's own top-1 and top-5 on the test clips are reported alongside
the pipeline's. Writes eval_results/sign_wlasl100.json (or sign_wlasl100_select.json).
"""
from pathlib import Path
import argparse, base64, json, logging, math, os, sys, time

import cv2
import numpy as np

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ap = argparse.ArgumentParser()
ap.add_argument("--select", action="store_true", help="only settle coordinate range and class order on the dev clips")
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--trained", action="store_true",
                help="score the weights trained on MediaPipe keypoints (sign_train_tgcn.py) instead of the published ones")
ARGS = ap.parse_args()
CKPT = os.path.join(BACKEND, "data", "tgcn", "asl100_mediapipe" if ARGS.trained else "asl100", "pytorch_model.bin")
os.environ["ACCESSIBILITY_TGCN_LOCAL_PATH"] = CKPT
os.environ["ACCESSIBILITY_ENABLE_TGCN"] = "1"      # the application leaves the tier off by default

import torch  # noqa: E402
from modules import tgcn_sign_model as tg  # noqa: E402
from modules.sign_language import SignLanguageRecognizer, _TasksHolistic  # noqa: E402

MAN = json.load(open("data/WLASL-master/start_kit/WLASL_v0.3.json"))
TOP100 = [e["gloss"] for e in MAN[:100]]
INST = {i["video_id"]: (e["gloss"], i) for e in MAN for i in e["instances"]}
TEST_DIR = os.path.join("data", "WLASL-master", "videos_test100")
DEV_DIR = os.path.join("data", "WLASL-master", "videos")
TEST_IDS = sorted(json.load(open(os.path.join(TEST_DIR, "labels.json"))))
DEV_IDS = sorted(p[:-4] for p in os.listdir(DEV_DIR) if p.endswith(".mp4") and p[:-4] in INST
                 and INST[p[:-4]][1]["split"] != "test" and INST[p[:-4]][0] in TOP100)
assert not set(DEV_IDS) & set(TEST_IDS)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def clip_frames(vid, directory, n):
    cap = cv2.VideoCapture(os.path.join(directory, f"{vid}.mp4"))
    allf = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        allf.append(fr)
    cap.release()
    inst = INST[vid][1]
    s, e = max(int(inst.get("frame_start", 1)) - 1, 0), int(inst.get("frame_end", -1))
    seg = allf[s:(e if e > 0 else None)] or allf
    if not seg:
        return []
    return [seg[i] for i in np.linspace(0, len(seg) - 1, n).astype(int)]


HOLISTIC = _TasksHolistic(os.path.join(BACKEND, "data"))


def landmarks(frames):
    return [HOLISTIC.process(np.ascontiguousarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))) for fr in frames]


def keypoints(results, layout):
    out = []
    for res in results:
        kp = tg.extract_55_keypoints(res, layout)
        if kp is not None:
            out.append(kp)
    return out


REC = tg.TGCNSignRecognizer("asl100")
assert REC.load(), "the TGCN checkpoint did not load"
state = torch.load(CKPT, map_location="cpu", weights_only=False)
state = state.get("state_dict", state) if isinstance(state, dict) else state
model_keys = set(REC.model.state_dict())
ckpt_keys = {k.replace("module.", "") for k in state}
LOAD = dict(missing=len(model_keys - ckpt_keys), unexpected=len(ckpt_keys - model_keys))
assert LOAD["missing"] == 0, f"checkpoint does not match the architecture: {LOAD}"
assert list(tg.WLASL100_DEFAULT_ORDER) == TOP100, "default class order differs from the manifest order"

ORDERS = {"manifest": list(TOP100), "alphabetical": sorted(TOP100)}


def model_topk(kps, coords, order, k=5):
    if len(kps) < 8:
        return []
    x = torch.from_numpy(tg._resample_frames(tg.to_model_coords(kps, coords), 50)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(REC.model(x).squeeze(0), dim=-1).numpy()
    return [ORDERS[order][i] for i in np.argsort(probs)[::-1][:k]]


LAYOUTS = ("legacy", "openpose", "openpose_swapped")
VARIANTS = [(l, c, o) for l in LAYOUTS for c in ("unit", "signed") for o in ("manifest", "alphabetical")]

if ARGS.select:
    dev = {vid: (INST[vid][0], landmarks(clip_frames(vid, DEV_DIR, 50))) for vid in DEV_IDS}
    scores = {}
    for l, c, o in VARIANTS:
        preds = [(g, model_topk(keypoints(res, l), c, o)) for g, res in dev.values()]
        scores[f"{l}/{c}/{o}"] = dict(top1=sum(1 for g, p in preds if p[:1] == [g]),
                                      top5=sum(1 for g, p in preds if g in p), n=len(preds))
    best = max(scores, key=lambda k: (scores[k]["top5"], scores[k]["top1"]))
    json.dump(dict(dev_clips=DEV_IDS, checkpoint_load=LOAD, scores=scores, chosen=best),
              open(os.path.join(SP, "sign_wlasl100_select.json"), "w"), indent=1)
    print("dev clips:", len(DEV_IDS), "| load:", LOAD, "| scores:", scores, "| chosen:", best, flush=True)
    sys.exit(0)

if ARGS.trained:
    CFG = json.load(open(os.path.join(os.path.dirname(CKPT), "config.json")))
    LAYOUT, COORDS, ORDER = CFG["layout"], CFG["coords"], CFG["class_order"]
    assert (REC.coords, REC.threshold) == (COORDS, CFG["threshold"]), "the recogniser ignored the saved settings"
    VARIANT = f"{LAYOUT}/{COORDS}/{ORDER} (trained, threshold {CFG['threshold']})"
else:
    sel = json.load(open(os.path.join(SP, "sign_wlasl100_select.json")))
    LAYOUT, COORDS, ORDER = sel["chosen"].split("/")
    assert (tg.KEYPOINT_LAYOUT, tg.KEYPOINT_COORDS, tg.CLASS_ORDER) == (LAYOUT, COORDS, ORDER), \
        "the application's TGCN settings differ from the dev-selected variant"
    VARIANT = sel["chosen"]
print("using", VARIANT, flush=True)

records, t0 = [], time.time()
slr_off = SignLanguageRecognizer()
slr_off._tgcn = tg.TGCNSignRecognizer("asl100")
slr_off._tgcn._load_failed = True                     # the TGCN tier as it was: unavailable
slr_on = SignLanguageRecognizer()
slr_on._ensure_tgcn()
assert slr_on._tgcn is not None and slr_on._tgcn.is_available(), "TGCN tier did not come up in the application class"
for k, vid in enumerate(TEST_IDS):
    gloss = INST[vid][0]
    frames = clip_frames(vid, TEST_DIR, ARGS.frames)
    b64 = [base64.b64encode(cv2.imencode(".jpg", f)[1].tobytes()).decode() for f in frames]
    row = dict(video=vid, gloss=gloss, frames=len(frames))
    for name, slr in (("before", slr_off), ("after", slr_on)):
        t = time.perf_counter()
        seq, diag = slr.classify_all_frames_combined(b64, window_size=8, stride=3)
        labels = [s["label"].lower() for s in seq]
        row[name] = dict(labels=labels[:5], top1=labels[:1] == [gloss], top3=gloss in labels[:3], top5=gloss in labels[:5],
                         tgcn=diag.get("tgcn_top_prediction"), ms=round((time.perf_counter() - t) * 1000))
    kps = keypoints(landmarks(frames), LAYOUT)
    top5 = model_topk(kps, COORDS, ORDER)
    row["model"] = dict(keypoint_frames=len(kps), top5=top5, top1_hit=top5[:1] == [gloss], top3_hit=gloss in top5[:3],
                        top5_hit=gloss in top5)
    records.append(row)
    if k % 10 == 0:
        print(f"{k}/{len(TEST_IDS)} clips, {time.time() - t0:.0f}s", flush=True)


def summary(key, hit):
    n = len(records)
    c = sum(1 for r in records if r[key][hit])
    return dict(n=c, of=n, rate=c / n, ci95=wilson(c, n))


out = dict(
    method=__doc__, checkpoint=("data/tgcn/asl100_mediapipe, trained by sign_train_tgcn.py" if ARGS.trained
                                else "huggingface.co/sharonn18/tgcn-wlasl checkpoints/asl100"), checkpoint_load=LOAD,
    variant=VARIANT, test_clips=len(TEST_IDS), official_test_clips=258,
    before=dict(top1=summary("before", "top1"), top3=summary("before", "top3"), top5=summary("before", "top5")),
    after=dict(top1=summary("after", "top1"), top3=summary("after", "top3"), top5=summary("after", "top5")),
    model=dict(top1=summary("model", "top1_hit"), top3=summary("model", "top3_hit"), top5=summary("model", "top5_hit")),
    median_ms=dict(before=float(np.median([r["before"]["ms"] for r in records])),
                   after=float(np.median([r["after"]["ms"] for r in records]))),
    records=records,
)
json.dump(out, open(os.path.join(SP, "sign_wlasl100_trained.json" if ARGS.trained else "sign_wlasl100.json"), "w"), indent=1)
for key in ("before", "after", "model"):
    print(key, {m: f"{v['n']}/{v['of']} ({v['rate']:.1%})" for m, v in out[key].items()})
print("DONE sign", flush=True)
