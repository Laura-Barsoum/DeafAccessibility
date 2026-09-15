"""[Final report: Section 5.2 signing]

Train the WLASL TGCN on the keypoints the application extracts.

The published asl100 weights score at chance on MediaPipe keypoints
(sign_wlasl100.py), most likely because they were learned from OpenPose. Here the
same network (tgcn_sign_model.build_TGCN at the asl100 size) is trained on
MediaPipe keypoints from the obtainable WLASL100 train clips (sign_keypoints.py).
Frames with nothing detected are dropped and the rest resampled to 50, exactly
as TGCNSignRecognizer.predict does.

Every choice uses the validation clips only: whether to start from the published
weights or from random weights, whether keypoints stay in image coordinates or
are normalised to the signer's shoulders, the epoch to keep, and the confidence
above which the application shows a TGCN sign (the lowest at which at least half
of accepted validation predictions are right). Learning rates, augmentation and
the epoch budget were fixed before training. The test clips are scored once, with
the chosen configuration, at the end.

Writes the checkpoint and its settings to data/tgcn/asl100_mediapipe/ and results
to eval_results/sign_tgcn_train.json.
"""
from pathlib import Path
import json, math, os, sys, time

import numpy as np

BACKEND = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)
SP = os.path.join(BACKEND, "eval_results")
OUT_DIR = os.path.join(BACKEND, "data", "tgcn", "asl100_mediapipe")
PUBLISHED = os.path.join(BACKEND, "data", "tgcn", "asl100", "pytorch_model.bin")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from modules import tgcn_sign_model as tg  # noqa: E402

torch.set_num_threads(8)
CLASSES = sorted(tg.WLASL100_DEFAULT_ORDER)            # the published head's order, kept for both starts
INDEX = {g: i for i, g in enumerate(CLASSES)}
EPOCHS, PATIENCE, BATCH = 150, 30, 32
LR = {"published": 3e-4, "random": 1e-3}
MIN_FRAMES = 8                                         # TGCNSignRecognizer.predict needs at least 8
LAYOUT = tg.KEYPOINT_LAYOUT                            # the layout matched to the published weights on dev clips


def load(split):
    z = np.load(os.path.join(SP, f"sign_keypoints_{split}.npz"), allow_pickle=True)
    clips = []
    for k in range(len(z["video"])):
        pose, right, left, has = z["pose"][k], z["right"][k], z["left"][k], z["has"][k]
        T = pose.shape[0]
        kp = np.zeros((T, 55, 2), np.float32)
        for i, src in enumerate(tg.OPENPOSE_POSE_13):
            kp[:, i] = pose[:, list(src)].mean(axis=1) if isinstance(src, tuple) else pose[:, src]
        first, second = (left, right) if LAYOUT == "openpose_swapped" else (right, left)
        kp[:, 13:34], kp[:, 34:55] = first, second     # as extract_55_keypoints builds LAYOUT
        valid = has.any(axis=1)
        clips.append(dict(video=str(z["video"][k]), y=INDEX[str(z["gloss"][k])], frames=kp[valid]))
    return clips


def augment(frames, rng):
    """Random temporal crop and small scale, rotation and shift of detected points, in [0, 1] image space."""
    n = len(frames)
    a = rng.uniform(0, 0.1) * (n - 1)
    b = (n - 1) - rng.uniform(0, 0.1) * (n - 1)
    idx = np.clip(np.round(np.linspace(a, b, 50)).astype(int), 0, n - 1)
    x = frames[idx].copy()
    present = np.any(x != 0, axis=2)
    th = np.deg2rad(rng.uniform(-8, 8))
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], np.float32) * rng.uniform(0.9, 1.1)
    moved = (x - 0.5) @ rot.T + 0.5 + rng.uniform(-0.05, 0.05, size=2).astype(np.float32)
    x[present] = moved[present]
    return list(x)


def tensor(seqs, coords):
    return torch.from_numpy(np.stack([tg._resample_frames(tg.to_model_coords(s, coords), 50) for s in seqs]))


def evaluate(model, clips, coords):
    model.eval()
    usable = [i for i, c in enumerate(clips) if len(c["frames"]) >= MIN_FRAMES]
    probs = np.zeros((len(clips), len(CLASSES)), np.float32)
    with torch.no_grad():
        for s in range(0, len(usable), 64):
            part = usable[s:s + 64]
            p = torch.softmax(model(tensor([list(clips[i]["frames"]) for i in part], coords)), dim=-1).numpy()
            for i, row in zip(part, p):
                probs[i] = row
    ys = np.array([c["y"] for c in clips])
    ok = np.array([len(c["frames"]) >= MIN_FRAMES for c in clips])
    order = np.argsort(-probs, axis=1)
    top1 = ok & (order[:, 0] == ys)
    top5 = ok & np.any(order[:, :5] == ys[:, None], axis=1)
    return dict(top1=float(top1.mean()), top5=float(top5.mean()), n=len(clips), no_prediction=int((~ok).sum()),
                conf=probs.max(axis=1).tolist(), correct=top1.tolist(), usable=ok.tolist(),
                loss=float(F.cross_entropy(torch.from_numpy(probs[ok] + 1e-9).log(), torch.from_numpy(ys[ok])).item())
                if ok.any() else None)


def train(clips, val, init, coords, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = tg.build_TGCN(num_class=100, hidden_feature=64, num_stage=20)
    if init == "published":
        state = torch.load(PUBLISHED, map_location="cpu", weights_only=False)
        state = {k.replace("module.", ""): v for k, v in state.get("state_dict", state).items()}
        model.load_state_dict(state, strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR[init], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    usable = [c for c in clips if len(c["frames"]) >= MIN_FRAMES]
    best, best_state, curve, since = None, None, [], 0
    for epoch in range(EPOCHS):
        model.train()
        perm = rng.permutation(len(usable))
        for s in range(0, len(perm), BATCH):
            part = [usable[i] for i in perm[s:s + BATCH]]
            x = tensor([augment(c["frames"], rng) for c in part], coords)
            y = torch.tensor([c["y"] for c in part])
            loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        v = evaluate(model, val, coords)
        curve.append(dict(epoch=epoch, val_top1=v["top1"], val_top5=v["top5"], val_loss=v["loss"]))
        key = (v["top1"], v["top5"])
        if best is None or key > best[0]:
            best, best_state, since = (key, epoch), {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            since += 1
        if epoch % 10 == 0:
            print(f"  {init}/{coords} epoch {epoch}: val top-1 {v['top1']:.3f} top-5 {v['top5']:.3f}", flush=True)
        if since >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, dict(best_epoch=best[1], val_top1=best[0][0], val_top5=best[0][1], epochs_run=len(curve), curve=curve)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


if __name__ == "__main__":
    tr, va, te = load("train"), load("val"), load("test")
    print(f"clips: train {len(tr)}, val {len(va)}, test {len(te)}", flush=True)
    runs, models, t0 = {}, {}, time.time()
    for init in ("published", "random"):
        for coords in ("signed", "body"):
            model, info = train(tr, va, init, coords)
            runs[f"{init}/{coords}"], models[f"{init}/{coords}"] = info, model
            print(f"{init}/{coords}: best epoch {info['best_epoch']}, val top-1 {info['val_top1']:.3f}, "
                  f"top-5 {info['val_top5']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    chosen = max(runs, key=lambda k: (runs[k]["val_top1"], runs[k]["val_top5"]))
    init, coords = chosen.split("/")
    model = models[chosen]

    # confidence for showing a sign, chosen on validation: lowest with at least half of accepted predictions right
    v = evaluate(model, va, coords)
    grid = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]

    def at(ev, t):
        acc = [c for c, u, conf in zip(ev["correct"], ev["usable"], ev["conf"]) if u and conf >= t]
        return dict(threshold=t, shown=len(acc) / ev["n"], precision=(sum(acc) / len(acc)) if acc else None)
    val_curve = [at(v, t) for t in grid]
    good = [r for r in val_curve if r["precision"] is not None and r["precision"] >= 0.5]
    threshold = good[0]["threshold"] if good else 0.95

    t = evaluate(model, te, coords)          # the only use of the test clips
    k1, k5, n = round(t["top1"] * t["n"]), round(t["top5"] * t["n"]), t["n"]
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "pytorch_model.bin"))
    config = dict(variant="asl100", layout=LAYOUT, coords=coords, class_order="alphabetical", init=init,
                  threshold=threshold, trained_by="backend/scripts/report_experiments/sign_train_tgcn.py",
                  training_data="WLASL100 train clips obtainable from the Voxel51 mirror (WLASL, C-UDA licence)")
    json.dump(config, open(os.path.join(OUT_DIR, "config.json"), "w"), indent=1)
    out = dict(method=__doc__, clips=dict(train=len(tr), val=len(va), test=len(te)),
               runs={k: {x: y for x, y in r.items()} for k, r in runs.items()}, chosen=chosen, config=config,
               val_threshold_curve=val_curve,
               test=dict(top1=dict(n=k1, of=n, rate=t["top1"], ci95=wilson(k1, n)),
                         top5=dict(n=k5, of=n, rate=t["top5"], ci95=wilson(k5, n)),
                         no_prediction=t["no_prediction"], at_threshold=at(t, threshold)))
    json.dump(out, open(os.path.join(SP, "sign_tgcn_train.json"), "w"), indent=1)
    print("chosen on validation:", chosen, "| threshold", threshold, "| test top-1", f"{k1}/{n}", "top-5", f"{k5}/{n}",
          "| shown at threshold", out["test"]["at_threshold"], flush=True)
    print("DONE train", flush=True)
