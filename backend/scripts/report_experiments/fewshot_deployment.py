"""[Final report: Section 5.2 personal sounds]

Personal-sound settings for AST embeddings as the application meets them.

fewshot_embeddings.py chose AST over YamNet on whole, clean ESC-50 clips with
three sounds enrolled at once. In the scripted fusion scenes those settings
matched a single enrolled alarm on every tick. With one enrolled sound the
softmax probability is always 1, so only the distance gate can reject, and a
gate chosen with three sounds enrolled is too loose on its own; the scenes also
deliver Opus-coded 2.8-second chunks, not clean five-second clips.

Here every ESC-50 clip is first prepared as the browser delivers a tick (the
loudest 2.8 seconds, levelled as fusion_scenes.py levels sound, Opus at
32 kbit/s, decoded by the server's own converter) and then embedded by AST.
Temperature, probability threshold and gate multiplier are chosen on the DEV
split (40 non-household classes) as the highest mean recall with a false-alarm
rate of at most 5% both with one and with three sounds enrolled, and only then
applied to the TEST split (10 household classes). Nothing is chosen on the test
split or on the fusion scenes. The earlier AST settings are scored on the same
trials for comparison. Writes eval_results/fewshot_deployment.json.
"""
from pathlib import Path
import concurrent.futures as cf
import csv, json, os, random, subprocess, sys, tempfile, time

import numpy as np
import soundfile as sf

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ESC = os.path.join(BACKEND, "data", "ESC-50")
meta = list(csv.DictReader(open(os.path.join(ESC, "meta", "esc50.csv"))))
DOMESTIC = ["door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks",
            "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm",
            "clock_tick", "glass_breaking"]
by = {}
for r in meta:
    by.setdefault(r["category"], []).append(r["filename"])
for c in by:
    by[c].sort()
DEV = sorted(c for c in by if c not in DOMESTIC)
SR, TICK_S = 16000, 2.8
PREVIOUS = json.load(open(os.path.join(SP, "fewshot_embeddings.json")))["results"]["ast_K3"]["params"]


# ── clips as the browser delivers a tick ────────────────────────────────────
TMP = tempfile.mkdtemp(prefix="fewshot_tick_")


def sh(*a):
    subprocess.run(a, check=True, capture_output=True)


def loudest(x, n):
    if len(x) <= n:
        return np.pad(x, (0, n - len(x)))
    starts = range(0, len(x) - n + 1, SR // 10)
    i = max(starts, key=lambda s: float(np.mean(x[s:s + n] ** 2)))
    return x[i:i + n]


def level(x, target=0.05):
    active = x[np.abs(x) > 0.01]
    rms = float(np.sqrt(np.mean(active ** 2))) if active.size else 0.0
    return x * (target / rms) if rms > 0 else x


def tick_webm(fname):
    base = os.path.join(TMP, fname[:-4])
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(ESC, "audio", fname),
       "-ar", str(SR), "-ac", "1", base + "_16k.wav")
    x, _ = sf.read(base + "_16k.wav", dtype="float32")
    x = level(loudest(x, int(TICK_S * SR)))
    peak = float(np.max(np.abs(x))) or 1.0
    if peak > 0.98:
        x = x * (0.98 / peak)
    sf.write(base + "_tick.wav", x, SR, subtype="PCM_16")
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", base + "_tick.wav", "-c:a", "libopus", "-b:a", "32k", base + ".webm")
    return open(base + ".webm", "rb").read()


cache = os.path.join(SP, "ast_esc50_tick_embeddings.npz")
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True)
    AST = dict(zip(z["files"].tolist(), z["embs"]))
else:
    import server
    files, t0 = [r["filename"] for r in meta], time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        wavs = dict(zip(files, ex.map(lambda f: server._ffmpeg_convert(tick_webm(f)), files)))
    print(f"prepared {len(wavs)} ticks in {time.time() - t0:.0f}s", flush=True)
    clf = server.get_audio_scene()
    clf._ensure_loaded()
    assert clf._backend == "ast", f"AST did not load ({clf.status()})"
    AST = {}
    for k, f in enumerate(files):
        emb, _speech = clf.embed_and_speech(wavs[f])
        assert emb is not None, f"no embedding for {f}"
        AST[f] = np.asarray(emb, dtype=np.float32)
        if k % 250 == 0:
            print(f"AST {k}/{len(files)} ticks, {time.time() - t0:.0f}s", flush=True)
    np.savez(cache, files=np.array(list(AST)), embs=np.stack(list(AST.values())))


# ── protocol, as in fewshot_embeddings.py, with the number enrolled varied ─
def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def trials(EMB, split, n_trials, K, seed, n_enrol):
    classes = DEV if split == "dev" else DOMESTIC
    for t in range(n_trials):
        rng = random.Random(seed + t)
        enrolled = rng.sample(classes, n_enrol)
        protos, radii, pos_e, pos_y = [], [], [], []
        for i, c in enumerate(enrolled):
            files = list(by[c]); rng.shuffle(files)
            ns = l2(np.stack([EMB[f] for f in files[:K]]))
            p = l2(ns.mean(axis=0))
            protos.append(p)
            radii.append(float(np.mean(np.sum((ns - p) ** 2, axis=1))))
            for f in files[K:]:
                pos_e.append(EMB[f]); pos_y.append(i)
        others = [c for c in classes if c not in enrolled]
        neg_files = [f for c in others for f in (by[c][:10] if split == "dev" else by[c])]
        P = np.stack(protos)
        pe, ne = l2(np.stack(pos_e)), l2(np.stack([EMB[f] for f in neg_files]))
        yield dict(y=np.array(pos_y), r=np.array(radii), Dp=2 - 2 * pe @ P.T, Dn=2 - 2 * ne @ P.T)


def counts_one(T, tau, pt, m):
    """The personaliser's decision rule: gate = max(radius * 3m, 0.2m)."""
    out = []
    for D in (T["Dp"], T["Dn"]):
        lg = -D / tau
        lg = lg - lg.max(axis=1, keepdims=True)
        pr = np.exp(lg); pr /= pr.sum(axis=1, keepdims=True)
        j = D.argmin(axis=1)
        pj = pr[np.arange(len(j)), j]
        gate = np.maximum(T["r"][j] * 3.0 * m, 0.20 * m)
        out.append(((pj >= pt) & (D[np.arange(len(j)), j] <= gate), j))
    (fp_fire, jp), (fn_fire, _jn) = out
    correct = jp == T["y"]
    tp = int(np.sum(fp_fire & correct))
    fp = int(np.sum(fp_fire & ~correct)) + int(np.sum(fn_fire))
    return np.array([tp, fp, int(np.sum(fn_fire)), len(jp), len(fn_fire)])


def metrics(c):
    tp, fp, fa, npos, nneg = (int(v) for v in c)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / npos
    return dict(recall=rec, precision=prec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                far=fa / nneg, n_pos=npos, n_neg=nneg)


def per_trial(S, params):
    return np.stack([counts_one(T, *params) for T in S])


def boot(per, fn, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    vals = [fn(per[rng.integers(0, len(per), len(per))].sum(axis=0)) for _ in range(n_boot)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


ENROLLED = (1, 3)
GRID = [(tau, pt, float(m)) for tau in (1.0, 0.5, 0.2, 0.1, 0.05)
        for pt in (0.40, 0.55, 0.70, 0.85)
        for m in np.round(np.geomspace(0.05, 4.0, 40), 4)]
dev = {n: list(trials(AST, "dev", 30, 3, 5000, n)) for n in ENROLLED}
test = {n: list(trials(AST, "test", 30, 3, 9000, n)) for n in ENROLLED}

ok = []
for g in GRID:
    dm = {n: metrics(per_trial(dev[n], g).sum(axis=0)) for n in ENROLLED}
    if all(dm[n]["far"] <= 0.05 for n in ENROLLED):
        ok.append((g, dm))
assert ok, "no setting keeps false alarms at 5% on the development split"
params, dev_m = max(ok, key=lambda x: (np.mean([x[1][n]["recall"] for n in ENROLLED]),
                                       -max(x[1][n]["far"] for n in ENROLLED)))


def scored(g):
    out = {}
    for n in ENROLLED:
        per = per_trial(test[n], g)
        out[f"enrolled_{n}"] = dict(
            dev=metrics(per_trial(dev[n], g).sum(axis=0)), test=metrics(per.sum(axis=0)),
            test_95={k: boot(per, lambda c, k=k: metrics(c)[k]) for k in ("recall", "precision", "f1", "far")})
    return out


out = dict(
    method=__doc__,
    chosen=dict(params=[float(v) for v in params], temperature=float(params[0]), prob_threshold=float(params[1]),
                radius_k=round(3.0 * params[2], 4), gate_floor=round(0.2 * params[2], 4), **scored(params)),
    previous=dict(params=[float(v) for v in PREVIOUS], **scored(tuple(PREVIOUS))),
)
json.dump(out, open(os.path.join(SP, "fewshot_deployment.json"), "w"), indent=1)
for name in ("chosen", "previous"):
    print(name, out[name]["params"], {k: {s: {m: round(v, 3) for m, v in out[name][k][s].items() if m in ("recall", "far")}
                                          for s in ("dev", "test")} for k in ("enrolled_1", "enrolled_3")})
print("DONE deployment", flush=True)
