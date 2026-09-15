"""[Final report: Section 5.2 personal sounds]

Which embedding and how many enrolment clips give the best personal-sound
operating point? Compares the shipped YamNet embedding with AST's pooled
embedding (the same forward pass the sound classifier already runs every tick),
at three and five enrolment clips, under the calibration protocol of
fewshot_calibrate.py.

For every (embedding, K) the prototypical rule's temperature, probability
threshold and gate multiplier are chosen on the DEV split (40 non-household
classes) as the highest recall with a false-alarm rate of at most 5%, and only
then applied to the TEST split (10 household classes). Nothing is chosen on test.
Test results carry 95% bootstrap intervals over the 30 trials, and the adopted
choice is compared with the shipped YamNet, K=3 rule by a paired bootstrap on
the same trials.

AST's speech-family probability for every ESC-50 clip is saved as well; none of
ESC-50 is speech, so it sets the transcript gate used by fusion.
Writes eval_results/fewshot_embeddings.json.
"""
from pathlib import Path
import csv, json, os, random, sys, time

import numpy as np

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

# ── embeddings ─────────────────────────────────────────────────────────────
yam_cache = os.path.join(SP, "yamnet_esc50_embeddings.npz")
assert os.path.exists(yam_cache), "run fewshot_calibrate.py first: it caches the YamNet embeddings"
z = np.load(yam_cache, allow_pickle=True)
YAM = dict(zip(z["files"].tolist(), z["embs"]))

ast_cache = os.path.join(SP, "ast_esc50_embeddings.npz")
if os.path.exists(ast_cache):
    z = np.load(ast_cache, allow_pickle=True)
    AST = dict(zip(z["files"].tolist(), z["embs"]))
    SPEECH = dict(zip(z["files"].tolist(), z["speech"].tolist()))
else:
    from modules.audio_scene import AudioSceneClassifier
    clf = AudioSceneClassifier()
    clf._ensure_loaded()
    assert clf._backend == "ast", f"AST did not load ({clf.status()})"
    AST, SPEECH, t = {}, {}, time.time()
    for k, r in enumerate(meta):
        emb, speech = clf.embed_and_speech(open(os.path.join(ESC, "audio", r["filename"]), "rb").read())
        AST[r["filename"]] = np.asarray(emb, dtype=np.float32)
        SPEECH[r["filename"]] = float(speech)
        if k % 250 == 0:
            print(f"AST {k}/2000 clips, {time.time() - t:.0f}s", flush=True)
    np.savez(ast_cache, files=np.array(list(AST)), embs=np.stack(list(AST.values())),
             speech=np.array([SPEECH[f] for f in AST]))


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def trials(EMB, split, n_trials, K, seed, n_enrol=3):
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
    """Same decision rule as fewshot_calibrate.proto_counts, for one trial."""
    out = []
    for D, is_pos in ((T["Dp"], True), (T["Dn"], False)):
        lg = -D / tau
        lg = lg - lg.max(axis=1, keepdims=True)
        pr = np.exp(lg); pr /= pr.sum(axis=1, keepdims=True)
        j = D.argmin(axis=1)
        pj = pr[np.arange(len(j)), j]
        gate = np.maximum(T["r"][j] * 3.0 * m, 0.20 * m)
        fire = (pj >= pt) & (D[np.arange(len(j)), j] <= gate)
        out.append((fire, j))
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


GRID = [(tau, pt, m) for tau in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02)
        for pt in (0.40, 0.55, 0.70, 0.85)
        for m in np.round(np.linspace(0.25, 4.0, 16), 3)]


def per_trial(S, params):
    return np.stack([counts_one(T, *params) for T in S])


def boot(per, fn, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    vals = [fn(per[rng.integers(0, len(per), len(per))].sum(axis=0)) for _ in range(n_boot)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


results = {}
PER = {}
for emb_name, EMB in (("yamnet", YAM), ("ast", AST)):
    for K in (3, 5):
        dev = list(trials(EMB, "dev", 30, K, seed=5000))
        test = list(trials(EMB, "test", 30, K, seed=9000))
        dev_rows = [(g, metrics(per_trial(dev, g).sum(axis=0))) for g in GRID]
        ok = [x for x in dev_rows if x[1]["far"] <= 0.05]
        params, dev_m = max(ok, key=lambda x: (x[1]["recall"], -x[1]["far"]))
        per = per_trial(test, params)
        PER[(emb_name, K)] = per
        tm = metrics(per.sum(axis=0))
        results[f"{emb_name}_K{K}"] = dict(
            params=[float(v) for v in params], dev=dev_m, test=tm,
            test_95={k: boot(per, lambda c, k=k: metrics(c)[k]) for k in ("recall", "precision", "f1", "far")},
        )
        print(f"{emb_name} K={K}: params {params} | dev R={dev_m['recall']:.3f} FAR={dev_m['far']:.3f} "
              f"| test R={tm['recall']:.3f} P={tm['precision']:.3f} F1={tm['f1']:.3f} FAR={tm['far']:.3f}", flush=True)

# the choice is made on DEV recall at the 5% false-alarm constraint, never on test
chosen = max(results, key=lambda k: (results[k]["dev"]["recall"], -results[k]["dev"]["far"]))
ship = PER[("yamnet", 3)]
emb_name, K = chosen.split("_K")[0], int(chosen.split("_K")[1])
cand = PER[(emb_name, K)]
paired = {}
if cand.shape == ship.shape:  # same trials only when K matches the shipped rule's query set
    rng = np.random.default_rng(2)
    for key in ("recall", "f1", "far"):
        d = []
        for _ in range(2000):
            idx = rng.integers(0, len(ship), len(ship))
            d.append(metrics(cand[idx].sum(axis=0))[key] - metrics(ship[idx].sum(axis=0))[key])
        paired[key] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

speech = np.array(list(SPEECH.values()))
out = dict(
    method=__doc__, results=results, chosen_on_dev=chosen, paired_vs_shipped_95=paired,
    ast_speech_probability_esc50=dict(
        percentiles={p: float(np.percentile(speech, p)) for p in (50, 90, 95, 99)},
        n=len(speech)),
)
json.dump(out, open(os.path.join(SP, "fewshot_embeddings.json"), "w"), indent=1)
print("chosen on dev:", chosen, "| paired vs shipped:", paired)
print("AST speech probability on ESC-50 percentiles:", out["ast_speech_probability_esc50"]["percentiles"])
print("DONE embeddings", flush=True)
