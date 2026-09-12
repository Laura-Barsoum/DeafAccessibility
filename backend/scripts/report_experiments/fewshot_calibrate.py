"""[Final report: Section 5.2 personal sounds, Table 5.2, Figure 5.2a]

Calibrate the personal-sound decision rules on a DEV split, report on a TEST split.

Why: in a unit-normalised embedding space, squared Euclidean distances lie in
[0, 4], so an untempered softmax over negative distances is nearly flat and a
0.55 probability threshold rejects many correct nearest-prototype decisions.
Snell et al. (2017) work in an unnormalised learned space where distances are
large and the softmax is sharp. A temperature restores that sharpness.

Protocol (K=3 support clips, 3 enrolled sounds per trial, 30 seeded trials):
  DEV  : enrol 3 of the 40 NON-domestic ESC-50 classes; positives are their
         remaining clips; negatives are 10 clips from each other dev class.
  TEST : enrol 3 of the 10 DOMESTIC classes; positives are their remaining
         clips; negatives are all 40 clips of each other domestic class.
No test clip or test class is used to choose any parameter.
Both matchers are tuned on DEV so the comparison is fair:
  cosine        : similarity threshold
  prototypical  : temperature tau, probability threshold, gate multiplier m
                  (gate = max(3*m*radius, 0.20*m); m=1, tau=1 is the shipped rule)
Selection criteria reported: max F1, and max recall subject to FAR <= 5 %.
"""
from pathlib import Path
import csv, json, os, random, sys, time
import numpy as np

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
os.makedirs(SP, exist_ok=True)
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)
from modules import personalizer as pm  # noqa: E402

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

cache = os.path.join(SP, "yamnet_esc50_embeddings.npz")
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True)
    EMB = dict(zip(z["files"].tolist(), z["embs"]))
else:
    import tempfile
    tmp = tempfile.mkdtemp()
    pm.PROFILE_DIR = tmp
    pm.PROFILE_PATH = os.path.join(tmp, "profile.json")
    P = pm.Personalizer()
    t = time.time()
    EMB = {}
    for r in meta:
        e = P._embed(open(os.path.join(ESC, "audio", r["filename"]), "rb").read())
        EMB[r["filename"]] = np.asarray(e, dtype=np.float32)
    assert P._yamnet not in (None, "placeholder"), "YamNet did not load"
    np.savez(cache, files=np.array(list(EMB.keys())), embs=np.stack(list(EMB.values())))
    print(f"embedded 2000 clips in {time.time()-t:.1f}s", flush=True)


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def trials(split, n_trials, K=3, n_enrol=3, seed=0):
    """Yield per-trial arrays: pos (embs, true idx), neg embs, protos, radii, means."""
    classes = DEV if split == "dev" else DOMESTIC
    for t in range(n_trials):
        rng = random.Random(seed + t)
        enrolled = rng.sample(classes, n_enrol)
        protos, radii, means, pos_e, pos_y = [], [], [], [], []
        for i, c in enumerate(enrolled):
            files = list(by[c]); rng.shuffle(files)
            sup = np.stack([EMB[f] for f in files[:K]])
            ns = l2(sup)
            p = l2(ns.mean(axis=0))
            protos.append(p)
            radii.append(float(np.mean(np.sum((ns - p) ** 2, axis=1))))
            means.append(sup.mean(axis=0))
            for f in files[K:]:
                pos_e.append(EMB[f]); pos_y.append(i)
        others = [c for c in classes if c not in enrolled]
        if split == "dev":
            neg_files = [f for c in others for f in by[c][:10]]
        else:
            neg_files = [f for c in others for f in by[c]]
        yield dict(pos=np.stack(pos_e), y=np.array(pos_y), neg=np.stack([EMB[f] for f in neg_files]),
                   P=np.stack(protos), r=np.array(radii), M=np.stack(means))


def score_trials(split, n_trials, seed):
    """Pre-compute distances / similarities once per trial."""
    out = []
    for T in trials(split, n_trials, seed=seed):
        Pn = T["P"]; Mn = l2(T["M"])
        pe, ne = l2(T["pos"]), l2(T["neg"])
        out.append(dict(
            y=T["y"], r=T["r"],
            Dp=2 - 2 * pe @ Pn.T, Dn=2 - 2 * ne @ Pn.T,     # squared euclid on unit vectors
            Sp=pe @ Mn.T, Sn=ne @ Mn.T,                     # cosine to raw mean (shipped cosine rule)
        ))
    return out


def proto_counts(S, tau, pt, m):
    tp = fp = fa = npos = nneg = 0
    for T in S:
        for D, is_pos in ((T["Dp"], True), (T["Dn"], False)):
            lg = -D / tau
            lg = lg - lg.max(axis=1, keepdims=True)
            pr = np.exp(lg); pr /= pr.sum(axis=1, keepdims=True)
            j = D.argmin(axis=1)
            pj = pr[np.arange(len(j)), j]
            gate = np.maximum(T["r"][j] * 3.0 * m, 0.20 * m)
            fire = (pj >= pt) & (D[np.arange(len(j)), j] <= gate)
            if is_pos:
                correct = j == T["y"]
                tp += int(np.sum(fire & correct)); fp += int(np.sum(fire & ~correct)); npos += len(j)
            else:
                fa += int(np.sum(fire)); fp += int(np.sum(fire)); nneg += len(j)
    return tp, fp, fa, npos, nneg


def cos_counts(S, thr):
    tp = fp = fa = npos = nneg = 0
    for T in S:
        Fp = T["Sp"] >= thr
        idx = np.arange(len(T["y"]))
        hit = Fp[idx, T["y"]]
        tp += int(hit.sum()); fp += int(Fp.sum() - hit.sum()); npos += len(idx)
        Fn = T["Sn"] >= thr
        fa += int(Fn.any(axis=1).sum()); fp += int(Fn.sum()); nneg += len(Fn)
    return tp, fp, fa, npos, nneg


def metrics(c):
    tp, fp, fa, npos, nneg = c
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / npos
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(recall=rec, precision=prec, f1=f1, far=fa / nneg, n_pos=npos, n_neg=nneg)


dev = score_trials("dev", 30, seed=5000)
test = score_trials("test", 30, seed=9000)

grid_p = [(tau, pt, m) for tau in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02)
          for pt in (0.40, 0.55, 0.70, 0.85)
          for m in np.round(np.linspace(0.25, 4.0, 16), 3)]
grid_c = np.round(np.linspace(0.50, 0.99, 50), 3)

dev_p = [(g, metrics(proto_counts(dev, *g))) for g in grid_p]
dev_c = [(g, metrics(cos_counts(dev, g))) for g in grid_c]


def pick(rows, crit):
    if crit == "f1":
        return max(rows, key=lambda x: (x[1]["f1"], -x[1]["far"]))
    ok = [x for x in rows if x[1]["far"] <= 0.05]
    return max(ok, key=lambda x: (x[1]["recall"], -x[1]["far"])) if ok else None


res = {"protocol": __doc__, "selection": {}}
shipped_p = (1.0, 0.55, 1.0)
res["shipped"] = dict(
    prototypical=dict(params=shipped_p, dev=metrics(proto_counts(dev, *shipped_p)), test=metrics(proto_counts(test, *shipped_p))),
    cosine=dict(params=0.90, dev=metrics(cos_counts(dev, 0.90)), test=metrics(cos_counts(test, 0.90))),
)
for crit in ("f1", "far<=5%"):
    bp, bc = pick(dev_p, crit), pick(dev_c, crit)
    res["selection"][crit] = dict(
        prototypical=dict(params=[float(v) for v in bp[0]], dev=bp[1], test=metrics(proto_counts(test, *bp[0]))),
        cosine=dict(params=float(bc[0]), dev=bc[1], test=metrics(cos_counts(test, bc[0]))),
    )

# test-set operating curves for the figure (parameters swept, not chosen, on test)
tau_f1 = res["selection"]["f1"]["prototypical"]["params"][0]
pt_f1 = res["selection"]["f1"]["prototypical"]["params"][1]
res["curves_test"] = dict(
    cosine=[dict(param=float(g), **metrics(cos_counts(test, g))) for g in np.round(np.linspace(0.30, 0.995, 60), 3)],
    prototypical_shipped=[dict(param=float(m), **metrics(proto_counts(test, 1.0, 0.55, m))) for m in np.round(np.geomspace(0.05, 12, 60), 3)],
    prototypical_calibrated=[dict(param=float(m), **metrics(proto_counts(test, tau_f1, pt_f1, m))) for m in np.round(np.geomspace(0.05, 12, 60), 3)],
)

# why the shipped rule misses: where do correct nearest-prototype decisions get rejected?
rej_prob = rej_gate = correct_nearest = 0
for T in test:
    D = T["Dp"]; lg = -D; lg = lg - lg.max(axis=1, keepdims=True)
    pr = np.exp(lg); pr /= pr.sum(axis=1, keepdims=True)
    j = D.argmin(axis=1); ok = j == T["y"]
    pj = pr[np.arange(len(j)), j]; gate = np.maximum(T["r"][j] * 3.0, 0.20)
    within = D[np.arange(len(j)), j] <= gate
    correct_nearest += int(ok.sum())
    rej_prob += int(np.sum(ok & (pj < 0.55)))
    rej_gate += int(np.sum(ok & (pj >= 0.55) & ~within))
res["shipped_rejection_breakdown_test"] = dict(
    correct_nearest=correct_nearest, rejected_by_probability=rej_prob, rejected_by_gate_only=rej_gate,
    median_top_probability=float(np.median(np.concatenate([
        (lambda D: (lambda pr: pr.max(axis=1))(np.exp(-D - (-D).max(axis=1, keepdims=True)) /
                                               np.exp(-D - (-D).max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)))(T["Dp"])
        for T in test]))),
)
json.dump(res, open(os.path.join(SP, "fewshot_tune.json"), "w"), indent=1, default=float)

def line(name, m):
    return f"{name:42s} R={m['recall']:.3f} P={m['precision']:.3f} F1={m['f1']:.3f} FAR={m['far']:.3f}"
print(line("TEST shipped prototypical (tau=1, pt=.55, m=1)", res["shipped"]["prototypical"]["test"]))
print(line("TEST shipped cosine (0.90)", res["shipped"]["cosine"]["test"]))
for crit, d in res["selection"].items():
    print(f"-- selected on DEV by {crit}: proto params {d['prototypical']['params']}  cosine thr {d['cosine']['params']}")
    print(line("   TEST prototypical", d["prototypical"]["test"]))
    print(line("   TEST cosine", d["cosine"]["test"]))
print("rejection breakdown (shipped, test):", res["shipped_rejection_breakdown_test"])
print("DONE tune", flush=True)
