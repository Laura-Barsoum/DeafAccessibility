"""[Final report: Section 5.2 personal sounds, Figure 5.2b]

Few-shot personal-sound evaluation on ESC-50 (no new recordings needed).

Compares the deployed cosine matcher against the deployed prototypical
matcher, under the YamNet embedding and the spectral fallback embedding,
by calling the REAL Personalizer.enrol / match / match_prototypical methods
with pre-computed embeddings patched in (so the decision code under test is
exactly the shipped code).

Protocol (per trial, 20 trials, seeded):
  - enrol 3 of the 10 ESC-50 "domestic" classes with K support clips each
  - positives: the remaining clips of the enrolled classes
  - negatives: every clip of the 7 non-enrolled domestic classes (40 each)
    plus 10 clips from each of the 40 non-domestic classes (open-set case)
Metrics: recall, precision, F1, false-acceptance rate (FAR, fraction of
negative clips that fire any alert), closed-set accuracy.
Also sweeps each matcher's decision threshold to trace recall vs FAR.
"""
from pathlib import Path
import csv, json, math, os, random, sys, tempfile, time
import numpy as np

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
os.makedirs(SP, exist_ok=True)
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from modules import personalizer as pm          # noqa: E402
from modules.events import Priority             # noqa: E402

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
OTHERS = sorted(c for c in by if c not in DOMESTIC)
clips = [(c, f) for c in DOMESTIC for f in by[c]] + \
        [(c, f) for c in OTHERS for f in by[c][:10]]

tmp = tempfile.mkdtemp(prefix="fewshot_")
pm.PROFILE_DIR = tmp
pm.PROFILE_PATH = os.path.join(tmp, "profile.json")
P = pm.Personalizer()
P.profile = {"sounds": {}}
real_embed = P._embed


def embed_all(fn):
    out = {}
    for c, f in clips:
        b = open(os.path.join(ESC, "audio", f), "rb").read()
        e = fn(b)
        out[f] = None if e is None else np.asarray(e, dtype=np.float32)
    return out


t = time.time()
EMB = {"yamnet": embed_all(real_embed)}
t_yam = time.time() - t
if P._yamnet in (None, "placeholder"):
    raise SystemExit("YamNet did not load; refusing to report fallback as YamNet")
t = time.time()
EMB["fallback"] = embed_all(P._embed_fallback)
t_fb = time.time() - t
print(f"embedded {len(clips)} clips: yamnet {t_yam:.1f}s, fallback {t_fb:.1f}s", flush=True)


def l2(a):
    return a / (np.linalg.norm(a) + 1e-9)


def run(emb_name, method, K, trials=20, n_enrol=3):
    E = EMB[emb_name]
    P._embed = lambda key: E[key.decode()]
    pooled = dict(tp=0, fp=0, fa=0, npos=0, nneg=0, closed_ok=0)
    per_trial = []
    # sweep accumulators (score-replicated, validated against method calls)
    cos_grid = np.round(np.linspace(0.30, 1.00, 71), 3)
    mult_grid = np.round(np.concatenate([np.linspace(0.05, 1.0, 20), np.linspace(1.25, 6.0, 20)]), 3)
    sweep = {"cosine": {float(g): [0, 0, 0] for g in cos_grid},          # tp, fp, fa
             "prototypical": {float(g): [0, 0, 0] for g in mult_grid}}
    for t in range(trials):
        rng = random.Random(1000 + t)
        enrolled = rng.sample(DOMESTIC, n_enrol)
        P.profile = {"sounds": {}}
        queries = []
        for c in enrolled:
            files = [f for f in by[c] if E.get(f) is not None]
            rng.shuffle(files)
            P.enrol(c, Priority.IMPORTANT, [f.encode() for f in files[:K]])
            queries += [(c, f) for f in files[K:]]
        negs = [(c, f) for c, f in clips if c not in enrolled and E.get(f) is not None]
        S = P.profile["sounds"]
        labs = list(S.keys())
        means = [np.array(S[l]["embedding"], dtype=np.float32) for l in labs]
        protos = [np.array(S[l]["prototype"], dtype=np.float32) for l in labs]
        radii = [float(S[l]["radius"]) for l in labs]

        tp = fp = fa = closed_ok = 0
        for is_pos, (c, f) in [(True, q) for q in queries] + [(False, n) for n in negs]:
            ev = P.match(f.encode()) if method == "cosine" else P.match_prototypical(f.encode())
            fired = [e.label.replace(" (personal)", "") for e in ev]
            if is_pos:
                if c in fired:
                    tp += 1
                fp += sum(1 for x in fired if x != c)
            else:
                fa += 1 if fired else 0
                fp += len(fired)

            # replicated scores for the sweep
            raw = E[f]
            sims = [float(np.dot(l2(raw), l2(m))) for m in means]
            e = l2(raw)
            d = np.array([float(np.dot(e - p, e - p)) for p in protos])
            lg = -d - (-d).max()
            pr = np.exp(lg); pr /= pr.sum()
            j = int(np.argmax(pr))
            if is_pos:
                nearest = labs[int(np.argmax(sims))] if method == "cosine" else labs[j]
                closed_ok += 1 if nearest == c else 0
            for g in sweep["cosine"]:
                fl = [labs[i] for i, s in enumerate(sims) if s >= g]
                acc = sweep["cosine"][g]
                if is_pos:
                    acc[0] += 1 if c in fl else 0
                    acc[1] += sum(1 for x in fl if x != c)
                else:
                    acc[2] += 1 if fl else 0
                    acc[1] += len(fl)
            for m in sweep["prototypical"]:
                gate = max(radii[j] * 3.0 * m, 0.20 * m)
                fires = pr[j] >= 0.55 and d[j] <= gate
                acc = sweep["prototypical"][m]
                if is_pos:
                    acc[0] += 1 if (fires and labs[j] == c) else 0
                    acc[1] += 1 if (fires and labs[j] != c) else 0
                else:
                    acc[2] += 1 if fires else 0
                    acc[1] += 1 if fires else 0
        npos, nneg = len(queries), len(negs)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / npos
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_trial.append(dict(enrolled=enrolled, recall=rec, precision=prec, f1=f1,
                              far=fa / nneg, closed_acc=closed_ok / npos))
        for k, v in dict(tp=tp, fp=fp, fa=fa, npos=npos, nneg=nneg, closed_ok=closed_ok).items():
            pooled[k] += v

    def ms(key):
        xs = [r[key] for r in per_trial]
        return [float(np.mean(xs)), float(np.std(xs))]

    prec = pooled["tp"] / (pooled["tp"] + pooled["fp"]) if (pooled["tp"] + pooled["fp"]) else 0.0
    rec = pooled["tp"] / pooled["npos"]
    curve = []
    fam = "cosine" if method == "cosine" else "prototypical"
    for g, (stp, sfp, sfa) in sorted(sweep[fam].items()):
        p_ = stp / (stp + sfp) if (stp + sfp) else 0.0
        r_ = stp / pooled["npos"]
        curve.append(dict(param=g, recall=r_, precision=p_, far=sfa / pooled["nneg"]))
    deployed = 0.90 if method == "cosine" else 1.0
    rep = next(x for x in curve if abs(x["param"] - deployed) < 1e-6)
    return dict(
        embedding=emb_name, method=method, K=K, trials=trials,
        mean_sd=dict(recall=ms("recall"), precision=ms("precision"), f1=ms("f1"),
                     far=ms("far"), closed_acc=ms("closed_acc")),
        pooled=dict(recall=rec, precision=prec,
                    f1=(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
                    far=pooled["fa"] / pooled["nneg"], closed_acc=pooled["closed_ok"] / pooled["npos"],
                    n_pos=pooled["npos"], n_neg=pooled["nneg"]),
        replicated_at_deployed=rep,
        curve=curve,
    )


results = []
for emb_name in ("yamnet", "fallback"):
    for method in ("cosine", "prototypical"):
        for K in (1, 3, 5):
            r = run(emb_name, method, K)
            pm_ = r["pooled"]
            print(f"{emb_name:8s} {method:12s} K={K}  recall={pm_['recall']:.3f} "
                  f"precision={pm_['precision']:.3f} F1={pm_['f1']:.3f} FAR={pm_['far']:.3f} "
                  f"closed={pm_['closed_acc']:.3f} | replicated R={r['replicated_at_deployed']['recall']:.3f} "
                  f"FAR={r['replicated_at_deployed']['far']:.3f}", flush=True)
            results.append(r)

out = dict(
    protocol="ESC-50 domestic classes; 3 enrolled per trial; 20 seeded trials; negatives = 7 other domestic x40 + 40 non-domestic x10",
    n_clips=len(clips), embed_seconds=dict(yamnet=t_yam, fallback=t_fb),
    results=results,
)
json.dump(out, open(os.path.join(SP, "fewshot_results.json"), "w"), indent=1)
print("DONE fewshot", flush=True)
