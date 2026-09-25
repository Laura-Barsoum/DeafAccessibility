"""[Final report: Section 5.2 people, Table 3.1, Table 5.1]

Face identification through the application's own registry, on Labeled Faces in
the Wild (Huang et al., 2007), the standard public benchmark for this task.

The application enrols a person from camera frames and later names the face it
sees (people.PeopleRegistry.enrol / identify_face, DeepFace VGG-Face embeddings
compared by cosine similarity against a fixed threshold). Nothing here reaches
into those internals: identities are enrolled through enrol() from JPEG frames
and scored through identify_face(), so what is measured is the deployed path.

Protocol. Identities with at least five photographs are split into a development
half and a test half, disjoint by person. In each half, 30 identities are enrolled
from 3 photographs each and probed with 2 held-out photographs; the same number of
photographs of people who were never enrolled act as impostors, because a home
assistant sees strangers far more often than it sees the four people it knows.
The acceptance threshold is chosen on the development half alone, as the value
that maximises correct decisions there, and the test half is scored once, at both
that value and the application's current 0.35.

Writes eval_results/face_lfw.json. The LFW archive is not committed
(backend/data/lfw, about 173 MB).
"""
from pathlib import Path
import base64
import json
import os
import random
import sys
import time

import numpy as np

BACKEND = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import cv2  # noqa: E402
from modules import people as pm  # noqa: E402

LFW = Path("data/lfw")
N_IDENTITIES = 30          # enrolled per half
N_ENROL = 3                # photographs used to enrol each identity
N_PROBE = 2                # held-out photographs of each enrolled identity
SEED = 0

people_dirs = sorted(p for p in LFW.iterdir() if p.is_dir())
assert people_dirs, f"no LFW identities under {LFW.resolve()}"
by_person = {p.name: sorted(str(f) for f in p.glob("*.jpg")) for p in people_dirs}
rich = sorted(n for n, f in by_person.items() if len(f) >= N_ENROL + N_PROBE)
singles = sorted(n for n, f in by_person.items() if len(f) == 1)
rng = random.Random(SEED)
rng.shuffle(rich)
rng.shuffle(singles)
assert len(rich) >= 2 * N_IDENTITIES, f"only {len(rich)} identities with enough photographs"

HALVES = {"development": (rich[:N_IDENTITIES], singles[:N_IDENTITIES * N_PROBE]),
          "test": (rich[N_IDENTITIES:2 * N_IDENTITIES], singles[N_IDENTITIES * N_PROBE:2 * N_IDENTITIES * N_PROBE])}


def frame_b64(path):
    img = cv2.imread(path)
    assert img is not None, path
    return base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()


def registry(tmp_name):
    """A registry with its own profile file, so nothing touches the user's."""
    pm.PROFILE_DIR = os.path.join("data", "eval_people")
    pm.PROFILE_PATH = os.path.join(pm.PROFILE_DIR, f"{tmp_name}.json")
    os.makedirs(pm.PROFILE_DIR, exist_ok=True)
    if os.path.exists(pm.PROFILE_PATH):
        os.unlink(pm.PROFILE_PATH)
    pm._registry_singleton = None
    reg = pm.get_people()
    reg.profile = {"people": {}, "self_name": None}
    return reg


def best_match(reg, b64):
    """identify_face without its threshold, so one pass scores every threshold."""
    emb = reg._face_embedding(b64)
    if emb is None:
        return None, None
    best_name, best_sim = None, -1.0
    for n, info in reg.profile["people"].items():
        ref = np.array(info["face_embedding"], dtype=np.float32)
        sim = pm._cosine(emb, ref)
        if sim > best_sim:
            best_sim, best_name = sim, n
    return best_name, best_sim


def run_half(name):
    enrolled, impostor_people = HALVES[name]
    reg = registry(name)
    t0 = time.time()
    n_enrolled = 0
    for person in enrolled:
        frames = [frame_b64(f) for f in by_person[person][:N_ENROL]]
        res = reg.enrol(person, frames)
        n_enrolled += 1 if res.get("ok") else 0
    print(f"{name}: enrolled {n_enrolled}/{len(enrolled)} in {time.time() - t0:.0f}s", flush=True)

    trials = []
    for person in enrolled:
        for f in by_person[person][N_ENROL:N_ENROL + N_PROBE]:
            nm, sim = best_match(reg, frame_b64(f))
            trials.append(dict(kind="genuine", truth=person, best=nm, sim=sim, photo=f))
    for person in impostor_people:
        f = by_person[person][0]
        nm, sim = best_match(reg, frame_b64(f))
        trials.append(dict(kind="impostor", truth=None, best=nm, sim=sim, photo=f))
    print(f"{name}: {len(trials)} trials in {time.time() - t0:.0f}s", flush=True)
    return trials


def score(trials, threshold):
    gen = [t for t in trials if t["kind"] == "genuine"]
    imp = [t for t in trials if t["kind"] == "impostor"]
    accepted = lambda t: t["sim"] is not None and t["sim"] >= threshold          # noqa: E731
    correct = sum(1 for t in gen if accepted(t) and t["best"] == t["truth"])
    wrong = sum(1 for t in gen if accepted(t) and t["best"] != t["truth"])
    missed = len(gen) - correct - wrong
    false_accept = sum(1 for t in imp if accepted(t))
    return dict(
        threshold=round(threshold, 3),
        genuine=len(gen), impostors=len(imp),
        correct=correct, correct_rate=correct / len(gen) if gen else 0.0,
        wrong_name=wrong, wrong_name_rate=wrong / len(gen) if gen else 0.0,
        no_match=missed, no_match_rate=missed / len(gen) if gen else 0.0,
        false_accept=false_accept, false_accept_rate=false_accept / len(imp) if imp else 0.0,
        correct_decisions=(correct + (len(imp) - false_accept)) / (len(gen) + len(imp)),
    )


dev_trials = run_half("development")
GRID = [round(x, 2) for x in np.arange(0.20, 0.91, 0.05)]
dev_curve = [score(dev_trials, t) for t in GRID]
chosen = max(dev_curve, key=lambda r: r["correct_decisions"])["threshold"]
print("chosen on development:", chosen, flush=True)

test_trials = run_half("test")
out = dict(
    method=__doc__,
    dataset="Labeled Faces in the Wild, funneled-free originals (data/lfw, not committed)",
    identities=dict(with_enough_photos=len(rich), enrolled_per_half=N_IDENTITIES,
                    enrol_photos=N_ENROL, probe_photos=N_PROBE, impostor_identities=N_IDENTITIES * N_PROBE),
    app_threshold=float(pm.FACE_THRESHOLD),
    development=dict(curve=dev_curve, chosen_threshold=chosen),
    test=dict(at_app_threshold=score(test_trials, float(pm.FACE_THRESHOLD)),
              at_chosen_threshold=score(test_trials, chosen)),
    trials=dict(development=dev_trials, test=test_trials),
)
json.dump(out, open(os.path.join(BACKEND, "eval_results", "face_lfw.json"), "w"), indent=1)
print("app threshold :", out["test"]["at_app_threshold"])
print("chosen        :", out["test"]["at_chosen_threshold"])
print("DONE face", flush=True)
