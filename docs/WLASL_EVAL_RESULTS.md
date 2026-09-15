# WLASL Evaluation Results

## Update, September 2026: official test clips, two silent defects and a retrained TGCN

This supersedes the May 2026 run further down, which is kept for the record.

**Clips.** 100 of the 258 official WLASL100 test clips, 744 of the 1,442
training clips and 163 of the 338 validation clips could still be obtained from
the Voxel51 mirror on Hugging Face. Each is cut to its annotated frame range.
None is committed (`backend/data/WLASL-master/videos_test100/`,
`videos_trainval100/`).

**Defect 1: a network no checkpoint fits.** `tgcn_sign_model.py` defined an
attention block, an extra graph convolution and a flattening classifier that no
published checkpoint contains, and loaded with `strict=False`, so those layers
would have kept random weights. The network now matches the released
`GCN_muti_att` key for key, and a checkpoint that does not fit is refused.

**Defect 2: no hands.** MediaPipe's Tasks API gives hand landmarks
`visibility=None`. The adapter in `sign_language.py` converted it with
`float()`, which raised inside a catch-all, so both hand slots were empty on
every frame and every hand-based sign tier ran on body pose alone. An earlier
version of this page concluded, from runs with this defect, that the published
weights were at chance on MediaPipe keypoints; that conclusion was wrong. The
adapter is now None-safe, with a regression test.

**Published weights, with hands.** The asl100 checkpoint at
`huggingface.co/sharonn18/tgcn-wlasl` expects OpenPose keypoints. Its keypoint
order (hands swapped), coordinates ([-1, 1]) and class order (alphabetical)
were chosen on 14 older local clips outside the test split.

**Retrained on MediaPipe keypoints.** `sign_train_tgcn.py` trains the same
network on keypoints extracted by the application's own adapter
(`sign_keypoints.py`). The starting weights (published or random), the
coordinates (image, or normalised to the signer's shoulders), the epoch and the
display threshold were all chosen on the validation clips: random start,
body-normalised keypoints, 60.1% validation top-1. The test clips were then
scored once.

| Configuration, 100 test clips | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| Sign cascade without the TGCN tier | 1% | 1% | 1% |
| Published weights, model alone | 8% | not recorded | 31% |
| Retrained weights, model alone | 60% (95% Wilson interval 50% to 69%) | 72% | 78% (69% to 85%) |
| Retrained weights, full sign cascade (as shipped) | 21% (14% to 30%) | 41% | 56% |

The full cascade keeps far fewer correct answers than the model because it
puts the TGCN's word first only when that word is more confident than the
curated hand-shape and gesture tiers, which cover everyday signs such as "ok"
and "hello" that are not among the 100 WLASL signs. In 38 of the 59 test clips
where the TGCN's answer was right, one of those labels came first. Letting the
TGCN lead would raise the word-level score but would mislabel signs outside its
vocabulary; that trade-off has not been measured.

The retrained weights and their settings are in
`backend/data/tgcn/asl100_mediapipe/`, and the tier runs by default when they
are present (`ACCESSIBILITY_ENABLE_TGCN=0` turns it off). They cover only the
100 WLASL100 signs and were trained and tested on WLASL's signers, a narrow
population; webcam signing by Deaf users is untested.

Commands, from `backend/`:
```
python scripts/report_experiments/sign_wlasl100.py --select
python scripts/report_experiments/sign_wlasl100.py
python scripts/report_experiments/sign_keypoints.py
python scripts/report_experiments/sign_train_tgcn.py
python scripts/report_experiments/sign_wlasl100.py --trained --frames 50
```
Results: `backend/eval_results/sign_wlasl100.json`, `sign_tgcn_train.json` and
`sign_wlasl100_trained.json`.

---

## May 2026 run (superseded)

**Date run:** 2026-05-13
**Subset:** WLASL v0.3 — top 25 most-instanced glosses, 2 videos per gloss attempted
**Final clips evaluated:** 17 (after dropping HTML/dead URLs)
**Command:**
```
python scripts/eval_wlasl.py --top-n 25 --videos-per-class 2 --frames-per-video 30
```

---

## Headline numbers

| Metric | Result |
|---|---|
| **Top-1 accuracy** | **1/17 = 5.9 %** |
| **Top-3 accuracy** | **1/17 = 5.9 %** |
| **Top-5 accuracy** | **1/17 = 5.9 %** |
| Mean latency | 798 ms / clip |
| P95 latency | 3 007 ms / clip |
| Glosses covered | 15 |

The single correct prediction was **`yes`**, which happens to overlap with the 7-word vocabulary of the MediaPipe pre-trained gesture model.

---

## Honest diagnosis — why the numbers are still low after the MediaPipe fix

The full pipeline is designed in four tiers:

| Tier | Source | Vocabulary | Status after fix |
|---|---|---|---|
| 0 | TGCN trained on WLASL | 100 – 2 000 signs | **Still blocked — checkpoint not findable** |
| 1 | MediaPipe pre-trained gesture model | 7 words | Active ✓ |
| 2 | HuggingFace ASL letter classifier | 26 letters | Active ✓ |
| 3 | Geometric landmark rules (Holistic) | ~25 handshapes | **Fixed ✓** — now firing |

**MediaPipe issue: FIXED.** A new `_TasksHolistic` shim composes `PoseLandmarker + HandLandmarker + FaceLandmarker` from `mediapipe.tasks.python.vision` and exposes the result in the legacy `holistic_result.pose_landmarks.landmark[i].x` shape via lightweight `_TasksLandmarkPoint` / `_TasksLandmarkList` adapters. Tier 3 (geometric) is now operational again — visible in the latest misclassifications (`fine → hello`, `help → ok`, `all → hello`).

**TGCN issue: still blocked.** All three configured HuggingFace repos (`kasrahabib/tgcn-wlasl`, `zhengshu/tgcn-wlasl`, `asl-research/tgcn-wlasl`) return HTTP 401. The dxli94/WLASL GitHub repo ships training code only, not pretrained `.bin` weights. Without the TGCN checkpoint, Tier 0 stays offline and arbitrary WLASL vocabulary (book, drink, computer, …) can only be guessed by Tiers 1-3 — whose vocabulary is small and curated, so most WLASL test glosses get collapsed onto one of the ~30 known handshape labels.

Result: only Tier 1 (the 7-word gesture model) and Tier 2 (the letter classifier) are operational, so any WLASL gloss outside that tiny vocabulary is mis-classified.

This is exactly what the per-class breakdown shows:
- `yes` → correctly recognised (in MediaPipe's 7-word set)
- `book`, `drink`, `go`, `clothes`, `cousin`, `fine`, `help`, `thin`, `walk`, `year` → mis-classified, all collapsed onto the same 7 words (`ok`, `no`, `you`)

---

## Comparison to Li et al. 2020 baseline

Li et al. (2020) report on the **full WLASL-2000** dataset:

| Model | Top-1 | Top-5 | Top-10 |
|---|---|---|---|
| TGCN (Li 2020) | 23.65 % | 51.75 % | 62.24 % |

These numbers are **not directly comparable** to ours because:
1. Li et al. evaluate all 2 000 classes, not 25.
2. Their TGCN tier is the one currently disabled here.
3. They use the official train/test split — we sampled the top-25 most-instanced glosses for a quick eval.

A fair re-run after fixing the Holistic issue (see remediation below) is the dissertation-relevant comparison.

---

## Remediation — what's done, what's left

✅ **DONE — MediaPipe Tasks-API port.** Implemented `_TasksHolistic` in `backend/modules/sign_language.py` (composes `PoseLandmarker + HandLandmarker + FaceLandmarker` and presents them in the legacy Holistic shape). All four landmark slots now populate; the geometric tier is firing again.

⏳ **REMAINING — TGCN WLASL checkpoint.** Three options, in increasing effort:

1. **Find or host a working public checkpoint.** Search HuggingFace / GitHub Releases / Zenodo for "TGCN WLASL pytorch_model.bin". If any are found, set `ACCESSIBILITY_TGCN_REPO=<username>/<repo>` in `.env` (no code change needed).
2. **Train one locally.** Use `dxli94/WLASL/code/TGCN/train_tgcn.py` against the downloaded video subset (needs ~21 K WLASL videos — most URLs are dead, so first re-scrape from working hosts only).
3. **Substitute with I3D-ResNet or a transformer baseline from `sign-language-transformers` (Camgöz et al. 2020)** — different architecture but same task; would also need its own pretrained weights.

Expected post-fix numbers once a TGCN checkpoint loads (rough, based on Li et al.'s reported TGCN performance scaled to WLASL-100):
- Top-1: **~30-40 %**
- Top-5: **~55-65 %**

The current run gives a believable lower bound — anything above this can be cleanly attributed to the TGCN tier once it loads.

---

## Data caveats — for the dissertation

- WLASL distributes only URLs, not videos. Of 50 attempted downloads from the manifest, **17 returned a real MP4** (3 4%). The rest were HTML 404 / login-wall pages from `aslpro.com`, `signingsavvy.com`, etc.
- Working hosts in this run: `aslbricks.org` (most reliable), `aslsignbank.haskins.yale.edu`, `handspeak.com`.
- Full WLASL-2000 evaluation would require ~21 000 video downloads; many URLs are now dead (the dataset is from 2020).
- The 17 clips covered 15 distinct glosses, so this is best read as a sanity-check, not a benchmark.

---

## Files produced

- `backend/wlasl_eval_results.json` — full per-class breakdown + misclassification list
- `backend/data/wlasl_eval_run2.log` — full stdout of the eval run
- `backend/data/WLASL-master/videos/*.mp4` — the 17 downloaded clips
