# WLASL Evaluation Results

## Update, September 2026: official test clips and the TGCN tier

This supersedes the May 2026 run further down, which is kept for the record.

**Test clips.** 100 of the 258 official WLASL100 test clips could still be
obtained (the Voxel51 mirror on Hugging Face), each trimmed to its annotated
frame range. They are not committed (`backend/data/WLASL-master/videos_test100/`).

**TGCN weights.** An asl100 checkpoint was found at
`huggingface.co/sharonn18/tgcn-wlasl` (64 hidden features, 20 stages, 50
frames). Loading it exposed a defect: `tgcn_sign_model.py` defined an attention
block, an extra graph convolution and a flattening classifier that no
checkpoint contains, and loaded with `strict=False`, so those layers would have
kept random weights. The network now matches the released `GCN_muti_att` key
for key, and a checkpoint that does not fit is refused.

**Input conventions.** The checkpoint was trained on OpenPose keypoints and its
release does not state the keypoint order, coordinate range or class order.
These were chosen on 14 older local WLASL clips outside the test split
(`backend/eval_results/sign_wlasl100_select.json`) before any test clip was
scored: OpenPose order, coordinates in [-1, 1], alphabetical classes. The best
variant got 3 of 14 right within its top five.

| Configuration, 100 test clips | Top-1 | Top-5 |
|---|---|---|
| Pipeline without the TGCN tier (as shipped) | 1% | 1% |
| Pipeline with the TGCN tier | 1% | 2% |
| TGCN model alone | 1% | 6% (95% Wilson interval 3% to 12%) |

Top-5 chance for 100 signs is 5%, so the published weights perform at chance
on MediaPipe keypoints, most likely because weights learned on OpenPose
keypoints do not transfer. The tier is therefore off by default
(`ACCESSIBILITY_ENABLE_TGCN=1` turns it on), because a confident wrong sign is
worse than none. The realistic fix is retraining the TGCN on MediaPipe
keypoints extracted from the WLASL training videos.

Commands, from `backend/`:
```
python scripts/report_experiments/sign_wlasl100.py --select
python scripts/report_experiments/sign_wlasl100.py
```
Results: `backend/eval_results/sign_wlasl100.json`.

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
