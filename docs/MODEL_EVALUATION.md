# Model selection and evaluation

**Project:** A Multimodal AI Accessibility Assistant for Deaf and Hard-of-Hearing Users
**Author:** Laura Barsoum, September 2026

This page records which models the system uses, which candidates were
rejected, and the evidence behind each decision. It matches Table 3.1 and
Section 5.2 of the final report. Every measured number was produced by a script
in `backend/scripts/report_experiments/` and is stored in `backend/eval_results/`;
that folder's README says which script produces which table or figure.

## Models in the running system

Seven pre-trained perception models and one language model, plus a sign
network (the TGCN) trained here on WLASL100 keypoints extracted by the
application, because the published weights, learned from OpenPose keypoints,
reached only 8% top-1 on them. YamNet is kept only as the fallback when AST
cannot load.

| Channel | Model | Role |
|---|---|---|
| Speech | Whisper via faster-whisper: `distil-small.en` for final captions, `tiny` for live partials | Captions |
| Sound events | Audio Spectrogram Transformer (AudioSet) | Generic sound labels |
| Personal sounds | AST embeddings with a prototypical-network rule calibrated on browser-coded audio | Few-shot enrolled sounds |
| Scene | BLIP base | Scene description, one tick in four |
| Hazards | YOLOv11n | Visual hazard detection |
| Sign | TGCN trained on MediaPipe keypoints from WLASL100, MediaPipe gesture model, geometric rules, fingerspelling | 100-sign vocabulary |
| Emotion | DeepFace (face) and wav2vec 2.0 (voice), confidence-weighted | Tone tags |
| Language | `openai/gpt-oss-20b` via Groq, with a fallback chain | Gloss polishing and alert composition |

## Selection decisions

"Measured" decisions were made by experiments on this project's hardware.
"Published" decisions rely on the cited literature listed in the report.

| Channel | Chosen | Rejected or demoted | Deciding evidence | Basis | Results file |
|---|---|---|---|---|---|
| Speech | `distil-small.en` finals, `tiny` partials | wav2vec 2.0 CTC; Conformer; Whisper `small` and `base`; repetition guards | WER on 73 LibriSpeech utterances: distil 6.9%, small 7.1%, base 9.0%, tiny 10.9%; distil 822 ms against small 1233 ms per 2.8 s chunk; the repetition guards had raised small to 15.2% | Measured and published | `whisper_wer.json`, `whisper_ablation.json`, `whisper_ablation_distil.json` |
| Sound events | AST | YamNet classifier; PANNs | ESC-50 top-3 on all 2,000 clips: AST 77.8%, YamNet 64.5%, same label map and scoring | Measured | `esc50_ast_results.json`, `yamnet_esc50_results.json` |
| Personal sounds | AST embedding, prototypical rule, settings chosen on browser-coded audio | YamNet and spectral embeddings; cosine threshold; settings chosen on clean clips with three sounds enrolled | With fallback features the cosine rule fired on 78% of never-enrolled clips; with YamNet, calibrated test F1 0.344 against 0.200 for cosine (difference 0.081 to 0.202, 95% bootstrap interval over the 30 trials). On clean clips AST raised test recall from 0.32 to 0.83 (paired gain 0.44 to 0.58), but those settings matched one enrolled alarm on every fusion-scene tick; re-chosen on Opus-coded 2.8 s ticks with one and three sounds enrolled, test recall is 0.64 at 4.9% false alarms with one sound, where the earlier settings raised 65.6% | Measured | `fewshot_results.json`, `fewshot_tune.json`, `fewshot_embeddings.json`, `fewshot_deployment.json` |
| Scene | BLIP base | CLIP retrieval | Open-ended captions need a generator; an uncached caption takes 225 ms, so BLIP runs one tick in four | Published and measured | `latency_results.json` |
| Hazards | YOLOv11n | Detectron2; EfficientDet | Single-stage detector built for CPU; 22 ms for three frames | Published and measured | `latency_results.json` |
| Sign | TGCN trained on MediaPipe keypoints, with landmarks and rules | Published TGCN weights; I3D; legacy Holistic API | Chosen on 163 validation clips (random start, body-normalised keypoints, 60.1% top-1); on 100 test clips 60% top-1 and 78% top-5, against 8% and 31% for the published weights once hands were detected; I3D needs RGB video; the Holistic API was removed from MediaPipe | Measured and published | `sign_tgcn_train.json`, `sign_wlasl100.json`, `sign_wlasl100_trained.json` |
| Emotion | DeepFace and wav2vec 2.0, confidence-weighted | Either channel alone | FER-2013 private test (3,589 faces): accuracy 54.7% (95% interval 53.1% to 56.3%), macro-F1 0.51, happy F1 0.76, fear F1 0.37; the neutral-bias recalibration left accuracy almost unchanged (54.9% without it); neutral bias also seen on live smiles | Measured and observed | `emotion_fer2013.json` |
| Diarization | Off by default | pyannote on every tick | First load of about 1 GB stalled every tick | Measured | none |
| Language model | `gpt-oss-20b` | Llama 3.3 70B; `gpt-oss-120b`; Qwen3.8-27B; `compound-mini` | 144 calls: 20b median 0.32 s, 120b 0.42 s, both preserved the meaning in 36 of 36 calls; `compound-mini` 0 of 36 (HTTP 400); Llama 3.3 decommissioned by the provider | Measured | `llm_bench.json` |
| Sound direction | Withdrawn | GCC-PHAT | Laptop microphones give no usable inter-channel timing | Prototype | none |

## Fusion

Fusion was scored separately, because the models can each be right while the
alerts are still wrong. `report_experiments/fusion_scenes.py` posts 35 scripted
scenes (ESC-50 clips and synthetic speech, with urgencies fixed before running)
through the real `/process` handler, three times each, with live captions off.
A second, held-out set with different clips and sentences was written and
measured before fusion was changed. The "after" runs include the fusion changes
(merging related labels and a caption with its alerts, AST speech labels
demoted, transcripts ignored when AST's speech probability is below 0.18) and
AST personal-sound embeddings. The scores are with live captions off: with them
on, the default, captions and their name and keyword alerts bypass fusion and
its speech gate. The gate was set on all 2,000 ESC-50 clips, which include every
clip in both scene sets.

| Measure | Original, before | Original, after | Held-out, before | Held-out, after |
|---|---|---|---|---|
| First headline is the most urgent real event | 79/93 (85%) | 84/93 (90%) | 72/93 (77%) | 78/93 (84%) |
| Headline slots repeating an event already shown | 84/218 (39%) | 0/105 | 63/191 (33%) | 0/93 |
| Headline slots with no urgent event behind them | 41/218 (19%) | 12/105 (11%) | 38/191 (20%) | 3/93 (3%) |
| Urgent events given no headline (none crowded out) | 15/108 | 15/108 | 18/108 | 18/108 |
| Extra detections of one event removed by the merge | 0/159 | 99/168 | 0/141 | 99/150 |

Results are in `backend/eval_results/` (`fusion_eval.json`,
`fusion_eval_after.json`, `fusion_eval_heldout_before.json`,
`fusion_eval_heldout_after.json`) and Section 5.4 of the report.

## Limits of this evidence

- ESC-50 and LibriSpeech are cleaner than a real home, so absolute accuracy in
  use will be lower.
- Personal sounds were evaluated on a household-class proxy built from ESC-50,
  not on recordings from users' homes. With one sound enrolled and clips passed
  through the browser codec, the adopted matcher recalls 0.64 at a 4.9%
  false-alarm rate, so it still misses about one play in three.
- Sign recognition covers only the 100 WLASL100 signs: 60% top-1 on 100 of the
  258 official test clips, and 59% through the full cascade once the trained
  network was allowed to lead, against 21% while the hand-shape rules outranked
  it. It was trained and tested on WLASL's signers, not on webcam signing by
  Deaf users
  (`docs/WLASL_EVAL_RESULTS.md`).
- Face identification was measured on Labeled Faces in the Wild, not on webcam
  frames at home, and with 30 people enrolled rather than the handful a home
  would hold. With 3 photographs each it names the right person in 80% of 60
  probes and accepts a stranger as someone enrolled in 12% of 60 impostor
  probes, at the 0.40 threshold chosen on a disjoint development half. The 0.35
  shipped before scored 82% but accepted 28% of strangers and misnamed 7% of
  genuine probes (`backend/scripts/report_experiments/face_identification_lfw.py`).
- Language-model latency depends on a hosted provider, which can withdraw
  models, as happened to Llama 3.3 during the project.
- Five Deaf and hard-of-hearing people gave informal feedback on successive
  interface versions. Nothing was recorded, and that feedback is not evidence
  of measured benefit.

## Reproducing the results

See `backend/scripts/report_experiments/README.md` for the command behind each
table and figure. Datasets are not committed; ESC-50 is fetched by
`backend/scripts/download_esc50.py` and the LibriSpeech sample by
`report_experiments/extract_librispeech_sample.py`.
