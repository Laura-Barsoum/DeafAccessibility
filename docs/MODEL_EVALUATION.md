# Model selection and evaluation

**Project:** A Multimodal AI Accessibility Assistant for Deaf and Hard-of-Hearing Users
**Author:** Laura Barsoum, September 2026

This page records which models the system uses, which candidates were
rejected, and the evidence behind each decision. It matches Table 3.1 and
Section 5.2 of the final report. Every measured number was produced by a script
in `backend/scripts/report_experiments/` and is stored in `backend/eval_results/`;
that folder's README says which script produces which table or figure.

## Models in the running system

Eight pre-trained perception models and one language model. No model is
trained from scratch.

| Channel | Model | Role |
|---|---|---|
| Speech | Whisper via faster-whisper: `distil-small.en` for final captions, `tiny` for live partials | Captions |
| Sound events | Audio Spectrogram Transformer (AudioSet) | Generic sound labels |
| Personal sounds | YamNet embeddings with a calibrated prototypical-network rule | Few-shot enrolled sounds |
| Scene | BLIP base | Scene description, one tick in four |
| Hazards | YOLOv11n | Visual hazard detection |
| Sign | MediaPipe Tasks landmark and gesture models, geometric rules, fingerspelling | Constrained sign vocabulary |
| Emotion | DeepFace (face) and wav2vec 2.0 (voice), confidence-weighted | Tone tags |
| Language | `openai/gpt-oss-20b` via Groq, with a fallback chain | Gloss polishing and alert composition |

## Selection decisions

"Measured" decisions were made by experiments on this project's hardware.
"Published" decisions rely on the cited literature listed in the report.

| Channel | Chosen | Rejected or demoted | Deciding evidence | Basis | Results file |
|---|---|---|---|---|---|
| Speech | `distil-small.en` finals, `tiny` partials | wav2vec 2.0 CTC; Conformer; Whisper `small` and `base`; repetition guards | WER on 73 LibriSpeech utterances: distil 6.9%, small 7.1%, base 9.0%, tiny 10.9%; distil 822 ms against small 1233 ms per 2.8 s chunk; the repetition guards had raised small to 15.2% | Measured and published | `whisper_wer.json`, `whisper_ablation.json`, `whisper_ablation_distil.json` |
| Sound events | AST | YamNet classifier; PANNs | ESC-50 top-3 on all 2,000 clips: AST 77.8%, YamNet 64.5%, same label map and scoring | Measured | `esc50_ast_results.json`, `yamnet_esc50_results.json` |
| Personal sounds | YamNet embedding, prototypical rule, temperature 0.5 | Spectral fallback features; cosine threshold | With fallback features the cosine rule fired on 78% of never-enrolled clips; calibrated test F1 0.344 against 0.200 for cosine chosen under the same development constraint | Measured | `fewshot_results.json`, `fewshot_tune.json` |
| Scene | BLIP base | CLIP retrieval | Open-ended captions need a generator; an uncached caption takes 225 ms, so BLIP runs one tick in four | Published and measured | `latency_results.json` |
| Hazards | YOLOv11n | Detectron2; EfficientDet | Single-stage detector built for CPU; 22 ms for three frames | Published and measured | `latency_results.json` |
| Sign | MediaPipe Tasks landmarks, rules, TGCN if weights exist | I3D; legacy Holistic API | I3D is more accurate on WLASL (32.48% against 23.65% top-1, Li et al. 2020) but needs RGB video; the Holistic API was removed from MediaPipe | Published and availability | `wlasl_results.json` |
| Emotion | DeepFace and wav2vec 2.0, confidence-weighted | Either channel alone | FER-2013 neutral bias seen on live smiles | Published and observed | none |
| Diarization | Off by default | pyannote on every tick | First load of about 1 GB stalled every tick | Measured | none |
| Language model | `gpt-oss-20b` | Llama 3.3 70B; `gpt-oss-120b`; Qwen3.8-27B; `compound-mini` | 144 calls: 20b median 0.32 s, 120b 0.42 s, both preserved the meaning in 36 of 36 calls; `compound-mini` 0 of 36 (HTTP 400); Llama 3.3 decommissioned by the provider | Measured | `llm_bench.json` |
| Sound direction | Withdrawn | GCC-PHAT | Laptop microphones give no usable inter-channel timing | Prototype | none |

## Fusion

Fusion was scored separately, because the models can each be right while the
alerts are still wrong. `report_experiments/fusion_scenes.py` posts 35 scripted
scenes (ESC-50 clips and synthetic speech, with urgencies fixed before running)
through the real `/process` handler, three times each, with live captions off.
Results are in `backend/eval_results/fusion_eval.json` and Section 5.4 of the
report.

| Measure | Result |
|---|---|
| First headline is the most urgent real event | 79 of 93 ticks (85%) |
| Headline slots repeating an event already shown | 84 of 218 (39%) |
| Headline slots with no urgent event behind them | 41 of 218 (19%), all Whisper transcripts of non-speech |
| Urgent events given no headline | 15 of 108, none crowded out |
| Extra detections of one event removed by the merge | 0 of 159 |

## Limits of this evidence

- ESC-50 and LibriSpeech are cleaner than a real home, so absolute accuracy in
  use will be lower.
- Personal sounds were evaluated on a household-class proxy built from ESC-50,
  not on recordings from users' homes. On the calibration test split the
  adopted matcher has recall 0.32 and a 20.6% false-alarm rate: it misses an
  enrolled sound roughly two times in three.
- Sign recognition did not meet its goal: 5.9% top-1 on 17 WLASL clips covering
  15 signs (`docs/WLASL_EVAL_RESULTS.md`). No public TGCN checkpoint could be
  obtained.
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
