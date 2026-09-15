# Report experiments

Scripts that produce every table and figure in Chapter 5 of the final report.
Each writes JSON to `backend/eval_results/`; the committed JSON files are the
exact results the report quotes. Run from `backend/` with the project venv.

| Script | Produces | Report location |
|---|---|---|
| `../eval_audio_scene.py --out eval_results/esc50_ast_results.json` | AST top-3 on all 2,000 ESC-50 clips | 5.2, Figure 5.1a |
| `../eval_audio_scene.py --top-k 1 --out eval_results/esc50_ast_top1.json` | AST top-1 on the same clips, planned in the preliminary report | Table 5.1 |
| `yamnet_esc50.py` | YamNet on the same clips and label map | 5.2, Figure 5.1a |
| `fewshot_esc50.py` | Cosine vs prototypical, YamNet vs fallback embedding | 5.2, Figure 5.2b |
| `fewshot_calibrate.py` | Development/test calibration of the matcher | 5.2, Table 5.3, Figure 5.2a |
| `fewshot_embeddings.py` | Personal sounds with AST against YamNet embeddings (K = 3 and 5), settings chosen on the development split; AST's speech probability on ESC-50, which sets fusion's transcript gate | 5.2, Table 5.3, Section 3.6 |
| `fewshot_deployment.py` | AST personal-sound settings re-chosen on ESC-50 clips passed through the browser codec as 2.8 s ticks, with one and three sounds enrolled; the earlier AST settings scored on the same trials | 5.2, Tables 5.1 and 5.6 |
| `sign_wlasl100.py` | WLASL100 test clips with and without the TGCN tier; `--select` first settles keypoint layout, coordinates and class order on other clips | 5.2, Table 5.1 |
| `extract_librispeech_sample.py` | 73 LibriSpeech utterances as WAV + transcripts | input to the speech scripts |
| `whisper_wer_first_run.py` | The first WER run that exposed the repetition guards | 5.2 |
| `whisper_ablation.py`, `whisper_ablation_distil.py` | WER by model and decoding configuration | 5.2, Table 5.1, Figure 5.1b |
| `llm_benchmark.py` | 144-call language-model benchmark (needs `GROQ_API_KEY`) | 5.2, Figure 5.1c |
| `latency_on_recorded_media.py` | Per-stage and end-to-end tick latency | 5.3, Figures 3.2 and 5.3 |
| `fusion_scenes.py` | Fusion on scripted scenes with known events: duplicate headlines, first-headline correctness, merge removals. `--scene-set heldout` uses different clips and sentences, written and measured before fusion was changed; `--out` names the result file | 5.4, Table 5.4, Figure 5.4 |
| `emotion_fer2013.py` | Facial emotion per-label precision, recall and F1 on the FER-2013 private test split, with and without the shipped recalibration | 5.2, Table 5.1, Figure 5.1d |
| `capture_hazard_diary.py` | Screenshots of the hazard channel and the Diary and Summary tabs, with a temporary diary and profiles | Figure 4.6 |
| `codec_margin_check.py` | One held-out clip against the calibrated YamNet gate, with and without Opus | Not in the final report (YamNet embeddings were replaced) |
| `capture_screenshots.py` | Live screenshots on recorded test media (needs Playwright) | Figures 4.3 to 4.5 |

Notes
- Datasets are not committed: ESC-50 is fetched by `scripts/download_esc50.py`;
  the LibriSpeech sample by `extract_librispeech_sample.py`; the FER-2013 private
  test split goes in `backend/data/emotion_eval/fer2013/privateTest/<label>/`
  (see that folder's README).
- `fewshot_esc50.py` stops if YamNet fails to load, rather than silently
  measuring the fallback embedding.
- Run the latency script on an otherwise idle machine; timings are hardware-specific
  (the report's figures come from one Apple Silicon laptop).
