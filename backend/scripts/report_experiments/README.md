# Report experiments

Scripts that produce every table and figure in Chapter 5 of the final report.
Each writes JSON to `backend/eval_results/`; the committed JSON files are the
exact results the report quotes. Run from `backend/` with the project venv.

| Script | Produces | Report location |
|---|---|---|
| `../eval_audio_scene.py` | AST top-3 on all 2,000 ESC-50 clips (`esc50_ast_results.json`) | 5.2, Figure 5.1a |
| `yamnet_esc50.py` | YamNet on the same clips and label map | 5.2, Figure 5.1a |
| `fewshot_esc50.py` | Cosine vs prototypical, YamNet vs fallback embedding | 5.2, Figure 5.2b |
| `fewshot_calibrate.py` | Development/test calibration of the matcher | 5.2, Table 5.2, Figure 5.2a |
| `extract_librispeech_sample.py` | 73 LibriSpeech utterances as WAV + transcripts | input to the speech scripts |
| `whisper_wer_first_run.py` | The first WER run that exposed the repetition guards | 5.2 |
| `whisper_ablation.py`, `whisper_ablation_distil.py` | WER by model and decoding configuration | 5.2, Table 5.1, Figure 5.1b |
| `llm_benchmark.py` | 144-call language-model benchmark (needs `GROQ_API_KEY`) | 5.2, Figure 5.1c |
| `latency_on_recorded_media.py` | Per-stage and end-to-end tick latency | 5.3, Figures 3.2 and 5.3 |
| `codec_margin_check.py` | One held-out clip against the calibrated gate, with and without Opus | 5.2 |
| `capture_screenshots.py` | Live screenshots on recorded test media (needs Playwright) | Figures 4.3 to 4.5 |

Notes
- Datasets are not committed: ESC-50 is fetched by `scripts/download_esc50.py`;
  the LibriSpeech sample by `extract_librispeech_sample.py`.
- `fewshot_esc50.py` stops if YamNet fails to load, rather than silently
  measuring the fallback embedding.
- Run the latency script on an otherwise idle machine; timings are hardware-specific
  (the report's figures come from one Apple Silicon laptop).
