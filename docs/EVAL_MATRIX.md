# Evaluation matrix (auto-generated)

Run via `python backend/scripts/run_eval_matrix.py`. This is the development harness: "Script ran" means the evaluation script finished, not that the target was met. Empty cells mean the script couldn't find its dataset; drop the data in and re-run. The results quoted in the final report come from `backend/scripts/report_experiments/` and are stored in `backend/eval_results/`.

| Component | Headline metric | Value | Target | Script ran |
|---|---|---|---|---|
| ESC-50 (AST) | Top-3 acc | 77.8 % | ≥ 60 % | ✓ |
| WLASL-100 (TGCN+geom) | Top-1 | 5.9 % | ≥ 30 % | ✓ |
| Custom sign set | — | (no data) | — | ✓ |
| LibriSpeech WER (Whisper) | — | (awaiting data) | — | ✓ |
| BLIP scene captions | — | (awaiting data) | — | ✓ |
| DeepFace 7-class | — | (awaiting data) | — | ✓ |
| YOLO hazard latency | — | — | — | ✓ |
| Personalizer P/R/F1 | — | (awaiting data) | — | ✓ |
