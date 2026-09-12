"""[Final report: Section 5.2 speech, Table 5.1, Figure 5.1b]

Ablation: which decoding settings cost Whisper accuracy?

Same 73 LibriSpeech validation-clean utterances as whisper_wer.py, with a
fairer normaliser (expands Mr/Mrs/Dr/St). Configurations:
  shipped               SpeechToText.transcribe exactly as deployed
  no_repetition_guards  shipped minus repetition_penalty and no_repeat_ngram_size
  library_defaults      faster-whisper defaults (beam 5, no VAD), English forced
Every output passes through collapse_repetition, so loops are still caught.
'truncated' counts utterances whose hypothesis has under 60% of the reference
words, the failure seen in the first run."""
from pathlib import Path
import json, os, re, sys, time
import numpy as np

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
os.makedirs(SP, exist_ok=True)
WAV = os.path.join(SP, "librispeech_wav")
sys.path.insert(0, BACKEND); os.chdir(BACKEND)
refs = json.load(open(os.path.join(WAV, "refs.json")))
from modules import stt as stt_mod                     # noqa: E402
from modules.stt import collapse_repetition            # noqa: E402

ABBR = {"mr": "mister", "mrs": "missus", "dr": "doctor", "st": "saint"}


def norm(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"\b(mr|mrs|dr|st)\.", lambda m: ABBR[m.group(1)], s)
    return re.sub(r"[^a-z' ]+", " ", s).split()


def edits(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)]


VAD = {"min_silence_duration_ms": 250, "speech_pad_ms": 200, "threshold": 0.35}
TEMPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CFGS = {
    "shipped": None,
    "no_repetition_guards": dict(beam_size=2, language="en", vad_filter=True, vad_parameters=VAD,
                                 condition_on_previous_text=False, no_speech_threshold=0.45,
                                 temperature=TEMPS, compression_ratio_threshold=2.4),
    "library_defaults": dict(language="en"),
}
out = {"n_utterances": len(refs), "audio_minutes": sum(r["seconds"] for r in refs) / 60, "results": {}}
for size in ("tiny", "base", "small"):
    s = stt_mod.SpeechToText(); s.model_size = size; s._ensure_loaded()
    for name, cfg in CFGS.items():
        errs = words = trunc = 0; per = []; t0 = time.time()
        for u in refs:
            path = os.path.join(WAV, u["file"])
            if cfg is None:
                hyp = s.transcribe(open(path, "rb").read(), with_timestamps=False).get("text", "")
            else:
                segs, _ = s._model.transcribe(path, **cfg)
                hyp = collapse_repetition(" ".join(x.text.strip() for x in segs).strip())
            r, h = norm(u["text"]), norm(hyp)
            e = edits(r, h); errs += e; words += len(r)
            trunc += 1 if len(h) < 0.6 * len(r) else 0
            per.append(e / max(1, len(r)))
        key = f"{size}/{name}"
        out["results"][key] = dict(wer=errs / words, median_utt_wer=float(np.median(per)), truncated=trunc,
                                   seconds=time.time() - t0)
        print(f"{key:30s} WER {100*errs/words:5.2f}%  median {100*np.median(per):5.2f}%  truncated {trunc:2d}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(SP, "whisper_ablation.json"), "w"), indent=1)
    del s
print("DONE ablation", flush=True)
