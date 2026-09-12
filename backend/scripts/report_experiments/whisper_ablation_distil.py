"""[Final report: Section 5.2 speech, Table 5.1]

distil-small.en under the same three decoding configurations as the main
ablation (it had only been measured with the harmful repetition guards)."""
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
            cur = d[j]; d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1])); prev = cur
    return d[len(h)]
VAD = {"min_silence_duration_ms": 250, "speech_pad_ms": 200, "threshold": 0.35}
TEMPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BASE = dict(beam_size=2, language="en", vad_filter=True, vad_parameters=VAD, condition_on_previous_text=False,
            no_speech_threshold=0.45, temperature=TEMPS, compression_ratio_threshold=2.4)
CFGS = {"shipped": dict(BASE, repetition_penalty=1.15, no_repeat_ngram_size=3),
        "no_repetition_guards": BASE, "library_defaults": dict(language="en")}
s = stt_mod.SpeechToText(); s.model_size = "distil-small.en"; s._ensure_loaded()
out = {"results": {}}
for name, cfg in CFGS.items():
    errs = words = trunc = 0; per = []; t0 = time.time()
    for u in refs:
        segs, _ = s._model.transcribe(os.path.join(WAV, u["file"]), **cfg)
        hyp = collapse_repetition(" ".join(x.text.strip() for x in segs).strip())
        r, h = norm(u["text"]), norm(hyp); e = edits(r, h); errs += e; words += len(r)
        trunc += 1 if len(h) < 0.6 * len(r) else 0; per.append(e / max(1, len(r)))
    out["results"][f"distil-small.en/{name}"] = dict(wer=errs / words, median_utt_wer=float(np.median(per)), truncated=trunc)
    print(f"distil-small.en/{name:22s} WER {100*errs/words:5.2f}%  median {100*np.median(per):5.2f}%  truncated {trunc}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(out, open(os.path.join(SP, "whisper_ablation_distil.json"), "w"), indent=1)
print("DONE distil", flush=True)
