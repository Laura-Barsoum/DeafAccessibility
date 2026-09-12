"""[Final report: Section 5.2 speech (the first run that exposed the problem)]

Word error rate of Whisper model sizes on real read speech.

Data: hf-internal-testing/librispeech_asr_dummy (73 utterances drawn from
LibriSpeech validation-clean), extracted to WAV beforehand. Each utterance is
passed through the SHIPPED SpeechToText.transcribe (VAD, repetition guards,
hallucination filter), so the figure describes the deployed configuration.
Normalisation: lower-case, punctuation stripped. No number normalisation,
which penalises all sizes equally when Whisper writes digits.
Timing is not reported here; latency comes from latency_bench.py."""
from pathlib import Path
import json, os, re, sys

import numpy as np

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
os.makedirs(SP, exist_ok=True)
WAV = os.path.join(SP, "librispeech_wav")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

refs = json.load(open(os.path.join(WAV, "refs.json")))
for r in refs:
    r["wav"] = open(os.path.join(WAV, r["file"]), "rb").read()
total_s = sum(r["seconds"] for r in refs)
print(f"{len(refs)} utterances, {total_s/60:.1f} min of audio", flush=True)


def norm(s):
    return re.sub(r"[^a-z' ]+", " ", s.lower()).split()


def edits(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)]


from modules import stt as stt_mod  # noqa: E402

results = {}
for size in ("tiny", "base", "small", "distil-small.en"):
    s = stt_mod.SpeechToText()
    s.model_size = size
    s._ensure_loaded()
    errs = words = empty = 0
    per = []
    for u in refs:
        hyp = s.transcribe(u["wav"], with_timestamps=False).get("text", "")
        r, h = norm(u["text"]), norm(hyp)
        e = edits(r, h)
        errs += e; words += len(r); empty += 0 if h else 1
        per.append(dict(id=u["id"], wer=e / max(1, len(r)), ref=u["text"], hyp=hyp))
    results[size] = dict(wer=errs / words, errors=errs, ref_words=words, empty_outputs=empty,
                         median_utt_wer=float(np.median([x["wer"] for x in per])),
                         worst=sorted(per, key=lambda x: -x["wer"])[:3])
    print(f"{size:16s} WER {100*errs/words:5.2f}%  ({errs}/{words})  empty={empty}", flush=True)
    del s

json.dump(dict(dataset="hf-internal-testing/librispeech_asr_dummy (LibriSpeech validation-clean sample)",
               n_utterances=len(refs), audio_minutes=total_s / 60, results=results),
          open(os.path.join(SP, "whisper_wer.json"), "w"), indent=1)
print("DONE wer", flush=True)
