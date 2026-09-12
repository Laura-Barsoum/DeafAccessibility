"""[Final report: Section 5.2 speech] Extract the 73-utterance LibriSpeech
validation-clean sample (hf-internal-testing/librispeech_asr_dummy) to WAV files
plus reference transcripts, for whisper_ablation.py. Needs pyarrow and soundfile."""
import io, json, os
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
p = hf_hub_download("hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet",
                    repo_type="dataset", local_dir=os.path.join(SP, "librispeech"))
out = os.path.join(SP, "librispeech_wav"); os.makedirs(out, exist_ok=True)
refs = []
for i, row in enumerate(pq.read_table(p).to_pylist()):
    y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    fn = f"{i:03d}.wav"; sf.write(os.path.join(out, fn), y, sr, subtype="PCM_16")
    refs.append(dict(id=row.get("id", str(i)), file=fn, text=row["text"], seconds=len(y) / sr))
json.dump(refs, open(os.path.join(out, "refs.json"), "w"), indent=1)
print("extracted", len(refs), "utterances")
