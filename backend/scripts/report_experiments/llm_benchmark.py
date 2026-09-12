"""[Final report: Section 5.2 language model, Figure 5.1c]

Benchmark candidate LLMs on the shipped gloss-to-English task.

Uses the production call parameters from modules/llm.py (_chat) and the
production system prompt from SignLanguageRecognizer.gloss_to_speech.
Scores each response with an objective content-preservation rubric and
records latency, finish reason and reasoning-token usage."""
from pathlib import Path
import json, os, re, sys, time

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
os.makedirs(SP, exist_ok=True)
sys.path.insert(0, BACKEND)
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(BACKEND, "..", ".env"))
from groq import Groq  # noqa: E402

client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYS = (
    "You translate ASL sign-language glosses into short English sentences.\n"
    "Rules:\n"
    "- Input: space-separated sign labels such as 'you ok' or 'hello you ok'.\n"
    "- Output: ONE natural English sentence. Nothing else. No explanation.\n"
    "- Examples:\n"
    "  you ok → Are you okay?\n"
    "  hello you ok → Hello — are you okay?\n"
    "  yes please → Yes please.\n"
    "  help me → Help me!\n"
    "  no stop → No, stop!\n"
    "  love you → I love you."
)
# Each gloss: groups of acceptable surface forms; every group must appear.
GLOSSES = {
    "you ok": [["you"], ["okay", "ok", "alright", "all right"]],
    "hello you ok": [["hello", "hi"], ["you"], ["okay", "ok", "alright", "all right"]],
    "help me": [["help"], ["me"]],
    "help me fire": [["help"], ["fire"]],
    "water please": [["water"], ["please"]],
    "love you": [["love"], ["you"]],
    "no stop": [["no"], ["stop"]],
    "thank you": [["thank", "thanks"]],
    "doctor need": [["doctor"], ["need"]],
    "where bathroom": [["where"], ["bathroom", "toilet", "restroom"]],
    "yes please": [["yes"], ["please"]],
    "danger go": [["danger", "dangerous"], ["go", "leave", "get out"]],
}
BAD = ["translate", "ready to", "asl gloss", "the sequence", "provide the",
       "sign language transl", "what's the", "please provide", "i need more"]
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "groq/compound-mini"]
REPEATS = 3


def score(gloss, text):
    if not text:
        return False, "empty"
    if len(text) > 200:
        return False, "too_long"
    low = text.lower()
    if any(b in low for b in BAD):
        return False, "meta_commentary"
    for group in GLOSSES[gloss]:
        if not any(re.search(rf"\b{re.escape(w)}\b", low) for w in group):
            return False, f"missing:{group[0]}"
    return True, "ok"


records = []
probe = {}
for m in ["llama-3.3-70b-versatile", "llama3-70b-8192"]:
    try:
        client.chat.completions.create(model=m, messages=[{"role": "user", "content": "hi"}], max_tokens=5)
        probe[m] = "available"
    except Exception as e:
        s = str(e)
        probe[m] = "model_decommissioned" if "decommission" in s else ("model_not_found" if "not_found" in s or "does not exist" in s else s[:120])
print("probe:", probe, flush=True)

for rep in range(REPEATS):
    for gloss in GLOSSES:
        for m in MODELS:
            t0 = time.perf_counter()
            rec = dict(model=m, gloss=gloss, rep=rep)
            try:
                r = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "system", "content": SYS}, {"role": "user", "content": gloss}],
                    max_tokens=150, temperature=0.1, reasoning_effort="low",
                )
                rec["latency_s"] = time.perf_counter() - t0
                ch = r.choices[0]
                rec["text"] = (ch.message.content or "").strip()
                rec["finish"] = ch.finish_reason
                u = getattr(r, "usage", None)
                rec["completion_tokens"] = getattr(u, "completion_tokens", None) if u else None
                det = getattr(u, "completion_tokens_details", None) if u else None
                rt = None
                if det is not None:
                    rt = det.get("reasoning_tokens") if isinstance(det, dict) else getattr(det, "reasoning_tokens", None)
                rec["reasoning_tokens"] = rt
                rec["error"] = None
            except Exception as e:
                rec["latency_s"] = time.perf_counter() - t0
                rec["text"] = ""
                rec["error"] = str(e)[:200]
            ok, why = score(gloss, rec["text"])
            rec["pass"], rec["why"] = ok, why
            records.append(rec)
            print(f"{m:22s} {gloss:15s} {rec['latency_s']:.2f}s pass={ok} {why} rt={rec.get('reasoning_tokens')} "
                  f"{(rec['text'] or rec['error'] or '')[:50]!r}", flush=True)
            time.sleep(1.1)

json.dump(dict(probe=probe, records=records), open(os.path.join(SP, "llm_bench.json"), "w"), indent=1)
print("DONE llm", flush=True)
