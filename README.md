# A Multimodal AI Accessibility Assistant for Deaf and Hard-of-Hearing Users

Final-year project, BSc Computer Science, University of London.
Module CM3020 Artificial Intelligence, Template 4.1: *Orchestrating AI Models
to Achieve a Goal*.

A browser-based assistant that runs **seven pre-trained AI models concurrently
on a laptop CPU** to provide situational awareness beyond speech captioning.
It transcribes speech, classifies environmental sounds, recognises sounds the
user has personally enrolled, detects visual hazards, describes the scene,
reads emotional tone, and interprets a constrained sign vocabulary. Detections
converge on a shared event bus that ranks them into four priority bands and
escalates genuinely urgent events with a sustained visual flash and a haptic
pulse.

No model is trained from scratch. The contribution is the orchestration layer:
concurrency under per-model timeouts, priority-based fusion, few-shot
personalisation with open-set rejection, and graceful degradation.

> **Disclaimer.** Research prototype. Not a medical device and not a substitute
> for hearing aids, cochlear implants, or human assistance.

---

## What it actually does

| Channel | Model | Status |
|---|---|---|
| Speech transcription | Whisper (faster-whisper, `distil-small.en` finals, `tiny` live partials) | Working, 6.9% WER on a LibriSpeech sample |
| Environmental sound classification | Audio Spectrogram Transformer (AudioSet) | Working, 77.8% top-3 on ESC-50 |
| Personal sound recognition | YamNet embeddings + prototypical networks | Working, few-shot enrolment |
| Visual hazard detection | YOLOv11 | Working |
| Scene description | BLIP | Working, throttled to every 4th tick |
| Emotion (tone) | DeepFace + wav2vec2, confidence-weighted fusion | Working |
| Sign recognition | MediaPipe Tasks + geometric rules + fingerspelling | Partial, see limitations |
| Sign to speech | LLM gloss polishing + neural TTS | Working |
| Name and keyword alerts | Transcript matching, word-boundary safe | Working |
| Caption reliability | Heuristic scorer from lip motion | Working (**not** lip reading) |

### Feedback from target users

The prototype has been demonstrated informally to Deaf and hard-of-hearing
users. This was unstructured formative feedback, with no measures collected
and nothing recorded, so it is design input rather than research data. The
reception was positive, and the response that best captures the intent was
that this is *"finally an app that lets us hear the world, not the other way
around"*.

Participants independently suggested the ability to enrol specific people so
the system recognises who is present and alerts on your own name, which
converges with the People feature and the name-alert channel already in the
build. A formal within-subjects study with measured outcomes is still to be
conducted; no claim of measured benefit is made on the basis of this feedback.

### Honest limitations

- **Sign recognition is constrained.** The TGCN tier trained on WLASL is
  inactive because no public checkpoint could be obtained, so vocabulary falls
  back to roughly 20 curated signs plus fingerspelling. Measured 5.9% top-1 on
  a 17-clip WLASL sample; see `docs/WLASL_EVAL_RESULTS.md`.
- **No lip reading.** `lip_reader.py` computes a transcript *reliability score*
  from mouth movement. It does not run AV-HuBERT or any audio-visual speech
  model.
- **Sound localisation was withdrawn.** GCC-PHAT direction-of-arrival is not
  feasible on laptop microphones (mono, closely spaced, browser audio
  processing removes the cues). It was replaced by name and keyword alerts.
- **No formal user study yet.** All current *quantitative* results are
  technical. Deaf and hard-of-hearing users have tried the prototype and
  responded positively, but that feedback was informal and unmeasured.

---

## Requirements

- Python 3.11 to 3.13
- `ffmpeg` on PATH (audio format conversion)
- A webcam and microphone
- Roughly 4 GB of disk for model weights, downloaded on first run
- No GPU required; everything runs on CPU

## Installation

```bash
git clone <your-repo-url>
cd DeafAccessibility

python3 -m venv backend/venv
source backend/venv/bin/activate          # Windows: backend\venv\Scripts\activate
pip install -r backend/requirements.txt

cp .env.example .env                      # then add your own API keys
```

`ffmpeg` must be installed separately (`brew install ffmpeg` on macOS,
`apt install ffmpeg` on Debian/Ubuntu).

### Configuration

Copy `.env.example` to `.env` and fill in the values you need. **Never commit
`.env`**; it is gitignored.

| Variable | Purpose | Required? |
|---|---|---|
| `GROQ_API_KEY` | LLM for gloss polishing and alert composition | Optional, degrades to plain text |
| `ACCESSIBILITY_LLM_MODEL` | Model id, default `openai/gpt-oss-20b` | Optional |
| `ELEVENLABS_API_KEY` | Neural voice for sign-to-speech | Optional, falls back to system TTS |
| `ACCESSIBILITY_WHISPER_SIZE` | `tiny`/`base`/`small`/`distil-small.en`, default `distil-small.en` | Optional |
| `HF_TOKEN` | HuggingFace, for gated model downloads | Optional |
| `PYANNOTE_TOKEN` | Speaker diarization | Optional, off by default |

Every external service is optional. Missing credentials degrade one channel
rather than breaking the application.

## Running

```bash
./start.sh
# or:  cd backend && source venv/bin/activate && python server.py
```

Then open **http://localhost:5051**.

On first start the server downloads model weights and prewarms them, which
takes roughly 20 to 60 seconds. Wait for `Models prewarmed ... ready` in the
log before clicking **Start Listening**, otherwise the first tick is slow.

### Usage walkthrough

1. **Live tab.** Click *Start Listening* and grant microphone and camera
   access. Speak and captions appear as you talk, first greyed (interim) then
   solid (final). Make a sharp sound such as a clap and it is classified in the
   Sounds panel. Point the camera at an object to populate Scene and Hazards.
2. **Personal Sounds tab.** Type a name, record three examples of a household
   sound, then *Enrol Sound*. Back on Live, trigger that sound: it fires with a
   confidence score and confirm/reject buttons. Confirming refines the stored
   prototype; rejecting raises that sound's threshold.
3. **People tab.** Set your own name so the system flashes when it is spoken.
   Optionally enrol a person's face and voice; when recognised, the Live view
   shows them and captions are attributed to them.
4. **Sign panel.** Click *Start signing*, sign a short phrase, then stop. The
   glosses are polished into a sentence and spoken aloud.
5. **Diary and Summary tabs.** Review a longitudinal log filtered by time and
   priority, or generate a one-paragraph recap.

## Testing

```bash
cd backend && source venv/bin/activate
python -m pytest tests/ -q
```

130 unit and integration tests. Unit tests cover pure logic (fusion ranking,
priority mapping, prototypical matching, the feedback loop, fingerspelling,
name matching). Integration tests exercise the HTTP layer, including graceful
degradation when audio or frames are absent.

## Evaluation

Scripts live in `backend/scripts/`. Each writes a JSON results file.

```bash
python scripts/eval_audio_scene.py     # ESC-50, sound classification
python scripts/eval_wlasl.py           # WLASL, sign recognition
python scripts/eval_personalizer.py    # personal sounds, precision/recall
python scripts/eval_whisper.py         # LibriSpeech, word error rate
python scripts/run_eval_matrix.py      # runs all, writes docs/EVAL_MATRIX.md
```

Datasets are not committed. Each folder under `backend/data/*_eval/` contains a
README with the recording protocol and expected layout.

The exact experiments behind the final report (model comparisons, the
personal-sound calibration, the Whisper decoding ablation, the language-model
benchmark and latency on recorded media) are in
`backend/scripts/report_experiments/`, with their JSON results committed in
`backend/eval_results/`. See that folder's README for which script produces
which table or figure.

## Project structure

```
backend/
  server.py            Flask app, 29 endpoints, the /process orchestration tick
  modules/             22 modules, one concern each, uniform Event interface
  scripts/             evaluation and dataset utilities
  tests/               unit and integration tests
frontend/
  index.html           single page, six live panels
  app.js               tick loop, streaming captions, rendering, enrolment
  style.css
docs/                  evaluation results and model notes
```

Architecture, data flow and the fusion algorithm are documented in the project
report (Chapter 3).

## Privacy

Audio and video are processed on the device and never transmitted or written
to disk. Only derived representations persist: mean embeddings for enrolled
sounds and people, and a diary of event labels, priorities and timestamps with
no audio. This is a structural property of the design, not a policy statement.

## Licence

Academic coursework, submitted for assessment. Third-party models remain under
their own licences (Whisper MIT, BLIP BSD-3, YOLOv11 AGPL-3.0, MediaPipe
Apache-2.0, DeepFace MIT). Note the AGPL-3.0 term attaching to YOLOv11 if this
work is ever distributed.
