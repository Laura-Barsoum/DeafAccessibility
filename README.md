# A Multimodal AI Accessibility Assistant for Deaf and Hard-of-Hearing Users

Final-year project, BSc Computer Science, University of London.
Project template: CM3020 Artificial Intelligence, 4.1 Project Idea 1:
*Orchestrating AI models to achieve a goal*.

A browser-based assistant that combines **seven pre-trained perception models,
a sign network trained here and a language model on a laptop CPU** to provide situational awareness beyond
speech captioning.
It transcribes speech, classifies environmental sounds, recognises sounds the
user has personally enrolled, detects visual hazards, describes the scene,
reads emotional tone, and recognises 100 common signs. Detections
converge on a shared event bus that ranks them into four priority bands and
escalates genuinely urgent events with a sustained visual flash (vibration is
coded for phone browsers, but this prototype is served to the laptop only).

The only model trained here is the small sign network, trained on the app's own
MediaPipe keypoints because the published weights, learned from OpenPose
keypoints, reached only 8% top-1 on them. The contribution is the orchestration layer:
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
| Personal sound recognition | AST embeddings + prototypical networks | Working, recall 0.64 at 4.9% false alarms on an ESC-50 household proxy |
| Visual hazard detection | YOLOv11n | Working |
| Scene description | BLIP | Working, throttled to every 4th tick |
| Emotion (tone) | DeepFace + wav2vec2, confidence-weighted fusion | Working, 54.7% on FER-2013 test faces (an upper bound for webcam frames) |
| Sign recognition | TGCN trained on MediaPipe keypoints (WLASL100) + geometric rules + fingerspelling | Working for 100 signs: 60% top-1 on WLASL100 test clips |
| Sign to speech | LLM gloss polishing (`gpt-oss-20b` via Groq) + neural TTS | Working |
| Name and keyword alerts | Transcript matching, word-boundary safe | Working |
| Caption reliability | Heuristic scorer from lip motion | Working (**not** lip reading) |

Which models were tested and rejected, and why, is summarised in
[`docs/MODEL_EVALUATION.md`](docs/MODEL_EVALUATION.md).

### Feedback from target users

Five Deaf and hard-of-hearing people tried the prototype informally at several
stages of development. This was unstructured feedback, with no measures
collected and nothing recorded; it was not collected as research data and is not
presented as research findings. The
reception was positive, and the response that best captures the intent was
that this is *"finally an app that lets us hear the world, not the other way
around"*.

Participants independently suggested the ability to enrol specific people so
the system recognises who is present and alerts on your own name. That became
the People feature and the name alerts, and the same testers used the interface
both before and after it was added. The formal within-subjects study planned in
the preliminary report was not run because too few participants could be
recruited in time; no claim of measured benefit is made on the basis of this
feedback.

### Honest limitations

- **Sign recognition covers 100 signs.** A TGCN trained on the application's
  own MediaPipe keypoints from WLASL100 recognises 60% of 100 official test
  clips first time and 78% within its top five. Through the full sign cascade it
  scored 21% until the cascade was changed to let the trained network lead
  whenever it fires, which raised the cascade to 59%; the unmeasured cost is
  that a sign outside those 100 words is now labelled with the nearest of them.
  It knows only those 100 signs, was trained and tested on
  WLASL's signers, and has not been tested on webcam signing by Deaf users. Two
  silent defects had held sign recognition at 1%: a network no checkpoint
  fitted, and hand landmarks dropped on every frame. See
  `docs/WLASL_EVAL_RESULTS.md`.
- **Personal sounds miss about a third of plays.** On a household-class proxy
  built from ESC-50 and passed through the browser's Opus codec, the matcher
  recalls 0.64 of an enrolled sound's plays at a 4.9% false-alarm rate (0.63 at
  7.9% with three sounds enrolled). AST embeddings replaced YamNet's, which
  recalled 0.32 at 20.6% even on clean clips. It has not been evaluated on
  recordings from real homes.
- **Fusion depends on detection.** Merging related labels and ignoring
  transcripts when AST hears no speech removed repeated headlines on 35 scripted
  scenes (39% of slots before, none after). On a held-out scene set written
  before the change, the most urgent event led 84% of ticks (77% before). The
  remaining failures are sounds AST never labels as urgent, and on the original
  set 12 transcripts of sirens and crying still passed the speech gate. These
  scores are with live captions off; with them on, the default, captions and
  their name and keyword alerts bypass fusion and its speech gate, so a phantom
  caption can still appear. See `backend/eval_results/fusion_eval*.json`.
- **No lip reading.** `lip_reader.py` computes a transcript *reliability score*
  from mouth movement. It does not run AV-HuBERT or any audio-visual speech
  model.
- **Sound localisation was withdrawn.** GCC-PHAT direction-of-arrival is not
  feasible on laptop microphones (mono, closely spaced, browser audio
  processing removes the cues). It was replaced by name and keyword alerts.
- **No formal user study yet.** Too few participants could be recruited in
  time, so all *quantitative* results are technical. Deaf and hard-of-hearing
  users have tried the prototype and responded positively, but that feedback
  was informal and unmeasured.
- **Scene and hazard accuracy are unmeasured.** BLIP and YOLOv11n have
  latency measurements only; facial emotion scored 54.7% on FER-2013 test
  faces, a dataset DeepFace was trained on.

---

## Requirements

- Python 3.11 to 3.13
- `ffmpeg` on PATH (audio format conversion)
- A webcam and microphone
- Roughly 4 GB of disk for model weights, downloaded on first run
- No GPU required; everything runs on CPU

## Installation

```bash
git clone https://github.com/Laura-Barsoum/DeafAccessibility.git
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

`http://localhost:5051/health` reports which sound classifier is active. If
the Hugging Face Hub cannot be reached, AST loads from the local cache;
`"degraded": true` means it could not load at all and a weaker fallback is
classifying sounds.

### Usage walkthrough

1. **Live tab.** Click *Start Listening* and grant microphone and camera
   access. Speak and captions appear as you talk, first greyed (interim) then
   solid (final). Make a sharp sound such as a clap and it is classified in the
   Sounds panel. Point the camera at an object to populate Scene and Hazards.
2. **Personal Sounds tab.** Type a name, record three examples of a household
   sound, then *Enrol Sound*. Back on Live, trigger that sound: it fires with a
   confidence score and confirm/reject buttons. Confirming refines the stored
   prototype and loosens its match gate; rejecting tightens the gate, never
   below the distance of the farthest enrolment clip.
3. **People tab.** Set your own name so the system flashes when it is spoken.
   Optionally enrol a person's face; when recognised, the Live view
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

177 unit and integration tests. Unit tests cover pure logic (fusion ranking,
priority mapping, prototypical matching, the feedback loop, fingerspelling,
name matching, the language-model fallback chain). Integration tests exercise
the HTTP layer, including graceful degradation when audio or frames are absent.

## Evaluation

The experiments behind every table and figure in the final report's evaluation
chapter are in `backend/scripts/report_experiments/`. Each writes JSON to
`backend/eval_results/`, and the committed JSON files are the exact results the
report quotes. That folder's README maps each script to its table or figure.

```bash
cd backend && source venv/bin/activate
python scripts/report_experiments/yamnet_esc50.py               # YamNet on ESC-50, compared with AST
python scripts/report_experiments/fewshot_calibrate.py          # personal-sound calibration
python scripts/report_experiments/extract_librispeech_sample.py # speech sample, needed by the next line
python scripts/report_experiments/whisper_ablation.py           # WER by Whisper model and decoding setting
python scripts/report_experiments/llm_benchmark.py              # language-model benchmark, needs GROQ_API_KEY
python scripts/report_experiments/latency_on_recorded_media.py  # per-stage and end-to-end tick latency
python scripts/report_experiments/fewshot_embeddings.py         # personal sounds: AST against YamNet embeddings
python scripts/report_experiments/fewshot_deployment.py         # AST personal-sound settings on browser-coded ticks
python scripts/report_experiments/sign_wlasl100.py --select     # then without --select: WLASL100 test clips
python scripts/report_experiments/sign_keypoints.py             # MediaPipe keypoints for WLASL100 clips
python scripts/report_experiments/sign_train_tgcn.py            # train the sign model; then sign_wlasl100.py --trained
python scripts/report_experiments/fusion_scenes.py              # fusion on scripted scenes with known events
python scripts/report_experiments/fusion_scenes.py --scene-set heldout --out fusion_eval_heldout_after.json
python scripts/report_experiments/first_tick_probe.py           # per-stage timing of the first ticks after start-up
```

The per-channel scripts directly in `backend/scripts/` (`eval_*.py`) came
first. `eval_audio_scene.py` still produces the AST result on ESC-50, and
`run_eval_matrix.py` runs the per-channel scripts and writes
`docs/EVAL_MATRIX.md`. Datasets are not committed: ESC-50 is fetched by
`scripts/download_esc50.py`, and each folder under `backend/data/*_eval/`
contains a README with the expected layout.

## Project structure

```
backend/
  server.py              Flask app, 29 endpoints, the /process orchestration tick
  modules/               23 modules, one concern each, uniform Event interface
  scripts/               per-channel evaluation and dataset utilities
    report_experiments/  the experiments behind the final report
  eval_results/          committed JSON results quoted in the report
  tests/                 unit and integration tests
frontend/
  index.html             single page, six live panels
  app.js                 tick loop, streaming captions, rendering, enrolment
  style.css
docs/                    model selection summary, WLASL results, evaluation matrix
```

Architecture, data flow and the fusion algorithm are documented in the project
report (Chapter 3).

## Privacy

Raw audio and video never leave the machine and are not kept. Camera frames
stay in memory; each audio chunk is written to temporary files for decoding,
which are deleted as soon as each decoding or model call finishes. Only derived
data persists: mean embeddings for enrolled sounds and people, a diary of event
labels, priorities and timestamps with no audio (for speech, the label is the
first 80 characters of each final caption), and cached speech for sentences
the assistant has already spoken. Short text does leave the machine: sign
glosses, alert and summary text go to the language model, and sentences to be
spoken go to an online voice service when one is used.

## Licence

Academic coursework, submitted for assessment. Third-party models remain under
their own licences (Whisper MIT, BLIP BSD-3, YOLOv11 AGPL-3.0, MediaPipe
Apache-2.0, DeepFace MIT). Note the AGPL-3.0 term attaching to YOLOv11 if this
work is ever distributed. `backend/data/audioset_ontology.json`, used by fusion to
relate sound labels, is Google's AudioSet ontology
(github.com/audioset/ontology, Gemmeke et al. 2017), under CC BY-SA 4.0.
