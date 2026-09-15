"""
server.py — Flask backend for the Deaf/HoH Accessibility Assistant.

Endpoints
    GET  /                          static frontend
    GET  /health                    health check
    POST /process                   main per-tick multimodal fusion
    POST /enrol                     enrol a new personal sound
    GET  /enrolled                  list enrolled custom sounds
    DEL  /enrolled/<label>          remove an enrolled sound
    POST /summary                   long-form summary of the past hour
    GET  /diary                     soundscape diary query
    POST /reset                     clear in-memory state

Per-tick payload to /process:
    {
      "audio_b64": "<base64 wav>",
      "frames_b64": ["<jpeg b64>", ...]   // optional video frames
    }
"""
from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Load .env early so module env vars are available at first construction.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from modules import (  # noqa: E402
    audio_scene, diarizer, diary, emotion as emotion_mod, events as events_mod,
    face_tracker, fusion, hazard_detector, lip_reader, llm, localization,
    personalizer, scene_describer, sign_language, stt, summarizer,
    tts as tts_mod,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("accessibility.server")

# Eagerly initialise the transformers lazy-module in THIS (main) import
# thread. When prewarm and the first request tick both import from
# transformers concurrently, one thread can see a partially-initialised
# module and raise "cannot import name 'BlipProcessor' from transformers",
# which silently disables scene description. Importing once up front makes
# all later concurrent imports safe.
try:
    import transformers  # noqa: F401
    from transformers import (  # noqa: F401
        BlipProcessor, BlipForConditionalGeneration,
    )
except Exception as _e:  # pragma: no cover
    log.warning("transformers eager import failed (%s) — scene may be disabled", _e)

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# -----------------------------------------------------------------------
# Lazy-loaded engines
# -----------------------------------------------------------------------

_stt = None; _scene = None; _audio_scene = None
_face = None; _lip = None; _llm = None
_personal = None; _summer = None; _fuser = None
_diary = None; _diar = None
_slr = None; _hazard = None; _emo_audio = None; _emo_visual = None; _localizer = None
_tts = None


# ---------------------------------------------------------------------------
# Lazy-singleton accessors. Each get_*() constructs its model wrapper on first
# use and caches it in the module-level global above, so heavy models load
# once (and only when actually needed) rather than at import time. This keeps
# server start-up fast and lets the prewarm thread control load ordering.
# ---------------------------------------------------------------------------
def get_stt():
    global _stt
    if _stt is None:
        _stt = stt.SpeechToText()
    return _stt


_stream_stt = None
def get_stream_stt():
    """A dedicated fast STT (tiny model) used only for low-latency streaming
    partials. Finals still use the accurate main model."""
    global _stream_stt
    if _stream_stt is None:
        s = stt.SpeechToText()
        s.model_size = "tiny"   # force fast model for interim captions
        s._model = None
        _stream_stt = s
    return _stream_stt


def get_diarized_stt():
    """Whisper + pyannote orchestrator — returns transcription with
    speaker labels per segment (Sarah: …, Mike: …)."""
    from modules.diarized_stt import get_diarized_stt as _get
    return _get()


def get_audio_scene():
    global _audio_scene
    if _audio_scene is None:
        _audio_scene = audio_scene.AudioSceneClassifier()
    return _audio_scene


def get_scene():
    global _scene
    if _scene is None:
        _scene = scene_describer.SceneDescriber()
    return _scene


def get_face():
    global _face
    if _face is None:
        _face = face_tracker.FaceTracker()
    return _face


def get_lip():
    global _lip
    if _lip is None:
        _lip = lip_reader.LipReader()
    return _lip


def get_llm():
    global _llm
    if _llm is None:
        _llm = llm.LLM()
    return _llm


def get_personal():
    global _personal
    if _personal is None:
        # Personal sounds are embedded with AST when AST is the loaded classifier.
        _personal = personalizer.Personalizer(embedder=lambda b: get_audio_scene().embed_and_speech(b)[0])
    return _personal


def get_summer():
    global _summer
    if _summer is None:
        _summer = summarizer.Summarizer()
    return _summer


def get_fuser():
    global _fuser
    if _fuser is None:
        _fuser = fusion.Fusion()
    return _fuser


def get_diary():
    global _diary
    if _diary is None:
        path = os.environ.get(
            "ACCESSIBILITY_DB_PATH",
            os.path.join(os.path.dirname(__file__), "data", "diary.db"),
        )
        _diary = diary.Diary(path)
    return _diary


def get_diarizer():
    global _diar
    if _diar is None:
        _diar = diarizer.Diarizer()
    return _diar


def get_slr():
    global _slr
    if _slr is None:
        _slr = sign_language.SignLanguageRecognizer()
    return _slr


def get_hazard():
    global _hazard
    if _hazard is None:
        _hazard = hazard_detector.HazardDetector()
    return _hazard


def get_emo_audio():
    global _emo_audio
    if _emo_audio is None:
        _emo_audio = emotion_mod.AudioEmotionAnalyzer()
    return _emo_audio


def get_emo_visual():
    global _emo_visual
    if _emo_visual is None:
        _emo_visual = emotion_mod.VisualEmotionAnalyzer()
    return _emo_visual


def get_localizer():
    global _localizer
    if _localizer is None:
        _localizer = localization.SoundLocalizer()
    return _localizer


def get_tts():
    global _tts
    if _tts is None:
        _tts = tts_mod.TTSEngine()
    return _tts


# -----------------------------------------------------------------------
# Static
# -----------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the single-page frontend."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health")
def health():
    """Liveness probe returning service metadata and the active sound
    classifier, so a silent downgrade from AST is visible."""
    return jsonify({
        "ok": True,
        "service": "Deaf/HoH Accessibility Assistant",
        "version": "0.1.0",
        "timestamp": time.time(),
        # Reads state only; it never triggers a model load.
        "sound_classifier": get_audio_scene().status(),
    })


# -----------------------------------------------------------------------
# Main per-tick fusion endpoint
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Parallel Execution Engine
# -----------------------------------------------------------------------

import concurrent.futures
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

def _ffmpeg_convert(raw_bytes: bytes) -> bytes:
    """Convert browser-recorded audio (webm/opus from Chrome, mp4 from Safari)
    into 16 kHz mono PCM WAV that Whisper and AST can read.

    Uses temp FILES rather than stdin pipes: ffmpeg cannot always seek a
    container from a pipe, and a piped conversion can also stall on larger
    clips. Failures are logged (previously they were swallowed silently,
    which is why a broken audio path looked like "nothing is detected").
    Returns b"" on failure so the downstream models skip cleanly.
    """
    if not raw_bytes:
        return b""
    import subprocess, tempfile, os
    in_path = out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fin:
            fin.write(raw_bytes)
            in_path = fin.name
        out_path = in_path + ".wav"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", in_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", out_path],
            capture_output=True, timeout=10,
        )
        if proc.returncode != 0 or not os.path.exists(out_path):
            log.warning(
                "audio convert failed (rc=%s, in=%dB): %s",
                proc.returncode, len(raw_bytes),
                proc.stderr.decode(errors="ignore")[:200],
            )
            return b""
        with open(out_path, "rb") as f:
            wav = f.read()
        if len(wav) < 100:
            log.warning("audio convert produced empty WAV (in=%dB)", len(raw_bytes))
            return b""
        return wav
    except FileNotFoundError:
        log.error("ffmpeg not found on PATH — audio cannot be decoded. "
                  "Install ffmpeg or add it to PATH.")
        return b""
    except Exception as e:
        log.warning("audio convert exception (in=%dB): %s", len(raw_bytes), e)
        return b""
    finally:
        for p in (in_path, out_path):
            if p:
                try: os.unlink(p)
                except Exception: pass

@app.route("/process", methods=["POST"])
def process():
    """The main per-tick multimodal endpoint.

    Takes one audio chunk plus a few webcam frames. Six inference tasks
    (Whisper, AST, the personaliser, BLIP, the face tracker and YOLO) are
    dispatched concurrently to the thread pool, each collected under its own
    timeout so a slow model cannot stall the tick. Sign landmarks, voice and
    facial emotion, and face identification then run on the request thread.
    Outputs are fused into priority-ranked headlines and returned as one JSON
    document the frontend renders across its live panels. This is the
    orchestration hot path; heavier or optional work (BLIP scene captioning,
    the LLM notification) is throttled or gated to stay within the latency
    budget.
    """
    t0 = time.perf_counter()
    data = request.get_json(force=True)
    audio_b64 = data.get("audio_b64", "")
    frames_b64 = data.get("frames_b64", []) or []
    # When the frontend runs the streaming-caption loop, it owns captions and
    # name-call detection (via /stt/stream), so /process skips its own STT to
    # avoid running Whisper twice on the same audio.
    stream_captions = bool(data.get("stream_captions", False))

    # 1. Start all AI tasks in parallel, including audio conversion
    raw_audio = base64.b64decode(audio_b64) if audio_b64 else b""
    f_audio = _executor.submit(_ffmpeg_convert, raw_audio)
    
    # We need audio_bytes for some tasks, so we wait for conversion 
    # but conversion is very fast (<50ms).
    try:
        audio_bytes = f_audio.result(timeout=2)
    except Exception:
        audio_bytes = raw_audio

    # 2. Start heavy models.
    # STT uses plain Whisper on the hot path for speed and reliability.
    # Speaker diarization (pyannote) is OPT-IN via ACCESSIBILITY_ENABLE_DIARIZATION,
    # because pyannote is a ~1GB gated model whose first-load can stall every
    # tick and starve the short-timeout sound/hazard models, producing an
    # all-zeros UI. The diarized path and /speakers endpoints remain available
    # for environments where pyannote is properly configured.
    if audio_bytes and not stream_captions:
        if os.environ.get("ACCESSIBILITY_ENABLE_DIARIZATION", "").lower() in ("1", "true", "yes"):
            f_stt = _executor.submit(get_diarized_stt().transcribe_with_speakers, audio_bytes)
        else:
            f_stt = _executor.submit(get_stt().transcribe, audio_bytes)
    else:
        f_stt = None
    f_sound = _executor.submit(get_audio_scene().analyse, audio_bytes) if audio_bytes else None

    # LATENCY OPTIMISATION: BLIP scene captioning is slow (~2-3 s).
    # Sample every Nth tick — scene rarely changes second-to-second — and
    # CACHE the last result so every tick (even skipped ones) returns a
    # non-empty caption to the frontend.
    _scene_tick_counter = getattr(process, "_scene_tick_counter", 0) + 1
    process._scene_tick_counter = _scene_tick_counter
    SCENE_INTERVAL = 4        # describe scene every 4th tick (BLIP is the slowest model)
    middle_frame = frames_b64[len(frames_b64)//2] if frames_b64 else None
    # Run BLIP on the FIRST tick (counter == 1) so the user sees a scene
    # immediately, then every 3rd tick after that.
    should_run_blip = middle_frame and (
        _scene_tick_counter == 1
        or _scene_tick_counter % SCENE_INTERVAL == 0
        or not getattr(process, "_last_scene_caption", "")   # no scene yet -> describe now
    )
    f_scene = (
        _executor.submit(get_scene().describe_jpeg, middle_frame)
        if should_run_blip else None
    )
    f_face = _executor.submit(get_face().analyse_frames, frames_b64) if frames_b64 else None
    f_hazard = _executor.submit(get_hazard().detect_to_events, frames_b64) if frames_b64 else None
    
    # 3. Wait with per-model timeouts
    def _safe(future, default, timeout, name):
        if future is None:
            return default
        try:
            res = future.result(timeout=timeout)
            return res if res is not None else default
        except concurrent.futures.TimeoutError:
            log.warning("%s timed out after %.1fs — skipping this tick", name, timeout)
            return default
        except Exception as e:
            log.warning("%s failed: %s", name, e)
            return default

    # Whisper 'base' on CPU with 4s of audio commonly takes 4-8s.
    # Give it 25s and let other panels render in the meantime.
    stt_result      = _safe(f_stt,      {"text": "", "segments": [], "segments_with_speakers": [], "speaker": None, "speaker_name": None}, 25, "STT")
    sound_result    = _safe(f_sound,    {"events": [], "speech_probability": None, "embedding": None}, 8, "audio-scene")
    sound_events    = sound_result["events"]
    # Personal sounds reuse the embedding from AST's pass over this audio.
    f_personal = (_executor.submit(get_personal().match_current, audio_bytes, sound_result.get("embedding"))
                  if audio_bytes else None)
    personal_events = _safe(f_personal, [], 8,  "personalizer")
    raw_scene       = _safe(f_scene,    "", 12, "scene-BLIP")
    face_attribution = _safe(f_face,    {"speaker_attribution": None, "faces": []}, 6, "face")
    hazard_events, hazards_for_radar = _safe(f_hazard, ([], []), 10, "hazard-YOLO")

    # Per-tick audio-pipeline diagnostic. If captions/sounds are empty this
    # one line shows whether the problem is upstream (no raw audio reaching
    # the server), conversion (wav=0), or the models (text empty / sounds 0).
    log.info(
        "tick audio: raw=%dB wav=%dB frames=%d | stt=%r sounds=%d",
        len(raw_audio), len(audio_bytes), len(frames_b64),
        (stt_result.get("text") or "")[:50], len(sound_events),
    )

    # Scene caching: if BLIP ran this tick, store the result.
    # Otherwise return whatever we last successfully captioned.
    if raw_scene:
        process._last_scene_caption = raw_scene
    scene_caption = raw_scene or getattr(process, "_last_scene_caption", "")

    # 4b) NEW — Sign Language Recognition (Buffer-based, keep serial for now)
    sign_event = None
    sign_label = None
    if frames_b64:
        get_slr().push_frames(frames_b64)
        sign_event = get_slr().classify_to_event()
        if sign_event:
            sign_label = sign_event.label
            get_slr().reset_buffer()

    # 4c) Emotion fusion (multi-frame averaging + cross-tick EMA).
    # Single-frame DeepFace is famously noisy (smile → "angry" on one frame).
    # We now sample up to 5 frames per tick + apply a 0.35α EMA across ticks.
    audio_emo = get_emo_audio().analyse(audio_bytes) if audio_bytes else {}
    visual_emo = {}
    if frames_b64:
        sample = frames_b64 if len(frames_b64) <= 5 else frames_b64[::max(1, len(frames_b64) // 5)]
        visual_emo = get_emo_visual().analyse_frames(sample)
    fused_emotion = emotion_mod.fuse_emotions(audio_emo, visual_emo)

    # 4d) Sound localisation (GCC-PHAT) was withdrawn after prototyping: laptop
    #     microphones are closely spaced, often exposed as mono, and browser
    #     audio processing removes the inter-channel timing cues it relies on.
    #     The field is kept so the response shape is unchanged.
    localization_result = {"compass": "unknown", "withdrawn": True}

    # 5) Lip reliability
    mar_series = [face_attribution["speaker_attribution"].get("mar", 0)] if face_attribution.get("speaker_attribution") else []
    lip_reliability = get_lip().reliability(mar_series, 0.02)

    # 5b) Person identification + name-call detection
    #     - Identify any enrolled person visible in the frames (so captions
    #       can attribute speech to them via face proximity).
    #     - Scan the STT transcript for the user's own name (CRITICAL) and
    #       for enrolled-person mentions (INFORM).
    from modules.people import get_people
    people_reg = get_people()
    identified_person = None
    # Only run face identification when at least one person is enrolled.
    # Otherwise identify_face() still computes a DeepFace embedding on every
    # tick (hundreds of ms) for nothing, adding latency to the hot path.
    if frames_b64 and people_reg.profile.get("people"):
        try:
            mid = frames_b64[len(frames_b64) // 2]
            hit = people_reg.identify_face(mid)
            if hit:
                identified_person = {"name": hit[0], "similarity": round(hit[1], 3)}
        except Exception as e:
            log.debug("face identify failed: %s", e)

    name_call_events: list = []
    name_call_info: dict = {"self_called": False, "keywords_heard": [], "mentioned": []}
    if stt_result.get("text"):
        try:
            nc = people_reg.detect_name_calls(stt_result["text"])
            name_call_events = nc["events"]
            name_call_info = {
                "self_called": nc["self_called"],
                "keywords_heard": nc.get("keywords_heard", []),
                "mentioned": nc["mentioned"],
            }
        except Exception as e:
            log.debug("name-call detect failed: %s", e)

    # 6) Multimodal fusion
    all_sound_events = sound_events + hazard_events + name_call_events
    if sign_event: all_sound_events.append(sign_event)

    fused = get_fuser().fuse(
        stt_result, all_sound_events, personal_events,
        lip_reliability, face_attribution, scene_caption,
        speech_probability=sound_result.get("speech_probability"),
    )

    # --- BRIDGE TO FRONTEND ---
    fused["stt"] = stt_result
    fused["sounds"] = [e for e in fused["events"] if e["source"] == "sound"]
    fused["scene_caption"] = scene_caption
    fused["face"] = face_attribution
    fused["identified_person"] = identified_person   # {name, similarity} or None
    fused["name_call"] = name_call_info              # {self_called, mentioned}
    fused["lip_reliability"] = lip_reliability.get("reliability", 0.5)
    fused["hazards"] = hazards_for_radar
    fused["sign"] = ({"label": sign_label, "confidence": sign_event.confidence} if sign_event else None)
    fused["emotion"] = fused_emotion
    fused["localization"] = localization_result
    
    if stt_result.get("text"):
        fused["caption_with_emotion"] = emotion_mod.tag_caption(stt_result["text"], fused_emotion)

    # 7) LLM-composed notification — only for CRITICAL events, since
    #    each LLM call adds ~700-1000ms of latency.
    notification = ""
    has_critical = any(
        h.get("priority_name") == "CRITICAL"
        for h in fused.get("headlines", [])
    )
    if has_critical:
        try:
            notification = get_llm().compose_notification(
                fused["headlines"], scene_caption or ""
            )
        except Exception as e:
            log.warning("LLM notification failed: %s", e)
    fused["notification"] = notification

    # 8) Diary + summary buffers
    raw_events = (
        list(sound_events) + list(personal_events)
        + list(hazard_events)
        + ([events_mod.Event(
            source="speech", label=stt_result["text"][:80],
            priority=events_mod.Priority.IMPORTANT,
            confidence=0.7, text=stt_result["text"],
        )] if stt_result.get("text") else [])
    )
    if raw_events:
        try:
            get_diary().log_many(raw_events)
            get_summer().add_many(raw_events)
        except Exception as e:
            log.warning("diary log failed: %s", e)

    fused["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return jsonify(fused)


# -----------------------------------------------------------------------
# Personal sound enrolment
# -----------------------------------------------------------------------


@app.route("/enrol", methods=["POST"])
def enrol():
    """Enrol a personal sound. Browser-recorded clips (webm/opus) are decoded
    and converted to WAV before the personalizer embeds them, since the
    embedding models read WAV, not webm."""
    data = request.get_json(force=True)
    label = data.get("label", "").strip()
    priority_int = int(data.get("priority", events_mod.Priority.IMPORTANT))
    clips_b64 = data.get("clips_b64", [])
    if not label or not clips_b64:
        return jsonify({"ok": False, "reason": "missing_label_or_clips"}), 400

    clips = []
    for c in clips_b64:
        try:
            raw = base64.b64decode(c)
        except Exception:
            continue
        wav = _ffmpeg_convert(raw)
        if wav:
            clips.append(wav)
    if not clips:
        return jsonify({"ok": False, "reason": "audio_conversion_failed"}), 400

    res = get_personal().enrol(
        label, events_mod.Priority(priority_int), clips,
    )
    return jsonify(res)


@app.route("/enrolled", methods=["GET"])
def enrolled():
    """List the user's enrolled personal sounds."""
    return jsonify({"sounds": get_personal().list_enrolled()})


@app.route("/enrolled/feedback", methods=["POST"])
def enrolled_feedback():
    """Continuous-learning loop. The frontend echoes back a match_id from
    a previous personal-sound alert plus a thumbs up/down. Positive
    feedback folds that audio sample into the running embedding mean."""
    data = request.get_json(force=True) or {}
    match_id = (data.get("match_id") or "").strip()
    is_positive = bool(data.get("is_positive", True))
    if not match_id:
        return jsonify({"ok": False, "error": "match_id required"}), 400
    return jsonify(get_personal().feedback(match_id, is_positive))


@app.route("/enrolled/<label>", methods=["DELETE"])
def remove_enrolled(label):
    """Delete an enrolled personal sound by label."""
    ok = get_personal().remove(label)
    return jsonify({"ok": ok})


# -----------------------------------------------------------------------
# Long-form summary mode
# -----------------------------------------------------------------------


@app.route("/summary", methods=["POST"])
def summary():
    """Compose an LLM one-paragraph recap of recent events for Summary mode."""
    summer = get_summer()
    text = summer.summarise(get_llm())
    return jsonify({
        "summary": text,
        "stats": summer.stats(),
    })


# -----------------------------------------------------------------------
# Soundscape diary query
# -----------------------------------------------------------------------


@app.route("/diary", methods=["GET"])
def diary_query():
    """Query the anonymised soundscape diary: recent events plus a label
    histogram, both filtered by time window and minimum priority."""
    since = float(request.args.get("since_s", 86400))
    min_priority = int(request.args.get("min_priority", 0))
    return jsonify({
        "events": get_diary().query(since, min_priority),
        "histogram": get_diary().label_histogram(since, min_priority),
    })


@app.route("/reset", methods=["POST"])
def reset():
    """Start a fresh session: clear the summary buffer, emotion EMA and sign
    buffer so state does not leak between sessions."""
    global _summer
    _summer = summarizer.Summarizer()
    # Clear emotion EMA + sign buffer so each fresh session starts clean
    try:
        if _emo_visual is not None:
            _emo_visual.reset()
    except Exception: pass
    try:
        if _slr is not None:
            _slr.reset_buffer()
    except Exception: pass
    return jsonify({"ok": True})


# -----------------------------------------------------------------------
# NEW — Sign Language Recognition: enrol + classify
# -----------------------------------------------------------------------


@app.route("/sign/enrol", methods=["POST"])
def sign_enrol():
    """
    Enrol a custom sign from N example clips.
    Payload:
      { "label": "thank_you",
        "clip_frames_b64_lists": [[<frame>, ...], [<frame>, ...], ...] }
    """
    data = request.get_json(force=True)
    label = (data.get("label") or "").strip()
    clips = data.get("clip_frames_b64_lists", [])
    if not label or not clips:
        return jsonify({"ok": False, "reason": "missing_label_or_clips"}), 400
    res = get_slr().enrol_sign(label, clips)
    return jsonify(res)


@app.route("/sign/enrolled", methods=["GET"])
def sign_list_enrolled():
    """List any user-enrolled custom signs."""
    return jsonify({"signs": get_slr().list_enrolled_signs()})


@app.route("/sign/speak", methods=["POST"])
def sign_speak():
    """
    Polish a glossed string (sequence of sign labels) into natural English,
    then synthesize it via ElevenLabs/OpenAI/macOS for richer voice output.
    """
    data = request.get_json(force=True)
    gloss = (data.get("gloss") or "").strip()
    speak_audio = bool(data.get("speak_audio", True))
    if not gloss:
        return jsonify({"text": "", "audio_b64": ""})

    text = get_slr().gloss_to_speech(gloss, get_llm())
    audio_b64 = ""
    source = "none"
    if speak_audio and text:
        try:
            res = get_tts().speak(text)
            audio_b64 = res.get("audio_b64", "")
            source = res.get("source", "none")
        except Exception as e:
            log.warning("/sign/speak TTS failed: %s", e)

    return jsonify({"text": text, "audio_b64": audio_b64, "tts_source": source})


@app.route("/sign/recognize", methods=["POST"])
def sign_recognize():
    """
    Classify a recorded signing clip and return ONE polished sentence.

    Uses classify_all_frames_combined() which processes every frame in the
    recording without the 30-frame deque cap — the whole sentence is
    analysed at once, regardless of how long the user signed.

    Returns
        label / gloss  — raw detected sign labels ("hello you ok")
        text           — LLM-polished natural English ("Are you okay?")
        sequence       — ordered list of detected signs with frame indices
        audio_b64      — spoken sentence (ElevenLabs → OpenAI → macOS)
    """
    data = request.get_json(force=True)
    frames_b64 = data.get("frames_b64", []) or []
    speak_audio = bool(data.get("speak_audio", True))
    if not frames_b64:
        return jsonify({
            "label": "", "gloss": "", "text": "", "audio_b64": "",
            "sequence": [], "reason": "missing_frames",
        }), 400

    slr = get_slr()
    slr.reset_buffer()

    # ── One-shot combined classifier over ALL frames (no deque cap) ──
    # This is what ensures a user can sign "hello → you → ok" across
    # several seconds and get all three signs back, not just the last one.
    sequence, diag = slr.classify_all_frames_combined(
        frames_b64, window_size=8, stride=3,
    )

    total_hand_frames = max(
        diag["frames_with_hand_pretrained"],
        diag["frames_with_hand_geometric"],
    )

    if not sequence:
        if total_hand_frames == 0:
            return jsonify({
                "label": "", "gloss": "", "text": "", "audio_b64": "",
                "sequence": [], "diagnostics": diag,
                "reason": "no_hands_visible",
            })
        return jsonify({
            "label": "", "gloss": "", "text": "", "audio_b64": "",
            "sequence": [], "diagnostics": diag,
            "reason": "no_match",
        })

    # Build gloss + polish into one natural English sentence
    gloss = slr.sequence_to_gloss(sequence)
    text = slr.gloss_to_speech(gloss, get_llm()) if gloss else ""

    # Synthesize speech (ElevenLabs → OpenAI → macOS say)
    audio_b64 = ""
    tts_source = "none"
    if speak_audio and text and not text.startswith("[LLM"):
        try:
            res = get_tts().speak(text)
            audio_b64 = res.get("audio_b64", "")
            tts_source = res.get("source", "none")
        except Exception as e:
            log.warning("/sign/recognize TTS failed: %s", e)

    top_conf = max((s["confidence"] for s in sequence), default=0.0)
    slr.reset_buffer()

    return jsonify({
        "label": gloss,
        "gloss": gloss,
        "text": text or gloss,
        "sequence": sequence,
        "confidence": float(top_conf),
        "audio_b64": audio_b64,
        "tts_source": tts_source,
        "n_signs": len(sequence),
        "diagnostics": diag,
    })


# -----------------------------------------------------------------------
# NEW — Localization-only and Emotion-only debug endpoints
# -----------------------------------------------------------------------


@app.route("/localize", methods=["POST"])
def localize():
    """Run only the 3D-sound localization on a stereo audio clip."""
    data = request.get_json(force=True)
    audio_b64 = data.get("audio_b64", "")
    if not audio_b64:
        return jsonify({"compass": "unknown"})
    audio_bytes = base64.b64decode(audio_b64)
    return jsonify(get_localizer().localize(audio_bytes))


@app.route("/emotion", methods=["POST"])
def emotion_only():
    """Debug endpoint: run audio and/or visual emotion analysis on a single
    clip/frame and return the fused result, without the full /process tick."""
    data = request.get_json(force=True)
    audio_b64 = data.get("audio_b64", "")
    frame_b64 = data.get("frame_b64", "")
    a = get_emo_audio().analyse(base64.b64decode(audio_b64)) if audio_b64 else {}
    v = get_emo_visual().analyse_jpeg(frame_b64) if frame_b64 else {}
    return jsonify(emotion_mod.fuse_emotions(a, v))


@app.route("/tts", methods=["POST"])
def tts_endpoint():
    """
    Synthesize speech via the TTS chain (ElevenLabs → OpenAI → macOS → gTTS).
    Returns MP3 base64 the browser can play directly.
    """
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    voice_id = data.get("voice_id") or None
    if not text:
        return jsonify({"audio_b64": "", "source": "none"})
    return jsonify(get_tts().speak(text, voice_id))


@app.route("/stt/stream", methods=["POST"])
def stt_stream():
    """Low-latency partial-caption endpoint for streaming captions.

    The client streams short PCM-WAV buffers as it speaks. Interim buffers
    are transcribed with the fast tiny model; the final buffer of an
    utterance uses the accurate main model and also runs name/keyword
    detection and diary logging (which /process no longer does while
    streaming is active).
    """
    data = request.get_json(force=True) or {}
    audio_b64 = data.get("audio_b64", "")
    is_final = bool(data.get("is_final", False))
    if not audio_b64:
        return jsonify({"text": "", "is_final": is_final, "name_call": {}})
    try:
        raw = base64.b64decode(audio_b64)
    except Exception:
        return jsonify({"text": "", "is_final": is_final, "name_call": {}})

    engine = get_stt() if is_final else get_stream_stt()
    try:
        out = engine.transcribe(raw, with_timestamps=False)
    except Exception as e:
        log.warning("stream STT failed: %s", e)
        return jsonify({"text": "", "is_final": is_final, "name_call": {}})
    text = out.get("text", "")

    name_call = {"self_called": False, "keywords_heard": [], "mentioned": []}
    if is_final and text:
        try:
            from modules.people import get_people
            nc = get_people().detect_name_calls(text)
            name_call = {
                "self_called": nc["self_called"],
                "keywords_heard": nc.get("keywords_heard", []),
                "mentioned": nc["mentioned"],
            }
            get_diary().log_many([events_mod.Event(
                source="speech", label=text[:80],
                priority=events_mod.Priority.IMPORTANT,
                confidence=0.7, text=text,
            )])
            get_summer().add_many([events_mod.Event(
                source="speech", label=text[:80],
                priority=events_mod.Priority.IMPORTANT,
                confidence=0.7, text=text,
            )])
        except Exception as e:
            log.debug("stream final post-processing failed: %s", e)
    return jsonify({"text": text, "is_final": is_final, "name_call": name_call})


@app.route("/tts/status", methods=["GET"])
def tts_status():
    """Lets the frontend warn the user if ElevenLabs isn't configured
    (so they understand why the voice sounds robotic)."""
    return jsonify(get_tts().status())


# -----------------------------------------------------------------------
# Speaker name management — friendly names for diarized speakers
# (raw pyannote labels SPEAKER_00, SPEAKER_01 → "Sarah", "Mike")
# -----------------------------------------------------------------------
@app.route("/speakers", methods=["GET"])
def speakers_list():
    """Return all current friendly-name mappings + any raw IDs we've
    observed this session but haven't named yet."""
    return jsonify(get_diarized_stt().list_speakers())


@app.route("/speakers/name", methods=["POST"])
def speakers_name():
    """Body: { "raw_id": "SPEAKER_00", "name": "Sarah" }"""
    data = request.get_json(force=True) or {}
    raw_id = (data.get("raw_id") or "").strip()
    name = (data.get("name") or "").strip()
    if not raw_id or not name:
        return jsonify({"ok": False, "error": "raw_id and name required"}), 400
    get_diarized_stt().name_speaker(raw_id, name)
    return jsonify({"ok": True, "raw_id": raw_id, "name": name})


@app.route("/speakers/<raw_id>", methods=["DELETE"])
def speakers_forget(raw_id: str):
    """Forget a diarization speaker-id → friendly-name mapping."""
    get_diarized_stt().forget_speaker(raw_id)
    return jsonify({"ok": True, "removed": raw_id})


# -----------------------------------------------------------------------
# People enrolment — face + voice + name. Powers the name-call flash.
# -----------------------------------------------------------------------
@app.route("/people", methods=["GET"])
def people_list():
    """List enrolled people and the user's own name (for name-call alerts)."""
    from modules.people import get_people
    p = get_people()
    return jsonify({
        "people": p.list_all(),
        "self_name": p.get_self_name(),
    })


@app.route("/people/enrol", methods=["POST"])
def people_enrol():
    """Body: {
        "name": "Sarah",
        "frames_b64": [<jpeg-b64>, ...],   # face frames (optional)
        "audio_clips_b64": [<wav-b64>, ...] # voice clips (optional)
    }
    """
    from modules.people import get_people
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    frames = data.get("frames_b64") or []
    audio_clips_b64 = data.get("audio_clips_b64") or []
    audio_bytes_list = []
    for c in audio_clips_b64:
        try:
            audio_bytes_list.append(base64.b64decode(c))
        except Exception:
            continue
    res = get_people().enrol(name, frames, audio_bytes_list)
    return jsonify(res), (200 if res.get("ok") else 400)


@app.route("/people/<name>", methods=["DELETE"])
def people_remove(name: str):
    """Remove an enrolled person by name."""
    from modules.people import get_people
    ok = get_people().remove(name)
    return jsonify({"ok": ok})


@app.route("/me/name", methods=["POST"])
def people_set_self():
    """Body: { "name": "Laura" } — sets the user's own name so when it's
    heard in any transcript the system emits a CRITICAL name-call event."""
    from modules.people import get_people
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    get_people().set_self_name(name)
    return jsonify({"ok": True, "self_name": name or None})


@app.route("/keywords", methods=["GET"])
def keywords_get():
    """Return the critical-keyword watch-list."""
    from modules.people import get_people
    return jsonify({"keywords": get_people().get_keywords()})


@app.route("/keywords", methods=["POST"])
def keywords_set():
    """Body: { "keywords": ["fire", "help", ...] } — words that trigger a
    CRITICAL alert (and screen flash) when spoken aloud."""
    from modules.people import get_people
    data = request.get_json(force=True) or {}
    words = data.get("keywords")
    if not isinstance(words, list):
        return jsonify({"ok": False, "error": "keywords must be a list"}), 400
    get_people().set_keywords(words)
    return jsonify({"ok": True, "keywords": get_people().get_keywords()})


@app.route("/hazards", methods=["POST"])
def hazards_only():
    """Debug endpoint: run YOLO hazard detection on frames only."""
    data = request.get_json(force=True)
    frames_b64 = data.get("frames_b64", []) or []
    events_, hazards = get_hazard().detect_to_events(frames_b64)
    return jsonify({
        "hazards": hazards,
        "events": [e.to_dict() for e in events_],
    })


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def _prewarm() -> None:
    """Load every model once at startup so the first real tick is fast.

    Without this, the first /process tick must cold-load all six models
    (~15-20s), so in a short session only one tick completes and the UI
    appears to update only when the user clicks Stop. We warm the models
    by running a single synthetic tick through the real pipeline.
    """
    import io, wave, time
    import numpy as np
    try:
        log.info("Prewarming models (first load can take ~20s)...")
        t0 = time.time()
        buf = io.BytesIO()
        w = wave.open(buf, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(np.zeros(8000, dtype=np.int16).tobytes()); w.close()
        audio_b64 = base64.b64encode(buf.getvalue()).decode()
        frames = []
        try:
            from PIL import Image
            im = Image.new("RGB", (320, 240), (127, 127, 127))
            fb = io.BytesIO(); im.save(fb, "JPEG")
            frames = [base64.b64encode(fb.getvalue()).decode()] * 3
        except Exception:
            pass
        with app.test_client() as c:
            c.post("/process", json={"audio_b64": audio_b64,
                                     "frames_b64": frames, "audio_rms_hint": 0.0})
        # Don't let the synthetic grey warm-up frame pollute the live scene
        # cache or consume the "run BLIP on tick 1" slot. Clear both the
        # server-side cache AND the SceneDescriber's own internal cache.
        try:
            process._scene_tick_counter = 0
            process._last_scene_caption = ""
            sd = get_scene()
            sd._last_caption = None
            sd._last_caption_ts = 0.0
        except Exception:
            pass
        # Warm the tiny model used for streaming-caption partials.
        try:
            get_stream_stt()._ensure_loaded()
        except Exception:
            pass
        log.info("Models prewarmed in %.0fs — ready.", time.time() - t0)
    except Exception as e:
        log.warning("Prewarm failed (models will load lazily on first tick): %s", e)


if __name__ == "__main__":
    port = int(os.environ.get("ACCESSIBILITY_PORT", 5051))
    log.info("Deaf/HoH Accessibility Assistant on http://127.0.0.1:%d", port)
    # Warm models in the background so the server starts serving immediately
    # but the first user tick is fast. Disable with ACCESSIBILITY_NO_PREWARM=1.
    if os.environ.get("ACCESSIBILITY_NO_PREWARM", "").lower() not in ("1", "true", "yes"):
        import threading
        threading.Thread(target=_prewarm, daemon=True).start()
    app.run(host="127.0.0.1", port=port, debug=False)
