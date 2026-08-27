"""
prewarm_models.py — Pre-downloads and caches all AI models for the assistant.
"""
import os
import sys
import logging

# Add current directory to path
sys.path.append(os.getcwd())

# Set SSL Certificates for the download
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("prewarm")

def prewarm():
    log.info("Starting model pre-warming. This may take a few minutes...")
    
    try:
        from modules import stt, audio_scene, scene_describer, face_tracker, hazard_detector
        
        log.info("--- [1/5] Loading Speech-to-Text (Whisper)...")
        stt_obj = stt.SpeechToText()
        stt_obj._ensure_loaded()
        
        log.info("--- [2/5] Loading Audio Scene Classifier (YamNet)...")
        asc_obj = audio_scene.AudioSceneClassifier()
        asc_obj._ensure_loaded()
        
        log.info("--- [3/5] Loading Scene Describer (BLIP)...")
        sd_obj = scene_describer.SceneDescriber()
        sd_obj._ensure_loaded()
        
        log.info("--- [4/5] Loading Face Tracker (MediaPipe)...")
        ft_obj = face_tracker.FaceTracker()
        ft_obj._ensure_loaded()
        
        log.info("--- [5/6] Loading Hazard Detector (YOLO)...")
        hd_obj = hazard_detector.HazardDetector()
        hd_obj._ensure_loaded()

        log.info("--- [6/6] Loading Emotion Analyzers...")
        from modules import emotion
        ae_obj = emotion.AudioEmotionAnalyzer()
        ae_obj._ensure_loaded()
        ve_obj = emotion.VisualEmotionAnalyzer()
        ve_obj._ensure_loaded()
        
        log.info("Success! All models are downloaded and cached.")
    except Exception as e:
        log.error(f"Error during pre-warm: {e}")
        # sys.exit(1) # Don't exit, allow the server to try later if just one fails

if __name__ == "__main__":
    prewarm()
