"""[Final report: Section 5.2 sound events, Figure 5.1a (AST figure: scripts/eval_audio_scene.py)]

Run the existing ESC-50 evaluation with YamNet forced as the classifier,
so AST (77.8% top-3, already measured) can be compared against YamNet on
the same 2,000 clips, the same label map and the same top-3 scoring."""
from pathlib import Path
import csv, os, ssl, sys

BACKEND = str(Path(__file__).resolve().parents[2])
SP = os.path.join(BACKEND, "eval_results")
os.makedirs(SP, exist_ok=True)
OUT = os.path.join(SP, "yamnet_esc50_results.json")
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))
os.chdir(BACKEND)
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass
ssl._create_default_https_context = ssl._create_unverified_context

from modules import audio_scene as asm  # noqa: E402


def _yamnet_only(self):
    if self._model is not None:
        return
    import tensorflow as tf
    import tensorflow_hub as hub
    from modules.model_cache import prepare_tfhub_cache
    print("tfhub cache:", prepare_tfhub_cache(), flush=True)
    self._model = hub.load("https://tfhub.dev/google/yamnet/1")
    path = self._model.class_map_path().numpy()
    path = path.decode() if isinstance(path, bytes) else path
    with tf.io.gfile.GFile(path) as fh:
        rows = list(csv.reader(fh))          # csv-aware: names contain commas
    self._labels = [r[2] for r in rows[1:]]
    self._backend = "yamnet"
    print(f"YamNet forced: {len(self._labels)} labels", flush=True)


asm.AudioSceneClassifier._ensure_loaded = _yamnet_only
import eval_audio_scene as ev  # noqa: E402

sys.argv = ["eval_audio_scene.py", "--top-k", "3", "--out", OUT]
ev.main()
print("DONE yamnet", flush=True)
