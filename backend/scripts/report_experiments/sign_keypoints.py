"""[Final report: Section 5.2 signing]

MediaPipe keypoints for WLASL100 clips, extracted as the application extracts them.

Each clip is cut to its annotated frame range and 50 frames are sampled evenly
(the TGCN's input length). Every frame goes through the application's own
MediaPipe Tasks adapter (sign_language._TasksHolistic) with its face detector
off, since no sign feature uses the face mesh. Per frame the script keeps the 33
pose landmarks and the two hand slots exactly as the adapter fills them, with
presence flags, so the 55-keypoint layout and coordinate conventions can be
varied later without running MediaPipe again.

Splits: train and val are the clips of the official WLASL100 train and
validation splits still on the Voxel51 mirror (data/WLASL-master/
videos_trainval100); test is data/WLASL-master/videos_test100. Neither folder is
committed. Writes eval_results/sign_keypoints_<split>.npz (not committed).
"""
from pathlib import Path
import argparse, json, os, sys, time
from multiprocessing import Pool

import numpy as np

BACKEND = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)
SP = os.path.join(BACKEND, "eval_results")
N_FRAMES = 50

MAN = json.load(open("data/WLASL-master/start_kit/WLASL_v0.3.json"))
INST = {i["video_id"]: (e["gloss"], i) for e in MAN[:100] for i in e["instances"]}
DIRS = {"train": "data/WLASL-master/videos_trainval100", "val": "data/WLASL-master/videos_trainval100",
        "test": "data/WLASL-master/videos_test100"}


def clip_ids(split):
    labels = json.load(open(os.path.join(DIRS[split], "labels.json")))
    ids = sorted(labels) if split == "test" else sorted(v for v, d in labels.items() if d["split"] == split)
    ids = [v for v in ids if os.path.exists(os.path.join(DIRS[split], v + ".mp4"))]
    assert all(INST[v][1]["split"] == split for v in ids), f"a {split} clip is not in the official {split} split"
    return ids


_H = None


def _init():
    global _H
    from modules.sign_language import _TasksHolistic
    _H = _TasksHolistic(os.path.join(BACKEND, "data"))
    try:
        _H._face.close()
    except Exception:
        pass
    _H._face = None          # the adapter catches the missing detector and leaves the face empty


def sampled_frames(split, vid):
    import cv2
    cap = cv2.VideoCapture(os.path.join(DIRS[split], vid + ".mp4"))
    allf = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        allf.append(fr)
    cap.release()
    inst = INST[vid][1]
    s, e = max(int(inst.get("frame_start", 1)) - 1, 0), int(inst.get("frame_end", -1))
    seg = allf[s:(e if e > 0 else None)] or allf
    if not seg:
        return []
    return [seg[i] for i in np.linspace(0, len(seg) - 1, N_FRAMES).astype(int)]


def one(job):
    import cv2
    split, vid = job
    pose = np.zeros((N_FRAMES, 33, 2), np.float32)
    right = np.zeros((N_FRAMES, 21, 2), np.float32)
    left = np.zeros((N_FRAMES, 21, 2), np.float32)
    has = np.zeros((N_FRAMES, 3), bool)
    frames = sampled_frames(split, vid)
    for t, fr in enumerate(frames):
        res = _H.process(np.ascontiguousarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        for k, (name, arr) in enumerate((("pose_landmarks", pose), ("right_hand_landmarks", right),
                                         ("left_hand_landmarks", left))):
            lms = getattr(res, name, None)
            if lms:
                arr[t] = [(p.x, p.y) for p in lms.landmark]
                has[t, k] = True
    return vid, INST[vid][0], len(frames), pose, right, left, has


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    for split in args.splits:
        ids, t0, out = clip_ids(split), time.time(), []
        with Pool(args.workers, initializer=_init) as pool:
            for k, r in enumerate(pool.imap(one, [(split, v) for v in ids], chunksize=2)):
                out.append(r)
                if k % 50 == 0:
                    print(f"{split}: {k}/{len(ids)} clips, {time.time() - t0:.0f}s", flush=True)
        vids, glosses, n_frames, P, R, L, H = zip(*out)
        np.savez_compressed(os.path.join(SP, f"sign_keypoints_{split}.npz"), video=np.array(vids), gloss=np.array(glosses),
                            n_frames=np.array(n_frames), pose=np.stack(P), right=np.stack(R), left=np.stack(L),
                            has=np.stack(H))
        print(f"{split}: {len(ids)} clips in {time.time() - t0:.0f}s; frames with a pose "
              f"{np.stack(H)[..., 0].mean():.2f}, with a hand {np.stack(H)[..., 1:].any(-1).mean():.2f}", flush=True)
