# Facial emotion evaluation set

Used by `backend/scripts/eval_emotion.py`, which reports per-label
precision, recall, F1 and a confusion matrix.

## Layout
One sub-directory per Ekman label, each containing .jpg samples:
  happy/  sad/  angry/  fear/  surprise/  disgust/  neutral/

Aim for at least 10 images per label. Only use images you have the right
to use: your own photographs with consent, or a licensed academic dataset
such as FER-2013. Note in the report which source was used, since the
DeepFace recalibration was tuned against FER-2013's known neutral bias.
