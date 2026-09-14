# Facial emotion evaluation set

Used by `backend/scripts/eval_emotion.py`, which reports per-label
precision, recall, F1 and a confusion matrix.

## Layout
One sub-directory per Ekman label, each containing .jpg samples:
  happy/  sad/  angry/  fear/  surprise/  disgust/  neutral/

For the report, `scripts/report_experiments/emotion_fer2013.py` uses the
FER-2013 private test split (3,589 faces) laid out as
`fer2013/privateTest/<label>/*.png`. It was taken from the Hugging Face copy
`Aaryan333/fer2013_train_publicTest_privateTest` (file
`data/privateTest-00000-of-00001-4b8a0715cf1b7560.parquet`) and is not committed.

Aim for at least 10 images per label. Only use images you have the right
to use: your own photographs with consent, or a licensed academic dataset
such as FER-2013. Note in the report which source was used, since the
DeepFace recalibration was tuned against FER-2013's known neutral bias.
