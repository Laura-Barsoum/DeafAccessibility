# Personal sound evaluation set

Used by `backend/scripts/eval_personalizer.py` to compute within-user
precision, recall and macro-F1 for the few-shot personal sound recogniser.

## Layout
One sub-directory per sound. Inside each:
  enrol_01.wav, enrol_02.wav, enrol_03.wav   <- build the prototype
  test_01.wav ... test_05.wav                <- held-out positives

The held-out positives of every OTHER label act as distractor negatives,
so precision is measured without needing separately recorded negatives.

## Recording protocol
- 3 distinct household sounds (for example: doorbell, microwave finish,
  smoke-alarm test button).
- Record each in 2 different rooms to capture acoustic variation.
- 8 clips per sound per room (3 enrolment + 5 held-out) = 48 clips total.
- 2 to 3 seconds per clip, 16 kHz mono WAV, recorded on the same laptop
  microphone the system uses, so the evaluation matches deployment.
- Record with normal background conditions, not silence, otherwise the
  result is an optimistic upper bound.

## Consent
Record only in your own household with the consent of anyone present.
No speech should be captured. Only derived embeddings are persisted by
the system itself; these raw files are evaluation data only and are
gitignored.
