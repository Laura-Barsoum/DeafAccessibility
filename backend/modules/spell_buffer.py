"""
spell_buffer.py — Buffer ASL fingerspelled letters into words.

When the user fingerspells (Y → O → U → pause → O → K), each per-frame letter
prediction from the ASL letter classifier streams in here. The buffer:

  1. Smooths consecutive predictions (3 frames of the same letter = accept)
  2. Detects word boundaries — a pause of N "no letter" frames flushes the
     current letter buffer into a word
  3. Returns the spelled word(s) on demand for LLM polishing

This is what gives the system unlimited vocabulary — any English word can be
spelled letter-by-letter.

Usage:
    buf = SpellBuffer()
    for frame in frames:
        buf.feed(per_frame_letter_or_none)
    words = buf.flush()         # ["YOU", "OK"]
"""
from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional

log = logging.getLogger("accessibility.spell")


class SpellBuffer:
    """
    Streaming letter → word buffer.

    Parameters
    ----------
    confirm_frames : int
        How many consecutive frames of the same letter are needed before
        the letter is "confirmed" and appended to the current word. Higher
        is more robust against single-frame noise but adds latency.
    word_pause_frames : int
        How many consecutive "no letter" frames signal a word boundary.
    max_word_length : int
        Safety cap to prevent runaway buffering on noisy detection.
    """

    def __init__(
        self,
        confirm_frames: int = 3,
        word_pause_frames: int = 8,
        max_word_length: int = 30,
    ) -> None:
        self.confirm_frames = confirm_frames
        self.word_pause_frames = word_pause_frames
        self.max_word_length = max_word_length

        self._recent: deque = deque(maxlen=confirm_frames)
        self._current_word: List[str] = []
        self._words: List[str] = []
        self._pause_count: int = 0
        self._last_committed: Optional[str] = None

    # ------------------------------------------------------------------

    def feed(self, letter: Optional[str]) -> None:
        """
        Feed one frame's letter prediction. `letter` may be None or "" if
        no clear letter was detected.

        Letter labels coming from the ASL letter classifier look like
        "letter_a" — we strip the prefix and uppercase for the word buffer.
        """
        if letter and letter.startswith("letter_"):
            ch = letter[7:].upper()
            if not ch.isalpha() or len(ch) != 1:
                ch = None
        elif letter and len(letter) == 1 and letter.isalpha():
            ch = letter.upper()
        else:
            ch = None

        # Track this letter in the recent-history window
        self._recent.append(ch)

        if ch is None:
            self._pause_count += 1
            if (
                self._pause_count >= self.word_pause_frames
                and self._current_word
            ):
                self._commit_word()
            return

        self._pause_count = 0

        # Confirm: the same letter must appear in all recent frames
        if (
            len(self._recent) >= self.confirm_frames
            and all(r == ch for r in self._recent)
            and ch != self._last_committed
        ):
            if len(self._current_word) < self.max_word_length:
                self._current_word.append(ch)
                self._last_committed = ch

    def _commit_word(self) -> None:
        if not self._current_word:
            return
        word = "".join(self._current_word)
        self._words.append(word)
        log.info("SpellBuffer committed word: %s", word)
        self._current_word = []
        self._last_committed = None

    # ------------------------------------------------------------------

    def flush(self) -> List[str]:
        """Force a word boundary and return the accumulated words list."""
        self._commit_word()
        out = list(self._words)
        return out

    def reset(self) -> None:
        self._recent.clear()
        self._current_word = []
        self._words = []
        self._pause_count = 0
        self._last_committed = None

    def current_state(self) -> dict:
        """Snapshot for debugging or UI display."""
        return {
            "current_word": "".join(self._current_word),
            "committed_words": list(self._words),
            "pause_frames": self._pause_count,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def from_letter_sequence(letter_seq: List[Optional[str]],
                              confirm_frames: int = 3,
                              word_pause_frames: int = 8) -> List[str]:
        """
        One-shot helper for a complete recording: pass the full
        per-frame letter list, get the word list back.
        """
        buf = SpellBuffer(
            confirm_frames=confirm_frames,
            word_pause_frames=word_pause_frames,
        )
        for letter in letter_seq:
            buf.feed(letter)
        return buf.flush()
