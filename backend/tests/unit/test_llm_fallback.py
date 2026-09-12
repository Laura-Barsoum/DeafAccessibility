"""Unit tests for the LLM model fallback chain, using a fake client.

Regression tests for two real defects:
  - a provider withdrew the configured model, and the error string was shown
    in the user interface;
  - a fallback model rejected the `reasoning_effort` parameter with HTTP 400,
    which the chain did not retry, so that fallback could never succeed.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules.llm import LLM  # noqa: E402


def _resp(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeCompletions:
    """Scripted behaviour per model; records every call's kwargs."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        action = self.behaviour[kwargs["model"]]
        return action(kwargs)


def _client(behaviour):
    comp = FakeCompletions(behaviour)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


def _raise(msg):
    def f(_kwargs):
        raise Exception(msg)
    return f


class LLMFallbackTests(unittest.TestCase):
    """The chain must degrade, never surface provider errors as text."""

    def _llm(self, models, behaviour):
        llm = LLM()
        llm._models = list(models)
        llm._model = models[0]
        llm._client, comp = _client(behaviour)
        return llm, comp

    def test_withdrawn_model_is_skipped_and_marked_dead(self):
        llm, _ = self._llm(["gone", "ok"], {
            "gone": _raise("Error code: 404 - model_not_found"),
            "ok": lambda k: _resp("Are you okay?"),
        })
        self.assertEqual(llm._chat([{"role": "user", "content": "you ok"}]), "Are you okay?")
        self.assertIn("gone", llm._dead)

    def test_decommissioned_model_is_marked_dead(self):
        llm, _ = self._llm(["old", "ok"], {
            "old": _raise("Error code: 400 - model_decommissioned"),
            "ok": lambda k: _resp("Hello."),
        })
        llm._chat([{"role": "user", "content": "hello"}])
        self.assertIn("old", llm._dead)

    def test_model_rejecting_reasoning_effort_is_retried_without_it(self):
        def picky(kwargs):
            if "reasoning_effort" in kwargs:
                raise Exception("Error code: 400 - `reasoning_effort` is not supported with this model")
            return _resp("Help me!")
        llm, comp = self._llm(["picky"], {"picky": picky})
        self.assertEqual(llm._chat([{"role": "user", "content": "help me"}]), "Help me!")
        self.assertEqual(len(comp.calls), 2)
        self.assertNotIn("reasoning_effort", comp.calls[1])
        self.assertNotIn("picky", llm._dead)

    def test_empty_content_moves_to_next_model(self):
        llm, _ = self._llm(["mute", "ok"], {
            "mute": lambda k: _resp(""),
            "ok": lambda k: _resp("Thank you."),
        })
        self.assertEqual(llm._chat([{"role": "user", "content": "thank you"}]), "Thank you.")

    def test_total_failure_returns_empty_string_not_error_text(self):
        llm, _ = self._llm(["a", "b"], {
            "a": _raise("Error code: 404 - model_not_found"),
            "b": _raise("Error code: 503 - service unavailable"),
        })
        out = llm._chat([{"role": "user", "content": "you ok"}])
        self.assertEqual(out, "")
        self.assertNotIn("error", out.lower())

    def test_dead_model_is_not_called_again(self):
        llm, comp = self._llm(["gone", "ok"], {
            "gone": _raise("model_not_found"),
            "ok": lambda k: _resp("Yes."),
        })
        llm._chat([{"role": "user", "content": "yes"}])
        llm._chat([{"role": "user", "content": "yes"}])
        self.assertEqual(sum(1 for c in comp.calls if c["model"] == "gone"), 1)


if __name__ == "__main__":
    unittest.main()
