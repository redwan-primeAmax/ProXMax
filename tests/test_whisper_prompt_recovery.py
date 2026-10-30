"""Regression coverage for Whisper decoder prompt-window exhaustion."""

from tests._support import *


class TestWhisperPromptRecovery(unittest.TestCase):
    def test_maximum_decoding_length_retries_without_hints(self):
        calls = []

        class _Model:
            def transcribe(self, samples, **kwargs):
                calls.append(dict(kwargs))

                def output():
                    if kwargs.get("hotwords") or kwargs.get("initial_prompt"):
                        raise ValueError("The maximum decoding length must be > 0")
                    yield types.SimpleNamespace(
                        text=" Hello.",
                        words=[types.SimpleNamespace(
                            start=0.1, end=0.5, word=" Hello.", probability=0.9
                        )],
                    )

                return output(), None

        stt = SpeechToText(model_size="tiny", hotwords="FC Mobile, Star Pass")
        stt._model = _Model()
        audio = np.zeros(28 * 16000, dtype=np.float32)
        words, text = stt._decode_words(
            audio, offset=2.0, initial_prompt="FC Mobile squad update"
        )

        self.assertEqual(len(calls), 3)
        self.assertIn("hotwords", calls[0])
        self.assertEqual(calls[1]["initial_prompt"], "FC Mobile squad update")
        self.assertNotIn("hotwords", calls[1])
        self.assertIsNone(calls[2]["initial_prompt"])
        self.assertEqual(text, "Hello.")
        self.assertAlmostEqual(words[0].start, 2.1)
        self.assertEqual(stt.last_hotwords_prompt, "")
        self.assertEqual(stt._prompt_fallback_stats["without_hotwords"], 1)
        self.assertEqual(stt._prompt_fallback_stats["without_context"], 1)
