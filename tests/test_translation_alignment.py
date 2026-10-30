"""Regression tests for stable Gemini paragraph-to-segment mapping."""

from tests._support import *


class TestTranslationAlignment(unittest.TestCase):
    def test_multi_paragraph_response_ids_keep_texts_attached_to_segments(self):
        def responder(_contents, _call_no):
            return json.dumps({"translations": [
                {"id": 2, "translation": "দুই"},
                {"id": 1, "translation": "এক"},
            ]})

        client = _FakeClient(responder)
        translator = _make_translator(client, batch_size=2)
        segments = _segments(["First source.", "Second source."])
        translator.translate_segments(segments)
        self.assertEqual([segment.text_bn for segment in segments], ["এক", "দুই"])

    def test_multi_paragraph_json_without_ids_falls_back_to_numbered_output(self):
        class _NoIdClient:
            def __init__(self):
                self.calls = []
                self.models = self

            def generate_content(self, *, model, contents, config):
                self.calls.append({"model": model, "contents": contents, "config": config})
                if len(self.calls) < 3:
                    return _FakeResponse(json.dumps({"translations": ["এক", "দুই"]}))
                return _FakeResponse("2. দুই\n1. এক")

        client = _NoIdClient()
        translator = _make_translator(client, batch_size=2, max_retries=4)
        segments = _segments(["First source.", "Second source."])
        with mock.patch("proxmax.translator.time.sleep"):
            translator.translate_segments(segments)
        self.assertEqual([segment.text_bn for segment in segments], ["এক", "দুই"])
        self.assertEqual(len(client.calls), 3)
        self.assertIn("numbered translations", client.calls[2]["contents"])

    def test_gemini_request_groups_are_capped_at_twelve_paragraphs(self):
        translator = _make_translator(
            _FakeClient(lambda _contents, _call_no: "{}"), batch_size=100
        )
        segments = _segments([f"Source paragraph {index}." for index in range(25)])
        groups = translator.batch_segments(segments)
        self.assertEqual([len(group) for group in groups], [12, 12, 1])
        self.assertEqual(translator.batch_size, 12)
