"""Focused regression tests for ProXMax."""





from tests._support import *





class TestTranslatorParsing(unittest.TestCase):
    def test_parse_plain_json(self):
        raw = json.dumps({"translations": ["এক", "দুই"]})
        self.assertEqual(parse_translations(raw, 2), ["এক", "দুই"])

    def test_parse_fenced_and_list(self):
        self.assertEqual(parse_translations("```json\n[\"এক\"]\n```", 1), ["এক"])

    def test_parse_object_items(self):
        raw = json.dumps({"translations": [{"bn": "এক"}, {"text": "দুই"}]})
        self.assertEqual(parse_translations(raw, 2), ["এক", "দুই"])

    def test_explicit_paragraph_ids_restore_order(self):
        raw = json.dumps({"translations": [
            {"id": 2, "translation": "দুই"},
            {"id": 1, "translation": "এক"},
        ]})
        self.assertEqual(parse_translations(raw, 2, require_ids=True), ["এক", "দুই"])

    def test_required_ids_reject_positional_json_arrays(self):
        raw = json.dumps({"translations": ["এক", "দুই"]})
        with self.assertRaises(TranslationError):
            parse_translations(raw, 2, require_ids=True)

    def test_required_ids_reject_duplicate_or_missing_ids(self):
        for items in (
            [{"id": 1, "translation": "এক"}, {"id": 1, "translation": "দুই"}],
            [{"id": 1, "translation": "এক"}, {"id": 3, "translation": "দুই"}],
        ):
            with self.subTest(items=items), self.assertRaises(TranslationError):
                parse_translations(json.dumps({"translations": items}), 2, require_ids=True)

    def test_partial_id_salvage_keeps_only_contiguous_input_prefix(self):
        raw = (
            '{"translations": [{"id": 1, "translation": "এক"}, '
            '{"id": 2, "translation": "দুই"}, '
            '{"id": 3, "translation": "তিন'
        )
        self.assertEqual(salvage_translations(raw, 3, require_ids=True), ["এক", "দুই"])
        reordered = (
            '{"translations": [{"id": 2, "translation": "দুই"}, '
            '{"id": 1, "translation": "এক"}, '
            '{"id": 3, "translation": "তিন'
        )
        self.assertEqual(salvage_translations(reordered, 3, require_ids=True), [])

    def test_parse_count_mismatch(self):
        with self.assertRaises(TranslationError):
            parse_translations(json.dumps({"translations": ["এক"]}), 2)

    def test_parse_invalid_json(self):
        with self.assertRaises(TranslationError):
            parse_translations("not json", 3)

    def test_parse_survives_wrapping_text_and_fences(self):
        raw = 'Here are the lines:\n```json\n{"translations": ["এক", "দুই"]}\n```\nHope it helps!'
        self.assertEqual(parse_translations(raw, 2), ["এক", "দুই"])

    def test_parse_numbered_lines_when_model_ignores_json(self):
        raw = "1. আমি একটি স্টার পাস দেব।\n2. দ্বিতীয় লাইন।"
        self.assertEqual(parse_translations(raw, 2), ["আমি একটি স্টার পাস দেব।", "দ্বিতীয় লাইন।"])

    def test_truncated_answer_keeps_complete_lines(self):
        raw = '{"translations": ["এক", "দুই", "তিন'
        self.assertEqual(salvage_translations(raw, 3), ["এক", "দুই"])
        self.assertEqual(salvage_translations('{"lines": [{"bn": "এক"}, {"bn": "দুই"}', 3),
                         ["এক", "দুই"])


class TestTranslatorFactory(unittest.TestCase):
    def test_gemini_backend(self):
        translator = create_translator(
            "gemini", gemini_api_key="test-key", gemini_model="gemini-test"
        )
        self.assertIsInstance(translator, GeminiTranslator)
        self.assertEqual(translator.describe(), "gemini:gemini-test")
        translator.close()

    def test_missing_api_key_raises(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(GeminiAuthError):
                create_translator("gemini", gemini_api_key="")

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            create_translator("google")
        with self.assertRaises(ValueError):
            create_translator("nope")

    def test_env_key_is_used(self):
        with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}, clear=True):
            translator = create_translator("gemini")
        self.assertEqual(translator._api_key, "env-key")
        translator.close()
