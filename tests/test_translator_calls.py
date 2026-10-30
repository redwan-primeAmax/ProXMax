"""Focused regression tests for ProXMax."""





from tests._support import *





class TestTranslatorCalls(unittest.TestCase):
    def test_batches_and_dedupe(self):
        def responder(contents, _n):
            return json.dumps(
                {"translations": [f"বাংলা: {t}" for t in _lines_from_contents(contents)]}
            )

        client = _FakeClient(responder)
        translator = _make_translator(client, batch_size=2)
        segments = _segments(["One.", "Two.", "One.", "Three.", "Four."])
        translator.translate_segments(segments)
        self.assertEqual(segments[0].text_bn, segments[2].text_bn)   # dedupe
        self.assertTrue(all(s.text_bn.startswith("বাংলা:") for s in segments))
        self.assertEqual(len(client.calls), 2)  # ৪টি অনন্য লাইন, ব্যাচ ২

    def test_empty_texts_are_not_sent(self):
        client = _FakeClient(lambda c, n: json.dumps({"translations": []}))
        translator = _make_translator(client)
        segments = _segments(["", "  "])
        translator.translate_segments(segments)
        self.assertEqual(client.calls, [])

    def test_network_error_is_retried(self):
        connect_error = type("ConnectError", (Exception,), {})
        connect_error.__module__ = "httpx"

        def responder(contents, call_no):
            if call_no == 1:
                raise connect_error("connection refused")
            return json.dumps({"translations": ["বাংলা"]})

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=3)
        with mock.patch("proxmax.translator.time.sleep"):
            result = translator.translate_text("Hello")
        self.assertEqual(result, "বাংলা")
        self.assertEqual(len(client.calls), 2)

    def test_all_retries_fail_is_graceful(self):
        def responder(contents, _n):
            raise TimeoutError("request timed out")

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=2)
        with mock.patch("proxmax.translator.time.sleep"):
            with self.assertRaises(TranslationError):
                translator.translate_text("Hello")
        self.assertEqual(len(client.calls), 2)

    def test_blocked_content_is_not_retried(self):
        class _Blocked(_FakeClient):
            def generate_content(self, *, model, contents, config):
                self.calls.append({"model": model, "contents": contents, "config": config})
                return _FakeResponse(None, block_reason="SAFETY")

        client = _Blocked(lambda c, n: "")
        translator = _make_translator(client, max_retries=5)
        with mock.patch("proxmax.translator.time.sleep"):
            with self.assertRaises(TranslationError):
                translator.translate_text("Hello")
        self.assertEqual(len(client.calls), 1)
        self.assertIsNone(translator._fatal_error)

    def test_pre_tts_shorten_predictor_is_disabled(self):
        translator = _make_translator(_FakeClient(lambda _c, _n: "{}"))
        original = "এটি একটি বেশ লম্বা বাক্য যা হয়তো পরে fit করবে"
        segment = Segment(index=0, start=0.0, end=0.5, text_bn=original)
        with mock.patch.object(translator, "_run_batches") as run:
            changed = translator.shorten_to_fit([segment], max_tempo=1.15)
        self.assertEqual(changed, 0)
        self.assertEqual(segment.text_bn, original)
        run.assert_not_called()

    def test_normal_gemini_request_uses_text_only_paragraphs(self):
        client = _FakeClient(lambda contents, _n: json.dumps({"translations": ["বাংলা।"]}))
        translator = _make_translator(client)
        segment = Segment(index=0, start=12.5, end=16.0, text_en="Welcome to FC Mobile.")
        translator.translate_segments([segment])
        contents = client.calls[0]["contents"]
        self.assertIn("Input paragraphs:\nParagraph 1:\nWelcome to FC Mobile.", contents)
        self.assertNotIn("12.5", contents)
        self.assertNotIn("16.0", contents)
        self.assertNotIn("Input lines:", contents)

    def test_token_budget_packs_complete_paragraphs_and_pipeline_groups(self):
        translator = _make_translator(_FakeClient(lambda _c, _n: "{}"))
        texts = [f"Commentary {index}: " + ("play " * 700) for index in range(2)]
        items = [(text, 0.0) for text in texts]
        singles = [translator._estimate_input_tokens([item]) for item in items]
        combined = translator._estimate_input_tokens(items)
        self.assertGreater(combined, max(singles))
        translator.max_input_tokens = max(singles)

        batches = translator._make_batches(items)
        self.assertEqual([[text for text, _ in batch] for batch in batches], [[texts[0]], [texts[1]]])
        segments = [
            Segment(index=i, start=i * 2.0, end=i * 2.0 + 1.0, text_en=text)
            for i, text in enumerate(texts)
        ]
        self.assertEqual(
            [[segment.text_en for segment in group] for group in translator.batch_segments(segments)],
            [[texts[0]], [texts[1]]],
        )

    def test_single_oversized_paragraph_is_kept_whole_then_skipped(self):
        client = _FakeClient(lambda _c, _n: "unexpected")
        translator = _make_translator(client, max_input_tokens=1000)
        long_text = "FC Mobile event " * 2000
        batches = translator._make_batches([(long_text, 0.0)])
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0][0], long_text)
        self.assertGreater(translator._estimate_input_tokens(batches[0]), 1000)
        self.assertEqual(translator._run_batches([(long_text, 0.0)]), [""])
        self.assertEqual(client.calls, [])
        self.assertEqual(translator._skipped_lines, 1)

    def test_measured_shortening_request_contains_no_duration_hints(self):
        client = _FakeClient(lambda contents, _n: json.dumps({"translations": ["ছোট বাক্য।"]}))
        translator = _make_translator(client)
        segment = Segment(index=0, start=12.5, end=16.0, text_bn="এটি অনেক লম্বা একটি বাংলা বাক্য।")
        translator.shorten_to_durations([segment], [0.72])
        contents = client.calls[0]["contents"]
        self.assertIn("Paragraph 1:", contents)
        self.assertIn("এটি অনেক লম্বা একটি বাংলা বাক্য।", contents)
        self.assertNotIn("0.72", contents)
        self.assertNotRegex(contents.lower(), r"duration|seconds|speaking time|timestamp")

    # ---- ফরম্যাট ভুল: অপেক্ষা নেই, শুধু বাকি লাইন আবার ------------------
    def test_format_error_retries_without_long_sleep(self):
        """ফরম্যাট ভুল হলে ২ সেকেন্ডের বেশি অপেক্ষা করা চলবে না।"""
        def responder(contents, call_no):
            if call_no <= 2:          # প্রথমে ফরম্যাট ভাঙা উত্তর
                return "sorry, I cannot do JSON"
            return "\n".join(       # তৃতীয়বার থেকে নম্বরযুক্ত লাইন (lines মোড)
                f"{index}. বাংলা: {text}"
                for index, text in enumerate(_lines_from_contents(contents), start=1)
            )

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=5, requests_per_minute=1_000_000)
        slept = []
        with mock.patch("proxmax.translator.time.sleep", side_effect=slept.append):
            result = translator.translate_text("Hello")
        self.assertTrue(result.startswith("বাংলা:"))
        self.assertTrue(all(delay <= 2.0 for delay in slept), slept)
        self.assertTrue(all(delay <= 0.3 for delay in slept), f"ফরম্যাট ভুলে অপেক্ষা: {slept}")

    def test_format_retry_does_not_wait_for_rate_limit_slot(self):
        """ফরম্যাট ভুলে আবার পাঠানো একই request — নতুন স্লটের জন্য অপেক্ষা নয়।"""
        def responder(contents, call_no):
            if call_no == 1:
                return "not json at all"
            return "\n".join(
                f"{index}. বাংলা: {text}"
                for index, text in enumerate(_lines_from_contents(contents), start=1)
            )

        client = _FakeClient(responder)
        # RPM খুব কম (১) → নতুন request হলে ৬০ সেকেন্ড অপেক্ষা করতে হতো
        translator = _make_translator(client, max_retries=3, requests_per_minute=1)
        slept = []
        with mock.patch("proxmax.translator.time.sleep", side_effect=slept.append):
            result = translator.translate_text("Hello")
        self.assertTrue(result.startswith("বাংলা:"))
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all(delay <= 0.3 for delay in slept),
                        f"ফরম্যাট retry-তে অপেক্ষা: {slept}")

    def test_retry_delay_never_exceeds_two_seconds(self):
        translator = _make_translator(_FakeClient(lambda c, n: "{}"), max_retries=7)
        for attempt in range(1, 8):
            for status in (None, 429, 500):
                delay = translator._retry_delay(attempt, TimeoutError("timed out"), status)
                self.assertLessEqual(delay, 2.0)
                self.assertGreaterEqual(delay, 0.0)

    def test_model_ignoring_json_falls_back_to_numbered_lines(self):
        """মডেল JSON না দিলে দুবার চেষ্টার পর নম্বরযুক্ত লাইনে চাওয়া হয়।"""
        def responder(contents, call_no):
            if call_no <= 2:
                return "Here is your Bangla text, sorry for the format."
            return "\n".join(
                f"{index}. বাংলা: {text}"
                for index, text in enumerate(_lines_from_contents(contents), start=1)
            )

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=5)
        with mock.patch("proxmax.translator.time.sleep"):
            result = translator.translate_text("Hello")
        self.assertTrue(result.startswith("বাংলা:"), result)
        self.assertTrue(any("numbered translations" in c["contents"] for c in client.calls[2:]),
                        [c["contents"][:60] for c in client.calls])

    def test_truncated_answer_only_retries_missing_lines(self):
        """কাটা উত্তরে শুরুর লাইনগুলো রেখে বাকিগুলোই আবার চাওয়া হয়।"""
        def responder(contents, call_no):
            if call_no == 1:
                return _FakeResponse(
                    '{"translations": ["বাংলা: One.", "বাংলা: Two.',
                    finish_reason="MAX_TOKENS",
                )
            return json.dumps({"translations": [f"বাংলা: {t}" for t in _lines_from_contents(contents)]})

        client = _FakeClient(responder)
        translator = _make_translator(client)
        segments = _segments(["One.", "Two.", "Three.", "Four."])
        with mock.patch("proxmax.translator.time.sleep"):
            translator.translate_segments(segments)
        self.assertEqual([s.text_bn for s in segments],
                         ["বাংলা: One.", "বাংলা: Two.", "বাংলা: Three.", "বাংলা: Four."])
        # প্রথম request-এ ৪টি লাইন ছিল, দ্বিতীয়টিতে শুধু বাকি লাইনগুলো
        self.assertEqual(len(_lines_from_contents(client.calls[0]["contents"])), 4)
        self.assertLess(len(_lines_from_contents(client.calls[1]["contents"])), 4)

    def test_one_unparseable_line_is_skipped_not_fatal(self):
        """এক লাইনে বারবার ভুল ফরম্যাট হলেও পুরো ভিডিও ব্যর্থ হবে না।"""
        def responder(contents, _n):
            lines = _lines_from_contents(contents)
            if any("BROKEN" in line for line in lines):
                return "no bengali here, sorry"
            return json.dumps({"translations": [f"বাংলা: {t}" for t in lines]})

        client = _FakeClient(responder)
        translator = _make_translator(client, max_retries=2, batch_size=2)
        segments = _segments(["One.", "BROKEN line.", "Three."])
        with mock.patch("proxmax.translator.time.sleep"):
            translator.translate_segments(segments)
        self.assertEqual(segments[0].text_bn, "বাংলা: One.")
        self.assertTrue(segments[1].text_bn in ("", "বাংলা: BROKEN line."))
        self.assertTrue(segments[2].text_bn.startswith("বাংলা:"))

    def test_thinking_config_is_not_sent_by_default_to_avoid_http_400(self):
        """নতুন Gemini মডেলে thinking_budget=0 পাঠালে 400 INVALID_ARGUMENT আসে — তাই ডিফল্টে বন্ধ।"""
        client = _FakeClient(lambda c, n: json.dumps({"translations": ["বাংলা"]}))
        translator = _make_translator(client)
        translator.translate_text("Hello")
        thinking = getattr(client.calls[0]["config"], "thinking_config", None)
        self.assertIsNone(thinking)

    def test_http_400_on_schema_falls_back_to_plain_json_without_failing(self):
        """HTTP 400 INVALID_ARGUMENT এলে response_schema বন্ধ করে সাধারণ JSON-এ সফল হয়।"""
        class _Http400SchemaClient(_FakeClient):
            def generate_content(self, *, model, contents, config):
                self.calls.append({"model": model, "contents": contents, "config": config})
                if getattr(config, "response_schema", None) is not None:
                    err = RuntimeError("400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Invalid argument'}}")
                    err.code = 400  # type: ignore[attr-defined]
                    raise err
                return _FakeResponse(json.dumps({"translations": ["বাংলা অনুবাদ।"]}))

        client = _Http400SchemaClient(lambda c, n: "")
        translator = _make_translator(client, max_retries=3, requests_per_minute=1_000_000)
        self.assertEqual(translator.translate_text("Hello"), "বাংলা অনুবাদ।")
        self.assertFalse(translator._use_schema)
        self.assertEqual(len(client.calls), 2)

    def test_http_400_or_404_on_model_switches_to_fallback_model(self):
        """প্রাথমিক মডেল 400/404 দিলে স্বয়ংক্রিয়ভাবে ব্যাকআপ মডেলে সুইচ করে।"""
        class _ModelFallbackClient(_FakeClient):
            def generate_content(self, *, model, contents, config):
                self.calls.append({"model": model, "contents": contents, "config": config})
                if model == "gemini-3.5-flash-lite":
                    err = RuntimeError("400 INVALID_ARGUMENT: model gemini-3.5-flash-lite not supported")
                    err.code = 400  # type: ignore[attr-defined]
                    raise err
                return _FakeResponse(json.dumps({"translations": ["স্বাগতম।"]}))

        client = _ModelFallbackClient(lambda c, n: "")
        translator = _make_translator(client, model="gemini-3.5-flash-lite", max_retries=5)
        result = translator.translate_text("Welcome.")
        self.assertEqual(result, "স্বাগতম।")
        self.assertNotEqual(translator.model, "gemini-3.5-flash-lite")

    def test_verify_api_checks_live_translation(self):
        client = _FakeClient(lambda c, n: json.dumps({"translations": ["এফসি মোবাইলে স্বাগতম।"]}))
        translator = _make_translator(client, model="gemini-3.5-flash-lite")
        info = translator.verify_api()
        self.assertTrue(info["ok"])
        self.assertEqual(info["active_model"], "gemini-3.5-flash-lite")
        self.assertIn("স্বাগতম", info["sample_reply"])

    def test_gemini_gets_only_matched_and_bounded_hotword_glossary(self):
        client = _FakeClient(lambda c, n: json.dumps({"translations": ["মেসি ও মাসচেরানো।"]}))
        translator = _make_translator(
            client,
            hotwords_file=str(REPO_ROOT / "hotwords.txt"),
            hotwords="Unrelated Player Name",
        )
        translator.translate_text("FC Mobile 27: Messi and Mascherano.")
        contents = client.calls[0]["contents"]
        self.assertIn("Recognized football/FC Mobile terms in this batch", contents)
        self.assertIn("Mascherano", contents)
        self.assertNotIn("Unrelated Player Name", contents)
        self.assertNotIn("Aitana Bonmati", contents)
        glossary = contents.split("Recognized football/FC Mobile terms in this batch", 1)[1]
        glossary = glossary.split("\nInput paragraphs:", 1)[0]
        self.assertLessEqual(len(glossary), 650)

    def test_gemini_glossary_caps_match_count_and_characters(self):
        response = lambda c, n: json.dumps({"translations": ["বাংলা অনুবাদ।"]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = (
                ([f"ModeTerm{i:02d}" for i in range(30)], 20),
                ([f"VeryLongGameMode{i:02d}_" + "x" * 40 for i in range(30)], None),
            )
            for index, (terms, expected_count) in enumerate(cases):
                hotwords_file = root / f"hotwords-{index}.txt"
                hotwords_file.write_text("\n".join(terms), encoding="utf-8")
                client = _FakeClient(response)
                translator = _make_translator(client, hotwords_file=str(hotwords_file))
                translator.translate_text(" / ".join(terms))

                contents = client.calls[0]["contents"]
                marker = "(keep names/terms accurate in Bangla): "
                self.assertIn(marker, contents)
                glossary = contents.split(marker, 1)[1].split("\nInput paragraphs:", 1)[0].rstrip(".")
                matched = glossary.split(", ")
                self.assertLessEqual(len(matched), 20)
                self.assertLessEqual(len(glossary), 600)
                if expected_count is not None:
                    self.assertEqual(len(matched), expected_count)
                else:
                    self.assertLess(len(matched), 20)

    def test_pre_tts_predictor_does_not_use_following_pause_or_call_gemini(self):
        translator = _make_translator(_FakeClient(lambda _c, _n: "{}"))
        long_line = "বাংলা বাক্য অনুবাদ " * 12
        first = Segment(index=0, start=0.0, end=1.0, text_en="A line.", text_bn=long_line)
        second = Segment(index=1, start=5.0, end=6.0, text_en="Next line.", text_bn="ছোট।")
        with mock.patch.object(translator, "_run_batches") as run:
            changed = translator.shorten_to_fit(
                [first, second], max_tempo=1.15, slot_gap=0.08, borrow=0.25
            )
        self.assertEqual(changed, 0)
        self.assertEqual(first.text_bn, long_line)
        run.assert_not_called()

    def test_output_budget_grows_with_batch_size(self):
        translator = _make_translator(_FakeClient(lambda c, n: "{}"))
        small = translator._output_token_budget([("short line", 1.0)] * 2)
        big = translator._output_token_budget([("a long line " * 20, 3.0)] * 20)
        self.assertGreaterEqual(small, 1024)
        self.assertGreater(big, small)
        self.assertLessEqual(big, translator.max_output_tokens)

    def test_unsupported_schema_disables_it_and_retries_immediately(self):
        class _NoSchemaClient(_FakeClient):
            def generate_content(self, *, model, contents, config):
                self.calls.append({"model": model, "contents": contents, "config": config})
                if getattr(config, "response_schema", None) is not None:
                    raise ValueError("response_schema is not supported for this model")
                return _FakeResponse(json.dumps({"translations": ["বাংলা"]}))

        client = _NoSchemaClient(lambda c, n: "")
        translator = _make_translator(client, max_retries=3, requests_per_minute=1_000_000)
        slept = []
        with mock.patch("proxmax.translator.time.sleep", side_effect=slept.append):
            self.assertEqual(translator.translate_text("Hello"), "বাংলা")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(slept, [])
        self.assertFalse(translator._use_schema)

    def test_close_releases_client(self):
        client = _FakeClient(lambda c, n: json.dumps({"translations": []}))
        translator = _make_translator(client)
        translator.close()
        translator.close()
        self.assertEqual(client.closed, 1)
