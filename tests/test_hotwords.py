"""Focused regression tests for ProXMax."""





from tests._support import *





class TestHotwords(unittest.TestCase):
    """আলাদা hotwords.txt ফাইল ও FC Mobile / খেলোয়াড় / ইভেন্ট হটওয়ার্ড পরীক্ষা।"""

    def test_hotwords_file_contains_fc_mobile_terms_events_and_players(self):
        from proxmax.hotwords import load_hotwords

        terms = load_hotwords(REPO_ROOT / "hotwords.txt")
        self.assertGreater(len(terms), 1_500)
        joined = " | ".join(terms)
        # FC Mobile UI, controls, and current game terms
        for term in (
            "FC Mobile", "OVR", "Star Pass", "Mascherano", "Skill Boost", "VSA", "H2H",
            "Exchange", "Player Exchange", "Play", "Game Plans", "Gauntlet Mode",
            "Football Centre", "Sprint & Skill", "Swipe Up", "Quick Match",
        ):
            self.assertIn(term, joined)
        # Current mobile PlayStyles, skill moves and events
        for term in (
            "Incisive Through Pass", "Dead Ball", "Finesse Expert", "Guardian",
            "Advanced Rainbow", "Lane Change", "Hocus Pocus", "3rd Anniversary",
            "Ballon d'Or 2026", "Unbreakable", "CONMEBOL Libertadores",
        ):
            self.assertIn(term, joined)
        # Events
        for event in ("TOTY", "UTOTY", "TOTS", "UTOTS", "Ballon d'Or", "Winter Wildcards", "Rulebreakers"):
            self.assertIn(event, joined)
        # Legends & current players
        for player in (
            "Pele", "Maradona", "Lionel Messi", "Cristiano Ronaldo", "Cruyff", "Zidane",
            "Ronaldinho", "Maldini", "Kylian Mbappe", "Erling Haaland", "Jude Bellingham",
            "Vinicius Junior", "Lamine Yamal", "Alexia Putellas", "Aitana Bonmati",
            "Aitana Bonmatí", "Ewa Pajor", "Michael Olise", "Willian Pacho",
        ):
            self.assertIn(player, joined)

    def test_context_prompt_is_bounded_and_selects_late_curated_names(self):
        from proxmax.hotwords import load_hotwords, select_whisper_hotwords

        terms = load_hotwords(REPO_ROOT / "hotwords.txt")
        prompt = select_whisper_hotwords(
            terms, max_chars=500, context="FC Mobile 27: Aitana Bonmati is on the squad."
        )
        self.assertLessEqual(len(prompt), 500)
        self.assertIn("Aitana Bonmati", prompt)
        self.assertIn("FC Mobile 27", prompt)

    def test_custom_hotwords_file_is_self_contained(self):
        from proxmax.hotwords import load_hotwords

        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom.txt"
            custom.write_text("Custom Club Term\n", encoding="utf-8")
            terms = load_hotwords(custom)
        self.assertEqual(terms, ["Custom Club Term"])

    def test_repair_transcript_hotwords_fixes_common_mishearings(self):
        from proxmax.hotwords import repair_transcript_hotwords

        raw = "We packed a 105 over card in the toty event with belling ham and mbappe!"
        fixed = repair_transcript_hotwords(raw)
        self.assertIn("105 OVR", fixed)
        self.assertIn("TOTY", fixed)
        self.assertIn("Bellingham", fixed)
        self.assertIn("Mbappe", fixed)

    def test_curated_player_aliases_get_exact_casing(self):
        from proxmax.hotwords import load_hotwords, repair_transcript_hotwords

        fixed = repair_transcript_hotwords(
            "dembélé and pacho", hotwords=load_hotwords(REPO_ROOT / "hotwords.txt")
        )
        self.assertEqual(fixed, "Dembélé and Pacho")

    def test_matched_hotwords_for_texts_extracts_relevant_terms(self):
        from proxmax.hotwords import load_hotwords, matched_hotwords_for_texts

        terms = load_hotwords(REPO_ROOT / "hotwords.txt")
        matched = matched_hotwords_for_texts(
            ["Ronaldinho and Gullit are insane in H2H during TOTY!"], terms
        )
        self.assertIn("Ronaldinho", matched)
        self.assertIn("Gullit", matched)
        self.assertIn("H2H", matched)
        self.assertIn("TOTY", matched)
