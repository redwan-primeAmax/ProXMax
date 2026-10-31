"""Frame and sample invariants for picture-window editing."""

import random
import unittest
from fractions import Fraction

from proxmax.models import Segment
from proxmax.video_timeline import build_video_plan, parse_fps, source_windows


class TestVideoTimeline(unittest.TestCase):
    def test_repeats_the_exact_last_frames_and_reuses_full_window_if_needed(self):
        fps = Fraction(10, 1)
        seg = Segment(0, 0, 1, stretched_duration=2.35)
        plan = build_video_plan([seg], [(0, 10)], 10, fps, 8000)
        self.assertEqual([(p.start, p.end, p.output_frames, p.kind) for p in plan.pieces],
                         [(0, 10, 10, "normal"), (0, 10, 10, "repeat"), (6, 10, 4, "repeat")])
        self.assertEqual(plan.output_frames, 24)  # ceil 2.35 seconds at 10 fps
        self.assertAlmostEqual(seg.video_added_seconds, 1.4)
        self.assertEqual(seg.video_speed, 1)

    def test_cap_is_115_percent_and_remaining_gap_is_permitted(self):
        fps = Fraction(25, 1)
        seg = Segment(0, 0, 10, stretched_duration=4.0)
        plan = build_video_plan([seg], [(0, 250)], 250, fps, 24000)
        self.assertEqual(plan.output_frames, 218)  # ceil(250/1.15)
        self.assertLessEqual(seg.video_speed, 1.15)
        self.assertGreater(seg.video_window_end - seg.placed_start - seg.stretched_duration, 4)
        with self.assertRaisesRegex(ValueError, "1.15"):
            build_video_plan([seg], [(0, 250)], 250, fps, 24000, 1.5)

    def test_fractional_frame_rate_uses_sample_accurate_nonoverlapping_starts(self):
        fps = Fraction(30000, 1001)
        segs = [Segment(0, 0, 1, stretched_duration=1.1234),
                Segment(1, 1, 2, stretched_duration=.25123)]
        n, windows = source_windows(segs, 2, fps, min_gap=0, borrow=0)
        plan = build_video_plan(segs, windows, n, fps, 24000)
        self.assertGreater(plan.output_frames, 0)
        self.assertGreaterEqual(segs[1].placed_start, segs[0].placed_start + segs[0].stretched_duration - 1/24000)
        for s in segs:
            self.assertGreaterEqual(s.video_window_end + 1 / 24000,
                                    s.placed_start + s.stretched_duration)

    def test_randomized_conflicting_windows_cover_source_and_all_speech(self):
        randomizer = random.Random(331)
        for _ in range(120):
            fps = randomizer.choice([Fraction(10), Fraction(25), Fraction(30000, 1001)])
            rate = randomizer.choice([8000, 24000, 44100])
            count = randomizer.randint(1, 16)
            starts = sorted(randomizer.choice([0, randomizer.uniform(0, 9.8)]) for _ in range(count))
            segs = [Segment(i, start, min(10, start + randomizer.uniform(0, 1.8)),
                            stretched_duration=randomizer.uniform(.01, 2.9))
                    for i, start in enumerate(starts)]
            source, windows = source_windows(segs, 10, fps, min_gap=.08, borrow=.25)
            plan = build_video_plan(segs, windows, source, fps, rate)
            covered = [0] * source
            for piece in plan.pieces:
                self.assertLessEqual(Fraction(piece.source_frames, piece.output_frames), Fraction(23, 20))
                if piece.kind != "repeat":
                    for frame in range(piece.start, piece.end):
                        covered[frame] += 1
            self.assertEqual(covered, [1] * source)
            self.assertEqual(sum(p.output_frames for p in plan.pieces), plan.output_frames)
            for i, seg in enumerate(segs):
                self.assertGreaterEqual(seg.video_window_end + 1 / rate,
                                        seg.placed_start + seg.stretched_duration)
                self.assertLessEqual(seg.video_speed, 1.15)
                if i:
                    prev = segs[i - 1]
                    self.assertGreaterEqual(seg.placed_start + 1 / rate,
                                            prev.placed_start + prev.stretched_duration)

    def test_tiny_window_borrows_nearby_motion_and_single_frame_input_warns(self):
        seg = Segment(0, .4, .44, stretched_duration=.5)
        plan = build_video_plan([seg], [(10, 11)], 50, Fraction(25), 8000)
        repeats = [p for p in plan.pieces if p.kind == "repeat"]
        self.assertGreater(repeats[0].source_frames, 1)
        self.assertIn("nearest available moving", seg.sync_recovery)
        self.assertEqual(sum(p.source_frames for p in plan.pieces if p.kind != "repeat"), 50)
        still = Segment(1, 0, .04, stretched_duration=.5)
        plan = build_video_plan([still], [(0, 1)], 1, Fraction(25), 8000)
        self.assertIn("only one picture frame", still.sync_recovery)
        self.assertEqual(plan.output_frames, 13)

    def test_invalid_fps_is_not_silently_accepted(self):
        for value in ("0/0", "0", "nan", "-1", "1000"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_fps(value)


if __name__ == "__main__":
    unittest.main()
