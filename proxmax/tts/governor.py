"""Adaptive speed governor that keeps TTS inside its wall-clock budget.

TTS-এর গতি বাজেটের মধ্যে রাখা হয়। সমস্যা: diffusion-ভিত্তিক TTS (IndicF5) কত দ্রুত চলবে তা ঠিক আগে থেকে
বলা যায় না — নির্ভর করে GPU, diffusion-এর ধাপসংখ্যা আর ``nfe``-এর উপর। ব্যবহারকারীর
চাহিদা স্পষ্ট: ১০ মিনিটের ভিডিও ২–৩ মিনিটে।

এই ক্লাস প্রতি ক্লিপের বাস্তব সময় মেপে (RTF = synthesis time / audio
length) ভবিষ্যৎ অনুমান করে। বাজেট ছাড়ানোর আশঙ্কা হলে ``nfe`` ধাপ
ধীরে ধীরে কমায় (১৬ → ১৩ → ১১ → ৯ → ৮) — এতে গুণমান সামান্য কমে,
কিন্তু গতি বাড়ে। বাজেটের ভেতরে থাকলে কিছুই বদলায় না।
"""

from __future__ import annotations

from typing import List, Optional

from ..utils import get_logger

logger = get_logger(__name__)


class SpeedGovernor:
    """Adaptive ``nfe`` controller driven by measured real-time factor."""

    def __init__(
        self,
        nfe_steps: int = 16,
        min_nfe_steps: int = 8,
        time_budget_seconds: Optional[float] = None,
        total_audio_seconds: float = 0.0,
        nominal_steps: Optional[List[int]] = None,
    ) -> None:
        """Args:
            nfe_steps: শুরুর nfe ধাপ (বেশি = ভালো মান, ধীর)।
            min_nfe_steps: এর নিচে নামবে না।
            time_budget_seconds: TTS-এর জন্য বরাদ্দ সময় (None হলে কোনো চাপ নেই)।
            total_audio_seconds: মোট কত সেকেন্ড কথা তৈরি হবে (অনুমান)।
            nominal_steps: ব্যবহারযোগ্য nfe ধাপের তালিকা (ডিফল্ট নিজে বানায়)।
        """
        self.nfe_steps = max(1, int(nfe_steps))
        self.min_nfe_steps = max(1, min(int(min_nfe_steps), self.nfe_steps))
        self.time_budget_seconds = time_budget_seconds
        self.total_audio_seconds = max(0.0, float(total_audio_seconds))
        self._steps: List[int] = nominal_steps or self._default_ladder()
        self._step_index = 0
        self._downgrades = 0

        self.synthesized_seconds = 0.0
        self.elapsed_seconds = 0.0
        self._samples = 0
        self._rtf_ema: Optional[float] = None

    # ------------------------------------------------------------------
    def _default_ladder(self) -> List[int]:
        ladder = [self.nfe_steps]
        for candidate in (13, 11, 9, 8, 6):
            if self.min_nfe_steps <= candidate < ladder[-1]:
                ladder.append(candidate)
        if self.min_nfe_steps not in ladder and self.min_nfe_steps < ladder[-1]:
            ladder.append(self.min_nfe_steps)
        return ladder

    @property
    def current_nfe(self) -> int:
        """এই মুহূর্তে ব্যবহার করার nfe ধাপ।"""
        return self._steps[self._step_index]

    @property
    def rtf(self) -> Optional[float]:
        """গড় real-time factor (কম = দ্রুত)। এখনো কিছু না হলে ``None``।"""
        return self._rtf_ema

    # ------------------------------------------------------------------
    def observe(self, audio_seconds: float, elapsed_seconds: float) -> None:
        """এক ক্লিপের পরিমাপ রেকর্ড করে এবং দরকার হলে ``nfe`` কমানো হয়।"""
        if audio_seconds <= 0 or elapsed_seconds <= 0:
            return
        rtf = elapsed_seconds / audio_seconds
        self.synthesized_seconds += audio_seconds
        self.elapsed_seconds += elapsed_seconds
        self._samples += 1
        self._rtf_ema = rtf if self._rtf_ema is None else (0.7 * self._rtf_ema + 0.3 * rtf)

        if self.time_budget_seconds is None or self._rtf_ema is None:
            return
        if self._samples < 2:  # প্রথম ক্লিপে সিদ্ধান্ত নয় (warm-up)
            return

        remaining_audio = max(0.0, self.total_audio_seconds - self.synthesized_seconds)
        projected = self.elapsed_seconds + remaining_audio * self._rtf_ema
        if projected <= self.time_budget_seconds:
            return
        if self._step_index + 1 >= len(self._steps):
            logger.warning(
                "⏱️ TTS বাজেট ছাড়াচ্ছে (আনুমানিক %.0fs > %.0fs) কিন্তু nfe আর কমানো যাবে না।",
                projected, self.time_budget_seconds,
            )
            return

        self._step_index += 1
        self._downgrades += 1
        logger.warning(
            "⏱️ গতি বাজেট রাখতে nfe %d → %d (আনুমানিক %.0fs > বাজেট %.0fs)।",
            self._steps[self._step_index - 1], self.current_nfe,
            projected, self.time_budget_seconds,
        )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def set_budget(
        self,
        time_budget_seconds: Optional[float],
        total_audio_seconds: float = 0.0,
    ) -> None:
        """নতুন ভিডিওর জন্য সময়ের বাজেট বসায় (শেখা RTF মনে রাখে)।

        ইতিমধ্যে GPU ধীর বলে প্রমাণিত হলে ``nfe`` নামানো অবস্থাতেই থাকে —
        তাই ব্যাচের পরের ভিডিওগুলো অপ্রয়োজনীয়ভাবে ধীর গতিতে শুরু করে না।
        """
        self.time_budget_seconds = time_budget_seconds
        self.total_audio_seconds = max(0.0, float(total_audio_seconds))
        self.synthesized_seconds = 0.0
        self.elapsed_seconds = 0.0
        self._samples = 0

    def summary(self) -> str:
        """এক লাইনের রিপোর্ট (লগ ও টাইমিং রিপোর্টে)।"""
        rtf = f"{self._rtf_ema:.2f}" if self._rtf_ema is not None else "n/a"
        return (
            f"TTS RTF={rtf} | অডিও {self.synthesized_seconds:.0f}s | "
            f"সময় {self.elapsed_seconds:.0f}s | nfe={self.current_nfe} "
            f"(নামানো হয়েছে {self._downgrades} বার)"
        )
