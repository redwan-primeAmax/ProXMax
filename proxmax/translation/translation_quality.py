"""Conservative checks for obvious translation errors, not semantic validation.

The source transcript itself may be wrong. These checks never infer words from
context or invent facts: they compare explicit source digits, detect unexpected
scripts and flag obvious repetitive/very short outputs. They cannot prove
meaning; ambiguous cases remain flagged for review.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import List

from .text_utils import normalise_bangla, number_to_bangla_words, prepare_for_speech

_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_BANGLA_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
_NUMBER = re.compile(r"(?<![A-Za-z0-9০-৯])(?:(?<![\w\u0980-\u09ff])[+\-−])?(?:[0-9০-৯]{1,3}(?:,[0-9০-৯]{2,3})+|[0-9০-৯]+)(?:\.[0-9০-৯]+)?")
_REPEATED_BENGALI_LETTER = re.compile(r"([\u0985-\u09b9])\1{4,}")
_NUMBER_ALIASES = {
    0: ("শূন্য", "জিরো"), 1: ("এক", "ওয়ান", "ওয়ান"), 2: ("দুই", "টু"),
    3: ("তিন", "থ্রি"), 4: ("চার", "ফোর"), 5: ("পাঁচ", "ফাইভ"),
    6: ("ছয়", "ছয়", "সিক্স"), 7: ("সাত", "সেভেন"), 8: ("আট", "এইট"),
    9: ("নয়", "নয়", "নাইন"), 10: ("দশ", "টেন"),
}


@dataclass(frozen=True)
class TranslationAssessment:
    text: str
    issues: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def _value(raw: str) -> str:
    try:
        return format(Decimal(raw.translate(_DIGITS).replace(",", "").replace("−", "-")).normalize(), "f")
    except InvalidOperation:
        return raw.translate(_DIGITS).replace(",", "").replace("−", "-")


def _number_matches(text: str) -> list[re.Match[str]]:
    return list(_NUMBER.finditer(text or ""))


def _word_number_present(value: str, text: str) -> bool:
    """Recognize common Bangla spoken forms, including normal number suffixes."""
    try:
        number = Decimal(value)
        if number != int(number) or abs(number) > 1_000_000_000:
            return False
        integer = abs(int(number))
    except (InvalidOperation, ValueError, OverflowError):
        return False
    forms = set(_NUMBER_ALIASES.get(integer, ()))
    spoken = number_to_bangla_words(integer)
    forms.update((spoken, spoken.replace("একশ ", "একশো ")))
    if number < 0:
        forms = {f"মাইনাস {form}" for form in forms}
    normalized = unicodedata.normalize("NFC", text)
    for form in forms:
        form = unicodedata.normalize("NFC", form)
        # Bangla classifiers are valid suffixes, e.g. আটটি or দুইবার.
        if re.search(r"(?<![\u0980-\u09ff])" + re.escape(form)
                     + r"(?:টি|টা|খানা|জন|বার|তম)?(?![\u0980-\u09ff])", normalized):
            return True
    return False


def _unexpected_scripts(source: str, candidate: str) -> List[str]:
    def scripts(text: str) -> set[str]:
        result = set()
        for char in text:
            if unicodedata.category(char).startswith("L"):
                name = unicodedata.name(char, "")
                script = name.split(" ", 1)[0] if name else "UNKNOWN"
                if script not in {"BENGALI", "LATIN"}:
                    result.add(script)
        return result

    return sorted(scripts(candidate) - scripts(source))


def assess_translation(source: str, candidate: str, *, repair_single_number: bool = True) -> TranslationAssessment:
    """Flag explicit number/script and obvious corruption signals, not semantic errors.

    Spelled-out source numbers and semantic correctness are deliberately not
    inferred. If several numbers are involved, no automatic substitution is made.
    """
    text = normalise_bangla(candidate)
    if not text:
        return TranslationAssessment(text)
    notes: list[str] = []
    source_numbers = _number_matches(source)
    output_numbers = _number_matches(text)
    expected = Counter(_value(m.group(0)) for m in source_numbers)
    actual = Counter(_value(m.group(0)) for m in output_numbers)

    if (repair_single_number and len(source_numbers) == len(output_numbers) == 1
            and expected != actual and not _word_number_present(next(iter(expected)), text)
            # Repair only a clear dropped digit, not an unrelated value that may
            # represent valid extra context. Other mismatches get a recheck.
            and next(iter(expected)).isdigit() and next(iter(actual)).isdigit()
            and next(iter(actual)) in next(iter(expected))
            and len(next(iter(actual))) < len(next(iter(expected)))):
        src = source_numbers[0].group(0).translate(_DIGITS).translate(_BANGLA_DIGITS)
        match = output_numbers[0]
        old = match.group(0)
        text = text[:match.start()] + src + text[match.end():]
        notes.append(f"Corrected a single explicit source number ({old} → {src}); review the translation.")
        actual = Counter(_value(m.group(0)) for m in _number_matches(text))

    missing: list[str] = []
    for value, count in expected.items():
        shortfall = max(0, count - actual[value])
        if shortfall and _word_number_present(value, text):
            shortfall -= 1  # repeated spoken forms require explicit review
        missing.extend([value] * max(0, shortfall))
    issues: list[str] = []
    if missing:
        issues.append("Explicit source number(s) not reliably preserved: " + ", ".join(missing))
    scripts = _unexpected_scripts(source, text)
    if scripts:
        issues.append("Unexpected writing system in Bangla output: " + ", ".join(scripts))
    if _REPEATED_BENGALI_LETTER.search(text):
        issues.append("Unusually repeated Bangla letters; possible corrupted response")
    source_length = len(re.sub(r"\s+", "", source or ""))
    output_length = len(re.sub(r"\s+", "", text))
    if source_length >= 40 and len(text.split()) == 1 and output_length <= source_length * 0.25:
        issues.append("Very short single-token output for a long source; possible missing content")
    return TranslationAssessment(text, tuple(issues), tuple(notes))


def assess_fit_rewrite(source: str, reference: str, current: str, candidate: str) -> TranslationAssessment:
    """Reject obvious loss in a measured-fit rewrite; this cannot prove semantic fidelity."""
    text = normalise_bangla(candidate)
    if not text:
        return TranslationAssessment(text, ("Fit rewrite was empty.",))
    issues = list(assess_translation(source, text, repair_single_number=False).issues)
    issues.extend(assess_translation(reference, text, repair_single_number=False).issues)
    reference_numbers = Counter(_value(match.group(0)) for match in _number_matches(reference))
    candidate_numbers = Counter(_value(match.group(0)) for match in _number_matches(text))
    for value, count in candidate_numbers.items():
        extra = max(0, count - reference_numbers[value])
        if extra and _word_number_present(value, reference):
            extra -= 1
        if extra:
            issues.append(f"Fit rewrite introduced an unverified number: {value}")
    negative = r"(?<![\u0980-\u09ff])(?:না|নেই|নাই|নয়|নয়|হয়নি|করেনি|করিনি|পারেনি|পাইনি|পায়নি|দেইনি|দেয়নি|হবেনা)(?![\u0980-\u09ff])"
    if re.search(negative, reference) and not re.search(negative, text):
        issues.append("Fit rewrite may have dropped negation.")
    if re.search(r"[\u0980-\u09ff]", reference) and not re.search(r"[\u0980-\u09ff]", text):
        issues.append("Fit rewrite does not contain Bangla script.")
    current_length = len(prepare_for_speech(current))
    reference_length = len(prepare_for_speech(reference))
    candidate_length = len(prepare_for_speech(text))
    if text == normalise_bangla(current) or candidate_length >= current_length:
        issues.append("Fit rewrite is not shorter in its spoken form.")
    if reference_length > 10 and candidate_length < reference_length * 0.65:
        issues.append("Fit rewrite is too short to safely retain the original details.")
    return TranslationAssessment(text, tuple(dict.fromkeys(issues)))
