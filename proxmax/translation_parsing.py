"""Gemini response parsing and recovery for complete line-to-line translations."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from .translation_errors import _PartialBatchError, _ResponseFormatError
from .translator_config import (
    _BANGLA_RE, _FENCE_RE, _JSON_START_RE, _NUMBERED_LINE_RE, _STRING_ITEM_RE,
    _OBJECT_TEXT_KEYS,
)

def normalise_text(text: Optional[str]) -> str:
    """Whitespace ঠিক করে এক লাইনের টেক্সট বানায় (``None`` → ``""``)."""
    return " ".join((text or "").split())

def _strip_fences(text: str) -> str:
    """Markdown code-fence (```json … ```) সরিয়ে ফেলে — যেখানেই থাকুক।"""
    cleaned = _FENCE_RE.sub("", text or "")
    cleaned = re.sub(r"```[A-Za-z]*", "", cleaned)
    return cleaned.replace("```", "").strip()

def _json_blobs(text: str) -> List[str]:
    """টেক্সট থেকে সম্ভাব্য JSON টুকরোগুলো (প্রথম ``{``/``[`` থেকে শুরু)।"""
    cleaned = _strip_fences(text)
    blobs = [cleaned]
    match = _JSON_START_RE.search(cleaned)
    if match and match.start() > 0:
        blobs.append(cleaned[match.start():])
    return blobs

def _decode_json(text: str) -> Any:
    """JSON ডিকোড করার সর্বাত্মক চেষ্টা (পেছনে বাড়তি লেখা থাকলেও চলে)।"""
    decoder = json.JSONDecoder()
    last_error: Optional[Exception] = None
    for blob in _json_blobs(text):
        if not blob:
            continue
        try:
            return json.loads(blob)
        except (TypeError, ValueError) as exc:
            last_error = exc
        start = _JSON_START_RE.search(blob)
        if start:
            try:
                value, _end = decoder.raw_decode(blob[start.start():])
                return value
            except (TypeError, ValueError) as exc:
                last_error = exc
    raise ValueError(str(last_error) if last_error else "JSON পাওয়া যায়নি")

def _salvage_string_array(text: str) -> List[str]:
    """কাটা/অসম্পূর্ণ JSON থেকে **শুরুর সম্পূর্ণ** স্ট্রিংগুলো উদ্ধার করে।"""
    cleaned = _strip_fences(text)
    match = _JSON_START_RE.search(cleaned)
    if not match:
        return []
    body = cleaned[match.start():]

    if body.startswith("{"):
        key = re.search(r'"(?:translations|lines|output|outputs|results)"\s*:\s*\[', body)
        if not key:
            return []
        body = body[key.end() - 1:]

    dict_mode = bool(re.match(r'\[\s*\{', body))
    values = _STRING_ITEM_RE.findall(body)
    if dict_mode:
        values = [v for v in values if v.strip() not in _OBJECT_TEXT_KEYS]
    items: List[str] = []
    for value in values:
        try:
            items.append(json.loads(f'"{value}"').strip())
        except (TypeError, ValueError):
            items.append(value.strip())
    return [item for item in items if item]

def _looks_like_bangla(lines: Sequence[str]) -> bool:
    """ফলাফলে অন্তত একটি বাংলা হরফ আছে কি না (মডেলের বাড়তি লেখা আলাদা করতে)।"""
    return any(_BANGLA_RE.search(line or "") for line in lines)

def parse_numbered_lines(
    raw: str, expected: int, require_numbered: bool = False
) -> List[str]:
    """Parse numbered lines and restore their exact input order."""
    cleaned = _strip_fences(raw or "")
    cleaned = re.sub(r"^\s*```.*$", "", cleaned, flags=re.M)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    numbered: Dict[int, str] = {}
    for line in lines:
        match = _NUMBERED_LINE_RE.match(line)
        if match:
            numbered.setdefault(int(match.group("num")), match.group("text").strip())
    ordered = [numbered[index] for index in range(1, expected + 1)
               if index in numbered]
    if len(ordered) == expected and _looks_like_bangla(ordered):
        return ordered
    if not require_numbered:
        plain = [re.sub(r"^\s*(?:[-*•]\s*)", "", line).strip() for line in lines]
        plain = [line for line in plain if line]
        if len(plain) == expected and _looks_like_bangla(plain):
            return plain
    snippet = " ".join((raw or "").split())[:160]
    raise _ResponseFormatError(
        f"ফরম্যাট বোঝা যায়নি: {expected} লাইন দরকার, নম্বরযুক্ত পাওয়া গেছে "
        f"{len(numbered)}টি (মোট লাইন {len(lines)}টি) — উত্তর: {snippet!r}"
    )

def _values_from_json(data: Any) -> Any:
    """JSON থেকে অনুবাদের list-টি বের করে (dict/একাধিক কী সহনশীলভাবে)।"""
    values: Any = data
    if isinstance(data, dict):
        for key in ("translations", "lines", "output", "results", "outputs"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                values = candidate
                break
        else:
            lists = [v for v in data.values() if isinstance(v, list)]
            values = lists[0] if len(lists) == 1 else data
    return values

def _normalise_items(
    values: Any, expected: int, require_ids: bool = False
) -> List[str]:
    """Normalize translations and restore order from explicit paragraph IDs."""
    if not isinstance(values, list):
        raise _ResponseFormatError(
            f"Gemini-র উত্তরে অনুবাদের list পাওয়া যায়নি (পাওয়া গেছে: {type(values).__name__})"
        )

    id_keys = ("id", "paragraph_id", "index")
    identified = [
        isinstance(item, dict) and any(key in item for key in id_keys)
        for item in values
    ]
    if any(identified):
        if not all(identified):
            raise _ResponseFormatError("Gemini-র উত্তরে কিছু paragraph ID অনুপস্থিত")
        by_id: Dict[int, str] = {}
        for item in values:
            assert isinstance(item, dict)
            raw_id = next((item.get(key) for key in id_keys if key in item), None)
            if isinstance(raw_id, bool):
                raise _ResponseFormatError("Gemini-র উত্তরে paragraph ID অবৈধ")
            if isinstance(raw_id, int):
                paragraph_id = raw_id
            elif isinstance(raw_id, str) and raw_id.isdigit():
                paragraph_id = int(raw_id)
            else:
                raise _ResponseFormatError("Gemini-র উত্তরে paragraph ID অবৈধ")
            if not 1 <= paragraph_id <= expected or paragraph_id in by_id:
                raise _ResponseFormatError(
                    f"Gemini-র উত্তরে paragraph ID অনুপযুক্ত বা duplicate: {paragraph_id}"
                )
            text = next(
                (item[key] for key in ("translation", "bn", "text", "bangla")
                 if isinstance(item.get(key), str)),
                None,
            )
            if text is None:
                raise _ResponseFormatError(
                    f"Gemini-র paragraph {paragraph_id}-এ অনুবাদ নেই"
                )
            by_id[paragraph_id] = text.strip()
        if set(by_id) != set(range(1, expected + 1)):
            raise _ResponseFormatError(
                f"Gemini-র paragraph ID অসম্পূর্ণ: দরকার 1–{expected}, পাওয়া গেছে {sorted(by_id)}"
            )
        return [by_id[index] for index in range(1, expected + 1)]

    if require_ids and expected > 1:
        raise _ResponseFormatError(
            "একাধিক paragraph-এর JSON উত্তরে ID-সহ অনুবাদ object পাওয়া যায়নি"
        )

    translations: List[str] = []
    for item in values:
        if isinstance(item, str):
            translations.append(item.strip())
        elif isinstance(item, dict):
            text = next(
                (item[k] for k in ("bn", "translation", "text", "bangla") if isinstance(item.get(k), str)),
                None,
            )
            if text is None:
                raise _ResponseFormatError("Gemini-র উত্তরের একটি item-এ টেক্সট নেই")
            translations.append(text.strip())
        else:
            raise _ResponseFormatError(
                f"Gemini-র উত্তরের item অবৈধ টাইপের: {type(item).__name__}"
            )
    return translations

def parse_translations(
    raw: str, expected: int, require_ids: bool = False
) -> List[str]:
    """Read translations and validate/order IDs when the batch protocol requires them."""
    cleaned = _strip_fences(raw or "")
    try:
        data = _decode_json(cleaned)
    except ValueError:
        return parse_numbered_lines(
            cleaned, expected, require_numbered=require_ids and expected > 1
        )

    translations = _normalise_items(_values_from_json(data), expected, require_ids)
    if translations and not _looks_like_bangla(translations):
        raise _ResponseFormatError(
            "উত্তরে বাংলা টেক্সট নেই (মডেল হয়তো অনুবাদ না করে অন্য কিছু লিখেছে)"
        )
    if len(translations) != expected:
        trimmed = list(translations)
        while len(trimmed) > expected and not trimmed[-1].strip():
            trimmed.pop()
        if len(trimmed) != expected:
            raise _ResponseFormatError(
                f"অনুবাদের সংখ্যা মেলেনি: দরকার {expected}, পাওয়া গেছে {len(trimmed)}"
            )
        translations = trimmed
    return translations

def _salvage_identified_prefix(raw: str, expected: int) -> tuple[bool, List[str]]:
    """Recover only a contiguous ID prefix from a truncated object array."""
    cleaned = _strip_fences(raw or "")
    key = re.search(r'"(?:translations|lines|output|outputs|results)"\s*:\s*\[', cleaned)
    if key:
        position = key.end()
    else:
        array = _JSON_START_RE.search(cleaned)
        if not array or cleaned[array.start()] != "[":
            return False, []
        position = array.end()

    decoder = json.JSONDecoder()
    entries: List[tuple[int, str]] = []
    id_keys = ("id", "paragraph_id", "index")
    while position < len(cleaned):
        while position < len(cleaned) and (cleaned[position].isspace() or cleaned[position] == ","):
            position += 1
        if position >= len(cleaned) or cleaned[position] == "]":
            break
        try:
            item, end = decoder.raw_decode(cleaned, position)
        except ValueError:
            break
        if not isinstance(item, dict):
            return bool(entries), []
        raw_id = next((item.get(name) for name in id_keys if name in item), None)
        if raw_id is None:
            return bool(entries), []
        if isinstance(raw_id, bool):
            return True, []
        if isinstance(raw_id, int):
            paragraph_id = raw_id
        elif isinstance(raw_id, str) and raw_id.isdigit():
            paragraph_id = int(raw_id)
        else:
            return True, []
        text = next(
            (item[name] for name in ("translation", "bn", "text", "bangla")
             if isinstance(item.get(name), str)),
            None,
        )
        if not 1 <= paragraph_id <= expected or text is None:
            return True, []
        entries.append((paragraph_id, text.strip()))
        position = end

    if not entries:
        return False, []
    ids = [paragraph_id for paragraph_id, _text in entries]
    if ids != list(range(1, len(ids) + 1)) or len(ids) >= expected:
        return True, []
    return True, [text for _paragraph_id, text in entries]


def salvage_translations(
    raw: str, expected: int, require_ids: bool = False
) -> List[str]:
    """Recover complete leading translations without misassigning paragraph IDs."""
    identified, ordered = _salvage_identified_prefix(raw, expected)
    if identified:
        return ordered[:expected]
    if require_ids and expected > 1:
        return []
    recovered = [text for text in _salvage_string_array(raw) if text]
    return recovered[:expected]
