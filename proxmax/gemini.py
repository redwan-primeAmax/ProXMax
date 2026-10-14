"""Gemini REST transport; credentials stay in worker memory, never Config/files."""
import http.client
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

LOG = logging.getLogger("proxmax")
_API_KEY = None
_ROOT = "https://generativelanguage.googleapis.com/v1beta/models/"


class GeminiError(RuntimeError):
    """Sanitized API failure: never include headers, response bodies or credentials."""


def require_api_key():
    global _API_KEY
    # Remove the key before any FFmpeg/model-download subprocess is launched.
    if "GEMINI_API_KEY" in os.environ:
        _API_KEY = os.environ.pop("GEMINI_API_KEY").strip()
    if not _API_KEY:
        raise GeminiError("Gemini API key নেই। Notebook-এর গোপন input-এ AI Studio key দিন; CLI-তে GEMINI_API_KEY দিন।")
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,256}", _API_KEY):
        raise GeminiError("API key-এর format ঠিক নেই। শুধু AI Studio key paste করুন; key চ্যাটে দেবেন না।")
    return _API_KEY


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward x-goog-api-key to a redirected host.
        return None


class GeminiClient:
    def __init__(self, config):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", config.translation_model):
            raise GeminiError("Gemini model ID-তে শুধু অক্ষর, সংখ্যা, dot, dash ও underscore ব্যবহার করুন।")
        self.model = config.translation_model
        self.timeout = config.gemini_timeout_seconds
        self.retries = config.gemini_http_retries
        self._key = require_api_key()
        self._opener = urllib.request.build_opener(_NoRedirect())

    def _request(self, suffix="", payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            _ROOT + self.model + suffix, data=data,
            headers={"Content-Type": "application/json", "x-goog-api-key": self._key},
            method="GET" if payload is None else "POST",
        )
        for attempt in range(self.retries + 1):
            status, wait = None, min(2 ** attempt, 30)
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise GeminiError("Gemini response অস্বাভাবিক বড়; গ্রহণ করা হয়নি।")
                if self._key.encode() in raw:
                    raise GeminiError("Gemini response-এ sensitive তথ্য পাওয়া গেছে; response সংরক্ষণ করা হয়নি।")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise GeminiError("Gemini response-এর format সঠিক নয়।")
                if self._key in json.dumps(value, ensure_ascii=False):
                    raise GeminiError("Gemini response-এ sensitive তথ্য পাওয়া গেছে; response সংরক্ষণ করা হয়নি।")
                return value
            except urllib.error.HTTPError as error:
                status = error.code
                retry_after = error.headers.get("Retry-After", "") if error.headers else ""
                if retry_after.isdigit():
                    wait = max(wait, int(retry_after))
                error.close()
            except (urllib.error.URLError, OSError, http.client.HTTPException):
                pass
            except (ValueError, UnicodeError):
                raise GeminiError("Gemini থেকে বৈধ JSON response পাওয়া যায়নি।") from None
            if status not in {None, 429, 500, 502, 503, 504}:
                if status in {401, 403}:
                    message = "API key, project/API restriction ও permission যাচাই করুন।"
                elif status == 404:
                    message = f"Model '{self.model}' বা এই API method পাওয়া যায়নি/এই key-তে উপলব্ধ নয়। AI Studio-তে ID ও access যাচাই করুন।"
                elif status == 400:
                    message = f"'{self.model}' request গ্রহণ করেনি। Model/API support, token limit ও structured-output settings যাচাই করুন।"
                else:
                    message = "API request প্রত্যাখ্যাত হয়েছে; model ও project settings যাচাই করুন।"
                raise GeminiError(f"Gemini HTTP {status}: {message}") from None
            if attempt == self.retries or wait > 60:
                if status == 429:
                    message = "Quota/rate limit শেষ। AI Studio quota/billing যাচাই করে পরে চালান।"
                else:
                    message = "Network/Google service সমস্যা। সংযোগ ঠিক করে পরে চালান।"
                raise GeminiError(f"Gemini request ব্যর্থ ({status or 'network'}): {message}") from None
            LOG.warning("Gemini সাময়িক সমস্যা (%s); %ss পরে retry %s/%s", status or "network", wait, attempt + 1, self.retries)
            time.sleep(wait)

    def check_model(self, config):
        info = self._request()
        methods = info.get("supportedGenerationMethods", [])
        if methods and "generateContent" not in methods:
            raise GeminiError(f"'{self.model}'-এ generateContent সমর্থিত নয়। অন্য model নিজে থেকে নেওয়া হয়নি।")
        for field, requested in (("inputTokenLimit", config.max_source_tokens),
                                 ("outputTokenLimit", config.max_target_tokens)):
            limit = info.get(field)
            if type(limit) is int and requested > limit:
                raise GeminiError(f"{field}={limit}; notebook-এর সংশ্লিষ্ট token limit কমিয়ে চালান।")

    def generate(self, messages, config):
        body = {
            "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
            "contents": [{"role": "user", "parts": [{"text": messages[1]["content"]}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": config.max_target_tokens,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "ARRAY", "items": {
                        "type": "OBJECT", "required": ["id", "bangla", "concise"],
                        "properties": {"id": {"type": "INTEGER"},
                                       "bangla": {"type": "STRING"},
                                       "concise": {"type": "STRING"}},
                    },
                },
            },
        }
        count = self._request(":countTokens", {"generateContentRequest": {
            "model": "models/" + self.model, **body,
        }})
        tokens = count.get("totalTokens")
        if type(tokens) is not int or tokens < 0:
            raise GeminiError("Gemini input token count পাওয়া যায়নি; truncation ছাড়াই কাজ বন্ধ হয়েছে।")
        if tokens > config.max_source_tokens:
            raise GeminiError("Gemini context token limit ছাড়িয়েছে; translation_batch_size কমান। Source কাটা হয়নি।")
        result = self._request(":generateContent", body)
        if result.get("promptFeedback", {}).get("blockReason"):
            raise GeminiError("Gemini prompt block করেছে; source/context যাচাই করুন।")
        candidates = result.get("candidates") or []
        if len(candidates) != 1 or not isinstance(candidates[0], dict):
            raise GeminiError("Gemini একটি সম্পূর্ণ translation candidate দেয়নি।")
        candidate = candidates[0]
        reason = candidate.get("finishReason")
        if reason != "STOP":
            if reason == "MAX_TOKENS":
                raise GeminiError("Gemini output token limit ছুঁয়েছে; batch ছোট করুন বা max_target_tokens বাড়ান। অসম্পূর্ণ লেখা গ্রহণ করা হয়নি।")
            raise GeminiError("Gemini translation সম্পূর্ণ করেনি বা safety filter করেছে; source যাচাই করুন।")
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(p["text"] for p in parts if isinstance(p, dict)
                       and not p.get("thought") and isinstance(p.get("text"), str))
        if not text.strip():
            raise GeminiError("Gemini খালি translation দিয়েছে।")
        version = result.get("modelVersion", "")
        version = version if isinstance(version, str) and re.fullmatch(r"[A-Za-z0-9._-]{1,128}", version) else None
        usage = result.get("usageMetadata") or {}
        usage = {k: usage[k] for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount")
                 if type(usage.get(k)) is int and usage[k] >= 0}
        return text, {"model_version": version, "usage": usage}
