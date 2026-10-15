"""Colab-এর native masked API-key box; custom JavaScript form নয়।"""
import re
import warnings
from getpass import GetPassWarning, getpass


def read_api_key():
    """Ask before setup; never echo, persist or include the key in an error."""
    key = None
    try:
        print("নিচের API-key box-এ AI Studio key paste করে Enter চাপুন। "
              "Key দেওয়ার আগে setup/অনুবাদ শুরু হবে না।", flush=True)
        with warnings.catch_warnings():
            # Never fall back to an echoing input if password entry is unavailable.
            warnings.simplefilter("error", GetPassWarning)
            key = getpass("Gemini API key: ").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{16,256}", key):
            raise ValueError("সঠিক AI Studio API key দিন। Cell আবার চালান; key চ্যাটে দেবেন না।")
        return key
    except GetPassWarning:
        raise RuntimeError("Masked API-key box পাওয়া যায়নি। Google Colab-এ page reload করে "
                           "cell আবার চালান; source code-এ key লিখবেন না।") from None
    except EOFError:
        raise RuntimeError("API-key input পাওয়া যায়নি। Colab-এ cell আবার চালান।") from None
    finally:
        key = None
