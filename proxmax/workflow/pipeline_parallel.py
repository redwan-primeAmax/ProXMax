"""Bounded Gemini batch concurrency with ordered delivery to TTS."""

from __future__ import annotations

import contextvars
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Sequence

from .models import Segment
from .translation_parsing import normalise_text


def translate_groups_in_order(translator, groups: Sequence[list[Segment]], stop: threading.Event,
                              emit: Callable[[list[Segment]], None], workers: int) -> None:
    """Run at most two cancellable independent batches; emit in source order.

    Groups containing the same source text do not run together, so the first
    result can populate the shared cache before the duplicate is considered.
    """
    call = getattr(translator, "translate_segments_cancellable", None)
    if not getattr(translator, "supports_parallel_batches", False) or not callable(call):
        for group in groups:
            if stop.is_set():
                break
            translator.translate_segments(group)
            if stop.is_set():
                break
            emit(group)
        return

    # Even sequential Gemini calls need a separate cancellable worker: the
    # producer can then notice TTS failure while an HTTP request remains in flight.
    limit = max(1, min(2, int(workers)))
    pool = ThreadPoolExecutor(max_workers=limit, thread_name_prefix="proxmax-gemini")
    pending = deque()
    active_text: set[str] = set()
    cursor = 0
    try:
        while (cursor < len(groups) or pending) and not stop.is_set():
            while cursor < len(groups) and len(pending) < limit:
                group = groups[cursor]
                keys = {normalise_text(segment.text_en) for segment in group}
                keys.discard("")
                if keys & active_text:
                    break
                context = contextvars.copy_context()
                future = pool.submit(context.run, call, group, stop)
                pending.append((group, keys, future))
                active_text.update(keys)
                cursor += 1
            if not pending:
                break
            group, keys, future = pending[0]
            # Check external TTS errors while waiting. An in-flight HTTP call
            # cannot be forcibly interrupted, but no later retry is started.
            while not future.done() and not stop.is_set():
                wait((future,), timeout=.1)
            if stop.is_set():
                break
            future.result()
            pending.popleft()
            active_text.difference_update(keys)
            emit(group)
    except BaseException:
        stop.set()
        raise
    finally:
        if stop.is_set():
            for _group, _keys, future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
        else:
            pool.shutdown(wait=True)
