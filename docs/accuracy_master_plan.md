# ProXMax dubbing accuracy master plan

**Status:** diagnostic plan only; the uploaded SRTs were reviewed, but this document does not change runtime behavior.

## Executive assessment

The main issue in this sample is not hotword-list size alone. The English transcript contains repeated recognition errors and a long likely hallucination/repetition run; Bengali then sometimes translates that bad source rather than repairing it. Separately, the English/Bengali SRT pair raises a serious local segment-to-time alignment flag around 08:49–09:37. The files alone cannot prove what is audible in the source or final dub, so the first implementation step must compare them with the original audio and rendered Bangla track.

The existing bounded-hotword and pause-budget safeguards are useful foundations, but they do not yet guarantee (a) reliable recognition of a noisy/repetitive region, (b) translation fidelity to corrected English, or (c) that real synthesized speech fits its own timestamp without being clipped.

## Findings from the uploaded files

- `video_en.txt`: **129 cues**. `video_bn.txt`: **211 cues**. Both cover `00:00:01.032–00:10:55.304`; cue counts are not expected to match because Bengali lines can be split differently.
- The English SRT repeatedly renders the likely term *Ballon d'Or* as variants such as “ball indoor,” “Ball of the Year,” “ball on door,” and “Volunder.” Candidate name errors include Yashin as “Yash,” Gullit as “Hulet/Hulett,” Matthäus as “Macteus,” and Figo as “Vigo.” These are hypotheses from context and need audio confirmation.
- Approximately **09:18–09:40** in the English file is a repeated, largely incoherent “truth/honest” run. Bengali also has repetitive material in the corresponding late section. This looks more like low-confidence ASR/hallucination than a missing vocabulary entry; translating it as fluent prose would make the dub confidently wrong.
- There are semantic fidelity flags in Bengali: the opening says “most” Ballon d'Or winners in English but “সব” (“all”) in Bengali; the 125-OVR/Star Pass giveaway condition is split and appears incompletely preserved. Preserve the condition and “per OVR below target” exactly.
- Cue-by-cue meaning appears locally displaced in the supplied SRTs around **08:49–09:37**. For example, the English Ronaldo line starts at `08:49.548`, while the corresponding Bengali line starts at `09:01.468` (about **11.9 s later**); the striker/Dembélé lines appear about **16.9–18.8 s later**; the “full squad” line is about **24.3 s later**. This may be subtitle/segment mapping or STT timing rather than audible dub drift, so verify against audio before changing placement logic.
- Bengali has **three subtitle gaps over 2 s**, the largest about **3.64 s** (`08:57.824–09:01.468`), and no overlapping cues. The English file has a maximum positive gap of about **1.68 s** and one roughly **20 ms** overlap. These intervals could be genuine source silence; do not fill them blindly.

**Important limitation:** only subtitle text and timecodes were uploaded—not source audio/video or the final dubbed waveform. Therefore this review cannot validate pronunciation, determine whether a gap is silence, or prove an audible A/V offset.

## Current foundation to build on

- `hotwords.txt` is a curated FC Mobile vocabulary; Whisper receives a context-aware prompt capped at 1,600 characters.
- Gemini receives only exact, boundary-aware matches from the current batch, capped at 20 terms / 600 characters—not the full hotwords file.
- Shortening, expansion, and sync windows are now based on the detected source speech interval plus a small configured borrow (250 ms by default), so a long pause is not treated as extra speaking time.
- These safeguards do not by themselves detect hallucinated English, enforce English-to-Bangla semantic equivalence, or prevent a too-long synthesized line from being clipped after TTS.

## Master plan

### Phase 0 — Establish a trustworthy benchmark and diagnose the failure layer

1. For this same video, retain locally (not in Git) the source audio/video, final mixed dub, both SRTs, and the exact config/commit used.
2. Manually review a compact gold set of roughly 40–60 utterances: the opening challenge/number condition, Yashin, Gullit, Matthäus, Figo, CR7/Dembélé, the 09:18–09:40 repeated section, and several real 2–5 s pauses.
3. Give every source utterance a stable ID and keep that ID through STT, translation, TTS, subtitle generation, and logs. Compare content by ID rather than aligning independently chunked SRT cue numbers.
4. Add per-segment diagnostics: start/end, source text, word confidence/log-probability where available, no-speech/compression indicators, translated text, TTS duration, allowed window, chosen tempo, clipped samples, and final placed interval.

### Phase 1 — Improve English STT and reject hallucinations

1. Use the video title/topic plus preceding **high-confidence** transcript as context; reset or shorten carried context after a long silence or a low-confidence/repetitive segment so a hallucination does not propagate.
2. Add a suspect-span detector for repeated n-grams, implausible repetition rate, low confidence, and speech/silence disagreement. Re-decode only flagged 2–5 s windows with surrounding context and a small topic-specific glossary; do not re-run the entire video unnecessarily.
3. Add conservative, context-gated alias repairs for this sample’s likely confusions (Ballon d'Or variants, Yashin, Gullit, Matthäus, Figo). Keep fuzzy guesses out of ordinary prose; log each repair and its before/after text.
4. If a span remains unintelligible after a second pass, mark it uncertain or omit it rather than translating hallucinated text as fact. Keep a human-review path for uncertain names and numbers.
5. Preserve word timestamps during repair. Any text correction must not silently shift the timestamps or reorder words.

### Phase 2 — Make translation faithful and terminology consistent

1. Translate corrected, ordered source units only. Enforce stable IDs and exact item counts/order at every batch/retry boundary.
2. Strengthen the translation contract: do not add examples or filler; do not change “most” to “all”; preserve conditions, negation, counts, OVRs, dates, currency, card names, and giveaway terms.
3. Maintain a small bilingual terminology map for canonical FC Mobile renderings and player-name spellings (for example OVR, Star Pass, Ballon d'Or, Yashin, Gullit, Matthäus, Figo, Dembélé). Use one policy for when proper names stay Latin versus Bengali transliteration.
4. Add a lightweight semantic QA pass for high-risk items: compare names, numbers, units, and negation/conditional phrases between English and Bangla. A mismatch should trigger re-translation or review, not automatic “correction” from uncertain ASR.
5. For low-confidence source text, do not let Gemini invent a plausible sentence. Carry an uncertainty flag through the pipeline and request human review or a cautious omission.

### Phase 3 — Close the gap between planned and actual TTS timing

1. Keep every dub unit anchored to its original source start and next-speaker deadline; retain genuine pauses and the configured small borrow.
2. After TTS, compare measured clip duration with the allowed window. If it cannot fit at the configured maximum tempo, shorten and re-synthesize before assembly. Treat end-trimming as a last resort and report the exact affected segment; the target for the gold benchmark is **zero clipped words/segments**.
3. Log and test the difference between predicted Bangla duration and actual TTS duration. Calibrate the speech-rate estimate per voice rather than assuming one character rate fits every sentence.
4. Generate Bangla subtitle times from measured TTS speech/word alignment where reliable. Character-count allocation across a whole speech unit is only an estimate and can put a subtitle line late/early when wording or pauses are uneven.
5. Add an integration regression with 2 s and 5 s source pauses: preserve the pause, keep the next line at its original onset, avoid overlap, and ensure no previous line is silently truncated.

### Phase 4 — Measure quality and prevent regressions

Track separate metrics; a single “accuracy” score hides the failure source:

- **STT:** word error rate on the reviewed set, plus exact accuracy for names, FC terms, OVRs, and numbers.
- **Translation:** human adequacy score and explicit checks for names, numbers, negation, conditions, and omissions/additions.
- **Timing:** median and 95th-percentile onset/end error against the reviewed audio; no semantic cue-order swaps; no unreported clipping; no cross-segment overlap.
- **Pause behavior:** preserve annotated 2–5 s source silences unless the source itself contains speech there.
- **Hallucinations:** the repeated 09:18–09:40 span should be recovered from audio or flagged—not emitted as invented/repeated Bangla.

Proposed initial acceptance for this clip: **zero critical name/number/condition errors in the gold set, no cue-order swaps, zero clipped TTS segments, and 95th-percentile onset error ≤0.5 s**. Revisit the threshold only after measuring actual source and dub audio.

### Phase 5 — Iterate on representative videos

After the gold-set fixes, test one name-heavy squad-building video, one fast gameplay clip with sparse commentary, and one ordinary talking-head clip. Review source, English transcript, Bangla text, and rendered audio together; update the glossary only for confirmed errors. Keep raw user videos and full transcripts out of Git; commit only small synthetic or permission-cleared regression fixtures.

## Suggested implementation sequence

1. **Diagnostic PR:** stable segment IDs, confidence/timing audit output, and a small fixture-based regression harness.
2. **STT PR:** confidence-gated targeted re-decode, context reset after silence/repetition, and conservative alias repairs.
3. **Translation PR:** high-risk semantic checks and a canonical English–Bangla game/name glossary.
4. **TTS/alignment PR:** post-synthesis fit-and-resynthesize loop, clipping guard, and subtitle/audio alignment validation.

Start with Phase 0 using the actual video and final dub. That will distinguish transcription errors from translation, TTS-duration, and subtitle-mapping errors before tuning any thresholds.
