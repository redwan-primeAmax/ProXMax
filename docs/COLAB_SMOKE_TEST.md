# Real-model Colab acceptance checklist

This is a **manual validation procedure**, not a claim that it has already passed. Lightweight repository tests do not download or evaluate the real models.

## Prepare

- [ ] Use a fresh current Colab runtime with a T4 GPU. Record Python, torch, CUDA and cuDNN versions plus GPU name.
- [ ] Upload/open the committed standalone notebook; do not rely on an existing repo clone.
- [ ] Have internet access to Hugging Face and enough local disk (10+ GB for a short trial).
- [ ] Use a permitted 20–40 second English clip with clear speech, two sentences, a pause, a name and a number. Prepare an English reference and ask a fluent Bengali reviewer to check it.
- [ ] Put it in `MyDrive/Video/input`. Start with just this one video; the batch processes every supported input file.
- [ ] Set `keep_local=True` for inspection/recovery tests; leave the remaining defaults initially.

## Fresh run

- [ ] Run every cell in order; installation, torch version check, Drive mount and model downloads succeed without secrets in the notebook.
- [ ] Verify the job and model caches are under `/content`, not `/content/drive`.
- [ ] ASR timestamps match source speech; there are no hallucinated sentences in silence.
- [ ] `translation.json` contains Bengali text with correct meaning. Review names/numbers rather than assuming they were pronounced correctly.
- [ ] Bengali synthesis works with the checkpoint tokenizer/uroman and has no NaNs/silent failures.
- [ ] Listen at the start, middle and end: complete phrases are audible, speech starts in its intended slot, pauses remain, and there is no accumulated drift.
- [ ] Review every speed flag and ensure extreme time compression has not made speech unintelligible.
- [ ] Confirm the source picture is intact and the original English/music track is intentionally absent.
- [ ] Confirm the output duration is within 250 ms of the source. This container-level check is not sufficient on its own; listen as well.
- [ ] Confirm output MP4 plays in Google Drive and both UTF-8 SRT files load in a subtitle-capable player.
- [ ] Confirm a complete manifest exists only after all declared artifacts are present and their hashes match.
- [ ] Source bytes are unchanged. No transcript, checkpoint, media or executed cell output is committed to Git.

## Recovery and quality controls

- [ ] Rerun with identical settings while keeping local work: logs show ASR and translation checkpoint reuse, and completed TTS clip hashes are reused.
- [ ] Edit one local `translation.json` Bengali phrase, preserving IDs/times. Rerun: that phrase is synthesized again, the other clip caches remain reusable, and the Drive output is updated.
- [ ] Set a low comfort limit (at least 1.0) and `overflow_policy="error"` on a deliberately long translation. Ensure it fails clearly, retains checkpoints, and does not publish a new completed bundle.
- [ ] Add a corrupt/no-audio input and a valid input. Confirm `batch_results.json` reports the former as failed while the latter completes.
- [ ] Test a video with an audio stream but no speech. It should emit a no-speech warning and a silent audio track, not invented dialogue.
- [ ] Test a second file with spaces, quotes and Bengali characters in its name.
- [ ] Test a non-H.264 source to exercise the documented re-encoding fallback.
- [ ] On a short clip, test `device="cpu"` and `asr_model="Systran/faster-whisper-small.en"`; expect lower speed/quality. CPU mode must not require CUDA.
- [ ] Turn `keep_local=False` back on for normal use: successful local job files should be removed only after publication. Failed job files should remain.

## Record results honestly

Record the source length, selected device, model revisions, package versions, cold/warm/cache state, elapsed time, peak host/GPU memory if measured, and fluent-review findings. Keep private media and transcripts out of a public report. The output manifest contains resolved model revisions and timing diagnostics.

Useful metrics for a controlled evaluation:

- ASR word error rate against a manually corrected English reference.
- Fluent-review scores for Bengali meaning, terminology and naturalness (with an explicit rubric).
- Fraction of phrases above the speed comfort limit, plus manual assessment of their intelligibility.
- Phrase-onset error against manually marked source intervals; container duration alone is not lip-sync quality.
- Processing time / source duration, with download and publication times separately identified.

Do not label a release “validated in Colab” until the fresh-run real-model checks actually pass. A dependency/model/runtime change should trigger another smoke test.
