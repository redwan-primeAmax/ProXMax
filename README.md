# ProXMax — ইংরেজি ভিডিও থেকে বাংলা

Google Colab-এ ইংরেজি ভিডিওর কথা শনাক্ত করে বাংলা অনুবাদ ও অডিও তৈরি করে। ভিডিও আগে local storage-এ নেওয়া হয়; কাজ শেষে ফলাফল Google Drive-এ সংরক্ষিত হয়।

[![Colab-এ খুলুন](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebooks/ProXMax_English_to_Bangla.ipynb)

> নতুন পরিবর্তন `main`-এ merge হওয়ার আগে [notebook ফাইলটি](notebooks/ProXMax_English_to_Bangla.ipynb) নামিয়ে **Colab → File → Upload notebook** দিয়ে খুলুন। Git clone প্রয়োজন নেই; অনুবাদের জন্য Google AI Studio API key লাগবে।

## ব্যবহার

১. **Runtime → Change runtime type → T4 GPU** নির্বাচন করুন।
২. ইংরেজি ভিডিও রাখুন **My Drive → Video → input** folder-এ। ভেতরের subfolder খোঁজা হয় না।
৩. একমাত্র **▶ ProXMax · Setup ও App** cell চালান। নিচে দৃশ্যমান form আসবে; processing code hidden থাকবে।
৪. **Domain dropdown** থেকে সাধারণ ভিডিওতে `general`, FC Mobile ভিডিওতে `football` বেছে নিন। **Gemini API key**-এর masked ঘরে AI Studio key paste করে **সেটআপ ও অনুবাদ চালু করুন** চাপুন। এরপর setup, Drive সংযোগ ও অনুবাদ হবে। প্রথমবার Drive-এর অনুমতি ও model download লাগবে। প্রতি run-এ key দিতে হবে।
৫. ফলাফল পাবেন **My Drive → Video → output → `<job>`** folder-এ।

| কী থাকে | Colab-এর অবস্থান |
|---|---|
| মূল ভিডিও | `/content/drive/MyDrive/Video/input` |
| অনুবাদ করা ভিডিও | `/content/drive/MyDrive/Video/output` |
| কাজের ফাইল ও checkpoint | `/content/proxmax-work` |
| model cache | `/content/proxmax-model-cache` |
| আলাদা environment ও worker | `/content/proxmax-app` |

Notebook-এ **একটিই cell**। Processing code স্বাভাবিক Colab ভিউতে লুকানো থাকে; শুধু Run, Domain/API-key form, প্রয়োজনীয় status ও ফলাফল দেখা যায়। এটি code-এর নিরাপত্তা-সীমা নয়—প্রয়োজনে **Show code** দিয়ে source দেখা যায়।

মূল ভিডিও বদলানো বা মুছে ফেলা হয় না। একবারে একটি ভিডিও চলে। Shared Drive হলে সেটিংসে input/output path বদলান।

## আবার চালালে কী হবে?

- **“Drive already mounted” কোনো error নয়।** এখন সক্রিয় mount থাকলে সেটিই ব্যবহার হয়; অকারণে আবার mount করা হয় না।
- একই requirements-এ সফল Setup থাকলে আবার package install হয় না। নতুন notebook-এ requirements বদলালে নিজস্ব `venv`-এ প্রয়োজনীয় package হালনাগাদ হয়। Colab kernel-এর torch/CUDA বদলানো হয় না।
- প্রতিবার অনুবাদ **নতুন Python worker process**-এ চলে। আগের run-এর import cache বা CUDA state নতুন worker-এ যায় না। শেষ হলে process-এর model/GPU memory ছেড়ে যায়।
- Stop চাপলে worker এবং তার FFmpeg subprocess-ও বন্ধ হয়। একই cell আবার চালাতে পারেন।
- অসম্পূর্ণ কাজের checkpoint থাকে। একই input/settings দিয়ে আবার চালালে সম্পন্ন ASR/translation ধাপ এবং তৈরি TTS clip পুনর্ব্যবহার হয়।
- নতুন notebook-এ `keep_local=True`: সফল কাজের local checkpoint-ও থাকে, যাতে লেখা সংশোধন করা যায়। জায়গা বাঁচাতে `False` করুন (CLI/core-এর default `False`)। সফল ভিডিও স্বয়ংক্রিয়ভাবে বাদ দেওয়া হয় না—পরের run-এ আবার render হতে পারে।
- সংস্করণ **2.1.0**-এর recipe আলাদা: পুরোনো দ্রুত audio/অনুবাদ checkpoint নতুন কাজে ব্যবহার হবে না। পুরোনো Drive output মুছে দেওয়া হবে না।
- Runtime **delete করার প্রয়োজন ছাড়াই সাধারণ rerun** সমর্থনের জন্য এই ব্যবস্থা। তবে প্রকৃত GPU OOM, driver failure, Drive/network সমস্যা বা নষ্ট ভিডিও আলাদা করে সমাধান করতে হবে। Runtime হারিয়ে গেলে local checkpoint-ও হারাবে; Drive-এর ফলাফল থাকবে।

একই work/output folder নিয়ে একসঙ্গে দুটি run চালাবেন না। আগে Stop হওয়া শেষ হতে দিন।

## ফলাফলের ফাইল

প্রতিটি `<job>` folder-এ পাবেন:

- `translated_bn.mp4` — বাংলা অডিওসহ ভিডিও।
- `english.srt`, `bangla.srt` — আলাদা subtitle ফাইল।
- `segments.json` — raw/normalized ইংরেজি, সংশোধনের তালিকা, পূর্ণ/সংক্ষিপ্ত অনুবাদ, আসলে বলা `bangla`, গতি ও review চিহ্ন।
- `asr.json` — ইংরেজি শনাক্তকরণের তথ্য।
- `manifest.json` — সম্পন্ন হওয়ার চিহ্ন, checksum, সংস্করণ ও সময়ের তথ্য।

ফাইল কপি ও checksum যাচাইয়ের **সবশেষে** manifest লেখা হয়। এটি ছাড়া folder-টিকে সম্পন্ন ধরবেন না। Batch-এর ফলাফল ও error সরাসরি cell/terminal output-এ দেখায়; `batch_results.json`, `failure.json` বা `.log` ফাইল লেখা হয় না। Drive mount থেকে checksum পড়া যাচাই করা হয়; Google server-এ তাৎক্ষণিক স্থায়ী সংরক্ষণের নিশ্চয়তা নয়।

## কাজের ধারা

```text
Drive → local copy → FFmpeg → ইংরেজি ASR → বাংলা অনুবাদ
      → বাংলা TTS → সময় অনুযায়ী অডিও বসানো → MP4 → Drive
```

- **faster-whisper large-v3-turbo:** ইংরেজি কথা ও word timestamp; GPU-তে INT8/FP16।
- **Gemini API — `gemini-3.5-flash-lite`:** আগের/পরের phrase ও সময়ের budget নিয়ে পূর্ণ এবং সংক্ষিপ্ত বাংলা তৈরি। সংখ্যা, negation ও glossary-র প্রাথমিক checks ব্যর্থ হলে সীমিত corrective retry। এই ID-ই ব্যবহার হয়; unavailable হলে অন্য model-এ fallback নেই।
- **NLLB distilled 600M:** explicit offline বিকল্প; context-aware concise generation নেই, মান কমতে পারে।
- **MMS Bengali + uroman:** স্থানীয়ভাবে একটি বাংলা কণ্ঠ তৈরি।
- ASR/TTS local model একটির পর একটি চলে; অনুবাদের জন্য local Qwen আর load হয় না। অডিও timeline disk-backed; সম্ভব হলে H.264 picture re-encode না করে কপি করা হয়।

[স্থাপত্যের বিস্তারিত](docs/ARCHITECTURE.md)

## প্রয়োজন ও সীমাবদ্ধতা

- Python 3.10+, torch 2.6+ এবং FFmpeg/ffprobe। Gemini অনুবাদে GPU লাগে না; local ASR/TTS দ্রুত চালাতে T4 প্রস্তাবিত। CPU-ও ব্যবহার করা যায়, তবে ধীর। বর্তমান CTranslate2 pin **CUDA 12/cuDNN 9** লক্ষ্য করে।
- ছোট পরীক্ষায় অন্তত **10 GB local disk** খালি রাখুন; দীর্ঘ ভিডিওতে আরও জায়গা লাগে। Gemini weights download হয় না। Whisper পুরো অডিও host memory-তে নেয়; দীর্ঘ ভিডিও ভাগ করুন।
- Google AI Studio key, model access, ইন্টারনেট ও পর্যাপ্ত API quota প্রয়োজন। ব্যবহারে চার্জ হতে পারে; free/unlimited ব্যবহারের নিশ্চয়তা নেই। Timeout/retry-তেও একাধিক request bill হতে পারে।
- শুধু ইংরেজি → বাংলা। একটি সাধারণ কণ্ঠ; speaker আলাদা করা, voice cloning বা lip-sync নেই।
- **মূল অডিও সম্পূর্ণ বদলে যায়**, অর্থাৎ ইংরেজি কথার সঙ্গে গান/পরিবেশের শব্দও থাকে না। প্রথম ভিডিও ও প্রথম audio track নেওয়া হয়।
- নাম, সংখ্যা, বাগ্‌ধারা, ভুল ASR ও দ্রুত উচ্চারণ নিজে যাচাই করুন। নির্দিষ্ট accuracy বা realtime speed-এর প্রতিশ্রুতি নেই।
- Audio track আছে কিন্তু কথা নেই—এমন ভিডিওতে নীরব output ও সতর্কতা আসবে। Audio track-ই না থাকলে error হবে।

### দরকারি সেটিংস

সাধারণ ব্যবহারে code খুলতে হবে না। উন্নত সেটিংস বদলাতে cell-এর **Show code** থেকে `CONFIG` অংশটি সম্পাদনা করুন।

| প্রয়োজন | সেটিংস |
|---|---|
| কম API request | default `translation_batch_size=4`; দীর্ঘ cue/response হলে `1` করুন |
| API ছাড়া local অনুবাদ | `translation_backend="nllb"`, `translation_model="facebook/nllb-200-distilled-600M"` |
| দ্রুত ASR/NLLB, কিছু মানের ছাড়সহ | `asr_beam_size=1`, `translation_beams=1` (beam শুধু NLLB-এ) |
| CPU-তে ছোট ASR model | `device="cpu"`, `asr_model="Systran/faster-whisper-small.en"` |
| কথার গতি | `min_speed=0.90`, `max_speed=1.25`; এর বাইরের মান গ্রহণ করা হয় না |
| নির্দিষ্ট নাম/পরিভাষা | `glossary={"English name": "বাংলা নাম"}` |
| সফল কাজের checkpoint রাখা | `keep_local=True` |
| API timeout/retry | `gemini_timeout_seconds=120`, `gemini_http_retries=2` |
| Local model snapshot ধরে রাখা | ASR/TTS ও optional NLLB-এর `*_revision`-এ Hub commit ID; Gemini-তে `translation_revision` প্রযোজ্য নয় |

### গতি ও অনুবাদের সুরক্ষা

- **সব fit retry-সহ সর্বোচ্চ 1.25×**, সর্বনিম্ন 0.90×। পুরোনো `stretch` setting দিলেও cap এড়ানো যায় না; পুরোনো `comfortable_speed=1.50` এখন invalid।
- Subtitle-এ অঙ্ক অক্ষত থাকে; TTS-এর জন্য cardinal/decimal/ordinal সংখ্যা বাংলা কথায় লেখা হয় (`tts_text`)। বিশেষ phone/date/code পড়ার ধরন মানুষের review লাগতে পারে।
- আগে পূর্ণ অনুবাদের TTS মাপা হয়। সেটি না ধরলে checks-পাস করা সংক্ষিপ্ত বিকল্পের TTS মাপা হয়; শেষে দরকার হলে silence যোগ হয়। বলা সংস্করণটিই `bangla.srt`-এ যায়।
- দুটিই না ধরলে **cue ID ও local checkpoint path দেখিয়ে থামে**; 2–3× গতি, speech কেটে দেওয়া, overlap বা video লম্বা করে “সফল” output প্রকাশ করে না। আগে তৈরি সফল clip পরের run-এ থাকে।
- ছোট reaction-phrase কাছে থাকলে যুক্ত হয়। নাম/negation-এর মাঝখানে cut এড়ায়; বাধ্যতামূলক সীমায় cut হলে review flag থাকে।
- `football` mode-এ ASR hint ও audit-সহ glossary corrections; `general` mode-এ এসব football substitution নেই। অস্পষ্ট নাম বা million/billion-এর source-বিরোধ নিজে অনুমান করে ঠিক করা হয় না।

**সীমা:** checks সম্পূর্ণ semantic verifier নয়—missing clause, ভুল নাম/ASR বা দৃশ্যের সঙ্গে timing সব ধরতে পারে না; বৈধ বিকল্প বানানও আটকে যেতে পারে। `status=complete` মানে rendering/publication শেষ, অনুবাদ নির্ভুল নয়; manifest-এর `quality_status=human_review_required` সেটিই জানায়। `review_required=false`-ও accuracy certificate নয়। Forced boundary, source correction/সন্দেহ ও concise speech নির্বাচন review flag দিতে পারে।

## সমস্যা হলে

`input_dir`, `device` ইত্যাদি থাকা JSON হলো settings, error নয়। আগের সংস্করণের `Systran/faster-whisper-large-v3-turbo` ID ভুল ছিল। এখন [faster-whisper-এর model তালিকা](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/utils.py) অনুযায়ী `mobiuslabsgmbh/faster-whisper-large-v3-turbo` ব্যবহৃত হয়; পুরোনো ID override করলেও সংশোধন হয়। Public model download-এ `token=False` ব্যবহৃত হয়, তাই আগের invalid cached token পাঠানো হয় না। এই সংস্করণ private/gated model authentication সমর্থন করে না।

Config/result worker-এর সঙ্গে **memory pipe** দিয়ে আদান-প্রদান হয়; আলাদা config/result/error/log ফাইল হয় না। Error ও traceback সরাসরি terminal-এ আসে। ভিডিও, subtitle, সফল output manifest এবং resume checkpoint আগের মতোই থাকবে। পুরোনো সংস্করণের লেখা diagnostic ফাইল স্বয়ংক্রিয়ভাবে মুছে ফেলা হয় না।

| সমস্যা | করণীয় |
|---|---|
| Drive সংযোগ নষ্ট | একই cell-এর form-এর **Drive-এর অতিরিক্ত সেটিংস** খুলে পুনরায় mount নির্বাচন করুন; ঠিক হলে পরের run-এ অনির্বাচিত রাখুন |
| GPU OOM | batch size `1`, ছোট model বা ছোট ভিডিও ব্যবহার করুন; একই setting বারবার চালালে OOM ঠিক হবে না |
| CUDA/cuDNN error | উপযুক্ত Colab GPU runtime বা `device="cpu"`; kernel-এর torch এলোমেলোভাবে বদলাবেন না |
| Gemini 401/403 | AI Studio key, project permission ও API restrictions যাচাই করুন; key চ্যাটে/লগে দেবেন না |
| Gemini 404 | `gemini-3.5-flash-lite` ID ও আপনার project-এর model access যাচাই করুন; নিজে থেকে model বদলানো হবে না |
| Gemini 429 | quota/rate limit/billing যাচাই করে পরে চালান; সীমিত backoff-এর পর থামে |
| Gemini token/JSON error | batch ছোট করুন বা model-এর অনুমোদিত token limit-এর মধ্যে settings ঠিক করুন |
| Download error | ইন্টারনেট/Hugging Face সংযোগ ঠিক করে আবার চালান |
| `open() got an unexpected keyword argument 'metadata_errors'` | পুরোনো faster-whisper/PyAV API mismatch। নতুন notebook চালান: setup `av==16.0.1` বসাবে, আর ASR-এ normalized WAV সরাসরি array হিসেবে যাবে—PyAV দিয়ে আবার decode হবে না। |
| ভিডিও পাওয়া যাচ্ছে না | সঠিক input folder, সমর্থিত extension ও সম্পূর্ণ upload নিশ্চিত করুন |
| অনুবাদ check/সময় মেলাতে ব্যর্থ | দেখানো local job-এর `translation.json` → `segments`-এ `bangla` ও প্রয়োজনে `bangla_concise` অর্থ রেখে ঠিক করুন; ID/time বদলাবেন না। একই settings-এ আবার চালান। JSON ID শূন্য থেকে, SRT cue এক থেকে। |
| অন্য error | cell/terminal-এ দেখানো error ও traceback কপি করুন; কোনো error ফাইল খুঁজতে হবে না |

সংশোধনের সময় phrase-এর ID/timestamp বদলাবেন না। Drive-এর `segments.json` বদলালে তা input checkpoint হিসেবে নেওয়া হয় না। Terminal output-এ transcript/path থাকতে পারে—শেয়ার করার আগে ব্যক্তিগত তথ্য সরান। তৃতীয় পক্ষের error/traceback ও কারিগরি field name মূল ভাষাতেই থাকে।

## Notebook হালনাগাদ

```bash
python scripts/build_notebook.py
python scripts/build_notebook.py --check
```

`proxmax/pipeline.py`, `language.py`, `gemini.py`, `colab_ui.py`, `colab_runtime.py`, `worker.py` বা dependency বদলালে notebook আবার generate করুন। Generator চালাতে শুধু Python লাগে; test dependency লাগে না। Notebook source সঙ্গে নিয়েই চলে; GitHub থেকে code download করে না। একমাত্র cell-টি Colab form view-তে থাকে; processing source সাধারণ ভিউতে দেখা যায় না।

ব্যবহারের আগে Colab-এ একটি ছোট ভিডিও দিয়ে model download, বাংলা উচ্চারণ, timing ও Drive output যাচাই করুন।

চাইলে repository থেকে আলাদা CLI-ও চালানো যায়; compatible torch, FFmpeg ও `requirements-colab.txt` আগে install করতে হবে:

```bash
python -m proxmax \
  --input-dir /path/to/input \
  --output-dir /path/to/output \
  --work-dir /path/to/local-work \
  --device auto
```

CLI-তে Gemini key `GEMINI_API_KEY` environment variable দিয়ে দিন; command argument/settings file-এ key লিখবেন না।

অতিরিক্ত setting দিতে `--config settings.json`; CLI path/device/keep-local argument অগ্রাধিকার পায়। কোনো ভিডিও ব্যর্থ হলে exit status nonzero হয়। CLI নিজে Drive mount করে না।

## গোপনীয়তা ও লাইসেন্স

নিজের অনুমোদিত ভিডিও ব্যবহার করুন। **Gemini mode-এ ইংরেজি transcript, আশপাশের context, সময়ের budget, glossary এবং corrective feedback Google-এ পাঠানো হয়।** Raw ভিডিও/audio Gemini-তে upload করা হয় না; ASR/TTS local থাকে। Google AI Studio/Gemini-এর আপনার account-এ প্রযোজ্য data-use, retention ও billing terms দেখে নিন। Colab ও Drive-ও cloud service।

API key দৃশ্যমান form-এর native `password` input থেকে শুধু worker-এর environment/memory-তে যায়; `CONFIG`, source, checkpoint, manifest বা Drive output-এ লেখা হয় না। Worker key environment থেকে সরিয়ে নেয়, যাতে FFmpeg subprocess সেটি না পায়। API URL-এ key নয়, HTTPS header ব্যবহৃত হয়; redirect অনুসরণ করা হয় না। Error body/headers প্রকাশ করা হয় না, terminal-এ sanitized error আসে। Key কোনো Colab `@param` বা saved ipywidget state নয়; field-এর value HTML/source-এ বসানো হয় না। Start/বাতিল করলে ঘরটি খালি ও disabled হয়। Form-এ অপেক্ষার সময় কোনো API call/setup শুরু হয় না। Key হারালে AI Studio থেকে revoke/rotate করুন।

কাজের শুরুতে exact model/key preflight হয়। Gemini-র সফল batch স্থানীয় `translation_progress.json` checkpoint-এ থাকে, যাতে মাঝপথে network/quota failure হলে আগের batch-এর API call পুনরায় না লাগে। `manifest.translation_api`-তে পাওয়া model version ও successful response-এর token usage থাকে—এটি চূড়ান্ত bill হিসাব নয়। Model alias ভবিষ্যতে বদলাতে পারে; Hub revision দিয়ে Gemini pin হয় না।

এই পরিবেশে live API/model availability, quota বা বাংলা output পরীক্ষা করা হয়নি; notebook preflight ও আপনার Colab run-এ যাচাই হবে।

- [Whisper conversion](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo)
- [Google AI Studio](https://aistudio.google.com/) — Gemini key/model access
- [Gemini API documentation](https://ai.google.dev/gemini-api/docs) — model availability, data ও quota/billing terms যাচাই করুন
- [NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M) — **CC-BY-NC-4.0**
- [MMS Bengali](https://huggingface.co/facebook/mms-tts-ben) — **CC-BY-NC-4.0**

শেষের দুটি model-এ non-commercial সীমাবদ্ধতা আছে; এই stack অবাধ বাণিজ্যিক ব্যবহারের জন্য নয়। Repository-এর code license মালিককে নির্ধারণ করতে হবে; সেটি model-এর license বদলাবে না।

ভিডিও, transcript, model weight, credential ও চালানো notebook-এর output Git-এ commit করবেন না। `.gitignore` সাধারণ runtime ফাইলগুলো বাদ দেয়; প্রকাশের আগে `git diff` দেখুন।
