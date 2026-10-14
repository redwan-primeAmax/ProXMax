# ProXMax — ইংরেজি ভিডিও থেকে বাংলা

Google Colab-এ ইংরেজি ভিডিওর কথা শনাক্ত করে বাংলা অনুবাদ ও অডিও তৈরি করে। ভিডিও আগে local storage-এ নেওয়া হয়; কাজ শেষে ফলাফল Google Drive-এ সংরক্ষিত হয়।

[![Colab-এ খুলুন](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebooks/ProXMax_English_to_Bangla.ipynb)

> নতুন পরিবর্তন `main`-এ merge হওয়ার আগে [notebook ফাইলটি](notebooks/ProXMax_English_to_Bangla.ipynb) নামিয়ে **Colab → File → Upload notebook** দিয়ে খুলুন। Git clone বা API key প্রয়োজন নেই।

## ব্যবহার

১. **Runtime → Change runtime type → T4 GPU** নির্বাচন করুন।
২. ইংরেজি ভিডিও রাখুন **My Drive → Video → input** folder-এ। ভেতরের subfolder খোঁজা হয় না।
৩. একমাত্র **▶ Setup ও App চালু করুন** cell চালান। Setup, Drive সংযোগ ও অনুবাদ পরপর নিজে থেকেই হবে। প্রথমবার Drive-এর অনুমতি ও কয়েক GB model download লাগবে।
৪. ফলাফল পাবেন **My Drive → Video → output → `<job>`** folder-এ।

| কী থাকে | Colab-এর অবস্থান |
|---|---|
| মূল ভিডিও | `/content/drive/MyDrive/Video/input` |
| অনুবাদ করা ভিডিও | `/content/drive/MyDrive/Video/output` |
| কাজের ফাইল ও checkpoint | `/content/proxmax-work` |
| model cache | `/content/proxmax-model-cache` |
| আলাদা environment, worker ও log | `/content/proxmax-app` |

Notebook-এ **একটিই cell**। Processing code স্বাভাবিক Colab ভিউতে লুকানো থাকে; শুধু Run, প্রয়োজনীয় status ও ফলাফল দেখা যায়। এটি code-এর নিরাপত্তা-সীমা নয়—প্রয়োজনে **Show code** দিয়ে source দেখা যায়।

মূল ভিডিও বদলানো বা মুছে ফেলা হয় না। একবারে একটি ভিডিও চলে। Shared Drive হলে সেটিংসে input/output path বদলান।

## আবার চালালে কী হবে?

- **“Drive already mounted” কোনো error নয়।** এখন সক্রিয় mount থাকলে সেটিই ব্যবহার হয়; অকারণে আবার mount করা হয় না।
- সফল Setup থাকলে আবার package install হয় না। প্যাকেজ notebook kernel-এ নয়, নিজস্ব `venv`-এ install হয়। Colab-এর torch/CUDA ব্যবহার করা হয়, বদলানো হয় না।
- প্রতিবার অনুবাদ **নতুন Python worker process**-এ চলে। আগের run-এর import cache বা CUDA state নতুন worker-এ যায় না। শেষ হলে process-এর model/GPU memory ছেড়ে যায়।
- Stop চাপলে worker এবং তার FFmpeg subprocess-ও বন্ধ হয়। একই cell আবার চালাতে পারেন।
- অসম্পূর্ণ কাজের checkpoint থাকে। একই input/settings দিয়ে আবার চালালে সম্পন্ন ASR/translation ধাপ এবং তৈরি TTS clip পুনর্ব্যবহার হয়।
- সফল কাজের local checkpoint সাধারণত মুছে যায়; সেগুলো রাখতে `keep_local=True` করুন। সফল ভিডিও স্বয়ংক্রিয়ভাবে বাদ দেওয়া হয় না—পরের run-এ আবার render হতে পারে।
- Runtime **delete করার প্রয়োজন ছাড়াই সাধারণ rerun** সমর্থনের জন্য এই ব্যবস্থা। তবে প্রকৃত GPU OOM, driver failure, Drive/network সমস্যা বা নষ্ট ভিডিও আলাদা করে সমাধান করতে হবে। Runtime হারিয়ে গেলে local checkpoint-ও হারাবে; Drive-এর ফলাফল থাকবে।

একই work/output folder নিয়ে একসঙ্গে দুটি run চালাবেন না। আগে Stop হওয়া শেষ হতে দিন।

## ফলাফলের ফাইল

প্রতিটি `<job>` folder-এ পাবেন:

- `translated_bn.mp4` — বাংলা অডিওসহ ভিডিও।
- `english.srt`, `bangla.srt` — আলাদা subtitle ফাইল।
- `segments.json` — কথার সময়, অনুবাদ, গতি ও যাচাইয়ের চিহ্ন।
- `asr.json` — ইংরেজি শনাক্তকরণের তথ্য।
- `manifest.json` — সম্পন্ন হওয়ার চিহ্ন, checksum, সংস্করণ ও সময়ের তথ্য।

ফাইল কপি ও checksum যাচাইয়ের **সবশেষে** manifest লেখা হয়। এটি ছাড়া folder-টিকে সম্পন্ন ধরবেন না। `Video/output/batch_results.json`-এ সর্বশেষ batch-এর ফলাফল থাকে। Drive mount থেকে checksum পড়া যাচাই করা হয়; Google server-এ তাৎক্ষণিক স্থায়ী সংরক্ষণের নিশ্চয়তা নয়।

## কাজের ধারা

```text
Drive → local copy → FFmpeg → ইংরেজি ASR → বাংলা অনুবাদ
      → বাংলা TTS → সময় অনুযায়ী অডিও বসানো → MP4 → Drive
```

- **faster-whisper large-v3-turbo:** ইংরেজি কথা ও word timestamp; GPU-তে INT8/FP16।
- **NLLB distilled 600M:** `eng_Latn → ben_Beng`, ছোট batch-এ অনুবাদ।
- **MMS Bengali + uroman:** স্থানীয়ভাবে একটি বাংলা কণ্ঠ তৈরি।
- Model-গুলো একটির পর একটি চলে। অডিও timeline disk-backed; সম্ভব হলে H.264 picture re-encode না করে কপি করা হয়।

[স্থাপত্যের বিস্তারিত](docs/ARCHITECTURE.md) · [Colab-এ পরীক্ষার তালিকা](docs/COLAB_SMOKE_TEST.md)

## প্রয়োজন ও সীমাবদ্ধতা

- Python 3.10+, torch 2.6+ এবং FFmpeg/ffprobe। T4 GPU প্রস্তাবিত; CPU ধীর। বর্তমান CTranslate2 pin **CUDA 12/cuDNN 9** লক্ষ্য করে।
- ছোট পরীক্ষার জন্য অন্তত **10 GB local disk** খালি রাখুন। দীর্ঘ ভিডিওতে বেশি জায়গা/RAM লাগবে। Whisper পুরো অডিও host memory-তে নেয়; খুব দীর্ঘ ভিডিও ভাগ করুন।
- শুধু ইংরেজি → বাংলা। একটি সাধারণ কণ্ঠ; speaker আলাদা করা, voice cloning বা lip-sync নেই।
- **মূল অডিও সম্পূর্ণ বদলে যায়**, অর্থাৎ ইংরেজি কথার সঙ্গে গান/পরিবেশের শব্দও থাকে না। প্রথম ভিডিও ও প্রথম audio track নেওয়া হয়।
- নাম, সংখ্যা, বাগ্‌ধারা, ভুল ASR ও দ্রুত উচ্চারণ নিজে যাচাই করুন। নির্দিষ্ট accuracy বা realtime speed-এর প্রতিশ্রুতি নেই।
- Audio track আছে কিন্তু কথা নেই—এমন ভিডিওতে নীরব output ও সতর্কতা আসবে। Audio track-ই না থাকলে error হবে।

### দরকারি সেটিংস

সাধারণ ব্যবহারে code খুলতে হবে না। উন্নত সেটিংস বদলাতে cell-এর **Show code** থেকে `CONFIG` অংশটি সম্পাদনা করুন।

| প্রয়োজন | সেটিংস |
|---|---|
| কম GPU memory | `translation_batch_size=1` |
| দ্রুত, কিছু মানের ছাড়সহ | `asr_beam_size=1`, `translation_beams=1` |
| CPU-তে ছোট ASR model | `device="cpu"`, `asr_model="Systran/faster-whisper-small.en"` |
| অতিরিক্ত দ্রুত উচ্চারণে থামা | `overflow_policy="error"` |
| সফল কাজের checkpoint রাখা | `keep_local=True` |
| একই model snapshot ধরে রাখা | তিনটি `*_revision`-এ Hub commit ID |

স্বাভাবিকভাবে কথাকে সময়ের মধ্যে ধরাতে গতি বাড়ে; `1.50×` ছাড়ালে যাচাইয়ের চিহ্ন দেওয়া হয়। কথা কেটে ছোট করা হয় না। ছোট অডিওর শেষে নীরবতা বসে; অংশগুলো মূল timestamp অনুযায়ী বসানো হয়।

## সমস্যা হলে

| সমস্যা | করণীয় |
|---|---|
| Drive সংযোগ নষ্ট | একই cell-এর `REMOUNT_DRIVE` ঘরটি নির্বাচন করে চালান; ঠিক হলে আবার `False` করুন |
| GPU OOM | batch size `1`, ছোট model বা ছোট ভিডিও ব্যবহার করুন; একই setting বারবার চালালে OOM ঠিক হবে না |
| CUDA/cuDNN error | উপযুক্ত Colab GPU runtime বা `device="cpu"`; kernel-এর torch এলোমেলোভাবে বদলাবেন না |
| Download error | ইন্টারনেট/Hugging Face সংযোগ ঠিক করে আবার চালান |
| ভিডিও পাওয়া যাচ্ছে না | সঠিক input folder, সমর্থিত extension ও সম্পূর্ণ upload নিশ্চিত করুন |
| অনুবাদ ভুল/খুব দ্রুত | `keep_local=True` রেখে local job-এর `translation.json`-এ শুধু `bangla` লেখা ঠিক করে আবার চালান |
| অন্য error | শেষে দেখানো `/content/proxmax-app/runs/*.log` এবং local `failure.json`-এর তথ্য দিন |

সংশোধনের সময় phrase-এর ID/timestamp বদলাবেন না। Drive-এর `segments.json` বদলালে তা input checkpoint হিসেবে নেওয়া হয় না। Log-এ transcript/path থাকতে পারে—শেয়ার করার আগে ব্যক্তিগত তথ্য সরান। তৃতীয় পক্ষের error/traceback ও কারিগরি field name মূল ভাষাতেই থাকে।

## উন্নয়ন ও পরীক্ষা

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
# OS package manager দিয়ে ffmpeg ও ffprobe install করুন।
python scripts/build_notebook.py
python scripts/build_notebook.py --check
python -m pytest -q
```

`proxmax/pipeline.py`, `colab_runtime.py`, `worker.py` বা dependency বদলালে notebook আবার generate করুন। Notebook source সঙ্গে নিয়েই চলে; GitHub থেকে code download করে না। একমাত্র cell-টি Colab form view-তে থাকে; processing source সাধারণ ভিউতে দেখা যায় না।

পরীক্ষায় timing, cache, FFmpeg, প্রকাশের checksum, পুনরায় setup, worker isolation ও notebook যাচাই হয়। **আসল model download/GPU inference/বাংলা উচ্চারণ এই lightweight test-এ যাচাই হয় না।** Release-এর আগে [Colab checklist](docs/COLAB_SMOKE_TEST.md) অনুসরণ করুন।

চাইলে repository থেকে আলাদা CLI-ও চালানো যায়; compatible torch, FFmpeg ও `requirements-colab.txt` আগে install করতে হবে:

```bash
python -m proxmax \
  --input-dir /path/to/input \
  --output-dir /path/to/output \
  --work-dir /path/to/local-work \
  --device auto
```

অতিরিক্ত setting দিতে `--config settings.json`; CLI path/device/keep-local argument অগ্রাধিকার পায়। কোনো ভিডিও ব্যর্থ হলে exit status nonzero হয়। CLI নিজে Drive mount করে না।

## গোপনীয়তা ও লাইসেন্স

নিজের অনুমোদিত ভিডিও ব্যবহার করুন। Model download Hugging Face থেকে হয়; speech/text কোনো hosted inference API-তে পাঠানো হয় না। তবে Colab ও Drive নিজেরাই cloud service।

- [Whisper conversion](https://huggingface.co/Systran/faster-whisper-large-v3-turbo)
- [NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M) — **CC-BY-NC-4.0**
- [MMS Bengali](https://huggingface.co/facebook/mms-tts-ben) — **CC-BY-NC-4.0**

শেষের দুটি model-এ non-commercial সীমাবদ্ধতা আছে; এই stack অবাধ বাণিজ্যিক ব্যবহারের জন্য নয়। Repository-এর code license মালিককে নির্ধারণ করতে হবে; সেটি model-এর license বদলাবে না।

ভিডিও, transcript, model weight, credential ও চালানো notebook-এর output Git-এ commit করবেন না। `.gitignore` সাধারণ runtime ফাইলগুলো বাদ দেয়; প্রকাশের আগে `git diff` দেখুন।
