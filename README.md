[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb)

# 🎙️ ProXMax — English → Bangla ভিডিও ডাবিং টুল

**English ভিডিওর কণ্ঠ স্বয়ংক্রিয়ভাবে বাংলায় ডাবিং করে নতুন ভিডিও তৈরি করে** — Google Drive-এ ভিডিও রাখলেই Google Colab (GPU) দিয়ে পুরো কাজ হয়ে যাবে, নিজের কম্পিউটারে কিছুই ইনস্টল করতে হবে না।

> 🌐 **অনুবাদে Google Gemini API** ব্যবহার করা হয় — দ্রুত, প্রসঙ্গ বুঝে অনুবাদ এবং ব্যর্থ call-এ স্বয়ংক্রিয় retry (সর্বোচ্চ ৭ বার)। Google Translate আর ব্যবহার হয় না।

---

## 📖 এক নজরে

- শুধুমাত্র একটি ভাষা-জোড়া: **English → Bangla** (সেই অনুযায়ী সর্বোচ্চ যত্নে টিউন করা)
- চলবে সম্পূর্ণ **Google Colab (T4 GPU)**-তে; ভিডিও আসবে **Google Drive** থেকে
- `Video/input` ফোল্ডারে যত ভিডিও থাকবে, সবগুলো **একটার পর একটা** ডাবিং হয়ে `Video/output` ফোল্ডারে যাবে
- আউটপুটের নাম মূল ফাইলের নামের শেষে `_bn` যোগ হয়ে হবে — যেমন `lecture1.mp4` → **`lecture1_bn.mp4`** (সাথে English ও Bangla সাবটাইটেল `.srt` ফাইলও)

---

## 🚀 কীভাবে ব্যবহার করবেন

### ধাপ ১: Google Drive-এ ভিডিও রাখুন
আপনার Google Drive-এ গিয়ে নিচের ফোল্ডারটি বানান এবং সেখানে English ভিডিওগুলো আপলোড করুন:

```
MyDrive/
└── Video/          ← V বড় হাতের
    ├── input/      ← এখানে English ভিডিও আপলোড করুন (.mp4, .mkv, .mov, .avi)
    └── output/     ← ডাবিং করা ভিডিও এখানে পাবেন
```

> ফোল্ডার দুটি আগে থেকে না থাকলেও সমস্যা নেই — Colab নোটবুক নিজে থেকেই তৈরি করে দেবে।

### ধাপ ২: Gemini API key নিন
[Google AI Studio](https://aistudio.google.com/app/apikey) থেকে একটি **Gemini API key** নিন (ফ্রি tier আছে)। এই key নোটবুকের **ধাপ ৬: Gemini API কনফিগারেশন** সেলে বসাতে হবে, অথবা Colab-এর Secrets (🔑) এ `GEMINI_API_KEY` নামে সেভ করতে হবে।

> ডিফল্ট অনুবাদ মডেল: **`gemini-3.5-flash-light`**। নোটবুকের ওই সেলেই `GEMINI_MODEL`-এর মান বদলে অন্য মডেল ব্যবহার করা যায়। key কারও সাথে শেয়ার করবেন না বা পাবলিক GitHub-এ কমিট করবেন না।

### ধাপ ৩: Colab নোটবুক খুলুন
এই পৃষ্ঠার একদম উপরের **"Open in Colab"** বাটনে ক্লিক করুন। নোটবুক খুললে প্রথমে উপরের মেনু থেকে **Runtime → Change runtime type → Hardware accelerator: T4 GPU** নির্বাচন করুন (GPU ছাড়া কাজ অনেক ধীর হবে)।

### ধাপ ৪: সেলগুলো উপর থেকে নিচে রান করুন
প্রতিটি সেলের বাম পাশের ▶ বাটনে ক্লিক করে একে একে রান করুন। সেলগুলোতে যা যা হবে:

1. Google Drive mount করা (অনুমতির পপ-আপে Allow করুন)
2. `Video/input` ও `Video/output` ফোল্ডার যাচাই/তৈরি করা
3. GPU (Tesla T4) ঠিকমতো পাওয়া গেছে কিনা যাচাই করা
4. এই রিপোজিটরি clone করা
5. সব dependency ও ffmpeg ইনস্টল/যাচাই করা
6. **Gemini API key ও মডেল কনফিগার করা** (একবারই লাগবে)
7. (ঐচ্ছিক) Whisper মডেল ও TTS ভয়েস পছন্দ করা
8. **মূল ডাবিং প্রোগ্রাম রান করা** — সব ভিডিও একে একে প্রসেস হবে
9. ফলাফলের তালিকা দেখা
10. কাজ শেষে GPU মেমরি খালি করা

### ধাপ ৫: ফলাফল সংগ্রহ করুন
প্রসেস শেষে Google Drive-এর **`Video/output`** ফোল্ডারে ডাবিং করা ভিডিওগুলো পাবেন। মাঝপথে কোনো কারণে Colab বন্ধ হয়ে গেলে শুধু **ধাপ ৮-এর সেলটি আবার রান করুন** — যে ভিডিওগুলোর আউটপুট তৈরি হয়ে গেছে সেগুলো নিজে থেকেই বাদ যাবে (resume সুবিধা), বাকিগুলো চলতে থাকবে।

> ⏳ **সময় কেমন লাগতে পারে:** প্রথমবার Whisper মডেল (~3 GB) ডাউনলোড হতে কিছু সময় লাগবে। এরপর T4 GPU-তে সাধারণত ভিডিওর দৈর্ঘ্যের প্রায় **৫–১৫ গুণ দ্রুত** গতিতে transcription হয় (যেমন ১০ মিনিটের ভিডিওতে মোটামুটি কয়েক মিনিট)।

---

## ⚙️ ভিতরে কীভাবে কাজ করে

```
input ভিডিও
   │
   ├─ ১. Audio Extraction ── ffmpeg দিয়ে 16kHz WAV অডিও আলাদা করা
   ├─ ২. Speech-to-Text ──── faster-whisper (large-v3, GPU) + word-level timestamp + VAD
   ├─ ৩. Translation ─────── English → Bangla — Google Gemini API (batch + retry)
   ├─ ৪. Text-to-Speech ──── edge-tts-এর Bangla neural ভয়েসে প্রতিটি segment-এর অডিও
   ├─ ৫. Sync Engine ─────── ffmpeg atempo দিয়ে প্রতিটি ক্লিপকে মূল সময়ে মেলানো
   └─ ৬. Final Muxing ────── মূল ভিডিওর picture (copy) + Bangla অডিও → *_bn.mp4
```

## ✨ ফিচারসমূহ

- 🎯 **উচ্চমানের transcription:** `large-v3` Whisper মডেল + word-level timestamp + VAD (নীরব অংশে ভুল টেক্সট আসে না)
- ⚡ **দ্রুত GPU প্রসেসিং:** CTranslate2 backend, `float16`, batched inference — সাধারণ Whisper-এর চেয়ে বহুগুণ দ্রুত
- 🤖 **Gemini অনুবাদ:** প্রসঙ্গ বুঝে স্বাভাবিক বাংলা; এক request-এ ২৫টি segment (API খরচ ও rate-limit কম), response schema দিয়ে সবসময় বৈধ JSON
- 🛡️ **শক্ত error handling:** rate limit মেনে চলা (ডিফল্ট ১৫ RPM), ব্যর্থ call-এ পূর্ণ error log + অপেক্ষা করে সর্বোচ্চ **৭ বার** retry; শেষ পর্যন্ত ব্যর্থ হলে প্রোগ্রাম crash না করে পরিষ্কার বার্তা দিয়ে পরের ভিডিওতে যায়
- 🔀 **সমান্তরাল কাজ:** TTS (async, একসঙ্গে একাধিক segment) ও time-stretch (একাধিক ffmpeg প্রসেস) — parallel
- 🧹 **GPU মুক্ত করা:** কাজ শেষে Whisper মডেল আনলোড করে VRAM ছেড়ে দেওয়া হয় — Colab-এ GPU ব্যবহার সাথে সাথে কমে যায়
- 🗣️ **সত্যিকারের Bangla কণ্ঠ:** Microsoft-এর neural ভয়েস — নারী (`bn-BD-NabanitaNeural`) ও পুরুষ (`bn-BD-PradeepNeural`)
- ⏱️ **সময় মিলিয়ে ডাবিং:** প্রতিটি Bangla ক্লিপ মূল English অংশের সময় অনুযায়ী টেনে/চেপে বসানো হয়, তাই ভিডিওর মোট দৈর্ঘ্য বদলায় না এবং lip-sync মোটামুটি বজায় থাকে
- 🎞️ **ভিডিওর মান হুবহু:** picture re-encode করা হয় না (`-c:v copy`) — মান একদম একই, সময়ও বাঁচে (বিশেষ কোডেকে স্বয়ংক্রিয় NVENC/x264 fallback আছে)
- 📦 **ব্যাচ প্রসেসিং ও resume:** পুরো ফোল্ডার একসঙ্গে; একটি ভিডিও নষ্ট/করাপ্ট থাকলেও বাকিগুলোর কাজ চলতে থাকে, শেষে সারসংক্ষেপ দেখা যায়
- 📝 **সাবটাইটেল ফ্রি:** প্রতিটি ভিডিওর সাথে English ও Bangla `.srt` ফাইলও তৈরি হয়
- 🧩 **পরিষ্কার modular কোড:** STT, Translation, TTS, Sync, Muxing — প্রতিটি ধাপ আলাদা মডিউলে, সব সেটিংস একটিমাত্র `config.yaml`-এ

## 🛠️ ব্যবহৃত টেকনোলজি ও মডেল

| কাজ | যা ব্যবহার করা হয়েছে |
|---|---|
| Speech-to-Text | **faster-whisper** (`large-v3`, চাইলে `medium`) — CTranslate2 engine-এ GPU acceleration |
| Translation (ডিফল্ট) | **Google Gemini API** (`google-genai` SDK) — ডিফল্ট মডেল **`gemini-3.5-flash-light`**, এক request-এ ২৫টি segment, ৭ বার পর্যন্ত retry |
| Translation (বিকল্প) | **IndicTrans2** (AI4Bharat) — English→Bangla-তে বিশেষ প্রশিক্ষিত লোকাল মডেল, offline চলতে পারে |
| Text-to-Speech | **edge-tts** — `bn-BD-NabanitaNeural` (নারী) / `bn-BD-PradeepNeural` (পুরুষ) |
| অডিও/ভিডিও প্রসেসিং | **ffmpeg** (extract, atempo time-stretch, muxing) + **pydub** (চূড়ান্ত ট্র্যাক সাজানো) |
| রানটাইম | **Google Colab (T4 GPU)** + Google Drive |

---

## 🔧 সেটিংস বদলানো (ঐচ্ছিক)

সব সেটিংস রিপোর **`config.yaml`** ফাইলে বাংলা কমেন্টসহ দেওয়া আছে। সাধারণ কয়েকটি:

- **বেশি speed দরকার হলে:** `stt → model: "medium"` করুন (accuracy সামান্য কমবে, গতি অনেক বাড়বে), অথবা নোটবুকের ধাপ ৭-এর সেলে `WHISPER_MODEL = 'medium'` লিখুন
- **পুরুষ কণ্ঠ চাইলে:** `tts → voice: "bn-BD-PradeepNeural"` করুন, অথবা ধাপ ৭-এর সেলে `TTS_VOICE` বদলান
- **অন্য Gemini মডেল ব্যবহার করতে:** `config.yaml`-এ `gemini → model` বদলান, অথবা নোটবুকের ধাপ ৬-এর `GEMINI_MODEL` বদলান, অথবা CLI-তে `--gemini-model <নাম>` দিন
- **Rate limit বেশি/কম হলে:** `gemini → requests_per_minute` আপনার API plan অনুযায়ী বদলান (ফ্রি tier-এ সাধারণত ১০–১৫ RPM)
- **IndicTrans2 অনুবাদ ব্যবহার করতে:** Colab-এ `pip install -r requirements-indictrans2.txt` চালিয়ে `translation → backend: "indictrans2"` করুন
- **আউটপুট থাকা সত্ত্বেও আবার প্রসেস করা:** ধাপ ৮-এর কমান্ডে `--force` যোগ করুন: `!python main.py --force`

## 📁 প্রজেক্ট স্ট্রাকচার

```
ProXMax/
├── notebook/
│   └── ProXMax_Colab.ipynb      ← Colab-এ চালানোর নোটবুক (এটাই মূল ব্যবহার)
├── proxmax/                     ← মূল পাইথন প্যাকেজ
│   ├── config.py                সেটিংস (dataclass + YAML লোডিং + validation)
│   ├── models.py                Segment ইত্যাদি শেয়ার করা ডেটা মডেল
│   ├── discovery.py             input ফোল্ডার স্ক্যান ও queue তৈরি
│   ├── audio_extraction.py      ffmpeg দিয়ে অডিও আলাদা করা
│   ├── stt.py                   faster-whisper (GPU, batched, VAD)
│   ├── translator.py            Gemini API (ডিফল্ট) / IndicTrans2 backend
│   ├── tts.py                   edge-tts (async, concurrent)
│   ├── sync_engine.py           atempo time-stretch + চূড়ান্ত অডিও ট্র্যাক
│   ├── muxer.py                 ভিডিও + Bangla অডিও মার্জ (stream copy)
│   ├── pipeline.py              পুরো pipeline ও batch controller
│   └── utils.py                 লগিং, ffmpeg হেল্পার, SRT রাইটার
├── main.py                      ← এন্ট্রি পয়েন্ট (CLI)
├── config.yaml                  ← সব সেটিংস (বাংলা কমেন্টসহ)
├── requirements.txt             pin করা dependencies
├── requirements-indictrans2.txt ঐচ্ছিক IndicTrans2 dependencies
└── tests/test_smoke.py          GPU/ইন্টারনেট ছাড়াই চালানো যায় এমন টেস্ট
```

---

## ⚠️ সতর্কতা ও সীমাবদ্ধতা

- 🌐 **ইন্টারনেট ও API key জরুরি:** অনুবাদ হয় Google Gemini API দিয়ে (API key লাগবে) এবং ভয়েস আসে edge-tts থেকে — দুটোই অনলাইন সার্ভিস। (IndicTrans2 backend নিলে অনুবাদ offlineও হয়, কিন্তু TTS-এর জন্য ইন্টারনেট লাগবেই।)
- 🔑 **API key গোপন রাখুন:** নোটবুকের key সেলটি কারও সাথে শেয়ার করবেন না বা পাবলিক রেপোতে কমিট করবেন না। ফাঁস হলে AI Studio থেকে key বাতিল করে নতুন নিন।
- ⏳ **Gemini rate limit:** ফ্রি tier-এ মিনিটে সীমিত সংখ্যক request পাওয়া যায় — কোড ডিফল্টে ১৫ RPM মেনে চলে ও ব্যর্থ call-এ অপেক্ষা করে retry করে; আপনার plan অনুযায়ী `gemini → requests_per_minute` ঠিক করে নিন।
- 🎵 **পেছনের মিউজিক/শব্দ থাকবে না:** আউটপুটে শুধু Bangla কণ্ঠ থাকবে; মূল ভিডিওর ব্যাকগ্রাউন্ড মিউজিক বা পরিবেশের শব্দ ডাবিং অডিওতে থাকে না।
- 👄 **Lip-sync আনুমানিক:** Bangla বাক্য English-এর চেয়ে লম্বা/ছোট হলে অডিও নির্দিষ্ট সীমার মধ্যে (0.65×–1.90×) দ্রুত/ধীর করে মেলানো হয়; খুব বেশি পার্থক্য হলে কিছু জায়গায় কণ্ঠ একটু দ্রুত শোনাতে পারে বা পরের বাক্যের আগে ছোট্ট বিরতি থাকতে পারে।
- 🗣️ **একটিমাত্র কণ্ঠ:** পুরো ভিডিও একই নির্বাচিত ভয়েসে ডাব হয় — একাধিক বক্তা থাকলেও আলাদা কণ্ঠ/আবেগ আসবে না, এবং গান/চিৎকার জাতীয় অংশ স্বাভাবিক কথার সুরে পড়া হবে।
- 📏 **খুব লম্বা ভিডিও:** Colab ফ্রি সেশনের সময়সীমা ও RAM সীমিত — ঘণ্টার পর ঘণ্টা লম্বা ভিডিও হলে ভেঙে আপলোড করা বা `medium` মডেল ব্যবহার করা ভালো। কাজ মাঝপথে থেমে গেলেও resume সুবিধায় ক্ষতি হবে না।
- 🔤 **অনুবাদের মান:** প্রযুক্তিগত শব্দ, নাম ও সংখ্যা মাঝে মাঝে হুবহু অনুবাদ নাও হতে পারে — পাশাপাশি থাকা `.srt` ফাইল দিয়ে সহজে যাচাই করা যায়।
- 📜 **কপিরাইট:** শুধুমাত্র নিজের তৈরি বা অনুমতি পাওয়া কনটেন্ট ডাবিং করুন।

## 🩺 সাধারণ সমস্যা ও সমাধান

| সমস্যা | সম্ভাব্য সমাধান |
|---|---|
| খুব ধীরে চলছে / "CUDA GPU পাওয়া যায়নি" বার্তা | Runtime → Change runtime type থেকে **T4 GPU** নির্বাচন করে নোটবুক আবার শুরু থেকে রান করুন |
| "Input ফোল্ডারে কোনো ভিডিও পাওয়া যায়নি" | Drive-এ `Video/input` (V বড় হাতের, input ছোট হাতের) ফোল্ডারেই ভিডিও রেখেছেন কিনা ও ফাইলের extension (.mp4/.mkv/.mov/.avi) ঠিক আছে কিনা দেখুন |
| Colab ডিসকানেক্ট হয়ে গেছে | নোটবুক আবার খুলে ধাপ ১–৬ দ্রুত রান করে ধাপ ৮ চালান — হয়ে যাওয়া ভিডিও বাদ গিয়ে বাকি কাজ চলবে |
| একটি ভিডিও "ব্যর্থ" দেখাচ্ছে | ভিডিওটি চালিয়ে দেখুন সচল ও অডিওসহ আছে কিনা; লগে নির্দিষ্ট কারণ লেখা থাকে (যেমন অডিও stream নেই / speech পাওয়া যায়নি) |
| GPU-এর মেমরি (VRAM) শেষ হয়ে যাওয়ার error | `config.yaml`-এ `stt → batch_size: 4` করুন অথবা মডেল `medium` ব্যবহার করুন (প্রোগ্রাম নিজে থেকেও non-batched পদ্ধতিতে fallback করার চেষ্টা করে) |
| কাজ শেষে GPU ব্যবহার কমছে না | প্রোগ্রাম এখন নিজেই কাজ শেষে Whisper আনলোড করে VRAM ছাড়ে (`runtime → release_gpu_when_done: true`)। তবুও কম না হলে নোটবুকের **ধাপ ১০** (GPU মেমরি পরিষ্কার) সেলটি রান করুন |
| "Gemini API key পাওয়া যায়নি" | নোটবুকের ধাপ ৬-এর সেলে `GEMINI_API_KEY` বসান, অথবা Colab Secrets-এ `GEMINI_API_KEY` যোগ করুন |
| "Gemini API error … 404/NOT_FOUND" | `GEMINI_MODEL`-এর নাম আপনার API key-তে নেই — নোটবুকের ধাপ ৬-এ মডেলের নাম বদলান |
| "429 RESOURCE_EXHAUSTED" / rate limit | কোড নিজেই অপেক্ষা করে retry করে। বারবার এলে `gemini → requests_per_minute` কমান বা কিছু সময় পর আবার চালান |

---

<details>
<summary>🧑‍💻 ডেভেলপারদের জন্য: লোকালি রান ও টেস্ট</summary>

```bash
git clone https://github.com/redwan-primeAmax/ProXMax.git
cd ProXMax
pip install -r requirements.txt     # সাথে সিস্টেমে ffmpeg থাকতে হবে
python main.py --help               # সব CLI অপশন
export GEMINI_API_KEY="your-key"
python main.py --input-dir ./input --output-dir ./output --model medium
python -m unittest discover -s tests -v   # GPU/ইন্টারনেট ছাড়া smoke টেস্ট
```

লোকাল মেশিনে GPU না থাকলে প্রোগ্রাম নিজে থেকেই CPU (int8) মোডে চলবে — কাজ হবে, তবে ধীরে।

</details>

---

❤️ দিয়ে তৈরি — English কনটেন্ট এখন বাংলাভাষী সবার জন্য!
