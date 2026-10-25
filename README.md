# 🎙️ ProXMax — English → Bangla ভিডিও ডাবিং

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb)
[![T4 GPU](https://img.shields.io/badge/GPU-T4%20(16GB)-76B900?logo=nvidia&logoColor=white)](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

ProXMax একটি English ভিডিওকে **বাংলা কণ্ঠে** ডাবিং করে। ভিডিও থেকে কথা শোনে
(faster-whisper), Gemini দিয়ে বাংলায় অনুবাদ করে, বাংলা neural ভয়েসে কথা
বানায় এবং এমনভাবে বসায় যাতে **কোনো দুই কণ্ঠ কখনো একসাথে না বলে**।

> **কণ্ঠের জন্য আপনাকে কিছু দিতে হবে না** — না রেকর্ডিং, না Hugging Face
> টোকেন, না কোনো লাইসেন্সের সম্মতি। বিল্ট-ইন বাংলা neural ভয়েস দিয়েই কাজ
> শুরু হয়ে যায়। চাইলে পরে আপনার নিজের কণ্ঠও ক্লোন করতে পারবেন (ঐচ্ছিক)।

পুরো কাজটি Google Colab-এর **T4 GPU**-তে চলে; ইনপুট ও আউটপুট দুটোই Google
Drive-এর ফোল্ডার থেকে।

---

## ☁️ এক ক্লিকে Colab-এ খুলুন

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb)

বাটনে ক্লিক করলেই সম্পূর্ণ নোটবুকটি Colab-এ খুলে যাবে — সবসময় GitHub-এর
`main` ব্র্যাঞ্চের **সর্বশেষ কোড** ব্যবহার হবে (আলাদা করে কিছু ডাউনলোড বা
আপলোড করতে হবে না)।

বাটনে ক্লিক করার পর মাত্র ২টি কাজ:

1. `Runtime → Change runtime type → **T4 GPU**` বেছে নিন।
2. নোটবুকের ধাপ ৬-এ **Gemini API key** দিন (Colab Secrets-এ `GEMINI_API_KEY`
   রাখলে নিজে থেকেই পড়ে নেবে) → তারপর ধাপ ৮-এ `python main.py` সেলটি রান করুন।

কণ্ঠের জন্য তৃতীয় কিছু লাগে না — বিল্ট-ইন বাংলা ভয়েস ডিফল্টেই কাজ করে।

> নোটবুকের লিংক: <https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb>

---

## 📖 এক নজরে

| আগে যে সমস্যা ছিল | এখন কী করা হয় |
|---|---|
| Whisper ৩০ সেকেন্ডের সীমায় **শব্দের মাঝখানে** কাটত (`XYZ` → `XY` + `Z`) | শুধু **৩০০ ms+ নীরবতার মাঝখানে** কাটা হয় (১৮–২৮s chunk), সীমানায় সন্দেহ হলে ছোট উইন্ডো আবার ডিকোড (seam repair) |
| পাংচুয়েশন হারিয়ে যেত, বাক্য ভাঙত | আগের লেখার শেষ ~২২০ অক্ষর **rolling prompt** হিসেবে যায় → বাক্যের ধারাবাহিকতা ও punctuation ঠিক থাকে |
| রোবটিক কণ্ঠ | **বিল্ট-ইন বাংলা neural ভয়েস** (`bn-BD-NabanitaNeural`) — কোনো সেটআপ লাগে না; tempo কখনো ১.২৫-এর বেশি নয় |
| দুই কণ্ঠ একসাথে বাজত | **Non-overlap sync solver** — প্রতিটি ক্লিপের জন্য আলাদা সময়-উইন্ডো; কঠোরভাবে কোনো overlap নেই |
| অনুবাদ সময়ের চেয়ে লম্বা/ছোট হতো | Gemini-কে প্রতিটি লাইনের **সময়ের বাজেট** জানানো হয়; না আঁটলে লাইনটি ছোট করে লেখানো হয় |
| ১০ মিনিটের ভিডিওতে অনেক সময় লাগত | অনুবাদ (নেটওয়ার্ক) ও TTS (GPU) **একসাথে** চলে; গতি বাজেট রাখতে একটি Speed Governor আছে → **২–৩ মিনিট** |
| টার্মিনালে আবার key/token চাইত, সেল রান করলে প্রম্পট আসত | **কোনো প্রম্পট নেই** — key নোটবুকের সেলেই পেস্ট করা হয়; HF token না দিলে অ্যাপ ধরে নেয় ক্লোনিং দরকার নেই এবং চুপচাপ বিল্ট-ইন ভয়েসে এগিয়ে যায় |
| অডিও extract করার সময় ঝুলে যেত, একই লগ বারবার আসত | ffmpeg-এ **`-nostdin`** + stdin বন্ধ (ব্লক হওয়ার মূল কারণ), **লাইভ শতাংশ লগ**, আর অগ্রগতি বন্ধ হলে **stall detector** — নিজেই থেমে software decoding-এ নামে বা স্পষ্ট error দেয় |
| Gemini key ভুলে গেলে অনেকক্ষণ পরে ফেইল করত | key না থাকলে **শুরুতেই** পরিষ্কার error (`exit=2`), কিছুই প্রসেস শুরু হয় না |

---

## ✅ যা যা লাগবে

1. **Google Colab** (Runtime → Change runtime type → **T4 GPU**)।
2. **ভিডিও** — `/content/drive/MyDrive/Video/input/` ফোল্ডারে (`.mp4/.mkv/.mov/.avi`)।
3. **Gemini API key** — https://aistudio.google.com/app/apikey (শুধু অনুবাদের জন্য)।
   নোটবুকের **ধাপ ৬**-এ `GEMINI_API_KEY = ''` এর ভেতরে পেস্ট করবেন — এটাই
   একমাত্র জায়গা, এরপর আর কোথাও কিছু টাইপ করতে হবে না।

**কণ্ঠের জন্য কিছুই লাগে না।** ডিফল্টে বিল্ট-ইন `bn-BD` neural ভয়েস ব্যবহার
হয়। চাইলে **ঐচ্ছিকভাবে** আপনার নিজের কণ্ঠ ক্লোন করতে পারেন — নিচের
**“ঐচ্ছিক: নিজের কণ্ঠে ডাবিং”** অংশ দেখুন।

---

## 🚀 কীভাবে ব্যবহার করবেন (Google Colab)

### ধাপ ১: Drive প্রস্তুত করুন

```
MyDrive/
└── Video/
    ├── input/
    │   └── lecture01.mp4          ← ডাবিং করতে হবে
    └── output/                    ← ফলাফল এখানে আসবে
```

### ধাপ ২: প্রজেক্টটি Colab-এ নিন

```bash
git clone https://github.com/redwan-primeAmax/ProXMax.git
cd ProXMax
pip install -r requirements.txt
```

> মাত্র একটি requirements ফাইল — কোনো বড় মডেল ডাউনলোড নেই, ইনস্টল
> ১–২ মিনিটে শেষ। (কণ্ঠ ক্লোনিং চাইলে অতিরিক্ত `requirements-indicf5.txt`।)
>
> Colab-এ `torch`/`torchaudio` আগে থেকেই CUDA-সহ ইনস্টল থাকে — সেগুলো pin করা
> হয়নি (করলে ~১ GB ডাউনলোড ও CUDA mismatch ঝুঁকি)।

### ধাপ ৩: Gemini API key

```bash
export GEMINI_API_KEY="your-key"       # অথবা GOOGLE_API_KEY
```

Colab Secrets-এ `GEMINI_API_KEY` রাখলে নোটবুক নিজেই পড়ে নেবে।

### ধাপ ৪: চালান

```bash
python main.py
```

`Video/input`-এর সব ভিডিও একটার পর একটা ডাবিং হবে। একটি ভিডিও ব্যর্থ হলে
পুরো কাজ থামে না — error টার্মিনালে লেখা হয়, পরের ভিডিও চলতে থাকে, শেষে
সারসংক্ষেপ দেখানো হয়।

### দরকারি CLI অপশন

```bash
python main.py --model medium                  # Whisper ছোট (দ্রুত)
python main.py --voice bn-BD-PradeepNeural     # পুরুষ কণ্ঠ (বিল্ট-ইন)
python main.py --gemini-model gemini-3.5-flash-lite
python main.py --force                         # আগের আউটপুট থাকলেও আবার
```

**Colab নোটবুক:** [▶ Colab-এ খুলুন](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb) ·
[`notebook/ProXMax_Colab.ipynb`](notebook/ProXMax_Colab.ipynb) — সব ধাপ
(GPU, Drive, ইনস্টল, Gemini key, রান, GPU ফ্রি) বাংলায় ব্যাখ্যা করা আছে।

---

## 🎉 ফলাফল

`Video/output/`-এ প্রতিটি ভিডিওর জন্য:

| ফাইল | কী |
|---|---|
| `lecture01_bn.mp4` | ডাবিং করা ভিডিও (মূল ভিডিও + বাংলা অডিও) |
| `lecture01_en.srt` | মূল English সাবটাইটেল (STT থেকে) |
| `lecture01_bn.srt` | বাংলা সাবটাইটেল |

---

## 🎤 কণ্ঠ: ডিফল্টে যা হয়

`tts.engine: auto` (ডিফল্ট) মানে — **আপনাকে কিছু বেছে নিতে বা দিতে হয় না**:

| পরিস্থিতি | কী হবে |
|---|---|
| ইনপুট ফোল্ডারে কণ্ঠের নমুনা নেই | **বিল্ট-ইন `bn-BD` neural ভয়েস** (Nabanita, মহিলা) — সঙ্গে সঙ্গে কাজ শুরু |
| ইনপুট ফোল্ডারে ৫s+ বাংলা কথার audio আছে | নিজে থেকেই **কণ্ঠ ক্লোনিং (IndicF5)** চেষ্টা করবে |
| ক্লোনিং চালু করা গেল না (HF অনুমতি/VRAM/ডিপেন্ডেন্সি) | স্পষ্ট মেসেজ দিয়ে বিল্ট-ইন ভয়েসে ফিরে আসবে — ভিডিও নষ্ট হবে না |

বিল্ট-ইন ভয়েস বদলাতে চাইলে `config.yaml`-এ `tts.voice` দিন:
`bn-BD-NabanitaNeural` (মহিলা) / `bn-BD-PradeepNeural` (পুরুষ)।

---

## ⚙️ ভিতরে কীভাবে কাজ করে

পুরো যাত্রা ৭টি ধাপে — প্রতিটির সময় লগে ও রিপোর্টে দেখানো হয়।

```
ভিডিও ─► [1] অডিও extract (16 kHz mono)
       ─► [2] ভয়েস প্রোফাইল (ঐচ্ছিক — নমুনা ও HF টোকেন থাকলে)
       ─► [3] STT (silence-aware chunk + rolling prompt + seam repair)
       ─► [4] অনুবাদ (Gemini, duration-aware) + TTS (ভয়েস)
       ─► [5] Sync (non-overlap উইন্ডো, tempo ≤ 1.25)
       ─► [6] Mux (ffmpeg) ─► [7] SRT
```

### ১) শব্দের মাঝখানে কাটা আর হবে না

Whisper একবারে ৩০ সেকেন্ডের বেশি দেখতে পারে না, তাই কাটা বাধ্যতামূলক। আগে
faster-whisper নিজেই জোর করে কেটে দিত — ফলে একটি শব্দ দুই টুকরো হয়ে যেত।
এখন:

* Silero VAD (না থাকলে RMS energy) দিয়ে **কথার অঞ্চল** বের করা হয়,
* অঞ্চলগুলো ১৮–২৮ সেকেন্ডের chunk-এ জোড়া লাগানো হয় — কাটা পড়ে **দুই অঞ্চলের
  মাঝের নীরবতায়** (অন্তত ৩০০ ms),
* একটানা ৩০s কথা বললেই কেবল জোর করে কাটা হয়, আর সেই chunk-এ `forced` ফ্ল্যাগ
  থাকে → সীমানার ±২.৫s আবার ডিকোড করে শব্দ ঠিক করা হয় (seam repair)।

### ১খ) "কাজ করতে করতে ঝুলে যায়" — আর হবে না

পুরোনো সমস্যা: অডিও extract করার সময় ffmpeg চুপচাপ আটকে যেত, লগ একই
লাইন বারবার ছাপত, আর কিছুই এগোত না। কারণ ও সমাধান:

* ffmpeg-এর **stdin খোলা** থাকত — নোটবুকে (`!python main.py`) সেটি TTY/pipe
  হওয়ায় ffmpeg কী-ইনপুটের অপেক্ষায় ব্লক করে বসত। এখন সব ffmpeg কলে
  **`-nostdin` + stdin=DEVNULL**, তাই ব্লক হওয়া অসম্ভব।
* **`-hwaccel auto`** বাদ দেওয়া হয়েছে (`-vn` দিয়ে ভিডিও stream ফেলাই দিচ্ছি,
  GPU decode-এ কোনো লাভ নেই, শুধু আটকে যাওয়ার ঝুঁকি) — দরকার হলে
  `runtime.use_hwaccel` দিয়ে চালু করা যায়, আর ঝুললে নিজেই software-এ নামে।
* **লাইভ অগ্রগতি**: `অডিও extract: 45% (04:30 / 10:00) — চলছে…` — ১০ সেকেন্ড
  পরপর, কেবল শতাংশ বদলালে। মোট অডিও decode আর muxing-এও একই।
* **Stall detector**: `runtime.extract_stall_seconds` (ডিফল্ট ৯০) সেকেন্ড
  কোনো অগ্রগতি না এলে প্রসেস kill হয় → পরের পথ চেষ্টা → শেষে স্পষ্ট error।
  অর্থাৎ অনন্তকাল অপেক্ষা নয়।
* **STT-তেও অগ্রগতি**: `STT 3/12 — 01:00–01:28` প্রতিটি chunk-এ, তাই মডেল
  ডাউনলোড/ডিকোডের সময়ও বোঝা যায় কাজ এগোচ্ছে কি না।

### ২) পাংচুয়েশন ও বাক্যের ধারাবাহিকতা

প্রতিটি chunk ডিকোড করার সময় আগের টেক্সটের শেষ ~২২০ অক্ষর
`initial_prompt` হিসেবে দেওয়া হয় — Whisper বাক্যের মাঝ থেকে শুরু করে না,
তাই যুক্তি/পাংচুয়েশন ঠিক থাকে। এরপর শব্দগুলোকে বাক্যে সাজানো হয়
(সর্বোচ্চ ~১২–১৫ শব্দ, এক আইডিয়া এক segment)।

### ৩) কণ্ঠ

* ডিফল্ট: **edge-tts**-এর `bn-BD` neural ভয়েস — কোনো নমুনা/টোকেন/লাইসেন্স
  লাগে না, ইন্টারনেট থাকলেই চলে (৫টি ক্লিপ একসাথে, retry সহ)।
* ঐচ্ছিক: ইনপুট ফোল্ডারে নমুনা থাকলে **IndicF5** দিয়ে zero-shot ভয়েস
  ক্লোনিং। নমুনা থেকে VAD দিয়ে সেরা একটানা কথা-উইন্ডো নেওয়া হয়, ffmpeg
  high-pass + loudnorm করা হয়, `ref_text` নিজেই বের করা হয় এবং ফলাফল ক্যাশ
  হয় — তাই পরের রানে সঙ্গে সঙ্গে প্রস্তুত।

### ৪) অনুবাদ সময় জানে

Gemini-কে প্রতিটি লাইনের সাথে তার **সময়সীমা (সেকেন্ড)** দেওয়া হয়, এবং
নির্দেশ থাকে সময়ে আঁটতে হবে। তারপরও কোনো লাইন ১.২৫× গতিতেও না আঁটলে
সেটি আরও সংক্ষেপে লেখানো হয়। তাই `atempo` খুব কম ক্ষেত্রে লাগে — কণ্ঠ
স্বাভাবিক শোনায়।

### ৫) কোনো overlap নেই

`sync_engine` প্রতিটি segment-এর জন্য আলাদা উইন্ডো বানায়:

```
উইন্ডো শেষ = min(পরের segment শুরুর আগে (৮০ ms ফাঁক), নিজের কথা + সর্বোচ্চ ২৫০ ms ধার)
```

ক্লিপ উইন্ডোর চেয়ে লম্বা হলে শেষে ছেঁটে fade দেওয়া হয় — **দুই কণ্ঠ একসাথে
বাজা অসম্ভব**। এরপর সব ক্লিপের RMS লেভেল মিলিয়ে দেওয়া হয়, যাতে এক বাক্য
উঁচু আরেক বাক্য নিচু না শোনায়।

### ৬) গতি

* অনুবাদ (নেটওয়ার্ক-বাউন্ড) আর TTS (নেটওয়ার্ক/GPU) **একসাথে** চলে,
* Speed Governor প্রথম কয়েকটি ক্লিপের বাস্তব গতি মেপে ভবিষ্যৎ অনুমান করে;
  (IndicF5 ব্যবহার হলে) বাজেট ছাড়ানোর আশঙ্কায় diffusion ধাপ `nfe` নামায়,
* কাজ শেষে মডেল আনলোড করে **VRAM খালি** করা হয়।

---

## 🛠️ ব্যবহৃত টেকনোলজি ও মডেল

| কাজ | টুল / মডেল |
|---|---|
| Speech-to-Text | `faster-whisper` 1.2.1 + `large-v3` (silence-aware chunking) |
| VAD | Silero VAD (faster-whisper-এর বান্ডল), fallback RMS energy |
| অনুবাদ | **Google Gemini API** — ডিফল্ট `gemini-3.5-flash-lite` |
| কণ্ঠ (ডিফল্ট) | `edge-tts` — `bn-BD-NabanitaNeural` / `bn-BD-PradeepNeural` |
| কণ্ঠ (ঐচ্ছিক) | **IndicF5** (`ai4bharat/IndicF5`) — zero-shot ভয়েস ক্লোনিং, ২৪ kHz |
| অডিও ইঞ্জিন | numpy + ffmpeg (`atempo`/`rubberband`, loudnorm) |
| সারিবদ্ধকরণ | নিজস্ব non-overlap window solver |

> ডিফল্ট অনুবাদ মডেল বদলাতে চাইলে `--gemini-model` অথবা `config.yaml`-এর
> `gemini.model` ব্যবহার করুন। `translator.backend: indictrans2` দিলে
> IndicTrans2-ও চালানো যায় (ঐচ্ছিক, `requirements-indictrans2.txt`)।

---

## 🔧 সেটিংস (config.yaml)

সব সেটিং এক জায়গায় — `config.yaml`। দরকার হলে বদলান, কোড ছোঁয়ার দরকার নেই।

| সেকশন | জরুরি কী | ডিফল্ট | মানে |
|---|---|---|---|
| `paths` | `input_dir`, `output_dir` | `/content/drive/.../Video/input` | Drive-এর পাথ |
| `stt` | `model` | `large-v3` | `medium` = দ্রুত, কম নির্ভুল |
| | `chunk_min_seconds` / `chunk_max_seconds` | 18 / 28 | শব্দের মাঝে কাটা রোধ |
| | `vad_min_silence_ms` | 300 | কাটার জন্য ন্যূনতম নীরবতা |
| | `carry_context_chars` | 220 | rolling prompt-এর দৈর্ঘ্য |
| `translation` | `backend` | `gemini` | `gemini` \| `indictrans2` |
| | `char_rate` | 14.0 | বাংলা কথার গতি (অক্ষর/সে) |
| `gemini` | `model` | `gemini-3.5-flash-lite` | অনুবাদ মডেল |
| | `max_retries` | 7 | ৪২৯/quota হলে কতবার চেষ্টা |
| `tts` | `engine` | `auto` | `auto` = নমুনা থাকলে ক্লোনিং, নাহলে বিল্ট-ইন; `edge` \| `indicf5` |
| | `voice` | `bn-BD-NabanitaNeural` | বিল্ট-ইন কণ্ঠ (পুরুষ: `bn-BD-PradeepNeural`) |
| | `nfe_steps` | 16 | IndicF5 diffusion ধাপ (মান ↑, গতি ↓) |
| | `time_budget_factor` | 0.35 | TTS সময় বাজেট (মোট কথার গুণক) |
| `voice` | `reference` | `""` | খালি = ইনপুট ফোল্ডার থেকে খুঁজবে (ঐচ্ছিক) |
| | `target_seconds` | 16 | ক্লোনিংয়ের রেফারেন্স দৈর্ঘ্য |
| `sync` | `min_tempo` / `max_tempo` | 0.95 / 1.25 | এর বাইরে গেলে কণ্ঠ বিকৃত |
| | `slot_gap_ms` / `borrow_silence_ms` | 80 / 250 | overlap রোধের নিয়ম |
| `runtime` | `release_gpu_when_done` | true | শেষে VRAM ছেড়ে দেওয়া |
| | `extract_stall_seconds` | 90 | ffmpeg এতক্ষণ অগ্রগতি না দেখালে থামিয়ে দেওয়া |
| | `enable_overlap` | true | অনুবাদ+TTS একসাথে (অডিও overlap নয়) |

---

## 🎤 ঐচ্ছিক: নিজের কণ্ঠে ডাবিং (voice cloning)

ডিফল্ট পথে **এটি দরকার নেই**। শুধু নিজের গলায় ডাবিং চাইলে:

1. **অনুমতি** — https://huggingface.co/ai4bharat/IndicF5-এ লগইন করে
   “Agree and access repository” চাপুন (একবারই)।
2. **HF token** — https://huggingface.co/settings/tokens (`read` যথেষ্ট)।
3. **অতিরিক্ত প্যাকেজ** — `pip install -r requirements-indicf5.txt`
4. **নমুনা** — `Video/input/`-এ ১০–২০ সেকেন্ডের পরিষ্কার বাংলা কথার
   `.wav/.mp3/.m4a` ফাইল রাখুন (৫s–১ মিনিট পর্যন্ত চলবে)।
5. **লগইন** (নোটবুকের ঐচ্ছিক সেল):

```python
from huggingface_hub import login
login(token="hf_...")
```

কিছু একটা না হলে অ্যাপ নিজেই বিল্ট-ইন ভয়েসে ফিরে যাবে — ভিডিও নষ্ট হবে না।
জোর করে শুধু ক্লোনিং চাইলে: `python main.py --tts-engine indicf5`; IndicF5 না
চললে ডিফল্টে (`tts.fallback_to_edge: true`) বিল্ট-ইন ভয়েসে ফিরে আসবে, আর
`false` করলে স্পষ্ট error দিয়ে থেমে যাবে।

---

## 📁 প্রজেক্ট স্ট্রাকচার

```
ProXMax/
├── main.py                       # CLI এন্ট্রি পয়েন্ট
├── config.yaml                   # সব সেটিংস
├── requirements.txt              # মূল dependencies (pin করা)
├── requirements-indicf5.txt      # ঐচ্ছিক: কণ্ঠ ক্লোনিং ইঞ্জিন
├── requirements-indictrans2.txt  # ঐচ্ছিক: বিকল্প অনুবাদ backend
├── notebook/ProXMax_Colab.ipynb  # Colab নোটবুক (বাংলা ব্যাখ্যা)
├── proxmax/
│   ├── config.py                 # কনফিগ লোড/ভ্যালিডেশন
│   ├── pipeline.py               # ৭ ধাপের অর্কেস্ট্রেশন + ব্যাচ কন্ট্রোল
│   ├── audio_fx.py               # numpy-only decode/encode/DSP
│   ├── chunking.py               # নীরবতা-সচেতন chunk plan
│   ├── segmenter.py              # শব্দ → বাক্য (segment)
│   ├── voice_profile.py          # ঐচ্ছিক কণ্ঠের নমুনা → রেফারেন্স (cache)
│   ├── text_utils.py             # বাংলা নরমালাইজ/সংখ্যা/সময় অনুমান
│   ├── stt.py                    # faster-whisper + rolling prompt + seam repair
│   ├── translator.py             # Gemini (duration-aware) + IndicTrans2
│   ├── sync_engine.py            # non-overlap time-stretch ও ট্র্যাক
│   ├── tts/                      # বিল্ট-ইন edge ভয়েস + IndicF5 + governor
│   ├── audio_extraction.py       # ffmpeg অডিও বের করা
│   ├── muxer.py                  # চূড়ান্ত MP4
│   ├── discovery.py / utils.py / models.py
└── tests/test_smoke.py           # ৮৫ টেস্ট (GPU/নেটওয়ার্ক ছাড়াই)
```

---

## ⏱️ প্রত্যাশিত গতি (১০ মিনিটের ভিডিও, T4)

| ধাপ | সময় |
|---|---|
| অডিও extract | ~৫s |
| STT (`large-v3`) + seam repair | ~৫০–৭০s |
| অনুবাদ + TTS (একসাথে) | ~৬০–১২০s |
| Sync + mux + SRT | ~২৫–৩৫s |
| **মোট** | **~২.৫–৩.৫ মিনিট** (প্রথম রানে মডেল ডাউনলোড আলাদা) |

বিল্ট-ইন ভয়েস ইন্টারনেট-বাউন্ড, তাই GPU-তে চাপ কম; ক্লোনিং (IndicF5)
ব্যবহার করলে Speed Governor সময়-বাজেটে রাখে।

---

## ⚠️ সীমাবদ্ধতা

* শুধু **English → Bangla**; অন্য ভাষা-জোড়া নেই।
* ডিফল্ট কণ্ঠ একটি নির্দিষ্ট neural ভয়েস (আপনার নিজের গলা নয়) — নিজের
  কণ্ঠ চাইলে ঐচ্ছিক ক্লোনিং অংশ দেখুন।
* গান/আবহসংগীত/ভিড়ের শব্দে STT নির্ভুলতা কমতে পারে।
* খুব দ্রুত কথা বলা ভিডিওতে (প্রতি মিনিটে ১৮০+ শব্দ) tempo সীমার কারণে কিছু
  লাইন ছোট করতে হয় — অর্থ ঠিক থাকে, শব্দচয়ন সংক্ষিপ্ত হয়।
* ইন্টারনেট দরকার (Gemini API + বিল্ট-ইন ভয়েস; ক্লোনিংয়ে মডেল ডাউনলোড)।

---

## 🩺 সাধারণ সমস্যা ও সমাধান

| সমস্যা | সমাধান |
|---|---|
| Gemini key দিইনি / ভুলে গেছি | চালু হওয়ার আগেই error দেখাবে: `Gemini API key পাওয়া যায়নি` → নোটবুকের ধাপ ৬-এ key পেস্ট করে আবার চালান |
| HF token দিইনি | কিছুই করতে হবে না — অ্যাপ নিজেই বিল্ট-ইন ভয়েসে চলে (এটা error নয়) |
| অডিও extract-এ ঝুলে যায় | আর অনন্তকাল ঝুলবে না; তবু হলে ভিডিওটি আবার আপলোড করুন (অর্ধেক আপলোড হওয়া ফাইল পড়া যায় না) |
| `Gemini API error (429 / quota)` | কিছুক্ষণ অপেক্ষা করুন — কোড নিজেই ৭ বার retry করে; plan-এর limit দেখুন |
| `edge-tts` ব্যর্থ / কণ্ঠ আসছে না | ইন্টারনেট সংযোগ দেখুন, `pip install -U edge-tts`; `tts.voice` বৈধ `bn-BD-*` কি না দেখুন |
| `CUDA out of memory` | Runtime → Restart; ক্লোনিং বন্ধ রাখলে (`auto` + নমুনা ছাড়া) GPU চাপ অনেক কম |
| `'ffmpeg' পাওয়া যায়নি` | `apt-get install -y ffmpeg` |
| ডাবিং ধীর | `--model medium` |
| নিজের কণ্ঠ ক্লোন হচ্ছে না | `Video/input`-এ নমুনা আছে কি না, HF অনুমতি/token ঠিক কি না দেখুন; অথবা `--tts-engine indicf5` দিয়ে সরাসরি চেষ্টা করুন |
| অনুবাদ ব্লক (`SAFETY`) | ওই অংশ বাদ যায়, বাকি ভিডিও চলতে থাকে; বাক্য পরিষ্কার করে আবার চালান |

---

## 🧪 ডেভেলপারদের জন্য

```bash
python -m unittest discover -s tests -v      # ৮৫ টেস্ট, GPU/নেটওয়ার্ক ছাড়াই
python -m compileall proxmax main.py
```

টেস্টগুলো মডেল/API mock করে, কিন্তু chunking, seam repair, rolling prompt,
duration-aware অনুবাদ, non-overlap sync, কণ্ঠ-ইঞ্জিন নির্বাচন এবং পুরো
pipeline (আসল ffmpeg দিয়ে একটি ছোট্ট ভিডিও) যাচাই করে।

---

## 📜 লাইসেন্স ও কৃতজ্ঞতা

* কোড: এই রিপোজিটরি।
* ব্যবহৃত ওপেন-সোর্স: faster-whisper, CTranslate2, edge-tts, google-genai,
  ffmpeg — নিজ নিজ লাইসেন্সের অধীন।
* Whisper মডেলের ওজন OpenAI-এর।
* ঐচ্ছিক ক্লোনিং: IndicF5 — AI4Bharat (MIT) — https://github.com/AI4Bharat/IndicF5
