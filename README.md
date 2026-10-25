# 🎙️ ProXMax — আপনার নিজের কণ্ঠে English → Bangla ভিডিও ডাবিং

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb)
[![T4 GPU](https://img.shields.io/badge/GPU-T4%20(16GB)-76B900?logo=nvidia&logoColor=white)](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

ProXMax একটি English ভিডিওকে **আপনার নিজের বাংলা কণ্ঠে** ডাবিং করে। ভিডিও থেকে
কথা শোনে (faster-whisper), Gemini দিয়ে বাংলায় অনুবাদ করে, **IndicF5** দিয়ে
আপনার কণ্ঠ ক্লোন করে কথা বানায় এবং এমনভাবে বসায় যাতে **কোনো দুই কণ্ঠ কখনো
একসাথে না বলে**।

পুরো কাজটি Google Colab-এর **T4 GPU**-তে চলে; ইনপুট ও আউটপুট দুটোই Google
Drive-এর ফোল্ডার থেকে।

---

## ☁️ এক ক্লিকে Colab-এ খুলুন

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb)

উপরের বাটনে ক্লিক করলেই সম্পূর্ণ নোটবুকটি Colab-এ খুলে যাবে — সবসময়
GitHub-এর `main` ব্র্যাঞ্চের **সর্বশেষ কোড** ব্যবহার হবে (আলাদা করে কিছু
ডাউনলোড বা আপলোড করতে হবে না)।

বাটনে ক্লিক করার পর মাত্র ৩টি কাজ:

1. `Runtime → Change runtime type → **T4 GPU**` বেছে নিন।
2. নোটবুকের ধাপ ৫ ও ৬-এ **Hugging Face token** ও **Gemini API key** দিন
   (Colab Secrets-এ রাখলে নিজে থেকেই পড়ে নেবে)।
3. ধাপ ৮-এ `python main.py` সেলটি রান করুন — বাকি সব নোটবুক নিজেই করে।

> নোটবুকের লিংক: <https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb>

---

## 📖 এক নজরে

| আগে যে সমস্যা ছিল | এখন কী করা হয় |
|---|---|
| Whisper ৩০ সেকেন্ডের সীমায় **শব্দের মাঝখানে** কাটত (`XYZ` → `XY` + `Z`) | শুধু **৩০০ ms+ নীরবতার মাঝখানে** কাটা হয় (১৮–২৮s chunk), সীমানায় সন্দেহ হলে ছোট উইন্ডো আবার ডিকোড (seam repair) |
| পাংচুয়েশন হারিয়ে যেত, বাক্য ভাঙত | আগের লেখার শেষ ~২২০ অক্ষর **rolling prompt** হিসেবে যায় → বাক্যের ধারাবাহিকতা ও punctuation ঠিক থাকে |
| রোবটিক/অপ্রাকৃত কণ্ঠ | **IndicF5 ভয়েস ক্লোনিং** — আপনার ১০–২০s নমুনা থেকে কণ্ঠ; tempo কখনো ১.২৫-এর বেশি নয় |
| দুই কণ্ঠ একসাথে বাজত | **Non-overlap sync solver** — প্রতিটি ক্লিপের জন্য আলাদা সময়-উইন্ডো; কঠোরভাবে কোনো overlap নেই |
| অনুবাদ সময়ের চেয়ে লম্বা/ছোট হতো | Gemini-কে প্রতিটি লাইনের **সময়ের বাজেট** জানানো হয়; না আঁটলে লাইনটি ছোট করে লেখানো হয় |
| ১০ মিনিটের ভিডিওতে অনেক সময় লাগত | অনুবাদ (নেটওয়ার্ক) ও TTS (GPU) **একসাথে** চলে; গতি বাজেট রাখতে একটি Speed Governor আছে → **২–৩ মিনিট** |

---

## ✅ যা যা লাগবে

1. **Google Colab** (Runtime → Change runtime type → **T4 GPU**)।
2. **ভিডিও** — `/content/drive/MyDrive/Video/input/` ফোল্ডারে (`.mp4/.mkv/.mov/.avi`)।
3. **আপনার কণ্ঠের নমুনা** — **একই `Video/input` ফোল্ডারে** একটি audio ফাইল
   (`.wav/.mp3/.m4a/.flac`, ১০ সেকেন্ড থেকে ১ মিনিট পর্যন্ত)। শান্ত ঘরে বলা
   পরিষ্কার বাংলা কথা থাকলেই হবে; কোড নিজেই সবচেয়ে ভালো অংশটি বেছে নেয়,
   বড় বিরতিগুলো ছেঁটে দেয়।
4. **Gemini API key** — https://aistudio.google.com/app/apikey
5. **Hugging Face token** — IndicF5 মডেলটি gated, তাই আগে অনুমতি নিতে হবে
   (নিচের ধাপ ৩)।

---

## 🚀 কীভাবে ব্যবহার করবেন (Google Colab)

### ধাপ ১: Drive প্রস্তুত করুন

```
MyDrive/
└── Video/
    ├── input/
    │   ├── lecture01.mp4          ← ডাবিং করতে হবে
    │   └── my_voice.m4a           ← আপনার কণ্ঠের নমুনা (১০–২০s)
    └── output/                    ← ফলাফল এখানে আসবে
```

### ধাপ ২: প্রজেক্টটি Colab-এ নিন

```bash
git clone https://github.com/redwan-primeAmax/ProXMax.git
cd ProXMax
pip install -r requirements.txt -r requirements-indicf5.txt
```

> `requirements.txt` = মূল পাইপলাইন; `requirements-indicf5.txt` = ভয়েস ক্লোনিং
> ইঞ্জিন। ইনস্টল প্রথমবার ২–৪ মিনিট নেয়।
>
> Colab-এ `torch`/`torchaudio` আগে থেকেই CUDA-সহ ইনস্টল থাকে — সেগুলো pin করা
> হয়নি (করলে ~১ GB ডাউনলোড ও CUDA mismatch ঝুঁকি)। লোকাল মেশিনে
> `pip install torch torchaudio` নিজের CUDA অনুযায়ী করুন।

### ধাপ ৩: IndicF5-এর অনুমতি + Hugging Face লগইন

1. https://huggingface.co/ai4bharat/IndicF5 — লগইন করে **“Agree and access
   repository”** চাপুন (একবারই)।
2. Token নিন: https://huggingface.co/settings/tokens (`read` permission যথেষ্ট)।
3. Colab-এ লগইন করুন:

```python
from huggingface_hub import login
login(token="hf_...")     # অথবা Colab Secrets-এ HF_TOKEN রেখে দিন
```

> অনুমতি না নিলে মডেল ডাউনলোড হবে না এবং অ্যাপটি পরিষ্কার error দেখিয়ে
> edge-tts fallback-এ যাবে (নিচে দেখুন)।

### ধাপ ৪: Gemini API key

```bash
export GEMINI_API_KEY="your-key"       # অথবা GOOGLE_API_KEY
```

Colab Secrets-এ `GEMINI_API_KEY` রাখলে নোটবুক নিজেই পড়ে নেবে।

### ধাপ ৫: চালান

```bash
python main.py
```

`Video/input`-এর সব ভিডিও একটার পর একটা ডাবিং হবে। একটি ভিডিও ব্যর্থ হলে
পুরো কাজ থামে না — error টার্মিনালে লেখা হয়, পরের ভিডিও চলতে থাকে, শেষে
সারসংক্ষেপ দেখানো হয়।

### দরকারি CLI অপশন

```bash
python main.py --model medium                  # Whisper ছোট (দ্রুত)
python main.py --reference my_voice.wav        # নমুনা নিজে বলে দেওয়া
python main.py --tts-engine edge               # ক্লোনিং ছাড়া edge-tts
python main.py --gemini-model gemini-3.5-flash-light
python main.py --force                         # আগের আউটপুট থাকলেও আবার
```

**Colab নোটবুক:** [▶ Colab-এ খুলুন](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb) ·
[`notebook/ProXMax_Colab.ipynb`](notebook/ProXMax_Colab.ipynb) — সব ধাপ
(GPU, Drive, ইনস্টল, HF লগইন, Gemini key, রান, GPU ফ্রি) বাংলায় ব্যাখ্যা করা আছে।

---

## 🎉 ফলাফল

`Video/output/`-এ প্রতিটি ভিডিওর জন্য:

| ফাইল | কী |
|---|---|
| `lecture01_bn.mp4` | ডাবিং করা ভিডিও (মূল ভিডিও + বাংলা অডিও) |
| `lecture01_en.srt` | মূল English সাবটাইটেল (STT থেকে) |
| `lecture01_bn.srt` | বাংলা সাবটাইটেল |

---

## ⚙️ ভিতরে কীভাবে কাজ করে

পুরো যাত্রা ৭টি ধাপে — প্রতিটির সময় লগে ও রিপোর্টে দেখানো হয়।

```
ভিডিও ─► [1] অডিও extract (16 kHz mono)
       ─► [2] ভয়েস প্রোফাইল (আপনার নমুনা → পরিষ্কার ২৪ kHz রেফারেন্স)
       ─► [3] STT (silence-aware chunk + rolling prompt + seam repair)
       ─► [4] অনুবাদ (Gemini, duration-aware) + TTS (IndicF5 ক্লোনিং)
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
* একটানা ৩০s কথা বললেই কেবল জোর করে কাটা হয়, আর সেই chunk-এ `forced` ফ্লাগ
  থাকে → সীমানার ±২.৫s আবার ডিকোড করে শব্দ ঠিক করা হয় (seam repair)।

### ২) পাংচুয়েশন ও বাক্যের ধারাবাহিকতা

প্রতিটি chunk ডিকোড করার সময় আগের টেক্সটের শেষ ~২২০ অক্ষর
`initial_prompt` হিসেবে দেওয়া হয় — Whisper বাক্যের মাঝ থেকে শুরু করে না,
তাই যুক্তি/পাংচুয়েশন ঠিক থাকে। এরপর শব্দগুলোকে বাক্যে সাজানো হয়
(সর্বোচ্চ ~১২–১৫ শব্দ, এক আইডিয়া এক segment)।

### ৩) আপনার কণ্ঠ (IndicF5)

* ইনপুট ফোল্ডার থেকেই নমুনা খুঁজে নেওয়া হয় (`reference/` ফোল্ডারও চলবে),
* ffmpeg high-pass + হালকা loudnorm দিয়ে গুঞ্জন/লেভেল ঠিক করা হয়,
* VAD দিয়ে সবচেয়ে ভালো **একটানা কথা**-র উইন্ডো বেছে নেওয়া হয়, ভেতরের বড়
  বিরতি ৮০০ ms-এ সীমিত করা হয় (স্বাভাবিক শ্বাস/থামা থাকে),
* স্বয়ংক্রিয়ভাবে নমুনার টেক্সট বের করা হয় (`ref_text` — ক্লোনিংয়ের মানের
  জন্য সবচেয়ে জরুরি),
* ফলাফল ক্যাশ হয় — পরের রানে সঙ্গে সঙ্গে প্রস্তুত।

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

* অনুবাদ (নেটওয়ার্ক-বাউন্ড) আর TTS (GPU-বাউন্ড) **একসাথে** চলে,
* Speed Governor প্রথম কয়েকটি ক্লিপের বাস্তব গতি মেপে ভবিষ্যৎ অনুমান করে;
  বাজেট ছাড়ানোর আশঙ্কা হলে diffusion ধাপ (`nfe`) ১৬ → ১৩ → ১১ → ৯ → ৮
  নামায় (গুণমান কিছুটা কমে, গতি বাড়ে),
* কাজ শেষে (এবং TTS-এর আগে Whisper) মডেল আনলোড করে **VRAM খালি** করা হয়।

---

## 🛠️ ব্যবহৃত টেকনোলজি ও মডেল

| কাজ | টুল / মডেল |
|---|---|
| Speech-to-Text | `faster-whisper` 1.2.1 + `large-v3` (silence-aware chunking) |
| VAD | Silero VAD (faster-whisper-এর বান্ডল), fallback RMS energy |
| অনুবাদ | **Google Gemini API** — ডিফল্ট `gemini-3.5-flash-light` |
| কণ্ঠ | **IndicF5** (`ai4bharat/IndicF5`) — zero-shot ভয়েস ক্লোনিং, ২৪ kHz |
| Fallback কণ্ঠ | `edge-tts` (bn-BD-Nabanita / Pradeep) |
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
| `gemini` | `model` | `gemini-3.5-flash-light` | অনুবাদ মডেল |
| | `max_retries` | 7 | ৪২৯/quota হলে কতবার চেষ্টা |
| `voice` | `reference` | `""` | খালি = ইনপুট ফোল্ডার থেকে খুঁজবে |
| | `target_seconds` | 16 | রেফারেন্সের কাঙ্ক্ষিত কথা-দৈর্ঘ্য |
| `tts` | `engine` | `indicf5` | `indicf5` \| `edge` |
| | `nfe_steps` | 16 | diffusion ধাপ (মান ↑, গতি ↓) |
| | `time_budget_factor` | 0.35 | TTS সময় বাজেট (মোট কথার গুণক) |
| | `fallback_to_edge` | true | IndicF5 না চললে ভিডিও নষ্ট না করা |
| `sync` | `min_tempo` / `max_tempo` | 0.95 / 1.25 | এর বাইরে গেলে কণ্ঠ বিকৃত |
| | `slot_gap_ms` / `borrow_silence_ms` | 80 / 250 | overlap রোধের নিয়ম |
| `runtime` | `release_gpu_when_done` | true | শেষে VRAM ছেড়ে দেওয়া |
| | `enable_overlap` | true | অনুবাদ+TTS একসাথে (অডিও overlap নয়) |

---

## 📁 প্রজেক্ট স্ট্রাকচার

```
ProXMax/
├── main.py                       # CLI এন্ট্রি পয়েন্ট
├── config.yaml                   # সব সেটিংস
├── requirements.txt              # মূল dependencies (pin করা)
├── requirements-indicf5.txt      # ভয়েস ক্লোনিং ইঞ্জিনের dependencies
├── requirements-indictrans2.txt  # ঐচ্ছিক বিকল্প backend
├── notebook/ProXMax_Colab.ipynb  # Colab নোটবুক (বাংলা ব্যাখ্যা)
├── proxmax/
│   ├── config.py                 # কনফিগ লোড/ভ্যালিডেশন
│   ├── pipeline.py               # ৭ ধাপের অর্কেস্ট্রেশন + ব্যাচ কন্ট্রোল
│   ├── audio_fx.py               # numpy-only decode/encode/DSP
│   ├── chunking.py               # নীরবতা-সচেতন chunk plan
│   ├── segmenter.py              # শব্দ → বাক্য (segment)
│   ├── voice_profile.py          # কণ্ঠের নমুনা → রেফারেন্স (cache)
│   ├── text_utils.py             # বাংলা নরমালাইজ/সংখ্যা/সময় অনুমান
│   ├── stt.py                    # faster-whisper + rolling prompt + seam repair
│   ├── translator.py             # Gemini (duration-aware) + IndicTrans2
│   ├── sync_engine.py            # non-overlap time-stretch ও ট্র্যাক
│   ├── tts/                      # IndicF5 + edge-tts + speed governor
│   ├── audio_extraction.py       # ffmpeg অডিও বের করা
│   ├── muxer.py                  # চূড়ান্ত MP4
│   ├── discovery.py / utils.py / models.py
└── tests/test_smoke.py           # ৭০+ টেস্ট (GPU/নেটওয়ার্ক ছাড়াই)
```

---

## ⏱️ প্রত্যাশিত গতি (১০ মিনিটের ভিডিও, T4)

| ধাপ | সময় |
|---|---|
| অডিও extract | ~৫s |
| STT (`large-v3`) + seam repair | ~৫০–৭০s |
| অনুবাদ + TTS (একসাথে) | ~১২০–১৪০s |
| Sync + mux + SRT | ~২৫–৩৫s |
| **মোট** | **~৩–৪ মিনিট** (প্রথম রানে মডেল ডাউনলোড আলাদা) |

Speed Governor বাজেট ছাড়ানোর আশঙ্কা করলে `nfe` কমিয়ে মোট সময়
**২.৫–৩ মিনিটে** নামিয়ে আনে।

---

## ⚠️ সীমাবদ্ধতা

* শুধু **English → Bangla**; অন্য ভাষা-জোড়া নেই।
* কণ্ঠ ক্লোনিংয়ের মান নির্ভর করে নমুনার মানের উপর — শান্ত ঘর, স্পষ্ট কথা,
  অন্তত ১০ সেকেন্ড।
* IndicF5 মডেলটি gated — অনুমতি ছাড়া ডাউনলোড হবে না।
* গান/আবহসংগীত/ভিড়ের শব্দে STT নির্ভুলতা কমতে পারে।
* খুব দ্রুত কথা বলা ভিডিওতে (প্রতি মিনিটে ১৮০+ শব্দ) tempo সীমার কারণে কিছু
  লাইন ছোট করতে হয় — অর্থ ঠিক থাকে, শব্দচয়ন সংক্ষিপ্ত হয়।
* ইন্টারনেট দরকার (Gemini API + প্রথম রানে মডেল ডাউনলোড)।

---

## 🩺 সাধারণ সমস্যা ও সমাধান

| সমস্যা | সমাধান |
|---|---|
| `IndicF5 মডেলটি gated` | https://huggingface.co/ai4bharat/IndicF5-এ অনুমতি নিন, `HF_TOKEN` দিয়ে লগইন করুন |
| `ইনপুট ফোল্ডারে কোনো ভয়েস নমুনা পাওয়া যায়নি` | `Video/input`-এ আপনার ১০–২০s বাংলা কথার audio ফাইল রাখুন |
| `CUDA out of memory` | Runtime → Restart করে আবার চালান; `stt.unload_before_tts: true` রাখুন |
| `Gemini API error (429 / quota)` | কিছুক্ষণ অপেক্ষা করুন — কোড নিজেই ৭ বার retry করে; plan-এর limit দেখুন |
| `'ffmpeg' পাওয়া যায়নি` | `apt-get install -y ffmpeg` |
| ডাবিং ধীর | `--model medium` অথবা `tts.time_budget_factor` কমান (`0.35` → `0.25`) |
| কণ্ঠ রোবটিক শোনায় | `sync.max_tempo` ১.২৫-এর বেশি করবেন না; নমুনা পরিষ্কার করুন |
| অনুবাদ ব্লক (`SAFETY`) | ওই অংশ বাদ যায়, বাকি ভিডিও চলতে থাকে; বাক্য পরিষ্কার করে আবার চালান |

---

## 🧪 ডেভেলপারদের জন্য

```bash
python -m unittest discover -s tests -v      # ৭০+ টেস্ট, GPU/নেটওয়ার্ক ছাড়াই
python -m compileall proxmax main.py
```

টেস্টগুলো মডেল/API mock করে, কিন্তু chunking, seam repair, rolling prompt,
duration-aware অনুবাদ, non-overlap sync এবং পুরো pipeline (আসল ffmpeg দিয়ে
একটি ছোট্ট ভিডিও) যাচাই করে।

---

## 📜 লাইসেন্স ও কৃতজ্ঞতা

* কোড: এই রিপোজিটরি।
* IndicF5: AI4Bharat (MIT) — https://github.com/AI4Bharat/IndicF5
* faster-whisper, CTranslate2, edge-tts, google-genai — নিজ নিজ লাইসেন্সের অধীন।
* Whisper মডেলের ওজন OpenAI-এর।
