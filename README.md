# 🎬 ProXMax — স্বয়ংক্রিয় English → Bangla ভিডিও ডাবিং (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb)

**ProXMax** একটি সম্পূর্ণ স্বয়ংক্রিয় পাইথন পাইপলাইন, যা Google Drive-এর ইনপুট ফোল্ডারে রাখা ইংরেজি ভিডিওগুলোকে বাংলা কণ্ঠে ডাব করে আউটপুট ফোল্ডারে সেভ করে। পুরো সিস্টেমটি Google Colab-এর ফ্রি **NVIDIA T4 GPU**-তে চলার জন্য অপ্টিমাইজড।

---

## ✨ মূল বৈশিষ্ট্য

| ধাপ | প্রযুক্তি | কেন এটি বেছে নেওয়া হয়েছে |
|---|---|---|
| **১. অডিও এক্সট্রাক্ট** | `ffmpeg` (progress + stall watchdog) | সোর্স ভিডিও কেবল **একবারই** পড়ে ১৬ kHz মনো WAV তৈরি করে; ড্রাইভ ফ্রিজ হলে পরিষ্কার বার্তা দেয় |
| **২. Speech-to-Text (ইংরেজি)** | `faster-whisper` (`large-v3`, প্রি-লোডেড ক্যাশ, নীরবতা-সচেতন ১৮–২৮s চাঙ্কিং + সীমান্ত মেরামত + `hotwords.txt`) | সেটআপ ধাপেই মডেল ক্যাশ হয়ে যায়; শব্দের মাঝখানে কখনো কাটে না এবং `hotwords.txt` থেকে FC Mobile টার্ম, ইভেন্ট ও ইতিহাসের সব ফুটবল খেলোয়াড়ের নাম নিখুঁতভাবে শনাক্ত করে |
| **৩. অনুবাদ (EN → BN)** | **Google Gemini API** (`google-genai`, ডিফল্ট `gemini-3.5-flash-lite`, স্লট-বাজেট ও হটওয়ার্ড সচেতন) | প্রতিটি লাইনের বরাদ্দ সময় হিসাব করে বাংলা অনুবাদ দেয়; লম্বা হলে ছোট করে, ছোট হলে ভরাট করে |
| **৪. বাংলা ভয়েস (TTS)** | **বিল্ট-ইন বাংলা neural ভয়েস** (`edge-tts` — পুরুষ কণ্ঠ `bn-BD-PradeepNeural` ডিফল্ট, নারী `bn-BD-NabanitaNeural`) | কোনো নমুনা রেকর্ডিং বা অতিরিক্ত টোকেন ছাড়াই সাথে সাথে স্পষ্ট বাংলা কণ্ঠ তৈরি হয় |
| **৫. অডিও-ভিডিও সিঙ্ক** | `ffmpeg` (`rubberband` / `atempo`) + একক ট্র্যাক অ্যাসেম্বলার | কণ্ঠের পিচ অক্ষুণ্ণ রেখে মূল টাইমস্ট্যাম্পে বসায়; **কোনো দুই ক্লিপ কখনো একে অপরের ওপর পড়ে না** |
| **৬. ভিডিও মাক্সিং** | `ffmpeg -c:v copy -c:a aac` | ভিডিও রি-এনকোড না করেই নতুন বাংলা অডিও যুক্ত করে — মান ১০০% অক্ষুণ্ণ ও অত্যন্ত দ্রুত |

---

## 📁 প্রজেক্ট স্ট্রাকচার

```text
ProXMax/
├── main.py                        # CLI এন্ট্রি পয়েন্ট
├── config.yaml                    # সব সেটিংস এক জায়গায় (কোড না ছুঁয়েই পরিবর্তনযোগ্য)
├── hotwords.txt                   # FC Mobile টার্ম, ইভেন্ট ও ইতিহাসের সব ফুটবল খেলোয়াড়ের হটওয়ার্ড তালিকা
├── requirements.txt               # দ্রুততম ইনস্টলের জন্য অপ্টিমাইজড ডিপেন্ডেন্সি
├── requirements-indictrans2.txt   # ঐচ্ছিক: লোকাল IndicTrans2 অনুবাদ ব্যাকএন্ডের জন্য
├── notebook/
│   └── ProXMax_Colab.ipynb        # এক-ক্লিক Google Colab নোটবুক (৫ ধাপ)
├── tests/
│   └── test_smoke.py              # স্বয়ংক্রিয় ইউনিট ও স্মোক টেস্ট স্যুট
└── proxmax/
    ├── config.py                  # AppConfig ডেটাক্লাস ও YAML লোডার
    ├── hotwords.py                # hotwords.txt লোডার, Whisper প্রম্পট নির্বাচন ও ট্রান্সক্রিপ্ট মেরামত
    ├── models.py                  # Segment, Word, VideoResult, StageTimings ডেটা মডেল
    ├── utils.py                   # লগিং, নিরাপদ সাবপ্রসেস, GPU মেমোরি ক্লিনআপ
    ├── audio_fx.py                # numpy/wave ভিত্তিক অডিও DSP (নীরবতা ছাঁটা, ফরম্যাট রূপান্তর)
    ├── media.py                   # ffmpeg/ffprobe র‍্যাপার ও SRT সাবটাইটেল লেখক
    ├── stt.py                     # faster-whisper ট্রান্সক্রিপশন ও মডেল ক্যাশ ম্যানেজার
    ├── speech_rate.py             # কণ্ঠের প্রকৃত বাংলা গতি (অক্ষর/সেকেন্ড) পরিমাপ
    ├── speech_units.py            # খণ্ড বাক্য জোড়া দিয়ে পূর্ণ বাক্য-ইউনিট তৈরি ও সাবটাইটেল আনপ্যাক
    ├── text_utils.py              # সংখ্যা/সংক্ষেপ/প্রতীক বাংলা উচ্চারণযোগ্য কথায় রূপান্তর
    ├── translator.py              # Google Gemini API অনুবাদ ও লাইভ API যাচাই
    ├── tts/
    │   ├── base.py                # TTS বেস ইন্টারফেস
    │   └── edge_engine.py         # বিল্ট-ইন বাংলা neural ভয়েস (edge-tts)
    ├── sync.py                    # টাইম-স্ট্রেচ ও ওভারল্যাপ-মুক্ত ট্র্যাক অ্যাসেম্বলার
    └── pipeline.py                # সমান্তরাল অনুবাদ+TTS অর্কেস্ট্রেটর ও ব্যাচ রানার
```

---

## 🚀 Google Colab-এ ব্যবহারবিধি (৫টি সহজ ধাপ)

1. উপরের **Open in Colab** ব্যাজে ক্লিক করে নোটবুকটি খুলুন।
2. Google Drive-এ দুটি ফোল্ডার রাখুন:
   - ইনপুট ভিডিও: `/content/drive/MyDrive/Video/input`
   - আউটপুট ভিডিও: `/content/drive/MyDrive/Video/output`
3. [Google AI Studio](https://aistudio.google.com/app/apikey) থেকে ফ্রি **Gemini API Key** নিন।
4. নোটবুকের **৫টি ধাপ** উপর থেকে নিচে চালান:
   - **ধাপ ১:** Google Drive মাউন্ট
   - **ধাপ ২:** রিপোজিটরি ক্লোন ও ফোল্ডার প্রস্তুত
   - **ধাপ ৩:** দ্রুত ডিপেন্ডেন্সি ইনস্টল ও Whisper `large-v3` প্রি-ডাউনলোড (যাতে ভিডিও প্রসেসিংয়ের সময় ডাউনলোড না হয়)
   - **ধাপ ৪:** Gemini API Key প্রদান ও **তাৎক্ষণিক লাইভ যাচাই** (পুরুষ `male` বা নারী `female` কণ্ঠ নির্বাচন)
   - **ধাপ ৫:** সব ভিডিও বাংলায় ডাব করুন!

---

## 💻 কমান্ড লাইন (CLI) উদাহরণ

```bash
# পুরো ইনপুট ফোল্ডার ডাব করুন (পুরুষ কণ্ঠ ডিফল্ট)
export GEMINI_API_KEY="AIza..."
python main.py

# নারী কণ্ঠে ডাব করুন
python main.py --gender female

# শুধু একটি নির্দিষ্ট ভিডিও প্রসেস করুন
python main.py --video /content/drive/MyDrive/Video/input/lecture.mp4
```

---

## 🧪 স্বয়ংক্রিয় পরীক্ষা (Tests)

```bash
python -m unittest discover -s tests -v
```
