# 🎬 ProXMax — স্বয়ংক্রিয় English → Bangla ভিডিও ডাবিং (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb)

**ProXMax** একটি সম্পূর্ণ স্বয়ংক্রিয় পাইথন পাইপলাইন, যা Google Drive-এর ইনপুট ফোল্ডারে রাখা ইংরেজি ভিডিওগুলোকে বাংলা কণ্ঠে ডাব করে আউটপুট ফোল্ডারে সেভ করে। পুরো সিস্টেমটি Google Colab-এর ফ্রি **NVIDIA T4 GPU**-তে চলার জন্য অপ্টিমাইজড।

---

## ✨ মূল বৈশিষ্ট্য

| ধাপ | প্রযুক্তি | কেন এটি বেছে নেওয়া হয়েছে |
|---|---|---|
| **১. অডিও এক্সট্রাক্ট** | `ffmpeg` (progress + stall watchdog) | সোর্স ভিডিও কেবল **একবারই** পড়ে ১৬ kHz মনো WAV তৈরি করে; ড্রাইভ ফ্রিজ হলে পরিষ্কার বার্তা দেয় |
| **২. Speech-to-Text (ইংরেজি)** | `faster-whisper` (`large-v3`, প্রি-লোডেড ক্যাশ, নীরবতা-সচেতন ১৮–২৮s চাঙ্কিং + সীমান্ত মেরামত + `hotwords.txt`) | সেটআপ ধাপেই মডেল ক্যাশ হয়; context-aware hotword prompt FC Mobile UI, event, gameplay ও curated player-name spelling-এ সাহায্য করে—তবে নাম শনাক্তকরণ probabilistic এবং তালিকাটি exhaustive নয় |
| **৩. অনুবাদ (EN → BN)** | **Google Gemini API** (`google-genai`, ডিফল্ট `gemini-3.5-flash-lite`, bounded glossary ও input-token budget) | সর্বোচ্চ ৫ সেকেন্ডের paragraph, request-এ সর্বোচ্চ ১২টি; stable paragraph ID দিয়ে output order যাচাই হয়, প্রথম অনুবাদে কোনো timing hint পাঠানো হয় না |
| **৪. বাংলা ভয়েস (TTS)** | **বিল্ট-ইন বাংলা neural ভয়েস** (`edge-tts` — পুরুষ কণ্ঠ `bn-BD-PradeepNeural` ডিফল্ট, নারী `bn-BD-NabanitaNeural`) | কোনো নমুনা রেকর্ডিং বা অতিরিক্ত টোকেন ছাড়াই সাথে সাথে স্পষ্ট বাংলা কণ্ঠ তৈরি হয় |
| **৫. অডিও-ভিডিও সিঙ্ক** | `ffmpeg` (`rubberband` / `atempo`) + একক ট্র্যাক অ্যাসেম্বলার | কণ্ঠের পিচ অক্ষুণ্ণ রেখে মূল টাইমস্ট্যাম্পে বসায়; clip overlap করে না এবং কোনো clip নিরাপদ window-তে না আঁটলে sentence কেটে না দিয়ে review flag করে |
| **৬. ভিডিও মাক্সিং** | `ffmpeg -c:v copy -c:a aac` | ভিডিও রি-এনকোড না করেই নতুন বাংলা অডিও যুক্ত করে — মান ১০০% অক্ষুণ্ণ ও অত্যন্ত দ্রুত |
| **৭. ডিটেইল রিপোর্ট** | secret-safe Markdown (`*_report.markdown`) | source ও dubbed media metadata, ইংরেজি/বাংলা transcript, timestamp, word timing, TTS/sync window ও audio activity এক ফাইলে দেয়; ঐচ্ছিক post-mux Bengali ASR coverage চালু থাকলে যোগ হয়; আলাদা SRT তৈরি করে না |

### FC Mobile হটওয়ার্ড ও timing safeguards

- `hotwords.txt`-এ UI/button actions (যেমন **Exchange**, **Play**), modes, events, gameplay terms, Skill Moves, PlayStyles এবং বাছাই করা current/legend player names ও aliases আছে। এটি **curated, exhaustive roster নয়**—সব card/player বা সব উচ্চারণভেদ অন্তর্ভুক্ত থাকার দাবি করে না।
- পুরো hotword list Gemini-তে পাঠানো হয় না। Gemini-র প্রতিটি translation batch-এ শুধু source text-এ boundary-aware exact match পাওয়া term যায়; সর্বোচ্চ **২০টি term / ৬০০ অক্ষর**।
- Colab-এর ধাপ ৪-এ **সর্বোচ্চ Gemini input token** editable (ডিফল্ট **৫,০০০**, minimum **১,০০০**); conservative local estimate-এ instruction, glossary ও পুরো paragraph ধরা হয়। প্রতি request-এ সর্বোচ্চ **১২টি paragraph** পাঠানো হয় এবং response-এ ID চাওয়া/যাচাই করা হয়। Exact tokenization Gemini নির্ধারণ করে; budget ছাড়ালে chunk ভাঙা/কাটা নয়—নতুন request-এ সরানো হয়, আর একা fit না হলে কেবল সেটি বাদ যায়।
- Gemini-তে API, network, safety বা response-format error হলে প্রতি request-এ প্রতিবার **১ সেকেন্ড অপেক্ষা করে সর্বোচ্চ ১০০ বার retry** হয় (প্রথম request সহ সর্বোচ্চ ১০১টি চেষ্টা)। আংশিক উত্তর পাওয়া গেলে বৈধ অনুবাদ রেখে শুধু বাকি অংশ আবার চাওয়া হয়; প্রয়োজনে ভুল ফরম্যাটের ব্যাচ ছোট করে বা অনূদিত না হওয়া অংশ উদ্ধার করতেও আলাদা request হতে পারে। Retry-র বিরতিতে সাধারণ RPM limiter বা server-এর `Retry-After` আর অতিরিক্ত অপেক্ষা যোগ করে না; এতে quota error চলতে পারে। স্বাভাবিক নতুন request-এ RPM limit বহাল থাকে। প্রতিটি request-এর নিজস্ব timeout থাকায় মোট সময় ১০০ সেকেন্ডের বেশি হতে পারে। সব retry ব্যর্থ হলে error লগ হয়; API error-এ ভিডিওটি ব্যর্থ চিহ্নিত হলেও পরের ভিডিওগুলো চলতে পারে। `config.yaml`-এর `gemini.max_retries` ও retry delay সেটিং বদলানো যায়।
- Gemini-র output-এ source-এর **অঙ্কে লেখা সংখ্যা** ও অপ্রত্যাশিত লিপির সীমিত heuristic পরীক্ষা হয়। একটিমাত্র স্পষ্ট বাদ-পড়া অঙ্ক কখনো স্থানীয়ভাবে ঠিক হয়; অনির্দিষ্ট সমস্যা থাকলে কেবল সংশ্লিষ্ট paragraph আবার অনুবাদ হয়। সমাধান না হলে report-এ review flag, আর স্বয়ংক্রিয় পরিবর্তনে note থাকে। কথায় লেখা সব সংখ্যা, নাম বা বাক্যের অর্থ এভাবে যাচাই হয় না; source STT ভুল হলেও তা ঠিক বলে ধরে নেওয়া যায় না।
- অনুবাদের প্রথম ব্যাচ পাওয়া মাত্রই TTS শুরু হয়। প্রস্তুত ব্যাচগুলো সীমিত আকারে একত্র করা হয়, যাতে আগের ১২টি clip-এর সবচেয়ে ধীরটির জন্য পরের সব clip অপেক্ষা না করে। ডিফল্টে একসঙ্গে সর্বোচ্চ **১২টি** Edge TTS request চলে; `config.yaml`-এর `tts.concurrency` বা নোটবুকে `config.tts_concurrency` দিয়ে বদলানো যায়। ব্যর্থ clip কম concurrency-তে পুনরায় চেষ্টা হয়; অনেক clip ব্যর্থ হলে ওই ভিডিওর বাকি অংশে concurrency ৫-এ নেমে যায়। প্রতিটি clip-এর টেক্সট, ভয়েস ও timestamp আলাদা থাকে। প্রকৃত সময় Edge সার্ভিস ও ইন্টারনেটের গতির উপর নির্ভর করে।
- Whisper-কে সর্বোচ্চ **৪০০ অক্ষরের** bounded hotword prompt দেওয়া হয়, যাতে context window-তে transcript-এর জন্য জায়গা থাকে; game terms অগ্রাধিকার পায় এবং chunk/video context-এ থাকা নামগুলো নির্বাচন করা হয়। এগুলো recognition-এ সাহায্য করে, নির্ভুলতার নিশ্চয়তা দেয় না। Seam re-decode আংশিক হলে পুরোনো শব্দ রেখে দেওয়া হয়; শুধু একই সময়ে প্রায় সম্পূর্ণ overlap করা একই শব্দ deduplicate হয়, পাশাপাশি বাস্তব পুনরাবৃত্তি নয়।
- নির্ভুলতা অগ্রাধিকার হলে Colab-এর ধাপ ৩-এ `WHISPER_MODEL = "large-v3"` রাখুন (`config.yaml`-এ `stt.model`); `distil-large-v3` দ্রুততর হতে পারে, তবে ভিডিওভেদে বেশি ভুল করতে পারে। Report কম recognizer-confidence শব্দগুলো review cue হিসেবে চিহ্নিত করে, এটি ভুল বা নির্ভুলতার প্রমাণ নয়। যেকোনো মডেলের transcript মূল অডিওর সঙ্গে যাচাই করা দরকার।
- ডাবের window detected source speech interval ও configured ছোট borrow-এর মধ্যে সীমিত থাকে (ডিফল্ট **২৫০ ms**); পরের বক্তব্যের আগে থাকা ২–৫ সেকেন্ডের pause আর পুরোপুরি ব্যবহার করা হয় না। এটি ওই ধরনের drift কমায় এবং প্রকৃত pause ধরে রাখে, কিন্তু ভুল STT timestamp বা speech boundary থেকে হওয়া সব sync error দূর করার নিশ্চয়তা দেয় না।
- বাংলা TTS-এর প্রকৃত WAV duration প্রতিটি source timeline window-র সাথে মেলানো হয় (১০ সেকেন্ড কোনো বাঁধা দৈর্ঘ্য নয়)। Overrun হলে শেষ থেকে শুরুর দিকে আগে tail, তারপর ভেতরের দীর্ঘ বিরতি এবং শেষে head-এর **শনাক্ত silence** শুধু যতটুকু দরকার বাদ যায়। ১০ ms peak frame ও সর্বোচ্চ −55 dBFS threshold ব্যবহার হয়; শনাক্ত কথার দুই পাশে ডিফল্ট ২৫ ms করে padding রেখে বাকি নীরবতা প্রয়োজনমতো সরানো যায়। তবুও খুব নিচু স্বর নীরবতা হিসেবে ভুল শনাক্ত হতে পারে—report-এর trim flag শুনে যাচাই করুন।
- নীরবতা কমিয়েও না আঁটলে tempo সর্বোচ্চ **১.২০×** পর্যন্ত বাড়ে; ছোট clip আগে যেমন সর্বোচ্চ ০.৯০× ধীর হতে পারে। `sync.max_tempo` কম সেট করা থাকলে সেই কম সীমাই মানা হয়। FFmpeg-এর প্রকৃত output duration আলাদা হতে পারে বলে দরকার হলে অনুমোদিত সর্বোচ্চ tempo-তে পুনরায় render ও মাপা হয়; তা মাপা না গেলে rewrite না করে review flag দেওয়া হয়। Frame rounding-এর জন্য সর্বোচ্চ ১০ ms শুধু সংরক্ষিত ফাঁক বা ভিডিওর বাকি সময় থেকে ব্যবহার হতে পারে, ভিডিওর শেষ বা পরের clip পেরিয়ে নয়।
- সর্বোচ্চ অনুমোদিত tempo-তেও clip fit না হলে কেবল সেই clip-এর ইংরেজি source ও আগের বাংলা দিয়ে অর্থ বজায় রেখে measured-fit rewrite এবং নতুন TTS চাওয়া হয়। নতুন WAV-এর প্রকৃত duration ও প্রয়োজনীয় silence বাদ দেওয়ার পর অবশিষ্ট সময় মাপা হয়, তারপর একই window-তে আবার fit পরীক্ষা হয়। সংখ্যা/লিপি/নেতিবাচক অর্থে স্পষ্ট সমস্যা, অত্যধিক ছোট rewrite বা বাস্তবে উন্নতি না হওয়া audio বাতিল করে আগের text/audio রাখা হয়। প্রথম গ্রহণযোগ্য rewrite-ও unfit থাকলে সর্বোচ্চ আরেকটি retry হতে পারে। চূড়ান্ত report-এ আগের বাংলা ও review note থাকে। এসব heuristic অর্থ অক্ষুণ্ণ থাকার প্রমাণ নয়; তবুও না আঁটলে speech কেটে না দিয়ে clip unplaced ও review flag হয়।
- Sync retry-তে অপরিবর্তিত clip-এর source/artifact signature, নির্দিষ্ট window ও settings মিললে আগের প্রস্তুত audio reuse হয়; placement ও loudness আবার হিসাব করা হয়। Report-এ sync pass ও fit retry-র সময়, overlap হওয়া translation/TTS call time এবং আলাদা mux ও output-copy সময় দেখায়—overlapping সময়গুলো যোগ করলে মোট সময় হয় না।
- ছোট clip ভরাতে text বাড়ানো হয় না। Post-mux Bengali ASR coverage check ঐচ্ছিক এবং default-এ বন্ধ—এটি পুরো dubbed audio আরেকবার Whisper-এ চালায় বলে সময় উল্লেখযোগ্য বাড়ে; চালু করলে ফল heuristic, semantic validation নয়, তাই flagged সময় শুনে যাচাই করা উচিত।

---

## 📁 প্রজেক্ট স্ট্রাকচার

Python source ও test modules দায়িত্ব অনুযায়ী ছোট ফাইলে ভাগ করা; প্রতিটি `.py` ফাইল ৪০০ লাইন বা তার কম। পুরনো import path সহজ রাখতে `translator.py`-র মতো facade রাখা হয়েছে।

```text
ProXMax/
├── main.py                        # CLI এন্ট্রি পয়েন্ট
├── config.yaml                    # সব সেটিংস এক জায়গায় (কোড না ছুঁয়েই পরিবর্তনযোগ্য)
├── hotwords.txt                   # FC Mobile UI/mode/event/skill/PlayStyle এবং curated football names/aliases
├── requirements.txt               # দ্রুততম ইনস্টলের জন্য অপ্টিমাইজড ডিপেন্ডেন্সি
├── requirements-indictrans2.txt   # ঐচ্ছিক: লোকাল IndicTrans2 অনুবাদ ব্যাকএন্ডের জন্য
├── notebook/
│   └── ProXMax_Colab.ipynb        # এক-ক্লিক Google Colab নোটবুক (৫ ধাপ)
├── tests/
│   ├── _support.py                # shared fakes and test imports
│   └── test_*.py                  # focused unit/regression suites by component
└── proxmax/
    ├── config.py + config_validation.py  # settings, YAML loading, validation
    ├── hotwords.py + hotword_repair.py   # dedicated glossary, bounded prompts, safe normalization
    ├── models.py                  # Segment, Word, VideoResult, StageTimings data models
    ├── utils.py + ffmpeg_utils.py  # logging/SRT helpers and guarded FFmpeg wrappers
    ├── audio_quality.py           # heuristic post-mux Bangla token/tail coverage check
    ├── stt.py                     # public STT interface and sentence orchestration
    ├── stt_inference.py           # Whisper loading, multilingual decoding and model lifecycle
    ├── stt_seams.py + whisper_model.py  # chunk seam repair and Whisper cache/device helpers
    ├── translator.py              # stable import facade for translation backends
    ├── gemini_*.py + translation_*.py  # Gemini protocol, batching, repair, parsing and errors
    ├── sync_engine.py + sync_math.py    # safe time-stretching and non-overlapping track assembly
    ├── reporting.py + reporting_*.py    # detailed Markdown report data/rendering sections
    ├── pipeline.py + pipeline_*.py      # reusable orchestration split by processing stage
    └── tts/                        # TTS base interface and built-in Bengali voice engine
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

প্রতিটি সফল ভিডিওর পাশে `Video/output/<stem>_report.markdown` তৈরি হবে। এতে English ও Bangla transcript, timestamp/word timing, TTS ও sync diagnostics এবং media metadata থাকবে; আলাদা `_en.srt` / `_bn.srt` ফাইল আর তৈরি হবে না। Spoken-content timeline আছে, তবে visual action recognition নয়।

আপডেটের পর Colab runtime restart করে আবার ধাপ ১–৫ চালান, যাতে নতুন কোড লোড হয়। পুরোনো সফল আউটপুট আবার বানাতে ধাপ ৫-এ `config.skip_existing = False` সক্রিয় করুন; এতে একই নামের আউটপুট ও রিপোর্ট পুনর্লিখিত হবে, দরকার হলে আগে backup নিন। CLI-তে `--force` ব্যবহার করা যায়।

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
# অফলাইন, কৃত্রিম অডিওতে sync rebuild তুলনা (numpy ও ffmpeg প্রয়োজন)
python benchmarks/sync_rebuild.py --clips 171 --changed 53 --workers 4
```

Benchmark কোনো live ভিডিও/Gemini/Edge-এর গতি নয়; fresh ও cached output waveform এবং sync statistics মিলেছে কি না সেটিও যাচাই করে।
