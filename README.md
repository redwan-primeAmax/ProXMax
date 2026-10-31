# 🎬 ProXMax — স্বয়ংক্রিয় English → Bangla ভিডিও ডাবিং (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/redwan-primeAmax/ProXMax/blob/main/notebook/ProXMax_Colab.ipynb)

**ProXMax** একটি সম্পূর্ণ স্বয়ংক্রিয় পাইথন পাইপলাইন, যা Google Drive-এর ইনপুট ফোল্ডারে রাখা ইংরেজি ভিডিওগুলোকে বাংলা কণ্ঠে ডাব করে আউটপুট ফোল্ডারে সেভ করে। সীমিত worker-সহ Google Colab-এর **NVIDIA T4** ও অনুরূপ পরিবেশে চালানোর জন্য তৈরি; GPU, RAM, encoder ও ইন্টারনেটের ওপর প্রকৃত সময় ও resource ব্যবহার নির্ভর করে।

---

## ✨ মূল বৈশিষ্ট্য

| ধাপ | প্রযুক্তি | কেন এটি বেছে নেওয়া হয়েছে |
|---|---|---|
| **১. অডিও এক্সট্রাক্ট** | `ffmpeg` (progress + stall watchdog) | ১৬ kHz মনো WAV তৈরি করে, stream-এর শুরুর offset ধরে রাখে; ড্রাইভ ফ্রিজ হলে পরিষ্কার বার্তা দেয় |
| **২. Speech-to-Text (ইংরেজি)** | `faster-whisper` (`distil-large-v3` ডিফল্ট, `large-v3` বিকল্প, প্রি-লোডেড ক্যাশ, নীরবতা-সচেতন ১৮–২৮s চাঙ্কিং + সীমান্ত মেরামত + `hotwords.txt`) | সেটআপ ধাপেই মডেল ক্যাশ হয়; context-aware hotword prompt FC Mobile UI, event, gameplay ও curated player-name spelling-এ সাহায্য করে—তবে নাম শনাক্তকরণ probabilistic এবং তালিকাটি exhaustive নয় |
| **৩. অনুবাদ (EN → BN)** | **Google Gemini API** (`google-genai`, ডিফল্ট `gemini-3.5-flash-lite`, bounded glossary ও input-token budget) | সর্বোচ্চ ৫ সেকেন্ডের paragraph, request-এ সর্বোচ্চ ১২টি; stable paragraph ID দিয়ে output order যাচাই হয়, প্রথম অনুবাদে কোনো timing hint পাঠানো হয় না |
| **৪. বাংলা ভয়েস (TTS)** | **বিল্ট-ইন বাংলা neural ভয়েস** (`edge-tts` — পুরুষ কণ্ঠ `bn-BD-PradeepNeural` ডিফল্ট, নারী `bn-BD-NabanitaNeural`) | আলাদা voice sample বা voice token ছাড়াই বাংলা কণ্ঠ তৈরি করে; উচ্চারণ ও নাম শুনে যাচাই করতে হবে |
| **৫. অডিও-ভিডিও সিঙ্ক** | `ffmpeg` (`rubberband` / `atempo`) + একক ট্র্যাক অ্যাসেম্বলার | শনাক্ত নীরবতা ছেঁটে মাপা speech সর্বোচ্চ ০.৯০× পর্যন্ত ধীর; লম্বা speech-এর জন্য ওই window-এর moving video tail পুনরাবৃত্তি; audio speedup বা fit rewrite নেই |
| **৬. ভিডিও মাক্সিং** | `ffmpeg` (frame-based edit, concat, AAC) | অপরিবর্তিত picture copy হতে পারে; অন্যথায় window-ভিত্তিক ভিডিও re-encode, bounded workers ও frame-count/duration যাচাই |
| **৭. ডিটেইল রিপোর্ট** | secret-safe Markdown (`*_report.markdown`) | source ও dubbed media metadata, ইংরেজি/বাংলা transcript, timestamp, word timing, TTS/sync window ও audio activity এক ফাইলে দেয়; ঐচ্ছিক post-mux Bengali ASR coverage চালু থাকলে যোগ হয়; আলাদা SRT তৈরি করে না |

### FC Mobile হটওয়ার্ড ও timing safeguards

- `hotwords.txt`-এ UI/button actions (যেমন **Exchange**, **Play**), modes, events, gameplay terms, Skill Moves, PlayStyles এবং বাছাই করা current/legend player names ও aliases আছে। এটি **curated, exhaustive roster নয়**—সব card/player বা সব উচ্চারণভেদ অন্তর্ভুক্ত থাকার দাবি করে না।
- পুরো hotword list Gemini-তে পাঠানো হয় না। Gemini-র প্রতিটি translation batch-এ শুধু source text-এ boundary-aware exact match পাওয়া term যায়; সর্বোচ্চ **২০টি term / ৬০০ অক্ষর**।
- Colab-এর ধাপ ৪-এ **সর্বোচ্চ Gemini input token** editable (ডিফল্ট **৫,০০০**, minimum **১,০০০**); conservative local estimate-এ instruction, glossary ও পুরো paragraph ধরা হয়। প্রতি request-এ সর্বোচ্চ **১২টি paragraph** পাঠানো হয় এবং response-এ ID চাওয়া/যাচাই করা হয়। Exact tokenization Gemini নির্ধারণ করে; budget ছাড়ালে chunk ভাঙা/কাটা নয়—নতুন request-এ সরানো হয়, আর একা fit না হলে কেবল সেটি বাদ যায়।
- Gemini-তে API, network, safety বা response-format error হলে প্রতি request-এ প্রতিবার **১ সেকেন্ড অপেক্ষা করে সর্বোচ্চ ১০০ বার retry** হয় (প্রথম request সহ সর্বোচ্চ ১০১টি চেষ্টা)। আংশিক উত্তর পাওয়া গেলে বৈধ অনুবাদ রেখে শুধু বাকি অংশ আবার চাওয়া হয়; প্রয়োজনে ভুল ফরম্যাটের ব্যাচ ছোট করে বা অনূদিত না হওয়া অংশ উদ্ধার করতেও আলাদা request হতে পারে। Retry-র বিরতিতে সাধারণ RPM limiter বা server-এর `Retry-After` আর অতিরিক্ত অপেক্ষা যোগ করে না; এতে quota error চলতে পারে। স্বাভাবিক নতুন request-এ RPM limit বহাল থাকে। প্রতিটি request-এর নিজস্ব timeout থাকায় মোট সময় ১০০ সেকেন্ডের বেশি হতে পারে। সব retry ব্যর্থ হলে error লগ হয়; API error-এ ভিডিওটি ব্যর্থ চিহ্নিত হলেও পরের ভিডিওগুলো চলতে পারে। `config.yaml`-এর `gemini.max_retries` ও retry delay সেটিং বদলানো যায়।
- Gemini-র output-এ source-এর **অঙ্কে লেখা সংখ্যা** ও অপ্রত্যাশিত লিপির সীমিত heuristic পরীক্ষা হয়। একটিমাত্র স্পষ্ট বাদ-পড়া অঙ্ক কখনো স্থানীয়ভাবে ঠিক হয়; অনির্দিষ্ট সমস্যা থাকলে কেবল সংশ্লিষ্ট paragraph আবার অনুবাদ হয়। সমাধান না হলে report-এ review flag, আর স্বয়ংক্রিয় পরিবর্তনে note থাকে। কথায় লেখা সব সংখ্যা, নাম বা বাক্যের অর্থ এভাবে যাচাই হয় না; source STT ভুল হলেও তা ঠিক বলে ধরে নেওয়া যায় না।
- অনুবাদের প্রথম ব্যাচ পাওয়া মাত্রই TTS শুরু হয়। প্রস্তুত ব্যাচগুলো সীমিত আকারে একত্র করা হয়, যাতে আগের ১২টি clip-এর সবচেয়ে ধীরটির জন্য পরের সব clip অপেক্ষা না করে। ডিফল্টে একসঙ্গে সর্বোচ্চ **১২টি** Edge TTS request চলে; `config.yaml`-এর `tts.concurrency` বা নোটবুকে `config.tts_concurrency` দিয়ে বদলানো যায়। ব্যর্থ clip কম concurrency-তে পুনরায় চেষ্টা হয়; অনেক clip ব্যর্থ হলে ওই ভিডিওর বাকি অংশে concurrency ৫-এ নেমে যায়। প্রতিটি clip-এর টেক্সট, ভয়েস ও timestamp আলাদা থাকে। প্রকৃত সময় Edge সার্ভিস ও ইন্টারনেটের গতির উপর নির্ভর করে।
- Whisper-কে সর্বোচ্চ **৪০০ অক্ষরের** bounded hotword prompt দেওয়া হয়, যাতে context window-তে transcript-এর জন্য জায়গা থাকে; game terms অগ্রাধিকার পায় এবং chunk/video context-এ থাকা নামগুলো নির্বাচন করা হয়। এগুলো recognition-এ সাহায্য করে, নির্ভুলতার নিশ্চয়তা দেয় না। Seam re-decode আংশিক হলে পুরোনো শব্দ রেখে দেওয়া হয়; শুধু একই সময়ে প্রায় সম্পূর্ণ overlap করা একই শব্দ deduplicate হয়, পাশাপাশি বাস্তব পুনরাবৃত্তি নয়।
- ডিফল্ট `distil-large-v3` শুধু ইংরেজি সমর্থন করে এবং কম resource/সময় লাগতে পারে; Colab-এর ধাপ ৩-এ `large-v3` বেছে নেওয়া যায়। দুটিই `faster-whisper` backend ব্যবহার করে। Distil কিছু ভিডিওতে বেশি ভুল করতে পারে; একই accuracy বা speed সব ভিডিওতে নিশ্চিত নয়। নাম/শব্দের ক্ষেত্রে source শুনে দেখুন।
- প্রতিটি TTS clip তৈরির সঙ্গে সঙ্গেই head, internal ও tail-এর শনাক্ত নীরবতা বাদ দেওয়া হয়; custom TTS হলে sync ধাপে একই নিয়ম প্রয়োগ হয়। ১০ ms peak frame ও −৭০ dBFS threshold ব্যবহার হয়; speech-এর দুই পাশে ডিফল্ট **২৫ ms padding** (কমপক্ষে ১০ ms) রাখা হয়। এটি heuristic: খুব নিচু speech ভুল করে নীরবতা ধরা পড়তে পারে।
- নীরবতা বাদ দিয়ে **প্রকৃত WAV** মাপার পর দীর্ঘ clip-এর speech দ্রুত বা Gemini দিয়ে ছোট করা হয় না। Positive `tts.rate` সেট করলেও provider-এ `+0%` পাঠানো হয়; custom negative voice rate থাকলে সেটি তৈরির সময় প্রযোজ্য, আর sync-এর ০.৯০× সীমা তৈরি হওয়া audio-র তুলনায় মাপা হয়। ওই source window-এর শেষের চলমান ফুটেজ যতটুকু দরকার পুনরাবৃত্তি করে ঠিক window-এর পর বসানো হয়; দরকারি সময় window-এর চেয়েও বেশি হলে পুরো tail একাধিকবার ব্যবহার হয়। পরের ছবি ও speech একসাথে সরে।
- ছোট clip-কে কেবল যতটা দরকার ধীর করা হয়, **০.৯০×-এর নিচে নয়**। মাপা FFmpeg ফল নিরাপদ না হলে `rubberband` থেকে `atempo` চেষ্টা হয়; কোনোটিই যাচাই না হলে মূল speech রেখে warning দেওয়া হয়। তারপরও gap থাকলে শুধু সেই video window সর্বোচ্চ **১.১৫×** দ্রুত হয়; সীমায় gap থাকলে সেটি থাকে। দীর্ঘ speech-এর audio tempo সবসময় ১.০০×।
- Sync window-তে source speech-এর পর সর্বোচ্চ **২৫০ ms** silence borrow ও পরের window-এর আগে **৮০ ms** gap রাখার চেষ্টা ডিফল্ট; `sync` সেটিংস দিয়ে বদলানো যায়। ছবি পূর্ণ frame-এ বাঁধা: প্রয়োজনীয় বাড়তি সময় উপরের frame-এ round হয়, তাই প্রতি edit-এ সর্বোচ্চ এক frame-এর কাছাকাছি পার্থক্য থাকতে পারে। Audio sample অনুযায়ী বসে; frame rounding ও একাধিক edit-এর প্রভাবে পরের timing বদলাতে পারে। Edited video গড় frame rate-এ constant-frame-rate হয়; variable-frame-rate source-এর প্রতিটি আসল frame অপরিবর্তিত থাকার নিশ্চয়তা নেই।
- শূন্য বা এক-frame window-তে moving tail না থাকলে কাছের উপলব্ধ ছবি ব্যবহার ও report-এ warning দেওয়া হয়। পুরো source-এ মাত্র একটি frame থাকলে বাস্তব motion তৈরি করা সম্ভব নয়। Repeated footage-এ দৃশ্যমান jump হতে পারে। Source picture দেরিতে শুরু হলে অনুপস্থিত শুরুর অংশে কালো frame এবং audio দেরিতে শুরু হলে নীরবতা যোগ করে মূল alignment রাখা হয়; source picture audio-র আগে শেষ হলে তার tail পুনরাবৃত্তি করা হয়।
- ছবি edit না হলে video copy হতে পারে; edit হলে H.264 re-encode করতে হয়। ডিফল্টে একসঙ্গে **৪টি audio preparation** ও **২টি video encoder** চলে (GPU-তে সর্বোচ্চ ২; CPU-তে কনফিগমতো সর্বোচ্চ ৪); অসমর্থিত NVENC ব্যর্থ হলে CPU চেষ্টা করা হয়। কাজের ধরন, encoding, video, GPU ও network অনুযায়ী সময়, disk ও RAM/VRAM বদলায়—আগের copy-only পদ্ধতির চেয়ে দ্রুত বা ১২ GB-তে সব video চলবে, এমন নিশ্চয়তা নেই।
- Report-এ মূল source time, বাস্তব placed time, video window, speed, repeat, silence ও warning থাকে; Bengali report cue-ও placed time ব্যবহার করে। কোনো clip উদ্ধার করা বা overlap ছাড়া সম্পূর্ণ বসানো অসম্ভব হলে success দেখানোর বদলে স্পষ্ট failure হয়। Waveform ও duration পরীক্ষা অর্থ, নাম বা নিখুঁত pronunciation প্রমাণ করে না।
- খুব ছোট/শূন্য STT segment-ও ফেলা হয় না। অনুপস্থিত translation বা TTS কেবল সংশ্লিষ্ট segment-এ retry হয়; না পেলে স্পষ্ট failure। আগের report-এ unplaced/অনুপস্থিত unit জানা থাকলে `skip_existing` চালু থাকলেও পুনরায় প্রসেস হয়; পুরোনো report-এ তথ্য না থাকলে `--force` ব্যবহার করুন। Sync retry-তে অপরিবর্তিত clip-এর মাপা প্রস্তুত audio ও settings মিললে reuse হয়। Report-এর translation ও TTS call time overlap হতে পারে, তাই একসাথে যোগ করে wall time ধরবেন না।
- TTS-এ শুধু আলাদা `দিব` শব্দ কথ্য `দিবো` হিসেবে পাঠানো হয়; প্রদর্শিত অনুবাদ বদলায় না, নাম বা সব `ব`-শেষ শব্দ স্বয়ংক্রিয়ভাবে বদলায় না। শেষের fade ৫০ ms থেকে ৮ ms করা হয়েছে। `output.keep_temp_files: true` দিলে raw Edge MP3 এবং প্রস্তুত WAV রেখে source/transcript, Edge pronunciation, silence trim, stretch ও mux তুলনা করে শুনতে পারবেন। `ইয়াসিন`-এর মতো নামের উচ্চারণ সঠিক হবে—এমন নিশ্চয়তা নেই।
- Post-mux Bengali ASR coverage ঐচ্ছিক, ডিফল্ট বন্ধ। ইংরেজি-only Distil নির্বাচিত থাকলে এই পরীক্ষা চালু করলে আলাদা multilingual `large-v3` নামানো/লোড ও আরেকটি সম্পূর্ণ pass হতে পারে; সময়/VRAM বাড়ে। ফল কেবল review signal, semantic বা pronunciation validation নয়।

### Source audio ও মডেল তুলনা

ডিফল্ট extraction ১৬ kHz mono; source-এর গতি বদলানো বা বাধ্যতামূলক denoise করা হয় না। Audio stream দেরিতে শুরু হলে timestamp alignment-এর জন্য শুরুতে নীরবতা থাকে। পরীক্ষামূলক `stt.audio_filter: highpass60` ৬০ Hz high-pass ব্যবহার করে, কিন্তু বাস্তব speech/নাম শনাক্তকরণে ক্ষতি বা লাভ হতে পারে—নিজের ডেটায় যাচাই ছাড়া চালু করবেন না।

প্রতিনিধিত্বমূলক ভিডিও ও হাতে যাচাই করা **ইংরেজি** transcript দিয়ে নিজের মেশিনে গতি ও word-error rate তুলনা করুন (মডেল প্রথমবার download হতে পারে):

```json
[{"audio": "/path/to/video.mp4", "reference": "Manually checked spoken English"}]
```

```bash
python benchmarks/stt_compare.py --manifest /path/to/manifest.json
```

স্ক্রিপ্ট উভয় মডেল ও `none`/`highpass60` তুলনা করে; model-load time আলাদা এবং extraction + STT realtime factor ও aggregated WER দেখায়। নেটওয়ার্ক ও hardware ভেদে একই sample কয়েকবার পরীক্ষা করুন। Benchmark বা automated checks উচ্চারণ/semantic accuracy প্রমাণ করে না; এই repo-তে ব্যবহারকারীর আসল অডিও ছাড়া accuracy বা runtime regression নেই বলে দাবি করা যাচ্ছে না।

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
    ├── sync_engine.py + sync_prepare.py  # silence trimming, safe slowdown and track assembly
    ├── video_timeline.py + video_render.py  # frame-based edits and bounded parallel encodes
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
   - **ধাপ ৩:** dependency install ও নির্বাচিত Whisper (`distil-large-v3` ডিফল্ট / `large-v3`) প্রি-ডাউনলোড (যাতে ভিডিও প্রসেসিংয়ের সময় ডাউনলোড না হয়)
   - **ধাপ ৪:** Gemini API Key প্রদান ও **তাৎক্ষণিক লাইভ যাচাই** (পুরুষ `male` বা নারী `female` কণ্ঠ নির্বাচন)
   - **ধাপ ৫:** সব ভিডিও বাংলায় ডাব করুন!

প্রতিটি সফল ভিডিওর পাশে `Video/output/<stem>_report.markdown` তৈরি হবে। এতে English ও Bangla transcript, timestamp/word timing, TTS ও sync diagnostics এবং media metadata থাকবে; আলাদা `_en.srt` / `_bn.srt` ফাইল আর তৈরি হবে না। Spoken-content timeline আছে, তবে visual action recognition নয়।

Colab-এ update pull ব্যর্থ হলে error দেখাবে; কোড import হয়ে থাকলে runtime restart করে আবার ধাপ ১–৫ চালান, যাতে নতুন কোড লোড হয়। পুরোনো সফল আউটপুট আবার বানাতে ধাপ ৫-এ `config.skip_existing = False` সক্রিয় করুন; এতে একই নামের আউটপুট ও রিপোর্ট পুনর্লিখিত হবে, দরকার হলে আগে backup নিন। CLI-তে `--force` ব্যবহার করা যায়।

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
