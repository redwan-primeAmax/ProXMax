# Colab-এ বাস্তব পরীক্ষার তালিকা

এটি পরীক্ষার পদ্ধতি—ইতিমধ্যে সব পরীক্ষা পাস করেছে এমন দাবি নয়। Repository-এর হালকা test আসল model/GPU দিয়ে অনুবাদের মান যাচাই করে না।

## প্রস্তুতি

- [ ] বর্তমান Colab-এ T4 GPU নিন; Python, torch, CUDA/cuDNN ও GPU সংস্করণ লিখে রাখুন।
- [ ] Generated notebook সরাসরি খুলুন; repository clone-এর ওপর নির্ভর করবেন না।
- [ ] Hugging Face-এ সংযোগ ও অন্তত 10 GB local খালি জায়গা নিশ্চিত করুন।
- [ ] অনুমোদিত 20–40 সেকেন্ডের ইংরেজি clip নিন: দুটি বাক্য, একটি বিরতি, নাম ও সংখ্যা থাকবে। ইংরেজি reference এবং বাংলা জানা reviewer রাখুন।
- [ ] `MyDrive/Video/input`-এ শুধু পরীক্ষার ভিডিও রাখুন। পরীক্ষার সময় `keep_local=True` দিন।

## প্রথমবার

- [ ] Notebook-এ একটিই cell এবং স্বাভাবিক Colab ভিউতে processing code দেখা যায় না। একবার Run দিলে setup, Drive ও অনুবাদ পরপর চলে। Kernel-এ torch/transformers import হয় না; package install হয় `/content/proxmax-app/venv`-এ।
- [ ] Model ও media cache `/content`-এ আছে, Drive mount-এ নয়।
- [ ] ASR-এর কথা/সময় reference-এর সঙ্গে মেলে; silence-এ বানানো কথা নেই।
- [ ] অনুবাদের অর্থ, নাম ও সংখ্যা ঠিক আছে; uroman/VITS বাংলা উচ্চারণ করতে পারে।
- [ ] শুরু, মাঝ ও শেষ শুনে দেখুন: কথা সম্পূর্ণ, phrase যথাসময়ে শুরু হয়, জমতে থাকা timing drift নেই।
- [ ] সব দ্রুত উচ্চারণের flag শুনে যাচাই করুন। Flag না থাকলেই অর্থ ঠিক—এমন নয়।
- [ ] ছবি ঠিক আছে; মূল ইংরেজি/গান/পরিবেশের অডিও ইচ্ছাকৃতভাবে বাদ গেছে।
- [ ] Output duration source-এর 250 ms-এর মধ্যে। পাশাপাশি শুনেও যাচাই করুন; এটি lip-sync পরীক্ষা নয়।
- [ ] Drive-এ MP4 চলে, দুই SRT সঠিক UTF-8-এ খোলে, সব artifact-এর hash manifest-এর সঙ্গে মেলে।
- [ ] মূল input অপরিবর্তিত; manifest সবশেষে এসেছে।

## আবার চালানো ও থামানো — বিশেষভাবে যাচাই করুন

- [ ] Runtime delete/restart না করে একই **Setup ও App চালু করুন** cell দ্বিতীয়বার চালান। Setup আবার install করে না; সক্রিয় Drive-এ আবার mount prompt/message আসে না।
- [ ] একই cell আবার চালান। নতুন worker চলে; একই input/settings-এ রাখা ASR/translation ও TTS checkpoint ব্যবহার হয়।
- [ ] ASR বা TTS চলাকালে Stop দিন। Worker ও FFmpeg process আর চলছে না, GPU ব্যবহার কমেছে।
- [ ] Stop-এর পরে একই cell আবার চালান; আগের run-এর অসম্পূর্ণ result-কে সফল দেখায় না।
- [ ] ইচ্ছাকৃত একটি ব্যর্থ কাজের পরে settings/input ঠিক করে আবার দিন; notebook kernel মুছতে হয় না।
- [ ] Drive connection সত্যিই নষ্ট হলে `REMOUNT_DRIVE=True` দিয়ে পুনঃসংযোগ হয়। কেবল “already mounted” দেখে force-remount করতে হয় না।
- [ ] একটি local `translation.json`-এর শুধু `bangla` phrase বদলান। নতুন অডিও হয়; অপরিবর্তিত clip পুনর্ব্যবহার হয়।

## অন্য failure ও সীমা

- [ ] দীর্ঘ অনুবাদে `overflow_policy="error"` দিন। অতিরিক্ত গতি প্রয়োজন হলে error, checkpoint এবং log থাকে; নতুন complete manifest আসে না।
- [ ] একটি corrupt/no-audio input ও একটি valid input দিন। Batch প্রথমটির error জানিয়ে পরেরটি সম্পন্ন করতে পারে।
- [ ] Audio track আছে কিন্তু কথা নেই—এমন ভিডিওতে নীরব output হয়, বানানো dialogue নয়।
- [ ] Space, quote ও বাংলা নামের file এবং non-H.264 input পরীক্ষা করুন।
- [ ] ছোট clip-এ `device="cpu"`, `asr_model="Systran/faster-whisper-small.en"` দিয়ে চালান। CUDA প্রয়োজন হবে না; গতি/মান কমতে পারে।
- [ ] `keep_local=False` দিলে সফল job শুধু প্রকাশের পরে মুছে যায়; ব্যর্থ job থাকে।
- [ ] `/content/proxmax-app/runs/*.log` ও `failure.json` থেকে error খুঁজে পাওয়া যায়। ব্যক্তিগত transcript প্রকাশ করবেন না।

## কী লিখে রাখবেন

ভিডিওর দৈর্ঘ্য, device, model commit, package version, cold/warm/cache অবস্থা, মোট সময় এবং reviewer-এর মতামত লিখুন। Memory মাপলে পদ্ধতি ও সর্বোচ্চ মান দিন। Download বা resume সময়কে পূর্ণ inference সময় বলে দেখাবেন না।

ভালো মূল্যায়নের জন্য ইংরেজি reference-এর বিপরীতে ASR word error, বাংলা অর্থ/স্বাভাবিকতা, দ্রুত phrase-এর অনুপাত এবং হাতে চিহ্নিত speech onset-এর সঙ্গে timing তুলনা করুন।

আসল model-সহ এই পরীক্ষা না চালিয়ে “Colab-এ সম্পূর্ণ যাচাই করা” লিখবেন না। Dependency, model বা Colab image বদলালে আবার পরীক্ষা করুন।
