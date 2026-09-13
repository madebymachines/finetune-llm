# Gemma-4 Finetune Studio

Streamlit tool untuk run, test, dan evaluate LoRA finetune Gemma-4 (E2B/E4B/31B/26B-A4B) —
**Text, Vision, dan Audio** — diadaptasi dari Unsloth's Gemma4 (E4B) Text/Vision/Audio notebooks.

## Requirement

- **GPU NVIDIA (CUDA)**. Unsloth memakai bitsandbytes 4-bit + xformers/triton yang CUDA-only.
  App ini tidak bisa training di CPU/Apple Silicon — cocok dijalankan di Colab, RunPod, Lambda,
  atau server GPU on-prem.

## Instalasi

```bash
pip install -r requirements.txt
```

Jika `torch` belum terpasang sesuai versi CUDA di mesinmu, install dulu torch yang sesuai
sebelum menjalankan `pip install -r requirements.txt` (lihat https://pytorch.org untuk perintah
yang cocok dengan driver CUDA-mu).

## Menjalankan

```bash
streamlit run app.py
```

## Alur pemakaian

1. **⚙️ Setup**
   - Pilih **modalitas**: Text / Vision / Audio — menentukan loader model (`FastModel` vs
     `FastVisionModel`), default LoRA (r/alpha), dan target modules yang dipakai.
   - Pilih/masukkan model Gemma-4, load model, lalu apply LoRA adapters.
2. **📊 Data** — tergantung modalitas:
   - **Text**: satu alur — upload **data custom apa pun** (format bebas: CSV/Excel/JSON/JSONL
     data percakapan siap pakai, tabel data mentah seperti katalog produk yang perlu dipetakan
     jadi tanya-jawab, atau dokumen PDF/DOCX/TXT/MD) dan diubah jadi dataset percakapan
     `user`/`assistant` (format ShareGPT — persis seperti `FineTome-100k` di notebook Unsloth,
     **tanpa** role `system`). Auto-detect: kolom `conversations` atau `user`+`assistant`
     langsung dipakai; tabel dengan kolom lain (mis. `nama_produk`/`deskripsi`/`harga`) perlu
     dipetakan manual lewat Template Builder (`{nama_kolom}` placeholder); dokumen dengan pola
     `User: "..." AI: "..."` di-auto-extract ke tabel yang bisa diedit langsung di UI sebelum
     dipakai. Atau pakai dataset HF Hub (default `mlabonne/FineTome-100k`).
     **Untuk chatbot katalog produk, jalur yang direkomendasikan adalah dataset builder di
     `dataset/` (lihat bagian "Dataset builder" di bawah)**: hasilnya `train.jsonl` dengan kolom
     `conversations` (termasuk role `system`) yang tinggal di-upload di tab ini.
   - **Vision**: dataset HF Hub dengan kolom gambar+teks (default `unsloth/LaTeX_OCR`), atau
     upload banyak gambar + CSV/Excel berisi nama file, pertanyaan (opsional), dan jawaban.
   - **Audio**: dataset HF Hub dengan kolom audio+transkrip (default
     `kadirnar/Emilia-DE-B000000`), atau upload banyak file audio + CSV/Excel transkrip.
   - Untuk Text, chat template Gemma-4 diterapkan otomatis; untuk Vision/Audio, dataset
     langsung dalam format `messages` yang dipakai `UnslothVisionDataCollator`.
3. **🚀 Train** — atur SFT config (default per modalitas mengikuti notebook Unsloth: Text pakai
   `warmup_steps`+`linear` scheduler, Vision/Audio pakai `warmup_ratio`+`cosine` scheduler +
   `max_length`), lalu jalankan training dengan progress bar & loss chart real-time. Ada opsi
   simpan LoRA adapter lokal setelah training selesai.
4. **💬 Test**
   - Text: chat multi-turn interaktif (streaming token-by-token).
   - Vision: upload satu gambar + pertanyaan, lihat jawaban model.
   - Audio: upload satu file audio + pertanyaan (mis. transkripsi), lihat jawaban model.
   - Toggle adapter (base vs hasil finetune). Default `temperature=0.3` (bukan 1.0 rekomendasi
     Gemma-4) karena untuk chatbot katalog akurasi fakta lebih penting daripada variasi; naikkan
     lewat slider kalau perlu. Field "System prompt" default berisi `dataset/system_prompt_short.txt`,
     yaitu system prompt yang **sama** dengan yang ditanam di data training oleh dataset builder —
     jangan diganti persona lain saat test, mismatch system prompt bikin jawaban ngaco.
5. **📈 Evaluate** — **Before vs After**: bandingkan output base model vs hasil finetune pada
   prompt/gambar/audio yang sama (adapter di-nonaktifkan sementara via `model.disable_adapter()`,
   tanpa perlu load model dua kali), hasil bisa diunduh sebagai CSV. Untuk Text, upload
   `dataset/eval_realistic.csv` (prompt gaya user asli yang tidak ada di training, termasuk
   multi-turn lewat kolom `history`) supaya skor cek-fakta-nya jujur; eval split acak dari template
   yang sama cenderung terlalu optimis.

## Struktur

```
app.py                   # UI Streamlit (semua tab)
src/constants.py          # daftar model, chat template, default per modalitas (LoRA & SFT)
src/gpu_utils.py           # cek CUDA & memory stats, pembersih cache CUDA
src/data_utils.py          # load dataset HF / upload custom / template builder / builder Vision & Audio
src/train_utils.py         # load model (modality-aware), LoRA, SFTTrainer, callback progress
src/eval_utils.py          # streaming chat, before-vs-after compare (modality-aware)
dataset/                   # dataset builder untuk chatbot katalog (lihat bagian di bawah)
```

## Dataset builder (Text, chatbot katalog produk)

Latar belakang: dataset tanya-jawab yang jawabannya nilai kolom mentah (`"Emina"`, `"Deco"`,
`"50000.0"`) bikin model hafal fakta, tapi nggak bisa merespons curhat ("aku bete, kulitku
kering") dengan rekomendasi, dan gaya jawabnya jadi pendek/kaku. Builder ini membangun dataset
percakapan yang lebih lengkap dari katalog + kurasi manual:

```
dataset/
  build_chat_dataset.py       # builder: katalog + mapping + persona -> out/train.jsonl, val.jsonl
  templates_id.py             # cara nanya & cara ngomong (tanpa fakta apa pun)
  extract_persona_examples.py # parse PDF simulasi curhat -> persona_examples.csv
  system_prompt_short.txt     # system prompt pendek, dipakai SAMA di training & test
  needs_mapping.csv           # KURASI MANUAL: keluhan -> produk -> alasan (sumber rekomendasi)
  persona_examples.csv        # hasil extract PDF (curhat tanpa produk + jawaban "nggak tahu")
  eval_realistic.csv          # set evaluasi: prompt gaya user asli, tidak ada di training
  out/                        # hasil build: train.jsonl, val.jsonl, stats.json, preview.md
```

Alur:

```bash
python dataset/extract_persona_examples.py   # hanya kalau PDF simulasinya berubah
python dataset/build_chat_dataset.py         # baca products_*.xlsx terbaru di root repo
```

Lalu di app: **Data** → upload `dataset/out/train.jsonl` → "Gunakan data ini" → terapkan chat
template. **Train** → sebelum training ada expander "Cek bagian yang dilatih": pastikan yang masuk
loss cuma jawaban asisten (system prompt + pesan user harus di bagian DI-MASK). **Test** → biarkan
system prompt default. **Evaluate** → upload `dataset/eval_realistic.csv`.

Jenis baris yang dihasilkan (lihat `stats.json`): fakta per produk (jawaban kalimat natural,
harga terformat `Rp50.000`, jujur kalau kolom kosong), daftar produk per kategori, curhat →
empati + rekomendasi + alasan (+harga) + pertanyaan lanjutan, multi-turn ("harganya berapa?"
setelah produk disebut), curhat murni tanpa produk (dari PDF + topik tambahan di
`templates_id.py` seperti putus cinta, insecure penampilan, burnout), jembatan curhat → user pivot
ke kulit/penampilan → rekomendasi ringan (multi-turn, produk tidak pernah ditawarkan di giliran
curhat pertama), guardrail (self-harm, medis, ilegal, manipulasi), pertanyaan yang memang nggak
bisa dijawab (stok/diskon), dan produk di luar katalog (jujur nggak ada). Kolom katalog yang kosong
(kandungan, keunggulan, cocok untuk) dicoba diambil dari teks `description` dulu sebelum dijawab
"belum ada data"; harga tidak punya fallback. Semua fakta selalu dari katalog atau `needs_mapping.csv`, nggak ada
yang di-generate LLM, jadi nggak bisa ngarang produk.

Yang perlu dirawat manual: `needs_mapping.csv` (kolom `alasan` harus tetap sesuai deskripsi
katalog; kolom `reviewed` buat menandai baris yang sudah dicek), `system_prompt_short.txt`
(kalau diubah, build ulang dataset DAN pakai versi yang sama saat test), dan `eval_realistic.csv`.
Kalau katalog diperbarui, cukup taruh `products_<tanggal>.xlsx` baru di root dan build ulang;
nama produk di `needs_mapping.csv` harus persis sama dengan kolom `name` (builder menolak kalau
ada yang nggak cocok).

## Catatan desain

- Tab Export (save/merge/GGUF/push-to-hub) sengaja tidak ada di v1 — training dijalankan
  langsung di tool ini, tapi kalau butuh export model, notebook Unsloth aslinya sudah
  punya cell untuk itu; bisa ditambahkan kembali ke tool ini kalau memang dibutuhkan.
- Data training **selalu** murni percakapan `user`/`assistant` — tidak ada role `system` yang
  ikut di-training, persis mengikuti format `FineTome-100k` di notebook referensi. Semua yang
  perlu diketahui/dilakukan model (persona, batasan, fakta produk) harus direpresentasikan
  sebagai baris `user`→`assistant` yang eksplisit di data training, bukan lewat system prompt
  atau retrieval saat inferensi — tool ini tidak punya fitur RAG/knowledge-base terpisah.
- Dokumen naratif (PDF/DOCX/TXT) di-parse otomatis (regex best-effort, lihat
  `parse_user_ai_examples` di `src/data_utils.py`) mencari pola `User: "..." AI: "..."` —
  hasilnya masuk ke tabel yang bisa diedit di UI, bukan langsung jadi dataset final, karena
  parsing seperti ini tidak dijamin cocok untuk semua format dokumen. Kalau dokumenmu pakai
  format lain yang tidak kena pola ini, isi tabelnya manual atau upload spreadsheet siap pakai.
