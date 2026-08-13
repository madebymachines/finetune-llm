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
   - Toggle adapter (base vs hasil finetune), parameter generation sesuai rekomendasi Gemma-4
     (`temperature=1.0, top_p=0.95, top_k=64`). Field "System prompt" murni opsional untuk
     eksperimen manual saat testing — bukan bagian dari data training.
5. **📈 Evaluate** — **Before vs After**: bandingkan output base model vs hasil finetune pada
   prompt/gambar/audio yang sama (adapter di-nonaktifkan sementara via `model.disable_adapter()`,
   tanpa perlu load model dua kali), hasil bisa diunduh sebagai CSV.

## Struktur

```
app.py                   # UI Streamlit (semua tab)
src/constants.py          # daftar model, chat template, default per modalitas (LoRA & SFT)
src/gpu_utils.py           # cek CUDA & memory stats, pembersih cache CUDA
src/data_utils.py          # load dataset HF / upload custom / template builder / builder Vision & Audio
src/train_utils.py         # load model (modality-aware), LoRA, SFTTrainer, callback progress
src/eval_utils.py          # streaming chat, before-vs-after compare (modality-aware)
```

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
