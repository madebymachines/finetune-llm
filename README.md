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
   - **Text** — dipecah jadi dua sub-tab yang terpisah karena tujuannya beda:
     - **🎭 Persona & Guardrail (Training)**: data yang benar-benar di-**finetune** ke model
       (gaya bicara & batasan aman/tidak aman). Dataset HF Hub (default
       `mlabonne/FineTome-100k`), atau upload dokumen persona/rule (PDF/DOCX/TXT) — kalau
       dokumennya berisi contoh dialog berpola `User: "..." AI: "..."` (termasuk beberapa
       variasi jawaban bernomor untuk satu pesan user), contoh itu **otomatis ditarik**
       jadi baris percakapan training di tabel yang bisa diedit langsung di UI (tambah/hapus
       baris, atau isi manual kalau tidak ada dokumen sama sekali) sebelum dipakai — jadi
       tidak perlu bikin file CSV terpisah kalau contohnya sudah ada di dalam dokumen. Upload
       file percakapan siap pakai (CSV/Excel/JSON/JSONL, kolom `conversations` atau
       `user`+`assistant`+`system`) tetap tersedia sebagai jalur alternatif untuk yang sudah
       punya spreadsheet.
     - **📦 Knowledge Base (Katalog / Chat with Document)**: data yang **tidak** di-training —
       upload katalog produk (CSV/Excel/JSON) dan/atau dokumen apapun (PDF/DOCX/TXT/MD),
       bisa banyak sekaligus. Setiap sumber otomatis dipecah jadi potongan-potongan kecil
       (baris tabel, atau paragraf dokumen), lalu diindeks dengan embedding multilingual
       (`intfloat/multilingual-e5-small`, termasuk Bahasa Indonesia). Saat chat di tab
       Test/Evaluate, potongan yang paling relevan dengan pertanyaan diambil otomatis
       (cosine similarity, top-k) dan disisipkan ke context — jadi jawabannya selalu
       berdasarkan data terbaru tanpa training ulang, dan tetap akurat walau pertanyaannya
       diparafrase.
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
     (`temperature=1.0, top_p=0.95, top_k=64`).
   - Kalau Knowledge Base sudah dibangun di tab Data, ada toggle "Sertakan Knowledge Base" —
     tiap pertanyaan otomatis diambilkan potongan paling relevan (expander "🔍 Konteks
     Knowledge Base yang dipakai" menampilkan potongan mana saja yang disisipkan).
5. **📈 Evaluate**
   - **Eval loss / perplexity**: `trainer.evaluate()` pada eval split, dicatat per-label
     (mis. jalankan sebelum & sesudah training untuk dibandingkan).
   - **Before vs After**: bandingkan output base model vs hasil finetune pada prompt/gambar/audio
     yang sama (adapter di-nonaktifkan sementara via `model.disable_adapter()`, tanpa perlu load
     model dua kali), hasil bisa diunduh sebagai CSV.

## Struktur

```
app.py                   # UI Streamlit (semua tab)
src/constants.py          # daftar model, chat template, default per modalitas (LoRA & SFT), default Knowledge Base
src/gpu_utils.py           # cek CUDA & memory stats
src/data_utils.py          # load dataset HF / upload custom / template builder / builder Vision & Audio / chunking KB
src/retrieval.py           # embedding (sentence-transformers) + cosine similarity retrieval untuk Knowledge Base
src/train_utils.py         # load model (modality-aware), LoRA, SFTTrainer, callback progress
src/eval_utils.py          # eval loss/perplexity, streaming chat, before-vs-after (modality-aware)
```

## Catatan desain

- Tab Export (save/merge/GGUF/push-to-hub) sengaja tidak ada di v1 — training dijalankan
  langsung di tool ini, tapi kalau butuh export model, notebook Unsloth aslinya sudah
  punya cell untuk itu; bisa ditambahkan kembali ke tool ini kalau memang dibutuhkan.
- Dokumen naratif persona/rule (PDF/DOCX/TXT) di-parse otomatis (regex best-effort, lihat
  `parse_user_ai_examples` di `src/data_utils.py`) mencari pola `User: "..." AI: "..."` —
  hasilnya masuk ke tabel yang bisa diedit di UI, bukan langsung jadi dataset final, karena
  parsing seperti ini tidak dijamin cocok untuk semua format dokumen. Kalau dokumenmu pakai
  format lain yang tidak kena pola ini, isi tabelnya manual atau upload spreadsheet siap
  pakai. Untuk dokumen naratif sebagai referensi/katalog (bukan contoh dialog training),
  upload saja langsung ke Knowledge Base — di situ dokumen memang diproses apa adanya
  (di-chunk, bukan di-parse jadi Q&A).
- Knowledge Base pakai retrieval brute-force (numpy cosine similarity, exact — bukan
  approximate), bukan FAISS/Chroma: cukup cepat & akurat untuk skala katalog produk biasa
  (ratusan–ribuan baris/potongan), tanpa dependency vector-DB tambahan. Model embedding-nya
  jalan di CPU supaya tidak berebut VRAM dengan LLM 4-bit yang sedang di-load.
