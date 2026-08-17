# Tes diagnostik: subset 5 baris dari Question Answer V1

File pasangannya: **`test-subset-5-rows.csv`** (5 baris `user,assistant`, diambil verbatim dari
`Question Answer V1.csv` — 4 baris Fresh Drop Tint + 1 baris Tone Up Sunscreen, biar bisa langsung
dibandingkan apel-ke-apel dengan hasil yang sudah gagal sebelumnya).

## Tujuan

Isolasi: apakah model BISA menghafal sempurna kalau datanya cuma sedikit (5 baris) + epoch banyak?
- Kalau **bisa** → mekanisme training-nya jalan, tinggal soal skala (LoRA rank / epoch) buat 211 baris penuh.
- Kalau **tetap gagal** (bahkan 5 baris ini) → ada masalah lain di pipeline yang belum ketemu, perlu digali lagi.

## Langkah

1. **Setup tab** → reload model dari awal (fresh, bukan reuse adapter lama).
2. **Setup tab** → di "2. LoRA adapters", set **`r (rank)` = 32** dan **`lora_alpha` = 32** (naik dari default
   8/8). Biarkan `lora_dropout` = 0.00 (jangan dinaikkan — tujuan tes ini justru MAU overfit sempurna ke
   5 baris, dropout akan menghalangi itu), `random_state` dibiarkan default, 3 checkbox
   (attention/MLP/language layers) tetap semua dicentang. Klik "Apply LoRA Adapters".
3. **Data tab** → upload `test-subset-5-rows.csv` — bakal langsung kedetect "✅ sudah format tanya-jawab"
   (kolom `user`/`assistant`), tidak perlu proses konversi apa pun.
4. **Data tab** → "Terapkan chat template & siapkan train/eval set" dengan **eval_ratio = 0** (datanya
   cuma 5 baris, semua harus masuk training, jangan ada yang di-hold-out).
5. **Train tab** → centang "Gunakan num_train_epochs", set ke **20-30 epoch** (aman — datanya cuma 5
   baris jadi tiap epoch cepat banget, dan tujuan tes ini memang menghafal total, bukan generalisasi).
6. **Train tab** → Start Training.
7. **Test tab** → kosongkan system prompt dulu (biar bersih, fokus ke fakta doang), lalu tanya
   pertanyaan-pertanyaan di bawah satu per satu (chat baru / "Bersihkan riwayat chat" dulu supaya
   tidak kebawa konteks giliran sebelumnya) dan bandingkan ke jawaban yang diharapkan.

## Pertanyaan buat ditest & jawaban yang diharapkan

| # | Pertanyaan (persis dari CSV) | Jawaban yang diharapkan (ground truth) |
|---|---|---|
| 1 | Ceritakan secara singkat mengenai Fresh Drop Tint dari Emina ini. | Fresh Drop Tint adalah lip tint dengan **buildable intensity** untuk bibir tampak lebih merata dan sehat sesuai yang kamu inginkan. |
| 2 | Untuk aktivitas apa saja Fresh Drop Tint cocok digunakan? | Fresh Drop Tint cocok untuk kegiatan seperti **sekolah, Pilates, dan Hot mat yoga** karena merupakan lip tint paling natural. |
| 3 | Berapa harga dari Fresh Drop Tint? | Harga Fresh Drop Tint adalah **50000.0**. |
| 4 | Sebutkan beberapa keunggulan utama yang dimiliki oleh Fresh Drop Tint. | Keunggulan Fresh Drop Tint meliputi **tahan stain hingga 12 jam**, effortless chic look in one swipe, satset natural glow for lip & cheek, serta ringan seperti tidak memakai apa-apa. |
| 5 | Ceritakan secara singkat mengenai Tone Up Sunscreen SPF 50 PA++++ dari Emina! | Tone Up Sunscreen SPF 50 PA++++ dari Emina adalah produk kategori Skincare, specialty **CERAH INSTANT NATURAL TANPA ABU-ABU**, memberikan perlindungan maksimal harian. |

Bonus check (opsional, tapi informatif): tanya juga pertanyaan yang **beda kalimat** tapi maksudnya
sama, mis. "apa itu Fresh Drop Tint" (bukan "Ceritakan secara singkat mengenai...") — kalau versi
persis-kalimat BENAR tapi versi parafrase SALAH, itu baru soal generalisasi/variasi kalimat (fix-nya:
augmentasi parafrase). Kalau versi persis-kalimat pun masih SALAH, itu murni soal
kapasitas/konvergensi training (fix-nya: rank/epoch), bukan soal variasi kalimat.

## Cara baca hasilnya

- **Semua 5 jawaban persis/dekat sekali dengan ground truth** → training mechanism sehat. Kembali ke
  211 baris penuh dengan rank yang sudah dinaikkan (dan epoch cukup, mis. 8-15) — kemungkinan besar
  akan membaik signifikan.
- **Ada yang masih meleset walau cuma 5 baris** → kasih tahu saya baris mana yang gagal + jawaban
  yang keluar. Itu petunjuk kuat masih ada bug pipeline (bukan cuma soal skala), dan saya perlu lihat
  training log per-step buat gali lebih dalam.
