"""Tarik contoh curhat (tanpa produk) dari PDF "EXPECTED DIALOGUE FROM EMINA GPT"
jadi tabel persona_examples.csv.

Yang diambil:
- 10 blok "Simulasi #N": satu kalimat user + 51 variasi jawaban AI.
- Jawaban "nggak tahu" (bagian "When users ask you something you don't
  understand") — dipasangkan oleh build_chat_dataset.py dengan pertanyaan
  katalog yang memang nggak bisa dijawab (stok, diskon, dll).

Contoh guardrail (self-harm, medis, dst) sengaja TIDAK di-parse dari PDF karena
formatnya nggak konsisten; versi rapinya ditulis tangan di templates_id.py.

Nama "Emina GPT"/"EminaGPT" di jawaban diganti "Mina" supaya konsisten dengan
system prompt. Output ditulis ke dataset/persona_examples.csv dan bisa
di-review/diedit manual sebelum dipakai builder.

Jalankan:
    python dataset/extract_persona_examples.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "EXPECTED DIALOGUE FROM EMINA GPT.pdf"
OUT_PATH = ROOT / "dataset" / "persona_examples.csv"

QUOTE_OPEN = "[“\"]"
QUOTE_CLOSE = "[”\"]"


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" *\n{2,} *", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _tidy(s: str) -> str:
    s = s.strip().strip("“”\"").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("Emina GPT", "Mina").replace("EminaGPT", "Mina")
    return s


def extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    raw = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return _clean_text(raw)


def parse_simulations(text: str) -> list[dict]:
    """Setiap blok 'Simulasi #N: judul' berisi 'User: “...” AI: 1. “...” 2. “...” ...'."""
    rows = []
    headers = list(re.finditer(r"Simulasi #(\d+):\s*([^\n]+)", text))
    for i, h in enumerate(headers):
        sim_no = int(h.group(1))
        title = h.group(2).strip()
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]

        m_user = re.search(rf"User:\s*{QUOTE_OPEN}(.+?){QUOTE_CLOSE}\s*AI:", block, re.DOTALL)
        if not m_user:
            print(f"[warn] Simulasi #{sim_no}: kalimat user tidak ketemu, dilewati", file=sys.stderr)
            continue
        user_msg = _tidy(m_user.group(1))
        answers_block = block[m_user.end():]

        # Jawaban bernomor: 1. “...” 2. “...” — ambil isi di antara kutip.
        seen = set()
        for m in re.finditer(rf"(?:^|\s)(\d{{1,2}})\.\s*{QUOTE_OPEN}(.+?){QUOTE_CLOSE}", answers_block, re.DOTALL):
            ans = _tidy(m.group(2))
            if not ans or len(ans) > 600 or ans.lower() in seen:
                continue
            seen.add(ans.lower())
            rows.append({
                "source": f"simulasi_{sim_no}",
                "topic": title,
                "user": user_msg,
                "assistant": ans,
            })
    return rows


def parse_dont_know(text: str) -> list[dict]:
    """Bagian 'When users ask you something you don't understand then say: 1. “...” ...'."""
    m = re.search(r"don.t understand then say:(.+?)(?:iii\.|Give list verbatim)", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return []
    rows = []
    for a in re.finditer(rf"\d\.\s*{QUOTE_OPEN}(.+?){QUOTE_CLOSE}", m.group(1), re.DOTALL):
        rows.append({"source": "dont_know", "topic": "Tidak tahu jawabannya", "user": "", "assistant": _tidy(a.group(1))})
    return rows


def main() -> None:
    if not PDF_PATH.exists():
        sys.exit(f"PDF tidak ditemukan: {PDF_PATH}")
    text = extract_pdf_text(PDF_PATH)
    rows = parse_simulations(text) + parse_dont_know(text)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "topic", "user", "assistant"])
        w.writeheader()
        w.writerows(rows)

    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"Tersimpan {len(rows)} baris ke {OUT_PATH}")
    for k, v in sorted(by_source.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
