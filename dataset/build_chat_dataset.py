"""Bangun dataset percakapan (format ShareGPT `conversations`, role/content) untuk
finetune Gemma-4 Text via Unsloth, dari:

  products_*.xlsx           katalog produk (sumber SEMUA fakta)
  dataset/needs_mapping.csv keluhan/kebutuhan -> produk -> alasan (kurasi manual)
  dataset/persona_examples.csv  contoh curhat tanpa produk (dari PDF simulasi)
  dataset/system_prompt_short.txt  system prompt yang SAMA dipakai training & test
  dataset/templates_id.py   cara nanya / cara ngomong (tanpa fakta)

Output (folder dataset/out/):
  train.jsonl, val.jsonl   satu baris = {"conversations": [{"role","content"}, ...]}
  stats.json               jumlah baris per jenis, panjang karakter, dsb.
  preview.md               beberapa contoh per jenis buat dicek manusia

Jenis baris yang dibuat (kolom "kind" di stats):
  identity        siapa Mina
  fact            fakta per produk, jawaban kalimat natural
  category_list   daftar produk per kategori
  need_rec        curhat/keluhan -> empati + rekomendasi + alasan (+harga) + follow-up
  need_rec_multi  need_rec lalu giliran ke-2 nanya harga/kandungan/... pakai "nya"
  fact_multi      definisi produk lalu giliran ke-2 nanya detail pakai "nya"
  persona         curhat murni tanpa produk (PDF)
  guardrail       penolakan aman (self-harm, medis, dst)
  dont_know       pertanyaan katalog yang nggak bisa dijawab (stok, diskon, ...)
  vague           pesan user yang nggak jelas -> minta diperjelas (jawaban PDF)
  out_of_catalog  produk yang nggak ada di katalog -> jujur
  emotional       curhat topik di luar PDF (putus, insecure, burnout, ...) tanpa produk
  bridge          curhat -> user pivot ke kulit/penampilan -> rekomendasi ringan (multi-turn)

Kolom katalog yang kosong TIDAK langsung dijawab "nggak ada data": kandungan,
keunggulan, dan "cocok untuk" dicoba diambil dulu dari teks description
(bagian "Kandungan Utama"/"Specially Made With"/"Keunggulan:"/"Cocok Untuk:",
atau kalimat "Dengan/Mengandung/Diformulasikan dengan ..."), dikutip apa adanya.
Harga tidak punya fallback: kalau kolom price kosong, jawabannya jujur.

Jalankan:
    python dataset/build_chat_dataset.py
    python dataset/build_chat_dataset.py --catalog products_2026-08-10.xlsx --val-ratio 0.05 --seed 3407
Opsi --no-system: tanpa role system (kalau mau eksperimen tanpa system prompt,
tapi ingat: saat test pun field system prompt harus dikosongkan).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import templates_id as T  # noqa: E402

OUT_DIR = HERE / "out"

# Kolom katalog -> intent fakta
FIELD_BY_INTENT = {
    "definisi": "description",
    "harga": "price",
    "kategori": "category",
    "keunggulan": "product_specialty",
    "kandungan": "ingredients",
    "shades": "shades",
    "cara_pakai": "cara_pakai",
    "aktivitas": "activity_related",
    "merek": "brand",
    "detail": "description",
}
# Intent yang, kalau kolomnya kosong, tetap dibuat barisnya dengan jawaban jujur "nggak ada info".
ASK_EVEN_IF_EMPTY = {"harga", "shades", "kandungan"}
MAX_ANSWER_CHARS = 900          # jaga-jaga max_seq_length 1024 token
MAX_INGREDIENT_ITEMS = 12       # daftar INCI panjang dipotong, disebut "dan lainnya"


# ---------------------------------------------------------------------------
# Pembersih teks katalog
# ---------------------------------------------------------------------------
def clean_text(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("⁠", "").replace("�", "").replace("^", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


_MARKETING_PREFIX = re.compile(
    r"^(?:about\s+[^:]{0,80}:\s*|new formula!?\s*|new!?\s*|baru!?\s*)+", re.IGNORECASE
)


def first_paragraph(desc: str, max_chars: int = 380) -> str:
    """Ambil paragraf pertama deskripsi (sebelum bullet 'Keunggulan:' dll),
    buang prefix marketing ("About X✨:", "NEW!"), dipotong di batas kalimat."""
    desc = clean_text(desc)
    para = re.split(r"\n\s*\n|\n(?=[A-Z][^\n]{0,30}:)", desc)[0].strip()
    lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
    # Baris-baris pendek = bullet marketing ("5 Seconds for EVEN blend", "BLURRING ..."):
    # gabung pakai "; " supaya nggak jadi satu kalimat aneh. Baris panjang = kalimat biasa.
    joiner = "; " if len(lines) > 1 and all(len(ln) < 90 for ln in lines) else " "
    para = joiner.join(lines).replace("✨", "")
    para = _MARKETING_PREFIX.sub("", para).strip()
    if len(para) <= max_chars:
        return para
    cut = para[:max_chars]
    m = re.search(r"^(.+[.!?])\s", cut)
    return (m.group(1) if m else cut.rsplit(" ", 1)[0] + "…").strip()


def lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


_BULLET_PREFIX = re.compile(r"^[^\w(]+")  # emoji/bullet/angka di awal baris


def _looks_like_header(line: str) -> bool:
    ln = line.strip()
    return bool(ln) and len(ln) < 40 and not ln.endswith((",", ".", "!", ";")) and ":" not in ln and not _BULLET_PREFIX.match(ln)


def description_section(desc: str, headers: list[str], max_chars: int = 600) -> str | None:
    """Ambil isi bagian ber-judul di description, mis. 'Keunggulan:' / 'Kandungan
    Utama' / 'Cocok Untuk:'. Berhenti di baris kosong, di judul berikutnya
    ('Xxx:' atau baris pendek tanpa tanda baca), atau di batas max_chars."""
    desc = clean_text(desc)
    lines = desc.split("\n")
    hdr = re.compile(r"^\s*(?:%s)\s*:?\s*(.*)$" % "|".join(re.escape(h) for h in headers), re.IGNORECASE)
    for i, ln in enumerate(lines):
        m = hdr.match(ln)
        if not m:
            continue
        out = [m.group(1).strip()] if m.group(1).strip() else []
        for nxt in lines[i + 1:]:
            t = nxt.strip()
            if not t:
                if out:
                    break
                continue
            if re.match(r"^[A-Za-z][^\n]{0,40}:\s*$", t) or (out and _looks_like_header(t)):
                break
            out.append(_BULLET_PREFIX.sub("", t) if len(t) < 200 else t)
        text = "\n".join(x for x in out if x)
        text = text.replace("✨", "").strip()
        if text:
            return text if len(text) <= max_chars else text[:max_chars].rsplit("\n", 1)[0].rstrip(",;") + "\n(…)"
    return None


_INGREDIENT_KEYWORDS = [
    "mengandung", "kandungan", "diperkaya", "dilengkapi dengan", "diformulasikan dengan",
    "formulated with", "infused with", "dengan ",
]


def ingredient_sentence(desc: str) -> str | None:
    """Kalimat di description yang menyebut bahan ('Dengan Oat Amino, Advanced
    Ceramide ... menjadikan kulit ...'), dikutip utuh — bukan di-parse jadi
    daftar, supaya nggak salah potong."""
    desc = clean_text(desc).replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", desc)
    for kw in _INGREDIENT_KEYWORDS:
        for s in sentences:
            low = s.lower().strip()
            if not (20 <= len(s) <= 300):
                continue
            # "dengan" terlalu umum ("pembersih dengan format gel") — hanya terima
            # kalau kalimatnya DIMULAI "Dengan <bahan>, ..." (pola khas daftar bahan).
            hit = low.startswith("dengan ") if kw == "dengan " else kw in low
            if hit:
                return s.strip()
    return None


def format_price(v) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return str(v)
    return "Rp" + f"{n:,}".replace(",", ".")


def format_ingredients(s: str) -> str:
    items = [i.strip() for i in clean_text(s).split(",") if i.strip()]
    if len(items) > MAX_INGREDIENT_ITEMS:
        return ", ".join(items[:MAX_INGREDIENT_ITEMS]) + ", dan lainnya"
    return ", ".join(items)


def format_specialty(s: str) -> str:
    s = clean_text(s)
    s = re.sub(r"\s*\|\s*", "; ", s)
    s = s.replace("\n", " ")
    # Teks katalog kadang ALL CAPS; rapikan supaya nggak kayak teriak.
    if sum(1 for c in s if c.isupper()) > 0.6 * max(1, sum(1 for c in s if c.isalpha())):
        s = s.lower().capitalize()
    return s.rstrip(".")


def definisi_value(name: str, category: str, desc: str) -> str:
    """Deskripsi yang selalu diawali nama produk, supaya template jawaban
    tinggal '{value}'. Kalau deskripsi katalog sudah mulai dengan 'Emina ...'
    atau nama produknya, dipakai apa adanya; kalau nggak, dibuka dengan
    '{name} itu produk {kategori} dari Emina.' (kategori = kolom category, fakta
    katalog juga) lalu deskripsinya."""
    para = first_paragraph(desc)
    short_words = name.replace("Emina ", "").lower().split()[:2]
    head = para.lower()
    if head.startswith("emina ") or head.startswith(name.lower()) or head.startswith(" ".join(short_words)):
        return para
    cat = f" produk {category.lower()}" if category else " produk"
    return f"{name} itu{cat} dari Emina. {para}"


def full_description(desc: str, max_chars: int = MAX_ANSWER_CHARS) -> str:
    """Seluruh description (bukan cuma paragraf pertama): bullet dirapikan,
    prefix marketing dibuang, dipotong di batas baris kalau kepanjangan."""
    desc = clean_text(desc).replace("✨", "")
    lines = []
    for ln in desc.split("\n"):
        t = ln.strip()
        if not t:
            continue
        t = _BULLET_PREFIX.sub("- ", t) if _BULLET_PREFIX.match(t) and not re.match(r"^\d+[.)]", t) else t
        lines.append(t)
    text = "\n".join(lines)
    text = _MARKETING_PREFIX.sub("", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0].rstrip(",;") + "\n(…selengkapnya di kemasan/official store)"


def format_value(intent: str, raw, name: str = "", category: str = "", group: str = "") -> str | None:
    s = clean_text(raw)
    if intent == "harga":
        return format_price(raw)
    if not s:
        return None
    if intent == "definisi":
        return definisi_value(name, category, s)
    if intent == "detail":
        return full_description(s)
    if intent == "kategori":
        return f"{s} ({group.lower()})" if group else s
    if intent == "kandungan":
        return format_ingredients(s)
    if intent == "keunggulan":
        return format_specialty(s)
    if intent == "cara_pakai":
        s = clean_text(s)
        return s if len(s) <= MAX_ANSWER_CHARS else s[:MAX_ANSWER_CHARS].rsplit("\n", 1)[0] + "\n(…lanjutannya bisa dicek di kemasan ya)"
    if intent == "aktivitas":
        return lower_first(s.replace("\n", " "))
    return s.replace("\n", " ")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class Builder:
    def __init__(self, catalog: pd.DataFrame, needs: pd.DataFrame, persona: pd.DataFrame,
                 system_prompt: str | None, rng: random.Random):
        self.catalog = catalog
        self.needs = needs
        self.persona = persona
        self.system_prompt = system_prompt
        self.rng = rng
        self.rows: list[dict] = []
        self.products = {clean_text(r["name"]): r for _, r in catalog.iterrows() if clean_text(r["name"])}
        self.by_category: dict[str, list[str]] = defaultdict(list)
        for name, r in self.products.items():
            self.by_category[clean_text(r["category"])].append(name)
        skincare = sorted({clean_text(r["category"]) for r in self.products.values() if clean_text(r["group_category"]) == "Skincare"})
        deco = sorted({clean_text(r["category"]) for r in self.products.values() if clean_text(r["group_category"]) == "Deco"} - set(skincare))
        self.skincare_cats = ", ".join(c.lower() for c in skincare)
        self.deco_cats = ", ".join(c.lower() for c in deco)

    # -- helpers ------------------------------------------------------------
    def pick(self, seq):
        return self.rng.choice(seq)

    def add(self, kind: str, turns: list[tuple[str, str]], meta: dict | None = None):
        convo = []
        if self.system_prompt:
            convo.append({"role": "system", "content": self.system_prompt})
        for user, assistant in turns:
            convo.append({"role": "user", "content": user.strip()})
            convo.append({"role": "assistant", "content": assistant.strip()})
        self.rows.append({"kind": kind, "conversations": convo, "meta": meta or {}})

    def fact_answer(self, name: str, intent: str) -> str | None:
        """Jawaban fakta untuk (produk, intent); None kalau kolom kosong dan
        intent bukan yang wajib ditanya."""
        row = self.products[name]
        value = format_value(
            intent, row.get(FIELD_BY_INTENT[intent]), name=name,
            category=clean_text(row.get("category")), group=clean_text(row.get("group_category")),
        )
        if value is None:
            # Kolom kosong -> coba gali dari teks description dulu (dikutip apa adanya).
            desc = clean_text(row.get("description"))
            if intent == "kandungan" and desc:
                sec = description_section(desc, ["Kandungan Utama", "Kandungan", "Specially Made With", "Key Ingredients"])
                if sec:
                    return self.pick(T.FACT_ANSWERS["kandungan"]).format(name=name, value=sec.replace(".\n", "\n").replace("\n", "; "))
                sent = ingredient_sentence(desc)
                if sent:
                    return self.pick(T.FACT_ANSWERS["kandungan_dari_deskripsi"]).format(name=name, value=sent)
            if intent == "keunggulan" and desc:
                sec = description_section(desc, ["Keunggulan Produk", "Keunggulan", "What You'll Love", "What Youll Love", "Manfaat Produk", "Manfaat"])
                if sec:
                    return self.pick(T.FACT_ANSWERS["keunggulan"]).format(name=name, value=sec.replace("\n", "; "))
            if intent == "aktivitas" and desc:
                sec = description_section(desc, ["Cocok Untuk", "Suitable for", "Tipe kulit", "Sunscreen ini untuk kamu yang"])
                if sec:
                    return self.pick(T.FACT_ANSWERS["aktivitas"]).format(name=name, value=lower_first(sec.replace("\n", ", ")))
            if intent == "harga":
                return self.pick(T.FACT_ANSWERS["harga_tidak_ada"]).format(name=name)
            if intent in ASK_EVEN_IF_EMPTY:
                return self.pick(T.FACT_ANSWERS["tidak_ada_info"]).format(name=name, field=T.FIELD_LABEL.get(intent, intent))
            return None
        return self.pick(T.FACT_ANSWERS[intent]).format(name=name, value=value)

    # -- generators ---------------------------------------------------------
    def gen_identity(self, repeat: int = 2):
        for _ in range(repeat):
            for q, a in T.IDENTITY_QA:
                self.add("identity", [(q, a)])

    def gen_facts(self, questions_per_intent: int = 3):
        for name in self.products:
            for intent, q_templates in T.FACT_QUESTIONS.items():
                answer = self.fact_answer(name, intent)
                if answer is None:
                    continue
                n_q = 1 if intent == "detail" else questions_per_intent
                qs = self.rng.sample(q_templates, min(n_q, len(q_templates)))
                for q in qs:
                    # Jawaban di-resample per pertanyaan supaya gaya kalimatnya ikut bervariasi.
                    self.add("fact", [(q.format(name=name), self.fact_answer(name, intent))], {"product": name, "intent": intent})

    def gen_category_lists(self):
        for cat, names in self.by_category.items():
            if len(names) < 2:
                continue
            names_str = ", ".join(names)
            for q in T.CATEGORY_QUESTIONS:
                a = self.pick(T.CATEGORY_ANSWERS).format(category=cat, names=names_str)
                self.add("category_list", [(q.format(category=cat.lower()), a)], {"category": cat})

    def _rec_answer(self, need_row, two: pd.DataFrame | None = None) -> str:
        empathy = self.pick(need_row["empati_list"])
        product = need_row["product"]
        price = format_price(self.products[product].get("price")) if product in self.products else None
        if two is not None and len(two) >= 2:
            r1, r2 = two.iloc[0], two.iloc[1]
            recommend = self.pick(T.RECOMMEND_TWO_FRAMES).format(
                product1=r1["product"], reason1=lower_first(r1["alasan"]),
                product2=r2["product"], reason2=lower_first(r2["alasan"]),
            )
            price_note = ""
        else:
            recommend = self.pick(T.RECOMMEND_FRAMES).format(product=product, reason=lower_first(need_row["alasan"]))
            note = self.pick(T.PRICE_NOTES)
            price_note = note.format(price=price) if (price and note) else ""
        layout = self.pick(T.ASSISTANT_REC_LAYOUTS)
        text = layout.format(empathy=empathy, recommend=recommend, price_note=price_note, followup=self.pick(T.FOLLOWUPS))
        return re.sub(r"[ ]{2,}", " ", text).replace(" \n", "\n").strip()

    def gen_need_recs(self, frames_per_example: int = 3, multi_turn_ratio: float = 0.35):
        for need_id, grp in self.needs.groupby("need_id", sort=False):
            grp = grp.sort_values("prioritas")
            examples = grp.iloc[0]["user_examples_list"]
            for complaint in examples:
                frames = self.rng.sample(T.COMPLAINT_FRAMES, min(frames_per_example, len(T.COMPLAINT_FRAMES)))
                for frame in frames:
                    user = frame.format(complaint=complaint)
                    # Prioritas 1 paling sering, sisanya kadang-kadang; sesekali dua produk sekaligus.
                    roll = self.rng.random()
                    if roll < 0.15 and len(grp) >= 2:
                        answer, chosen = self._rec_answer(grp.iloc[0], two=grp), grp.iloc[0]
                    elif roll < 0.65 or len(grp) == 1:
                        chosen = grp.iloc[0]
                        answer = self._rec_answer(chosen)
                    else:
                        chosen = grp.iloc[self.rng.randrange(1, len(grp))]
                        answer = self._rec_answer(chosen)
                    turns = [(user, answer)]
                    kind = "need_rec"
                    if self.rng.random() < multi_turn_ratio and chosen["product"] in self.products:
                        intent = self.pick(list(T.REC_FOLLOWUP_TURNS))
                        follow_a = self.fact_answer(chosen["product"], intent)
                        if follow_a:
                            turns.append((self.pick(T.REC_FOLLOWUP_TURNS[intent]), follow_a))
                            kind = "need_rec_multi"
                    self.add(kind, turns, {"need_id": need_id, "product": chosen["product"]})

    def gen_fact_multiturn(self, per_product: int = 3):
        intents = [i for i in T.MULTITURN_FOLLOWUPS if i != "definisi"]
        for name in self.products:
            for _ in range(per_product):
                q1 = self.pick(T.FACT_QUESTIONS["definisi"]).format(name=name)
                a1 = self.fact_answer(name, "definisi")
                if not a1:
                    continue
                turns = [(q1, a1)]
                for intent in self.rng.sample(intents, 2):
                    a = self.fact_answer(name, intent)
                    if a:
                        turns.append((self.pick(T.MULTITURN_FOLLOWUPS[intent]), a))
                if len(turns) >= 2:
                    self.add("fact_multi", turns, {"product": name})

    def gen_persona(self, answers_per_sim: int = 12):
        sims = self.persona[self.persona["source"].str.startswith("simulasi_")]
        for source, grp in sims.groupby("source", sort=False):
            answers = grp["assistant"].tolist()
            self.rng.shuffle(answers)
            answers = answers[:answers_per_sim]
            original_user = grp.iloc[0]["user"]
            variants = [original_user] + T.CURHAT_USER_VARIANTS.get(source, [])
            for i, a in enumerate(answers):
                user = variants[i % len(variants)]
                self.add("persona", [(user, a)], {"source": source})

    def gen_emotional(self, answers_per_user: int = 2):
        """Curhat topik di luar PDF: tiap kalimat user dipasangkan dengan
        beberapa jawaban validasi (tanpa produk)."""
        for topic, spec in T.EMOTIONAL_TOPICS.items():
            for user in spec["user"]:
                for a in self.rng.sample(spec["assistant"], min(answers_per_user, len(spec["assistant"]))):
                    self.add("emotional", [(user, a)], {"topic": topic})

    def gen_bridges(self, users_per_scenario: int = 3):
        """Curhat (giliran 1, tanpa produk) -> user pivot ke kulit/penampilan
        (giliran 2) -> validasi + rekomendasi ringan dari needs_mapping.csv."""
        for sc in T.BRIDGE_SCENARIOS:
            topic = T.EMOTIONAL_TOPICS.get(sc["topic"])
            grp = self.needs[self.needs["need_id"] == sc["need_id"]].sort_values("prioritas")
            if topic is None or grp.empty:
                print(f"[warn] bridge dilewati: topic={sc['topic']} need_id={sc['need_id']}", file=sys.stderr)
                continue
            for user1 in self.rng.sample(topic["user"], min(users_per_scenario, len(topic["user"]))):
                a1 = self.pick(topic["assistant"])
                for user2 in sc["pivot_user"]:
                    need_row = grp.iloc[0] if self.rng.random() < 0.7 or len(grp) == 1 else grp.iloc[self.rng.randrange(1, len(grp))]
                    recommend = self.pick(T.RECOMMEND_FRAMES).format(product=need_row["product"], reason=lower_first(need_row["alasan"]))
                    a2 = f"{self.pick(sc['bridge'])} {recommend} {self.pick(T.BRIDGE_FOLLOWUPS)}"
                    turns = [(user1, a1), (user2, a2)]
                    # Sesekali giliran ke-3: nanya harga/cara pakai produk yang baru disebut.
                    if self.rng.random() < 0.3:
                        intent = self.pick(["harga", "cara_pakai", "kandungan"])
                        a3 = self.fact_answer(need_row["product"], intent)
                        if a3:
                            turns.append((self.pick(T.MULTITURN_FOLLOWUPS[intent]), a3))
                    self.add("bridge", turns, {"topic": sc["topic"], "need_id": sc["need_id"], "product": need_row["product"]})

    def gen_guardrails(self, repeat: int = 2):
        for _ in range(repeat):
            for q, a in T.GUARDRAIL_QA:
                self.add("guardrail", [(q, a)])

    def gen_dont_know(self, per_question: int = 3):
        names = list(self.products)
        for q in T.UNANSWERABLE_QUESTIONS:
            for name in self.rng.sample(names, min(per_question, len(names))):
                a = self.pick(T.UNANSWERABLE_ANSWERS).format(name=name)
                self.add("dont_know", [(q.format(name=name), a)], {"product": name})

    def gen_vague(self, repeat: int = 1):
        answers = self.persona[self.persona["source"] == "dont_know"]["assistant"].tolist()
        if not answers:
            return
        for _ in range(repeat):
            for q in T.VAGUE_QUESTIONS:
                self.add("vague", [(q, self.pick(answers))])

    def gen_out_of_catalog(self, repeat: int = 2):
        for _ in range(repeat):
            for q in T.OUT_OF_CATALOG_QUESTIONS:
                a = self.pick(T.OUT_OF_CATALOG_ANSWERS).format(skincare_cats=self.skincare_cats, deco_cats=self.deco_cats)
                self.add("out_of_catalog", [(q, a)])


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_needs(path: Path, products: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"need_id", "user_examples", "empati", "product", "alasan", "prioritas"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"needs_mapping.csv kurang kolom: {sorted(missing)}")
    df["product"] = df["product"].str.strip()
    unknown = sorted(set(df["product"]) - products)
    if unknown:
        sys.exit("Nama produk di needs_mapping.csv tidak ada di katalog (harus persis sama):\n  " + "\n  ".join(unknown))
    df["prioritas"] = df["prioritas"].replace("", "9").astype(int)
    df["user_examples_list"] = df["user_examples"].apply(lambda s: [x.strip() for x in s.split("|") if x.strip()])
    df["empati_list"] = df["empati"].apply(lambda s: [x.strip() for x in s.split("|") if x.strip()] or [""])
    return df


def split_rows(rows: list[dict], val_ratio: float, rng: random.Random):
    """Split acak, stratified per kind supaya val punya semua jenis."""
    train, val = [], []
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    for kind, items in by_kind.items():
        rng.shuffle(items)
        n_val = int(round(len(items) * val_ratio)) if val_ratio > 0 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"conversations": r["conversations"]}, ensure_ascii=False) + "\n")


def write_preview(path: Path, rows: list[dict], per_kind: int, rng: random.Random):
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    lines = ["# Preview dataset (acak, per jenis)\n"]
    for kind in sorted(by_kind):
        lines.append(f"\n## {kind} ({len(by_kind[kind])} baris)\n")
        for r in rng.sample(by_kind[kind], min(per_kind, len(by_kind[kind]))):
            for m in r["conversations"]:
                if m["role"] == "system":
                    continue
                lines.append(f"- **{m['role']}**: {m['content']}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=None, help="path xlsx katalog (default: products_*.xlsx terbaru di root repo)")
    ap.add_argument("--needs", default=str(HERE / "needs_mapping.csv"))
    ap.add_argument("--persona", default=str(HERE / "persona_examples.csv"))
    ap.add_argument("--system-prompt", default=str(HERE / "system_prompt_short.txt"))
    ap.add_argument("--no-system", action="store_true", help="jangan sertakan role system")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--fact-questions", type=int, default=2, help="variasi pertanyaan per (produk, intent)")
    ap.add_argument("--rec-frames", type=int, default=4, help="variasi bingkai kalimat per contoh keluhan")
    ap.add_argument("--persona-per-sim", type=int, default=12, help="jawaban curhat murni per simulasi PDF")
    args = ap.parse_args()

    catalog_path = Path(args.catalog) if args.catalog else sorted(ROOT.glob("products_*.xlsx"))[-1]
    catalog = pd.read_excel(catalog_path)
    catalog.columns = [c.strip() for c in catalog.columns]
    if "name" not in catalog.columns:
        sys.exit("Katalog harus punya kolom 'name'")

    rng = random.Random(args.seed)
    system_prompt = None if args.no_system else Path(args.system_prompt).read_text(encoding="utf-8").strip()
    persona = pd.read_csv(args.persona, dtype=str).fillna("") if Path(args.persona).exists() else pd.DataFrame(columns=["source", "topic", "user", "assistant"])
    products = {clean_text(n) for n in catalog["name"] if clean_text(n)}
    needs = load_needs(Path(args.needs), products)

    b = Builder(catalog, needs, persona, system_prompt, rng)
    b.gen_identity()
    b.gen_facts(questions_per_intent=args.fact_questions)
    b.gen_category_lists()
    b.gen_need_recs(frames_per_example=args.rec_frames)
    b.gen_fact_multiturn()
    b.gen_persona(answers_per_sim=args.persona_per_sim)
    b.gen_emotional()
    b.gen_bridges()
    b.gen_guardrails()
    b.gen_dont_know()
    b.gen_vague()
    b.gen_out_of_catalog()

    # Buang duplikat persis (user+assistant sama).
    seen, uniq = set(), []
    for r in b.rows:
        key = json.dumps([(m["role"], m["content"]) for m in r["conversations"]], ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    train, val = split_rows(uniq, args.val_ratio, rng)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_preview(out / "preview.md", uniq, per_kind=4, rng=rng)

    def convo_chars(r):
        return sum(len(m["content"]) for m in r["conversations"])

    lengths = sorted(convo_chars(r) for r in uniq)
    stats = {
        "catalog": catalog_path.name,
        "products": len(products),
        "needs": int(needs["need_id"].nunique()),
        "system_prompt_chars": len(system_prompt or ""),
        "rows_total": len(uniq),
        "rows_train": len(train),
        "rows_val": len(val),
        "rows_by_kind": dict(Counter(r["kind"] for r in uniq)),
        "chars_per_convo": {
            "min": lengths[0], "median": lengths[len(lengths) // 2], "p95": lengths[int(len(lengths) * 0.95)], "max": lengths[-1],
            "est_tokens_max": lengths[-1] // 3,
        },
        "seed": args.seed,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nTersimpan ke {out}/train.jsonl, val.jsonl, stats.json, preview.md")


if __name__ == "__main__":
    main()
