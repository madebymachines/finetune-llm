from __future__ import annotations

import io
import json
import os
import re

import pandas as pd
from datasets import Dataset

from .constants import AUDIO_SAMPLING_RATE, CHAT_TEMPLATE

# ---------------------------------------------------------------------------
# Generic table loading (shared by all modalities)
# ---------------------------------------------------------------------------

def load_hf_dataset(dataset_name: str, num_rows: int | None, split: str = "train"):
    from datasets import load_dataset

    split_expr = f"{split}[:{num_rows}]" if num_rows else split
    return load_dataset(dataset_name, split=split_expr)


def read_uploaded_table(uploaded_file):
    """Read an uploaded CSV/Excel/JSON/JSONL file into a pandas DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    if name.endswith(".jsonl"):
        rows = [json.loads(line) for line in uploaded_file.read().decode("utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    if name.endswith(".json"):
        data = json.loads(uploaded_file.read().decode("utf-8"))
        return pd.DataFrame(data)
    raise ValueError(f"Format file tidak didukung: {uploaded_file.name}")


def _clean_extracted_text(text: str) -> str:
    """Normalize whitespace from PDF/DOCX extraction. Some PDFs (notably ones
    exported from Google Docs) make pypdf emit one word per line with extra
    spaces — this rejoins mid-paragraph line breaks and collapses runs of
    spaces, while still keeping blank-line paragraph breaks intact."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" *\n{2,} *", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_text_from_document(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF/DOCX/TXT (persona/rule/guardrail
    document). Plain text extraction — structured Q&A examples embedded in
    the narrative are pulled out separately by parse_user_ai_examples()."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(uploaded_file)
        raw = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    elif name.endswith(".docx"):
        from docx import Document as DocxDocument

        doc = DocxDocument(uploaded_file)
        raw = "\n".join(p.text for p in doc.paragraphs)
    elif name.endswith((".txt", ".md")):
        raw = uploaded_file.read().decode("utf-8")
    else:
        raise ValueError(f"Format dokumen tidak didukung: {uploaded_file.name}")
    return _clean_extracted_text(raw)


_USER_AI_BLOCK = re.compile(
    r'User\s*:\s*[“"](?P<user>.+?)[”"]\s*AI\s*:\s*(?P<ai>.+?)(?=User\s*:|\Z)', re.DOTALL
)
_NUMBERED_VARIANT = re.compile(r'\d+[.)]\s*[“"](?P<text>.+?)[”"]', re.DOTALL)

# A real chat turn in this pattern is a sentence or two. Anything past this is
# almost certainly the non-greedy quote match "running away" to some unrelated
# closing quote much later in the document (e.g. a citation or another quoted
# term elsewhere on the page) rather than genuine dialogue — better to drop it
# than silently produce a multi-thousand-character row that blows the training
# context budget without the user noticing.
_MAX_EXAMPLE_CHARS = 600


def parse_user_ai_examples(text: str) -> list[dict]:
    """Best-effort extraction of 'User: "..." AI: "..."' example dialogues out
    of a narrative document — including the common pattern of several numbered
    AI response variants for the same user line (data augmentation by
    phrasing), which each become a separate training row reusing that user
    message. This is a starting point for the user to review/edit in the UI,
    not a guaranteed-correct parse of any document."""
    rows = []
    for m in _USER_AI_BLOCK.finditer(text):
        user_msg = m.group("user").strip()
        if len(user_msg) > _MAX_EXAMPLE_CHARS:
            continue
        ai_block = m.group("ai").strip()
        variants = [v.group("text").strip() for v in _NUMBERED_VARIANT.finditer(ai_block)]
        if not variants:
            single = ai_block.strip(' "“”').strip()
            if single:
                variants = [single]
        for assistant_msg in variants:
            if user_msg and assistant_msg and len(assistant_msg) <= _MAX_EXAMPLE_CHARS:
                rows.append({"user": user_msg, "assistant": assistant_msg})
    return rows


# ---------------------------------------------------------------------------
# Text modality: tabular data -> ShareGPT-style conversations -> "text" field
# ---------------------------------------------------------------------------

def sharegpt_df_to_dataset(df: pd.DataFrame) -> Dataset:
    from unsloth.chat_templates import standardize_data_formats

    ds = Dataset.from_pandas(df.reset_index(drop=True))
    return standardize_data_formats(ds)


def resolve_conversation_columns(df: pd.DataFrame) -> dict:
    """A "conversations" column means the table is already final, multi-turn
    ShareGPT format -> use directly, never flattened into a Q&A table (that
    would silently drop turns). A table that already has clean user/assistant
    (or question/answer) columns is "qa_ready" -> also use directly, no need
    to run it through auto_generate_qa_rows()/Gemma just to get the exact
    same values back out. Anything else is "tabular" -> needs auto-conversion."""
    cols_lower = {c.lower() for c in df.columns}
    if "conversations" in cols_lower:
        return {"mode": "sharegpt"}
    if ("user" in cols_lower and "assistant" in cols_lower) or ("question" in cols_lower and "answer" in cols_lower):
        return {"mode": "qa_ready"}
    return {"mode": "tabular"}


# Column names that read naturally as a single short attribute ("Berapa harga
# X?", "Apa kategori X?") — used by auto_generate_qa_rows() to decide which
# extra columns earn their own focused question beyond the general overview.
_ATTRIBUTE_KEYWORDS = {
    "harga", "price", "biaya", "cost", "kategori", "category", "jenis", "tipe", "type",
    "stok", "stock", "warna", "color", "colour", "ukuran", "size", "berat", "weight",
    "rating", "diskon", "discount", "merek", "brand", "satuan", "unit",
}

# Maps a column name to the natural-language noun phrase that should slot
# into "Apa {phrase} dari {subject}?" / "Berapa {phrase} {subject}?" — a
# short-valued column earns its own question in auto_generate_qa_rows() even
# when its name isn't in _ATTRIBUTE_KEYWORDS above (see the `len(v) <= 30`
# check there), so without this map a column like "group_category" or
# "product_specialty" would leak its raw snake_case name straight into the
# question text ("Apa group_category dari X?") instead of reading naturally.
# Unmapped columns still fall back to a readable underscore/dash-stripped
# version of the column name rather than the raw name.
_COLUMN_PHRASE_MAP = {
    "harga": "harga", "price": "harga", "biaya": "harga", "cost": "harga",
    "kategori": "kategori", "category": "kategori", "product_category": "kategori",
    "group_category": "kelompok produk", "sub_category": "sub-kategori",
    "jenis": "jenis", "type": "jenis", "tipe": "jenis",
    "stok": "stok", "stock": "stok",
    "warna": "warna", "color": "warna", "colour": "warna", "shade": "shade",
    "ukuran": "ukuran", "size": "ukuran", "volume": "volume", "netto": "netto",
    "berat": "berat", "weight": "berat",
    "rating": "rating", "review": "ulasan",
    "diskon": "diskon", "discount": "diskon",
    "merek": "merek", "brand": "merek",
    "satuan": "satuan", "unit": "satuan",
    "product_specialty": "keunggulan", "specialty": "keunggulan", "keunggulan": "keunggulan",
    "benefit": "manfaat", "manfaat": "manfaat",
    "finish": "finish", "coverage": "coverage", "texture": "tekstur", "tekstur": "tekstur",
    "skin_type": "jenis kulit yang cocok", "untuk_kulit": "jenis kulit yang cocok",
    "ingredient": "kandungan", "ingredients": "kandungan", "kandungan": "kandungan",
    "cara_pakai": "cara pakai", "usage": "cara pakai", "how_to_use": "cara pakai",
}


def _column_phrase(col: str) -> str:
    key = col.lower().strip()
    if key in _COLUMN_PHRASE_MAP:
        return _COLUMN_PHRASE_MAP[key]
    return key.replace("_", " ").replace("-", " ")  # readable fallback for anything unmapped


def _looks_numeric(value) -> bool:
    s = str(value).strip().replace(".", "", 1).replace(",", "", 1)
    return s.isdigit()


# Starter keyword groups for the cross-product recommendation feature — each
# label maps to keyword variants that get substring-matched (case-insensitive)
# against a product's ENTIRE row text (name + description + every other
# column), not one single column's literal value. Real catalogs usually
# don't have one clean "category" column at all — concerns/benefits are
# scattered inside marketing-copy sentences instead ("...menyatu sempurna
# dengan kulit...", "melindungi dari sinar matahari...") — so matching by
# keyword-in-text is what actually finds the right products, instead of
# grouping by whatever a single column's raw (and possibly paragraph-long)
# value happens to be. The user edits/extends this list in the UI; these are
# just sane Indonesian skincare/makeup-catalog defaults to start from.
# Skincare concerns AND makeup-side categories share this one list on
# purpose — when type_col scoping is used (see generate_concern_qa_rows), a
# combination that doesn't make sense (e.g. a "makeup" product matching
# "kulit kering") naturally ends up with 0-1 matches and gets dropped by the
# existing <2-matches skip, so there's no need to tag each line with which
# product type it "belongs to".
DEFAULT_CONCERN_KEYWORDS = {
    "kulit kering": ["kulit kering", "kering"],
    "kulit berminyak": ["kulit berminyak", "berminyak", "oily"],
    "kulit kombinasi": ["kulit kombinasi", "kombinasi"],
    "kulit berjerawat": ["kulit berjerawat", "berjerawat", "jerawat", "acne", "breakout"],
    "melembapkan kulit": ["melembapkan", "melembabkan", "lembap", "lembab", "hydrating", "moistur"],
    "mencerahkan kulit": ["mencerahkan", "cerah", "brighten", "glowing"],
    "melindungi dari paparan sinar matahari": ["spf", "uv", "sinar matahari", "sunscreen", "tabir surya"],
    "full coverage dan tahan lama": ["full coverage", "coverage tinggi", "tahan lama", "stain lama", "long lasting", "long wear"],
    "acara formal atau graduation": ["graduation", "wisuda", "acara formal", "pesta", "acara resmi"],
}
DEFAULT_CONCERN_KEYWORDS_TEXT = "\n".join(f"{label}: {', '.join(kws)}" for label, kws in DEFAULT_CONCERN_KEYWORDS.items())


def parse_concern_keywords(text: str) -> dict[str, list[str]]:
    """Parse the UI's editable "label: kw1, kw2, ..." textarea format (one
    group per line) into the dict generate_concern_qa_rows() expects. Blank
    lines and lines without a ':' are skipped; a label with no keywords
    after it is dropped."""
    groups: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        label, _, kws = line.partition(":")
        label = label.strip()
        keywords = [k.strip() for k in kws.split(",") if k.strip()]
        if label and keywords:
            groups[label] = keywords
    return groups


def parse_type_map(text: str) -> dict[str, str]:
    """Parse the UI's "nilai=label, nilai=label" format (e.g. "Deco=makeup,
    Skincare=skincare") into the dict generate_concern_qa_rows()'s type_map
    expects. Malformed pairs (no '=', empty side) are skipped."""
    result: dict[str, str] = {}
    for pair in text.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        raw, _, label = pair.partition("=")
        raw, label = raw.strip(), label.strip()
        if raw and label:
            result[raw] = label
    return result


def generate_concern_qa_rows(
    df: pd.DataFrame,
    keyword_groups: dict[str, list[str]],
    name_col: str | None = None,
    type_col: str | None = None,
    type_map: dict[str, str] | None = None,
) -> list[dict]:
    """Rule-based, cross-product recommendation Q&A: for each `label` in
    `keyword_groups`, a product counts as matching if ANY of its keywords
    appears anywhere in that row's combined text (every column, lowercased,
    space-joined) — not tied to one column having a short, clean tag value.
    Every label with 2+ matching products gets one Q&A pair listing those
    products' names, e.g. "Ada rekomendasi produk untuk kulit kering?" ->
    "Beberapa produk yang cocok untuk kulit kering: A, B, C." Purely
    template + literal product names, no LLM call — can't hallucinate a
    product that isn't actually there. Labels matched by fewer than 2
    products are skipped (nothing to recommend "among"). `name_col` defaults
    to the first column, same subject convention as auto_generate_qa_rows().

    `type_col` (optional): a column that marks product type/segment (e.g. a
    "group_category" column with values like "Deco"/"Skincare"). When given,
    matching is scoped PER TYPE VALUE as well as per concern — a product
    only counts for a (type, concern) combo if it's both that type AND
    matches that concern's keywords — and the question spells the type out:
    "Produk {type} apa saja yang cocok untuk kulit kering?". Nonsensical
    combos (e.g. "makeup" x "kulit kering") just end up with <2 matches and
    get skipped like any other under-populated group — no special-casing
    needed for which concerns "belong" to which type.
    `type_map` (optional): raw type_col value -> display label (e.g.
    {"Deco": "makeup", "Skincare": "skincare"}); a value not in the map is
    used as-is, lowercased."""
    if name_col is None:
        name_col = df.columns[0]
    type_map = type_map or {}

    row_entries = []  # (name, lowercased combined text, raw type value or None)
    for _, row in df.iterrows():
        name = row[name_col]
        if name != name or not str(name).strip():  # != self filters NaN
            continue
        blob = " ".join(str(v) for v in row.values if v == v).lower()
        type_value = None
        if type_col is not None:
            raw_type = row[type_col]
            if raw_type == raw_type and str(raw_type).strip():  # not NaN
                type_value = str(raw_type).strip()
        row_entries.append((str(name).strip(), blob, type_value))

    type_values = sorted({t for _, _, t in row_entries if t is not None}) if type_col is not None else [None]

    rows = []
    for type_value in type_values:
        candidates = [(name, blob) for name, blob, t in row_entries if t == type_value]
        for label, keywords in keyword_groups.items():
            keywords_lower = [k.lower() for k in keywords if k.strip()]
            if not keywords_lower:
                continue
            matches = [name for name, blob in candidates if any(kw in blob for kw in keywords_lower)]
            matches = list(dict.fromkeys(matches))  # dedupe, keep first-seen order
            if len(matches) < 2:
                continue
            names_list = ", ".join(matches)
            if type_value is None:
                rows.append({
                    "user": f"Ada rekomendasi produk untuk {label}?",
                    "assistant": f"Beberapa produk yang cocok untuk {label}: {names_list}.",
                })
            else:
                type_label = type_map.get(type_value, type_value.lower())
                rows.append({
                    "user": f"Produk {type_label} apa saja yang cocok untuk {label}?",
                    "assistant": f"Beberapa produk {type_label} yang cocok untuk {label}: {names_list}.",
                })
    return rows


def auto_generate_qa_rows(df: pd.DataFrame) -> list[dict]:
    """Turn any tabular custom dataset into question/answer rows with zero
    configuration:
    - if the table already has recognizable Q&A columns (user/assistant or
      question/answer, case-insensitive), use those values directly per row.
    - otherwise (e.g. a raw product catalog with columns like nama_produk/
      deskripsi/harga), treat the first column as the subject and emit an
      overview pair ("Apa itu {subject}?" -> remaining columns joined
      "kolom: nilai, ..."), PLUS one focused pair per remaining column that
      reads as a short attribute (name matches a small keyword list like
      harga/kategori/stok, or its value is short — long free-text columns
      like "deskripsi" don't get a redundant near-duplicate question).
      "Berapa {kolom} {subject}?" for numeric-looking values, otherwise
      "Apa {kolom} dari {subject}?" — so one row can produce several rows.
      Every answer is still the literal column value: a plain, honest join,
      not LLM-rewritten prose, since this path makes no model call — the
      result is meant to be reviewed/edited before training, not used verbatim.
    Rows with an empty subject or an empty resulting answer are skipped."""
    cols_lower = {c.lower(): c for c in df.columns}
    user_col = cols_lower.get("user") or cols_lower.get("question")
    assistant_col = cols_lower.get("assistant") or cols_lower.get("answer")

    rows = []
    if user_col and assistant_col:
        for _, row in df.iterrows():
            u, a = row[user_col], row[assistant_col]
            if pd.isna(u) or pd.isna(a) or not str(u).strip() or not str(a).strip():
                continue
            rows.append({"user": str(u), "assistant": str(a)})
        return rows

    if len(df.columns) < 2:
        return rows
    subject_col, other_cols = df.columns[0], df.columns[1:]
    for _, row in df.iterrows():
        subject = row[subject_col]
        if pd.isna(subject) or not str(subject).strip():
            continue
        pairs = [(c, row[c]) for c in other_cols if not pd.isna(row[c]) and str(row[c]).strip()]
        if not pairs:
            continue
        overview = ", ".join(f"{c}: {v}" for c, v in pairs)
        rows.append({"user": f"Apa itu {subject}?", "assistant": overview})
        for c, v in pairs:
            if c.lower() not in _ATTRIBUTE_KEYWORDS and len(str(v)) > 30:
                continue  # long free-text column already covered by the overview
            phrase = _column_phrase(c)
            question = f"Berapa {phrase} {subject}?" if _looks_numeric(v) else f"Apa {phrase} dari {subject}?"
            rows.append({"user": question, "assistant": str(v)})
    return rows


def conversations_to_dataset(conversations: list[list[dict]]) -> Dataset:
    return Dataset.from_dict({"conversations": conversations})


def apply_chat_template_to_dataset(dataset: Dataset, tokenizer) -> Dataset:
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    return dataset.map(formatting_prompts_func, batched=True)


def get_chat_tokenizer(tokenizer):
    from unsloth.chat_templates import get_chat_template

    return get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)


def train_eval_split(dataset: Dataset, eval_ratio: float, seed: int = 3407):
    if eval_ratio <= 0:
        return dataset, None
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"]


# ---------------------------------------------------------------------------
# Vision modality: image + caption -> "messages" format
# ---------------------------------------------------------------------------

def build_vision_messages_from_hf(hf_dataset, image_column: str, text_column: str, instruction: str, system_prompt: str = ""):
    messages = []
    for row in hf_dataset:
        image = row[image_column]
        user_content = [{"type": "text", "text": instruction}, {"type": "image", "image": image}]
        convo = []
        if system_prompt.strip():
            convo.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        convo.append({"role": "user", "content": user_content})
        convo.append({"role": "assistant", "content": [{"type": "text", "text": str(row[text_column])}]})
        messages.append({"messages": convo})
    return messages


def build_vision_messages_from_upload(
    image_files: dict,
    df: pd.DataFrame,
    filename_column: str,
    question_column: str | None,
    answer_column: str,
    default_instruction: str,
    system_prompt: str = "",
):
    """image_files: {filename: uploaded file object (from st.file_uploader)}"""
    from PIL import Image

    messages = []
    for _, row in df.iterrows():
        fname = str(row[filename_column])
        if fname not in image_files:
            raise KeyError(f"Gambar '{fname}' tidak ditemukan di file yang diupload")
        image = Image.open(io.BytesIO(image_files[fname].getvalue())).convert("RGB")
        question = str(row[question_column]) if question_column else default_instruction
        answer = str(row[answer_column])
        user_content = [{"type": "text", "text": question}, {"type": "image", "image": image}]
        convo = []
        if system_prompt.strip():
            convo.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        convo.append({"role": "user", "content": user_content})
        convo.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        messages.append({"messages": convo})
    return messages


# ---------------------------------------------------------------------------
# Audio modality: audio + transcript -> "messages" format
# ---------------------------------------------------------------------------

def build_audio_messages_from_hf(hf_dataset, audio_column: str, text_column: str, instruction: str, system_prompt: str):
    from datasets import Audio

    hf_dataset = hf_dataset.cast_column(audio_column, Audio(sampling_rate=AUDIO_SAMPLING_RATE))
    messages = []
    for row in hf_dataset:
        audio_array = row[audio_column]["array"]
        convo = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio_array}, {"type": "text", "text": instruction}]},
            {"role": "assistant", "content": [{"type": "text", "text": str(row[text_column])}]},
        ]
        messages.append({"messages": convo})
    return messages


def build_audio_messages_from_upload(
    audio_files: dict,
    df: pd.DataFrame,
    filename_column: str,
    question_column: str | None,
    answer_column: str,
    default_instruction: str,
    system_prompt: str,
    tmp_dir: str,
):
    """audio_files: {filename: uploaded file object (from st.file_uploader)}.
    Uploaded audio is written to tmp_dir then decoded via datasets.Audio so we
    don't need an extra audio-decoding dependency."""
    from datasets import Audio

    os.makedirs(tmp_dir, exist_ok=True)
    paths = []
    rows = []
    for _, row in df.iterrows():
        fname = str(row[filename_column])
        if fname not in audio_files:
            raise KeyError(f"Audio '{fname}' tidak ditemukan di file yang diupload")
        path = os.path.join(tmp_dir, fname)
        with open(path, "wb") as f:
            f.write(audio_files[fname].getvalue())
        paths.append(path)
        rows.append(row)

    audio_ds = Dataset.from_dict({"audio": paths}).cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))

    messages = []
    for row, audio_row in zip(rows, audio_ds):
        audio_array = audio_row["audio"]["array"]
        question = str(row[question_column]) if question_column else default_instruction
        answer = str(row[answer_column])
        convo = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio_array}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        messages.append({"messages": convo})
    return messages


def load_audio_array(uploaded_file, tmp_dir: str):
    """Decode a single uploaded audio file to a numpy array via datasets.Audio
    (reuses the `datasets` dependency instead of adding a new audio library)."""
    from datasets import Audio

    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    ds = Dataset.from_dict({"audio": [path]}).cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
    return ds[0]["audio"]["array"]


def media_train_eval_split(messages: list[dict], eval_ratio: float, seed: int = 3407):
    """Plain-list split for Vision/Audio "messages" data (no HF Dataset needed
    since UnslothVisionDataCollator accepts a plain list of {"messages": [...]})."""
    if eval_ratio <= 0:
        return messages, None
    import random

    rng = random.Random(seed)
    shuffled = messages.copy()
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_ratio))
    return shuffled[n_eval:], shuffled[:n_eval]
