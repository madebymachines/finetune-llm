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


# Values that read as booleans/flags rather than meaningful category labels
# (e.g. a "prioritize_product" Yes/No column) — excluded from grouping-column
# detection below, since "produk apa saja yang termasuk kategori Ya?" isn't a
# useful recommendation question.
_BOOLEAN_LIKE_VALUES = {"true", "false", "yes", "no", "ya", "tidak", "1", "0", "1.0", "0.0"}

# Multiple natural phrasings for the auto-generated cross-product
# recommendation question — deliberately more than one template (unlike the
# single "Apa itu {subject}?" overview phrasing) because this is the exact
# question shape that was observed to fail hardest at inference time: an
# open-ended "recommend me a product for X" phrased differently than the
# training template caused the finetuned model to fabricate plausible-
# sounding but nonexistent product names instead of returning the real,
# grounded list. More surface variety on this one template gives the
# finetuning signal more chances to generalize to a real user's phrasing.
_RECOMMENDATION_QUESTION_TEMPLATES = [
    "Produk apa saja yang termasuk kategori {value}?",
    "Ada rekomendasi produk untuk kategori {value}?",
]


def _detect_grouping_columns(
    df: pd.DataFrame, name_col, min_group_size: int = 2, max_group_size: int = 15,
) -> list[str]:
    """Find columns that look like clean categorical fields worth grouping
    rows by — purely from the data's shape, no hardcoded domain/column-name
    knowledge, so this works the same on a skincare catalog, an electronics
    catalog, or anything else with a name column plus some attribute columns.

    A column qualifies if: it isn't the subject/name column; its non-empty
    values are short (not free-text paragraphs); it isn't purely numeric
    (price/rating columns aren't "categories") or boolean-like; it has more
    than one distinct value but not close to one distinct value per row
    (that's an identifier, like a SKU, not a category); and at least one
    value is shared by somewhere between `min_group_size` and
    `max_group_size` rows (too few = nothing to recommend "among"; too many
    = too broad to be a specific, useful recommendation grouping)."""
    candidates = []
    n_rows = len(df)
    for c in df.columns:
        if c == name_col:
            continue
        values = df[c].dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            continue
        distinct = values.unique()
        if all(_looks_numeric(v) for v in distinct):
            continue  # e.g. price, rating — not a meaningful "category"
        if all(v.lower() in _BOOLEAN_LIKE_VALUES for v in distinct):
            continue
        if len(distinct) < 2 or len(distinct) > max(10, n_rows * 0.4):
            continue
        if any(len(v) > 40 for v in distinct):
            continue
        counts = values.value_counts()
        if not ((counts >= min_group_size) & (counts <= max_group_size)).any():
            continue
        candidates.append(c)
    return candidates


def generate_recommendation_qa_rows(
    df: pd.DataFrame, name_col: str | None = None, min_group_size: int = 2, max_group_size: int = 15,
) -> list[dict]:
    """Rule-based, cross-product recommendation Q&A — fully automatic, zero
    configuration: auto-detects which columns look like clean category
    fields (see _detect_grouping_columns), then for every distinct value
    shared by min_group_size..max_group_size rows, emits Q&A pairs (using
    several phrasings, see _RECOMMENDATION_QUESTION_TEMPLATES) listing the
    matching product names verbatim. Purely template + literal data, no LLM
    call — can't hallucinate a product that isn't actually there. Reusable
    as-is on any tabular custom data with a name column plus category-like
    columns; nothing here is specific to any one catalog/domain."""
    if name_col is None:
        name_col = df.columns[0]
    rows = []
    for col in _detect_grouping_columns(df, name_col, min_group_size, max_group_size):
        groups: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            name, value = row[name_col], row[col]
            if name != name or not str(name).strip():  # != self filters NaN
                continue
            if value != value or not str(value).strip():
                continue
            names = groups.setdefault(str(value).strip(), [])
            nm = str(name).strip()
            if nm not in names:
                names.append(nm)
        for value, names in groups.items():
            if not (min_group_size <= len(names) <= max_group_size):
                continue
            names_list = ", ".join(names)
            for template in _RECOMMENDATION_QUESTION_TEMPLATES:
                rows.append({
                    "user": template.format(value=value),
                    "assistant": f"Produk yang termasuk kategori {value}: {names_list}.",
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
      like "deskripsi" don't get a redundant near-duplicate question), PLUS
      dynamically-detected cross-product recommendation Q&A (see
      generate_recommendation_qa_rows) appended at the end.
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

    rows.extend(generate_recommendation_qa_rows(df, subject_col))
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
