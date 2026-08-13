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
    would silently drop turns). Anything else is "tabular" -> always
    auto-converted into Q&A rows via auto_generate_qa_rows(), no manual
    mapping step."""
    if "conversations" in {c.lower() for c in df.columns}:
        return {"mode": "sharegpt"}
    return {"mode": "tabular"}


# Column names that read naturally as a single short attribute ("Berapa harga
# X?", "Apa kategori X?") — used by auto_generate_qa_rows() to decide which
# extra columns earn their own focused question beyond the general overview.
_ATTRIBUTE_KEYWORDS = {
    "harga", "price", "biaya", "cost", "kategori", "category", "jenis", "tipe", "type",
    "stok", "stock", "warna", "color", "colour", "ukuran", "size", "berat", "weight",
    "rating", "diskon", "discount", "merek", "brand", "satuan", "unit",
}


def _looks_numeric(value) -> bool:
    s = str(value).strip().replace(".", "", 1).replace(",", "", 1)
    return s.isdigit()


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
            question = f"Berapa {c} {subject}?" if _looks_numeric(v) else f"Apa {c} dari {subject}?"
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
