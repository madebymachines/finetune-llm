from __future__ import annotations

import io
import json
import os

import pandas as pd
from datasets import Dataset

from .constants import AUDIO_SAMPLING_RATE, CHAT_TEMPLATE


class _SafeDict(dict):
    """dict subclass so str.format_map leaves unknown {placeholders} untouched
    instead of raising, so template preview errors are surfaced explicitly
    by the caller rather than silently producing garbage."""

    def __missing__(self, key):
        raise KeyError(key)


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


def extract_text_from_document(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF/DOCX/TXT (persona/rule/guardrail
    document). This is plain text extraction, not structured Q&A parsing —
    narrative documents with numbered dialogue examples still need manual
    conversion to a spreadsheet if those examples should become individual
    training rows rather than just system-prompt context."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(uploaded_file)
        return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if name.endswith(".docx"):
        from docx import Document as DocxDocument

        doc = DocxDocument(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8").strip()
    raise ValueError(f"Format dokumen tidak didukung: {uploaded_file.name}")


# ---------------------------------------------------------------------------
# Text modality: tabular data -> ShareGPT-style conversations -> "text" field
# ---------------------------------------------------------------------------

def looks_like_sharegpt(df: pd.DataFrame) -> bool:
    return "conversations" in df.columns


def sharegpt_df_to_dataset(df: pd.DataFrame) -> Dataset:
    from unsloth.chat_templates import standardize_data_formats

    ds = Dataset.from_pandas(df.reset_index(drop=True))
    return standardize_data_formats(ds)


def build_conversations_from_template(
    df: pd.DataFrame,
    system_template: str,
    user_template: str,
    assistant_template: str,
) -> list[list[dict]]:
    """Turn each row of a tabular custom dataset (product catalog, instruction
    /rule + Q&A examples, etc.) into a ShareGPT-style conversation by
    filling {column_name} placeholders from the row's values."""
    conversations = []
    for _, row in df.iterrows():
        values = _SafeDict({k: ("" if pd.isna(v) else v) for k, v in row.items()})
        convo = []
        if system_template.strip():
            convo.append({"role": "system", "content": system_template.format_map(values)})
        convo.append({"role": "user", "content": user_template.format_map(values)})
        convo.append({"role": "assistant", "content": assistant_template.format_map(values)})
        conversations.append(convo)
    return conversations


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
