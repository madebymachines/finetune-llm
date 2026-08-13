from __future__ import annotations

import math
import re
import threading

import torch

from .constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P


def compute_eval_loss(trainer):
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss")
    perplexity = math.exp(loss) if loss is not None else None
    return metrics, perplexity


# Cap how many prior turns get replayed into the model on every new message.
# Without this, a long Test-tab chat session would make each turn's prompt
# grow without bound — slower every message, and eventually risks blowing
# past max_seq_length (the same truncation/OOM class of issue this project
# has hit before, just triggered by chat length instead of dataset rows).
_MAX_HISTORY_MESSAGES = 20  # ~10 back-and-forth exchanges


def build_messages(modality: str, text: str, image=None, audio=None, system_prompt: str = "", history: list[dict] | None = None):
    """Build a chat-template-ready messages list for the given modality.

    `history` (optional): prior turns as [{"role": "user"/"assistant", "content": str}, ...],
    oldest first, NOT including the current `text` turn (that's appended separately below as
    the final user message, along with any image/audio). Only the most recent
    `_MAX_HISTORY_MESSAGES` are kept. Without this, every reply is generated with zero
    awareness of earlier turns — the model can't refer back to anything said before, and
    tends to answer as if the conversation just started (e.g. opening with a greeting) every
    single time, since that's structurally what a single, history-less turn looks like."""
    user_content = []
    if modality == "Audio" and audio is not None:
        user_content.append({"type": "audio", "audio": audio})
    if modality == "Vision" and image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": text})

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    for turn in (history or [])[-_MAX_HISTORY_MESSAGES:]:
        messages.append({"role": turn["role"], "content": [{"type": "text", "text": turn["content"]}]})
    messages.append({"role": "user", "content": user_content})
    return messages


def _prepare_inputs(modality, processor, messages, image, device):
    if modality == "Vision":
        # Gemma's vision processor needs the raw image passed separately from
        # the rendered chat text (see Unsloth's Gemma4 Vision notebook).
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        return processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(device)

    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)


def generate_response(
    modality,
    model,
    processor,
    text: str,
    image=None,
    audio=None,
    system_prompt: str = "",
    history: list[dict] | None = None,
    max_new_tokens: int = 256,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    use_adapter: bool = True,
) -> str:
    messages = build_messages(modality, text, image=image, audio=audio, system_prompt=system_prompt, history=history)
    inputs = _prepare_inputs(modality, processor, messages, image, model.device)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        use_cache=True,
    )
    if use_adapter:
        out = model.generate(**gen_kwargs)
    else:
        with model.disable_adapter():
            out = model.generate(**gen_kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_batch(
    modality,
    model,
    processor,
    texts: list[str],
    system_prompt: str = "",
    max_new_tokens: int = 256,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    use_adapter: bool = True,
) -> list[str]:
    """Text-only batched generation: builds one left-padded batch and calls
    model.generate() ONCE for the whole list, instead of once per prompt like
    generate_response(). Far fewer generate() calls for the same work, and
    the GPU actually gets to parallelize across sequences instead of sitting
    at low utilization between one-at-a-time calls.

    Trade-off: every sequence in the batch holds its activations/KV-cache in
    VRAM at the same time, so a bigger batch uses more memory — callers
    should chunk large item lists into modest-sized batches (see
    before_after_compare's `batch_size` param) rather than passing everything
    at once, same headroom concerns as elsewhere in this app.

    Vision/Audio aren't supported here — image/audio batching would need
    per-modality collation this app doesn't implement, and the slow case
    this was built for (large eval-split comparisons) is Text-only anyway."""
    if modality != "Text":
        raise ValueError("generate_batch only supports modality='Text'")
    if not texts:
        return []

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Tokenize each prompt on its own first (unpadded) so we know how long
    # every sequence actually is, then left-pad them all to the batch's max
    # length ourselves. Left-padding (not right) is what makes batched
    # decoder-only generation correct: it keeps every prompt's last real
    # token flush against the same column, so `model.generate()` continues
    # all sequences from a shared position instead of continuing mid-padding
    # for the shorter ones.
    per_item_ids = []
    for text in texts:
        messages = build_messages(modality, text, system_prompt=system_prompt)
        encoded = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        )
        per_item_ids.append(encoded["input_ids"][0])

    max_len = max(ids.shape[0] for ids in per_item_ids)
    input_ids = torch.full((len(per_item_ids), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(per_item_ids), max_len), dtype=torch.long)
    for i, ids in enumerate(per_item_ids):
        n = ids.shape[0]
        input_ids[i, max_len - n:] = ids
        attention_mask[i, max_len - n:] = 1
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        use_cache=True,
        pad_token_id=pad_id,
    )
    if use_adapter:
        out = model.generate(**gen_kwargs)
    else:
        with model.disable_adapter():
            out = model.generate(**gen_kwargs)

    # Left-padding means every row's prompt ends at the same column, so one
    # shared slice index recovers each row's newly generated tokens.
    new_tokens = out[:, input_ids.shape[1]:]
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in new_tokens]


_QA_PAIR_RE = re.compile(r'Q\d*\s*:\s*(.+?)\s*\n\s*A\d*\s*:\s*(.+?)(?=\n\s*Q\d*\s*:|\Z)', re.DOTALL | re.IGNORECASE)
# The last matched answer absorbs any trailing text the model writes after the
# final Q/A block (the regex has no marker to stop at otherwise) — cap length
# so a stray trailing ramble can't turn into a multi-thousand-character row.
# This is the same failure mode (and same fix) as parse_user_ai_examples()'s
# _MAX_EXAMPLE_CHARS, which caused a real training OOM earlier in this project.
_MAX_LLM_QA_CHARS = 500


def generate_qa_pairs_llm(
    model, processor, df, questions_per_row: int = 2, max_new_tokens: int = 300, batch_size: int = 4,
) -> list[dict]:
    """Alternative to data_utils.auto_generate_qa_rows() that asks the
    already-loaded model to write `questions_per_row` naturally-varied Q&A
    pairs per row, strictly grounded in that row's own column values, instead
    of the fixed rule-based templates. It's an opt-in the caller should gate
    behind a model-loaded check and a progress spinner — this function has no
    Streamlit/UI awareness itself.

    Rows are processed `batch_size` at a time via generate_batch() — one
    model.generate() call per chunk of rows instead of one per row, same
    technique as before_after_compare's Evaluate-tab speedup. Still linear in
    row count (a big catalog is still a big catalog), just far fewer, better
    -utilized GPU calls to get there.

    use_adapter defaults to True (via generate_batch) rather than False:
    at Data-tab time a LoRA adapter may not even be attached yet (Setup's
    "Apply LoRA" step is independent of Data tab), and model.disable_adapter()
    raises on a plain (non-PEFT) model — True works unconditionally, and an
    untrained adapter (zero-initialized B matrix) behaves identically to the
    base model anyway.

    Every pair from a row becomes its own INDEPENDENT training row once
    flattened — there's no shared context between Q1/A1 and Q2/A2 of the same
    product, let alone across different products. So the prompt explicitly
    requires (a) one overview/identity question per row even when
    questions_per_row is small (otherwise "vary the style" alone can make the
    model skip introducing the product entirely), and (b) every question to
    name the product/item outright instead of a bare pronoun like "ini" —
    that pronoun would have no antecedent once the pair stands alone, so a
    real user's differently-phrased "ini" at inference time can't be
    resolved either."""
    if len(df.columns) == 0:
        return []
    subject_col = df.columns[0]
    prompts = []
    for _, row in df.iterrows():
        subject = row[subject_col]
        if subject != subject or not str(subject).strip():  # `!= self` filters NaN
            continue
        facts = "\n".join(f"- {c}: {v}" for c, v in row.items() if v == v and str(v).strip())  # v == v filters NaN
        if not facts:
            continue
        prompts.append(
            f'Berikut data satu produk/item bernama "{subject}":\n{facts}\n\n'
            f"Buat {questions_per_row} pasang pertanyaan dan jawaban dalam Bahasa Indonesia tentang data ini.\n"
            "ATURAN WAJIB:\n"
            f'1. Pasangan pertama (Q1/A1) HARUS pertanyaan pengenalan/ringkasan produk (boleh variasikan '
            f'kalimatnya, mis. "Apa itu {subject}?" atau "Ceritakan tentang {subject}").\n'
            f'2. SETIAP pertanyaan wajib menyebut nama "{subject}" secara eksplisit — JANGAN pakai kata ganti '
            'seperti "ini"/"produk ini" tanpa menyebut namanya, karena tiap pasangan dipakai terpisah tanpa '
            "konteks pasangan lain.\n"
            "3. Kalau ada pasangan setelah yang pertama, variasikan gayanya (jangan semua diawali 'Apa itu').\n"
            "4. Jawabannya HARUS hanya berdasarkan data di atas, jangan menambahkan informasi yang tidak ada.\n"
            "Format WAJIB persis seperti ini, satu pasang per blok:\n"
            "Q1: <pertanyaan>\nA1: <jawaban>\nQ2: <pertanyaan>\nA2: <jawaban>"
        )

    rows = []
    step = max(1, batch_size)
    for start in range(0, len(prompts), step):
        chunk = prompts[start:start + step]
        outputs = generate_batch("Text", model, processor, chunk, max_new_tokens=max_new_tokens, temperature=0.7)
        for text in outputs:
            for q, a in _QA_PAIR_RE.findall(text):
                q, a = q.strip(), a.strip()
                if q and a and len(q) <= _MAX_LLM_QA_CHARS and len(a) <= _MAX_LLM_QA_CHARS:
                    rows.append({"user": q, "assistant": a})
    return rows


def stream_chat_response(
    modality,
    model,
    processor,
    text: str,
    image=None,
    audio=None,
    system_prompt: str = "",
    history: list[dict] | None = None,
    max_new_tokens: int = 256,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    use_adapter: bool = True,
):
    """Generator yielding text chunks, for use with st.write_stream."""
    from transformers import TextIteratorStreamer

    messages = build_messages(modality, text, image=image, audio=audio, system_prompt=system_prompt, history=history)
    inputs = _prepare_inputs(modality, processor, messages, image, model.device)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        streamer=streamer,
        use_cache=True,
    )

    def _run():
        if use_adapter:
            model.generate(**gen_kwargs)
        else:
            with model.disable_adapter():
                model.generate(**gen_kwargs)

    thread = threading.Thread(target=_run)
    thread.start()
    for chunk in streamer:
        yield chunk
    thread.join()


# Small hardcoded fallback if NLTK's stopwords corpus can't be fetched (no
# internet) — keeps the fact-check usable offline, just less thorough.
_STOPWORDS_FALLBACK = {
    "yang", "dan", "atau", "dengan", "untuk", "dari", "kamu", "kami", "kita", "adalah",
    "akan", "bisa", "ada", "tidak", "juga", "saja", "kalau", "kalo", "ini", "itu", "aku",
    "nggak", "gak", "banget", "sama", "buat", "biar", "kayak", "the", "and", "or", "with",
    "for", "from", "this", "that", "also", "will", "can", "not", "are", "have", "has",
}
_stopwords_cache: set | None = None


def _get_stopwords() -> set:
    """Lazy-loaded, cached union of NLTK's Indonesian + English stopword
    lists. Downloaded on first use (needs internet once; NLTK caches the
    corpus to disk after that, same as the sentence-transformers-style
    on-demand downloads elsewhere in this app's dependency stack)."""
    global _stopwords_cache
    if _stopwords_cache is not None:
        return _stopwords_cache
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            words = set(stopwords.words("indonesian")) | set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("indonesian")) | set(stopwords.words("english"))
        _stopwords_cache = words
    except Exception:
        _stopwords_cache = _STOPWORDS_FALLBACK
    return _stopwords_cache


def extract_key_facts(text: str, max_facts: int = 8) -> list[str]:
    """Pull out the facts in `text` most worth checking for in a generated
    answer: numbers first (prices/quantities are usually the most concrete,
    verifiable thing in a product-catalog-style answer), then distinctive
    words (longer than 3 chars, not a common stopword). Order preserved,
    de-duplicated case-insensitively, capped at `max_facts` so a long prose
    answer doesn't turn into an unreasonably strict checklist."""
    numbers = re.findall(r"\d[\d.,]*\d|\d+", text)
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text)
    stopwords_set = _get_stopwords()
    significant = [w for w in words if len(w) > 3 and w.lower() not in stopwords_set]
    seen, facts = set(), []
    for f in numbers + significant:
        key = f.lower()
        if key not in seen:
            seen.add(key)
            facts.append(f)
    return facts[:max_facts]


def check_factual_grounding(expected: str, generated: str, max_facts: int = 8) -> dict:
    """Rough, automatic stand-in for manually reading whether a generated
    answer actually contains the facts from the expected (training-row)
    answer — a keyword/number presence check, not a meaning-aware judge.
    Returns {"score": float|None, "matched": [...], "missed": [...], "verdict": str}.
    `score` is None (verdict "–") when `expected` has no extractable facts to
    check at all, so callers can tell "nothing to verify" apart from "verified
    and failed"."""
    facts = extract_key_facts(expected, max_facts=max_facts)
    if not facts:
        return {"score": None, "matched": [], "missed": [], "verdict": "–"}
    gen_lower = generated.lower()
    matched = [f for f in facts if f.lower() in gen_lower]
    missed = [f for f in facts if f not in matched]
    score = len(matched) / len(facts)
    verdict = "✅" if score >= 0.7 else ("⚠️" if score > 0 else "❌")
    return {"score": score, "matched": matched, "missed": missed, "verdict": verdict}


def before_after_compare(
    modality,
    model,
    processor,
    items: list[dict],
    system_prompt: str = "",
    max_new_tokens: int = 256,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 4,
):
    """items: list of {"text": str, "image"?: PIL.Image, "audio"?: np.ndarray}.
    For each item, generate with the LoRA adapter disabled (base model
    behaviour) and enabled (finetuned), reusing the same loaded weights.

    Text modality processes items `batch_size` at a time via generate_batch()
    — 2 * ceil(len(items) / batch_size) total generate() calls instead of
    2 * len(items), which is what made large eval-split comparisons slow.
    Vision/Audio still run one item at a time (see generate_batch's
    docstring for why)."""
    if modality == "Text":
        prompts = [item["text"] for item in items]
        results = []
        step = max(1, batch_size)
        for start in range(0, len(prompts), step):
            chunk = prompts[start:start + step]
            shared_kwargs = dict(
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            base_outputs = generate_batch(modality, model, processor, chunk, use_adapter=False, **shared_kwargs)
            finetuned_outputs = generate_batch(modality, model, processor, chunk, use_adapter=True, **shared_kwargs)
            for prompt, base_out, finetuned_out in zip(chunk, base_outputs, finetuned_outputs):
                results.append({"prompt": prompt, "base_model": base_out, "finetuned_model": finetuned_out})
        return results

    results = []
    for item in items:
        kwargs = dict(
            text=item["text"],
            image=item.get("image"),
            audio=item.get("audio"),
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        base_output = generate_response(modality, model, processor, use_adapter=False, **kwargs)
        finetuned_output = generate_response(modality, model, processor, use_adapter=True, **kwargs)
        results.append({"prompt": item["text"], "base_model": base_output, "finetuned_model": finetuned_output})
    return results
