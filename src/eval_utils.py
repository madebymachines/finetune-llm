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


# Caps how long a single LLM-generated question paraphrase can be before it's
# dropped — same failure mode (and same fix) as parse_user_ai_examples()'s
# _MAX_EXAMPLE_CHARS, which caused a real training OOM earlier in this project:
# a model that doesn't stop cleanly can otherwise turn one "variant" into a
# multi-thousand-character ramble.
_MAX_LLM_QA_CHARS = 500


_PARAPHRASE_LINE_RE = re.compile(r'^\s*Q\s*:\s*(.+)$', re.MULTILINE | re.IGNORECASE)


def augment_qa_paraphrases(
    model, processor, base_rows: list[dict],
    paraphrases_per_row: int = 2, max_new_tokens: int = 200, batch_size: int = 4,
) -> list[dict]:
    """Given already-grounded {"user", "assistant"} rows (e.g. from
    data_utils.auto_generate_qa_rows()), ask the already-loaded model to
    write `paraphrases_per_row` additional natural rephrasings of each
    QUESTION — the ANSWER is always copied verbatim from the base row, never
    regenerated, so this step can't introduce a new hallucinated fact, only
    teach the model more ways of asking for a fact it already has grounded.

    This targets a specific, observed failure mode: a finetuned model
    reliably recalls a fact when asked close to the exact trained phrasing,
    but for a differently-worded (especially open-ended "give me a
    recommendation" style) question, it falls back to free-generating
    plausible-sounding but fabricated content instead of finding the correct
    grounded answer already in its training data. More phrasing variety per
    fact directly targets that gap. Returns each base row immediately
    followed by its own paraphrase variants (not all bases first, then all
    variants appended at the end) — so scanning the returned table top to
    bottom in the UI shows each fact's variants right next to it instead of
    the table opening with what looks like a purely rule-based table until
    scrolled far down.

    Rows are processed `batch_size` at a time via generate_batch() — same
    batching technique as before_after_compare's Evaluate-tab speedup.

    use_adapter defaults to True (via generate_batch) rather than False: at
    Data-tab time a LoRA adapter may not even be attached yet, and
    model.disable_adapter() raises on a plain (non-PEFT) model."""
    if paraphrases_per_row <= 0 or not base_rows:
        return list(base_rows)

    prompts = [
        f'Pertanyaan asli: "{row["user"]}"\n\n'
        f"Tulis {paraphrases_per_row} cara lain menanyakan HAL YANG PERSIS SAMA, dengan gaya kalimat "
        "berbeda-beda (mis. lebih santai, lebih formal, atau lebih singkat). JANGAN ubah maksud "
        "pertanyaannya sama sekali — ini murni variasi kalimat, bukan pertanyaan baru, dan jangan "
        "sertakan jawabannya. Format WAJIB persis seperti ini, satu variasi per baris, tanpa nomor:\n"
        "Q: <variasi 1>\nQ: <variasi 2>"
        for row in base_rows
    ]

    variants_by_index = [[] for _ in base_rows]
    step = max(1, batch_size)
    for start in range(0, len(prompts), step):
        chunk = prompts[start:start + step]
        chunk_rows = base_rows[start:start + step]
        outputs = generate_batch("Text", model, processor, chunk, max_new_tokens=max_new_tokens, temperature=0.7)
        for offset, (row, text) in enumerate(zip(chunk_rows, outputs)):
            seen = {row["user"].strip().lower()}
            added = 0
            for variant in _PARAPHRASE_LINE_RE.findall(text):
                if added >= paraphrases_per_row:
                    break
                variant = variant.strip().strip('"').strip()
                if not variant or len(variant) > _MAX_LLM_QA_CHARS or variant.lower() in seen:
                    continue
                seen.add(variant.lower())
                variants_by_index[start + offset].append({"user": variant, "assistant": row["assistant"]})
                added += 1

    result = []
    for base, variants in zip(base_rows, variants_by_index):
        result.append(base)
        result.extend(variants)
    return result


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
