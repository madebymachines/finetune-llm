from __future__ import annotations

import math
import re
import threading

from .constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P


def compute_eval_loss(trainer):
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss")
    perplexity = math.exp(loss) if loss is not None else None
    return metrics, perplexity


def build_messages(modality: str, text: str, image=None, audio=None, system_prompt: str = ""):
    """Build a chat-template-ready messages list for the given modality."""
    user_content = []
    if modality == "Audio" and audio is not None:
        user_content.append({"type": "audio", "audio": audio})
    if modality == "Vision" and image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": text})

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
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
    max_new_tokens: int = 256,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    use_adapter: bool = True,
) -> str:
    messages = build_messages(modality, text, image=image, audio=audio, system_prompt=system_prompt)
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


_QA_PAIR_RE = re.compile(r'Q\d*\s*:\s*(.+?)\s*\n\s*A\d*\s*:\s*(.+?)(?=\n\s*Q\d*\s*:|\Z)', re.DOTALL | re.IGNORECASE)
# The last matched answer absorbs any trailing text the model writes after the
# final Q/A block (the regex has no marker to stop at otherwise) — cap length
# so a stray trailing ramble can't turn into a multi-thousand-character row.
# This is the same failure mode (and same fix) as parse_user_ai_examples()'s
# _MAX_EXAMPLE_CHARS, which caused a real training OOM earlier in this project.
_MAX_LLM_QA_CHARS = 500


def generate_qa_pairs_llm(model, processor, df, questions_per_row: int = 2, max_new_tokens: int = 300) -> list[dict]:
    """Alternative to data_utils.auto_generate_qa_rows() that asks the
    already-loaded model to write `questions_per_row` naturally-varied Q&A
    pairs per row, strictly grounded in that row's own column values, instead
    of the fixed rule-based templates. Costs one generate() call per row, so
    it's an opt-in the caller should gate behind a model-loaded check and a
    progress spinner — this function has no Streamlit/UI awareness itself.

    use_adapter defaults to True (via generate_response) rather than False:
    at Data-tab time a LoRA adapter may not even be attached yet (Setup's
    "Apply LoRA" step is independent of Data tab), and model.disable_adapter()
    raises on a plain (non-PEFT) model — True works unconditionally, and an
    untrained adapter (zero-initialized B matrix) behaves identically to the
    base model anyway."""
    rows = []
    for _, row in df.iterrows():
        facts = "\n".join(f"- {c}: {v}" for c, v in row.items() if v == v and str(v).strip())  # v == v filters NaN
        if not facts:
            continue
        prompt = (
            f"Berikut data satu produk/item:\n{facts}\n\n"
            f"Buat {questions_per_row} pasang pertanyaan dan jawaban dalam Bahasa Indonesia tentang data ini. "
            "Pertanyaannya harus bervariasi gayanya (jangan semua diawali 'Apa itu'). "
            "Jawabannya HARUS hanya berdasarkan data di atas, jangan menambahkan informasi yang tidak ada. "
            "Format WAJIB persis seperti ini, satu pasang per blok:\n"
            "Q1: <pertanyaan>\nA1: <jawaban>\nQ2: <pertanyaan>\nA2: <jawaban>"
        )
        text = generate_response(
            "Text", model, processor, text=prompt, max_new_tokens=max_new_tokens, temperature=0.7,
        )
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
    max_new_tokens: int = 256,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    use_adapter: bool = True,
):
    """Generator yielding text chunks, for use with st.write_stream."""
    from transformers import TextIteratorStreamer

    messages = build_messages(modality, text, image=image, audio=audio, system_prompt=system_prompt)
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
):
    """items: list of {"text": str, "image"?: PIL.Image, "audio"?: np.ndarray}.
    For each item, generate with the LoRA adapter disabled (base model
    behaviour) and enabled (finetuned), reusing the same loaded weights."""
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
