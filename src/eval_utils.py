from __future__ import annotations

import math
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
    """items: list of {"text": str, "image"?: PIL.Image, "audio"?: np.ndarray,
    "system_prompt"?: str}. Each item may override `system_prompt` (e.g. with
    per-question Knowledge Base retrieval context); falls back to the shared
    `system_prompt` arg otherwise. For each item, generate with the LoRA
    adapter disabled (base model behaviour) and enabled (finetuned), reusing
    the same loaded weights."""
    results = []
    for item in items:
        kwargs = dict(
            text=item["text"],
            image=item.get("image"),
            audio=item.get("audio"),
            system_prompt=item.get("system_prompt", system_prompt),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        base_output = generate_response(modality, model, processor, use_adapter=False, **kwargs)
        finetuned_output = generate_response(modality, model, processor, use_adapter=True, **kwargs)
        results.append({"prompt": item["text"], "base_model": base_output, "finetuned_model": finetuned_output})
    return results
