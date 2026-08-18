from __future__ import annotations

import pandas as pd
from transformers import TrainerCallback

from .constants import AUDIO_TARGET_MODULES, INSTRUCTION_PART, RESPONSE_PART


def load_model_and_processor(modality: str, model_name: str, max_seq_length: int, load_in_4bit: bool):
    """Load base model + processor/tokenizer. Each modality uses a different
    Unsloth loader class, matching the Gemma4 (E4B) Text/Vision/Audio notebooks."""
    if modality == "Vision":
        from unsloth import FastVisionModel, get_chat_template

        model, processor = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )
        processor = get_chat_template(processor, "gemma-4")
        return model, processor

    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    model, processor = FastModel.from_pretrained(
        model_name=model_name,
        dtype=None,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )
    processor = get_chat_template(processor, "gemma-4")
    return model, processor


def apply_lora(
    modality: str,
    model,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    finetune_attention_modules: bool,
    finetune_mlp_modules: bool,
    finetune_language_layers: bool,
    finetune_vision_layers: bool = True,
    seed: int = 3407,
):
    if modality == "Vision":
        from unsloth import FastVisionModel

        return FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=finetune_vision_layers,
            finetune_language_layers=finetune_language_layers,
            finetune_attention_modules=finetune_attention_modules,
            finetune_mlp_modules=finetune_mlp_modules,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            random_state=seed,
            use_rslora=False,
            loftq_config=None,
            target_modules="all-linear",
        )

    from unsloth import FastModel

    if modality == "Audio":
        return FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=finetune_language_layers,
            finetune_attention_modules=finetune_attention_modules,
            finetune_mlp_modules=finetune_mlp_modules,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            random_state=seed,
            use_rslora=False,
            loftq_config=None,
            target_modules=AUDIO_TARGET_MODULES,
        )

    # Text
    return FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=finetune_language_layers,
        finetune_attention_modules=finetune_attention_modules,
        finetune_mlp_modules=finetune_mlp_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        random_state=seed,
    )


class StreamlitTrainerCallback(TrainerCallback):
    """Pushes live training progress — progress bar, latest-step status, loss
    chart, and a scrolling text log — into Streamlit placeholders during the
    synchronous trainer.train() call (Streamlit flushes placeholder updates
    as they happen within a single script run)."""

    def __init__(self, progress_bar, status_text, chart_placeholder, log_placeholder=None, max_log_lines=200):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.chart_placeholder = chart_placeholder
        self.log_placeholder = log_placeholder
        self.max_log_lines = max_log_lines
        self.history = []
        self.log_lines = []

    def _log(self, line: str):
        self.log_lines.append(line)
        del self.log_lines[: -self.max_log_lines]  # keep only the most recent lines
        if self.log_placeholder is not None:
            self.log_placeholder.code("\n".join(self.log_lines))

    def on_train_begin(self, args, state, control, **kwargs):
        total = args.max_steps if args.max_steps and args.max_steps > 0 else "?"
        self._log(f"🚀 Training dimulai — target {total} steps.")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        self.history.append({"step": state.global_step, "loss": logs["loss"]})
        df = pd.DataFrame(self.history).set_index("step")
        self.chart_placeholder.line_chart(df["loss"])

        total = state.max_steps or 0
        frac = min(state.global_step / total, 1.0) if total else 0.0
        self.progress_bar.progress(frac)
        lr = logs.get("learning_rate", "-")
        grad_norm = logs.get("grad_norm")
        line = f"[{frac * 100:5.1f}%] Step {state.global_step}/{total} | loss={logs['loss']:.4f} | lr={lr}"
        if grad_norm is not None:
            line += f" | grad_norm={grad_norm:.4f}"
        self.status_text.text(line)
        self._log(line)

    def on_train_end(self, args, state, control, **kwargs):
        self._log(f"✅ Training selesai di step {state.global_step}.")


def build_trainer(modality: str, model, processor, train_dataset, eval_dataset, sft_kwargs: dict):
    from trl import SFTConfig, SFTTrainer

    if modality in ("Vision", "Audio"):
        from unsloth.trainer import UnslothVisionDataCollator

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor.tokenizer,
            data_collator=UnslothVisionDataCollator(model, processor),
            args=SFTConfig(
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                report_to="none",
                # compute_eval_loss() only reads metrics["eval_loss"], never predictions —
                # without this, Trainer.evaluate() accumulates full [seq_len, vocab_size]
                # logits per example on GPU for the whole eval pass, which is enormous for
                # Gemma's ~256k-token vocab and was the real cause of Eval-Loss-only OOMs.
                prediction_loss_only=True,
                eval_accumulation_steps=1,
                **sft_kwargs,
            ),
        )
        return trainer

    from unsloth.chat_templates import train_on_responses_only

    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            report_to="none",
            prediction_loss_only=True,
            eval_accumulation_steps=1,
            **sft_kwargs,
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )
    return trainer


def save_lora(model, processor, path: str):
    model.save_pretrained(path)
    processor.save_pretrained(path)


def zip_lora_adapter(path: str) -> bytes:
    """Zip a saved LoRA adapter folder into in-memory bytes, ready for
    st.download_button — so the adapter can be downloaded and re-uploaded
    in a different session/runtime (Colab's local disk doesn't survive a
    runtime restart). The folder's own name is kept as the zip's top-level
    entry, so unzipping recreates the same folder structure."""
    import io
    import os
    import zipfile

    base_dir = os.path.dirname(os.path.normpath(path)) or "."
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(path):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, base_dir))
    return buf.getvalue()


def extract_adapter_zip(uploaded_file, dest_dir: str) -> str:
    """Extract an uploaded adapter .zip (from zip_lora_adapter, or hand-made
    by the user) into `dest_dir`, then return the path to whichever
    directory actually holds adapter_config.json — the zip might extract
    the adapter directly into dest_dir, or one level deeper inside a
    wrapping folder, depending on how it was zipped, so this walks the
    extracted tree instead of assuming one fixed layout."""
    import os
    import zipfile

    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(uploaded_file) as zf:
        zf.extractall(dest_dir)
    for root, _, files in os.walk(dest_dir):
        if "adapter_config.json" in files:
            return root
    return dest_dir
