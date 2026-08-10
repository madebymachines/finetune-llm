import pandas as pd
import streamlit as st

from src.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    GEMMA4_MODELS,
    LORA_DEFAULTS,
    MODALITIES,
    SFT_DEFAULTS,
)
from src.data_utils import (
    apply_chat_template_to_dataset,
    build_conversations_from_template,
    build_audio_messages_from_hf,
    build_audio_messages_from_upload,
    build_vision_messages_from_hf,
    build_vision_messages_from_upload,
    conversations_to_dataset,
    extract_text_from_document,
    load_audio_array,
    load_hf_dataset,
    looks_like_sharegpt,
    media_train_eval_split,
    read_uploaded_table,
    sharegpt_df_to_dataset,
    train_eval_split,
)
from src.eval_utils import before_after_compare, compute_eval_loss, stream_chat_response
from src.gpu_utils import check_cuda, memory_snapshot
from src.train_utils import (
    StreamlitTrainerCallback,
    apply_lora,
    build_trainer,
    load_model_and_processor,
    save_lora,
)

st.set_page_config(page_title="Gemma-4 Finetune Studio", layout="wide")


def _use_extracted_text_as_system():
    st.session_state["text_system_template"] = st.session_state.get("_extracted_doc_text", "")

SCRATCH_DIR = "/tmp/gemma4_finetune_studio"

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "modality": "Text",
    "model": None,
    "processor": None,
    "lora_applied": False,
    "train_dataset": None,
    "eval_dataset": None,
    "trainer": None,
    "trained": False,
    "start_gpu_mem": None,
    "eval_log": [],
    "chat_history": [],
    "last_system_prompt": "",
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

st.title("🦎 Gemma-4 Finetune Studio")
st.caption("Run, test, and evaluate LoRA finetunes of Gemma-4 (Text / Vision / Audio), adapted from Unsloth's Gemma4 (E4B) notebooks.")

cuda_info = check_cuda()
if not cuda_info["available"]:
    st.error(
        "Tidak ada CUDA GPU terdeteksi di environment ini. Unsloth (4-bit training, "
        "bitsandbytes, xformers) membutuhkan GPU NVIDIA. Jalankan tool ini di mesin "
        "dengan CUDA (Colab, RunPod, Lambda, atau server GPU on-prem)."
    )
else:
    st.success(f"GPU terdeteksi: {cuda_info['name']} ({cuda_info['total_gb']} GB)")

tab_setup, tab_data, tab_train, tab_test, tab_eval = st.tabs(
    ["⚙️ Setup", "📊 Data", "🚀 Train", "💬 Test", "📈 Evaluate"]
)

# ---------------------------------------------------------------------------
# Setup tab
# ---------------------------------------------------------------------------
with tab_setup:
    st.subheader("0. Modalitas")
    modality = st.radio("Pilih modalitas finetuning", MODALITIES, horizontal=True, key="modality")
    if modality == "Vision":
        st.caption("Data: pasangan gambar + teks (caption/Q&A). Contoh bawaan: `unsloth/LaTeX_OCR`.")
    elif modality == "Audio":
        st.caption("Data: pasangan audio + transkrip/teks. Contoh bawaan: `kadirnar/Emilia-DE-B000000`.")
    else:
        st.caption("Data: percakapan teks (ShareGPT-style).")

    st.subheader("1. Load base model")
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Model", GEMMA4_MODELS, index=1)
        custom_model = st.text_input("Atau isi nama model manual (opsional)")
        model_name = custom_model.strip() or model_choice
    with col2:
        max_seq_length = st.number_input(
            "Max sequence length",
            min_value=128, max_value=32768,
            value=8192 if modality == "Audio" else 1024,
            step=128,
            disabled=(modality == "Vision"),
            help="Tidak dipakai untuk modalitas Vision (mengikuti notebook Unsloth).",
        )
        load_in_4bit = st.checkbox("Load in 4-bit", value=True)

    if st.button("Load Model", type="primary", disabled=not cuda_info["available"]):
        with st.spinner(f"Loading {model_name} ({modality})..."):
            model, processor = load_model_and_processor(modality, model_name, int(max_seq_length), load_in_4bit)
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.lora_applied = False
        st.session_state.start_gpu_mem = memory_snapshot()
        st.success(f"Model '{model_name}' ({modality}) berhasil dimuat.")

    if st.session_state.model is not None:
        st.info(f"Model aktif: **{model_name}** ({modality}) | Reserved memory: {memory_snapshot()} GB")

        st.subheader("2. LoRA adapters")
        lora_defaults = LORA_DEFAULTS[modality]
        c1, c2, c3 = st.columns(3)
        with c1:
            r = st.number_input("r (rank)", min_value=1, max_value=256, value=lora_defaults["r"])
            lora_alpha = st.number_input("lora_alpha", min_value=1, max_value=256, value=lora_defaults["lora_alpha"])
        with c2:
            lora_dropout = st.number_input("lora_dropout", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            seed = st.number_input("random_state (seed)", min_value=0, value=3407)
        with c3:
            finetune_attention = st.checkbox("Finetune attention modules", value=True)
            finetune_mlp = st.checkbox("Finetune MLP modules", value=True)
            finetune_language = st.checkbox("Finetune language layers", value=True)
            finetune_vision = True
            if modality == "Vision":
                finetune_vision = st.checkbox("Finetune vision layers", value=True)

        if st.button("Apply LoRA Adapters", disabled=st.session_state.lora_applied):
            with st.spinner("Applying LoRA adapters..."):
                st.session_state.model = apply_lora(
                    modality,
                    st.session_state.model,
                    r=int(r),
                    lora_alpha=int(lora_alpha),
                    lora_dropout=float(lora_dropout),
                    finetune_attention_modules=finetune_attention,
                    finetune_mlp_modules=finetune_mlp,
                    finetune_language_layers=finetune_language,
                    finetune_vision_layers=finetune_vision,
                    seed=int(seed),
                )
            st.session_state.lora_applied = True
            st.success("LoRA adapters applied.")

        if st.session_state.lora_applied:
            st.success("LoRA sudah aktif pada model ini.")

# ---------------------------------------------------------------------------
# Data tab
# ---------------------------------------------------------------------------
with tab_data:
    modality = st.session_state.modality

    # ----- TEXT -----
    if modality == "Text":
        st.subheader("Sumber data")
        source = st.radio("Pilih sumber", ["Dataset Hugging Face Hub", "Upload Custom Data"], horizontal=True)

        if source == "Dataset Hugging Face Hub":
            hf_name = st.text_input("Nama dataset", value="mlabonne/FineTome-100k")
            num_rows = st.number_input("Jumlah baris (0 = semua)", min_value=0, value=3000, step=100)
            if st.button("Load dataset dari Hub"):
                with st.spinner("Loading dataset..."):
                    ds = load_hf_dataset(hf_name, num_rows or None)
                    ds = sharegpt_df_to_dataset(ds.to_pandas()) if "conversations" in ds.column_names else ds
                st.session_state["_raw_dataset"] = ds
                st.session_state["last_system_prompt"] = ""
                st.success(f"{len(ds)} baris dimuat.")
                st.dataframe(ds.to_pandas().head())

        else:
            uploaded = st.file_uploader(
                "Upload CSV / Excel / JSON / JSONL",
                type=["csv", "xlsx", "xls", "json", "jsonl"],
                help=(
                    "Contoh: (1) CSV/Excel data produk (nama, deskripsi, harga, ...), atau "
                    "(2) spreadsheet hasil konversi dokumen instruction/rule + contoh Q&A "
                    "(kolom mis. category/system/user/assistant)."
                ),
            )
            if uploaded is not None:
                df = read_uploaded_table(uploaded)
                st.write(f"{len(df)} baris, kolom: {list(df.columns)}")
                st.dataframe(df.head())

                if looks_like_sharegpt(df):
                    st.info("Kolom 'conversations' terdeteksi -> menggunakan format ShareGPT langsung.")
                    if st.button("Gunakan data ini"):
                        st.session_state["_raw_dataset"] = sharegpt_df_to_dataset(df)
                        st.session_state["last_system_prompt"] = ""
                        st.success("Dataset siap.")
                else:
                    st.markdown("**Template Builder** — petakan kolom ke percakapan training memakai `{nama_kolom}`.")
                    st.caption(
                        "Contoh data produk: system=`Kamu adalah asisten produk toko.`, "
                        "user=`Apa itu {nama_produk}?`, assistant=`{deskripsi} Harga: {harga}`.\n\n"
                        "Contoh spreadsheet hasil konversi dokumen persona/guardrail/simulasi: "
                        "system=`{system}`, user=`{user}`, assistant=`{assistant}`."
                    )
                    with st.expander("📄 Opsional: upload dokumen persona/rule (PDF/DOCX/TXT) untuk auto-isi system template"):
                        doc_file = st.file_uploader(
                            "Dokumen persona/rule/guardrail",
                            type=["pdf", "docx", "txt"],
                            help=(
                                "Teksnya diekstrak apa adanya (bukan di-parse jadi Q&A) dan dipakai sebagai "
                                "system prompt yang sama untuk semua baris data CSV/Excel di atas. Kalau dialog "
                                "contoh di dokumen ini mau jadi baris training terpisah, konversi dulu ke "
                                "spreadsheet (kolom category/system/user/assistant)."
                            ),
                        )
                        if doc_file is not None:
                            try:
                                extracted = extract_text_from_document(doc_file)
                            except Exception as e:
                                st.error(f"Gagal membaca dokumen: {e}")
                                extracted = ""
                            if extracted:
                                st.session_state["_extracted_doc_text"] = extracted
                                st.text_area("Preview teks hasil ekstraksi", extracted, height=150, disabled=True)
                                st.button(
                                    "Pakai teks ini sebagai System / rule template",
                                    on_click=_use_extracted_text_as_system,
                                )

                    cols = list(df.columns)
                    st.caption(
                        "Kolom tersedia di data-mu: " + ", ".join(f"`{{{c}}}`" for c in cols)
                    )
                    user_placeholder = f"Apa itu {{{cols[0]}}}?" if cols else "Apa itu {nama_kolom}?"
                    assistant_placeholder = f"{{{cols[1] if len(cols) > 1 else cols[0]}}}" if cols else "{nama_kolom}"

                    system_template = st.text_area(
                        "System / rule template (opsional)", key="text_system_template",
                        placeholder="Kamu adalah asisten produk toko yang ramah.",
                    )
                    user_template = st.text_area("User message template", value="", placeholder=user_placeholder)
                    assistant_template = st.text_area(
                        "Assistant response template", value="", placeholder=assistant_placeholder
                    )
                    st.caption(
                        "⚠️ Wajib diisi: User message template & Assistant response template "
                        "(pakai placeholder di atas). Kalau dikosongkan, hasil percakapan akan kosong."
                    )

                    colp, colb = st.columns(2)
                    with colp:
                        if st.button("Preview 3 contoh"):
                            if not user_template.strip() or not assistant_template.strip():
                                st.warning("Isi dulu User message template dan Assistant response template.")
                            else:
                                try:
                                    preview = build_conversations_from_template(
                                        df.head(3), system_template, user_template, assistant_template
                                    )
                                    st.json(preview)
                                except KeyError as e:
                                    st.error(f"Kolom {e} tidak ditemukan di data. Cek nama placeholder-mu.")
                    with colb:
                        if st.button("Bangun dataset dari template", type="primary"):
                            if not user_template.strip() or not assistant_template.strip():
                                st.warning("Isi dulu User message template dan Assistant response template.")
                            else:
                                try:
                                    convos = build_conversations_from_template(
                                        df, system_template, user_template, assistant_template
                                    )
                                    st.session_state["_raw_dataset"] = conversations_to_dataset(convos)
                                    st.session_state["last_system_prompt"] = system_template
                                    st.success(f"Dataset dibangun: {len(convos)} percakapan.")
                                except KeyError as e:
                                    st.error(f"Kolom {e} tidak ditemukan di data. Cek nama placeholder-mu.")

        st.divider()
        st.subheader("Chat template & split")
        eval_ratio = st.slider("Porsi data untuk evaluasi (eval split)", 0.0, 0.5, 0.1, 0.05)

        if st.button("Terapkan chat template & siapkan train/eval set", disabled=st.session_state.processor is None):
            raw_ds = st.session_state.get("_raw_dataset")
            if raw_ds is None:
                st.warning("Belum ada dataset dimuat/dibangun di atas.")
            else:
                with st.spinner("Applying chat template..."):
                    formatted = apply_chat_template_to_dataset(raw_ds, st.session_state.processor)
                    train_ds, eval_ds = train_eval_split(formatted, eval_ratio)
                st.session_state.train_dataset = train_ds
                st.session_state.eval_dataset = eval_ds
                st.success(f"Train: {len(train_ds)} baris | Eval: {len(eval_ds) if eval_ds else 0} baris")
                st.text_area("Contoh hasil chat template (baris 0)", train_ds[0]["text"], height=200)

    # ----- VISION -----
    elif modality == "Vision":
        st.subheader("Sumber data (gambar + teks)")
        source = st.radio("Pilih sumber", ["Dataset Hugging Face Hub", "Upload gambar + CSV/Excel"], horizontal=True)
        system_prompt = st.text_area("System prompt (opsional)", value="")
        instruction = st.text_area("Instruksi/pertanyaan default untuk tiap gambar", value="Write the LaTeX representation for this image.")

        if source == "Dataset Hugging Face Hub":
            hf_name = st.text_input("Nama dataset", value="unsloth/LaTeX_OCR")
            col1, col2, col3 = st.columns(3)
            with col1:
                image_column = st.text_input("Kolom gambar", value="image")
            with col2:
                text_column = st.text_input("Kolom teks/caption", value="text")
            with col3:
                num_rows = st.number_input("Jumlah baris", min_value=1, value=200, step=50)
            if st.button("Load & bangun dataset"):
                with st.spinner("Loading dataset & membangun messages..."):
                    ds = load_hf_dataset(hf_name, num_rows)
                    messages = build_vision_messages_from_hf(ds, image_column, text_column, instruction, system_prompt)
                st.session_state["_media_messages"] = messages
                st.session_state["last_system_prompt"] = system_prompt
                st.success(f"{len(messages)} contoh dibangun.")
                st.image(messages[0]["messages"][-2]["content"][-1]["image"], caption="Contoh gambar #1", width=200)

        else:
            image_files = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
            caption_file = st.file_uploader("Upload CSV/Excel caption", type=["csv", "xlsx", "xls"], key="vision_caption_file")
            if image_files and caption_file is not None:
                df = read_uploaded_table(caption_file)
                st.write(f"{len(df)} baris, kolom: {list(df.columns)}")
                st.dataframe(df.head())
                cols = list(df.columns)
                c1, c2, c3 = st.columns(3)
                with c1:
                    filename_column = st.selectbox("Kolom nama file gambar", cols)
                with c2:
                    question_column = st.selectbox("Kolom pertanyaan (opsional)", ["(pakai instruksi default)"] + cols)
                with c3:
                    answer_column = st.selectbox("Kolom jawaban", cols)
                if st.button("Bangun dataset dari upload", type="primary"):
                    try:
                        files_by_name = {f.name: f for f in image_files}
                        q_col = None if question_column == "(pakai instruksi default)" else question_column
                        messages = build_vision_messages_from_upload(
                            files_by_name, df, filename_column, q_col, answer_column, instruction, system_prompt
                        )
                        st.session_state["_media_messages"] = messages
                        st.session_state["last_system_prompt"] = system_prompt
                        st.success(f"{len(messages)} contoh dibangun.")
                    except KeyError as e:
                        st.error(str(e))

        st.divider()
        st.subheader("Split train/eval")
        eval_ratio = st.slider("Porsi data untuk evaluasi (eval split)", 0.0, 0.5, 0.1, 0.05, key="vision_eval_ratio")
        if st.button("Siapkan train/eval set", type="primary"):
            messages = st.session_state.get("_media_messages")
            if not messages:
                st.warning("Belum ada dataset dibangun di atas.")
            else:
                train_ds, eval_ds = media_train_eval_split(messages, eval_ratio)
                st.session_state.train_dataset = train_ds
                st.session_state.eval_dataset = eval_ds
                st.success(f"Train: {len(train_ds)} contoh | Eval: {len(eval_ds) if eval_ds else 0} contoh")

    # ----- AUDIO -----
    else:
        st.subheader("Sumber data (audio + transkrip)")
        source = st.radio("Pilih sumber", ["Dataset Hugging Face Hub", "Upload audio + CSV/Excel"], horizontal=True)
        system_prompt = st.text_area("System prompt", value="You are an assistant that transcribes speech accurately.")
        instruction = st.text_area("Instruksi/pertanyaan default untuk tiap audio", value="Please transcribe this audio.")

        if source == "Dataset Hugging Face Hub":
            hf_name = st.text_input("Nama dataset", value="kadirnar/Emilia-DE-B000000")
            col1, col2, col3 = st.columns(3)
            with col1:
                audio_column = st.text_input("Kolom audio", value="audio")
            with col2:
                text_column = st.text_input("Kolom transkrip", value="text")
            with col3:
                num_rows = st.number_input("Jumlah baris", min_value=1, value=200, step=50)
            if st.button("Load & bangun dataset"):
                with st.spinner("Loading dataset & membangun messages..."):
                    ds = load_hf_dataset(hf_name, num_rows)
                    messages = build_audio_messages_from_hf(ds, audio_column, text_column, instruction, system_prompt)
                st.session_state["_media_messages"] = messages
                st.session_state["last_system_prompt"] = system_prompt
                st.success(f"{len(messages)} contoh dibangun.")

        else:
            audio_files = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "m4a"], accept_multiple_files=True)
            transcript_file = st.file_uploader("Upload CSV/Excel transkrip", type=["csv", "xlsx", "xls"], key="audio_transcript_file")
            if audio_files and transcript_file is not None:
                df = read_uploaded_table(transcript_file)
                st.write(f"{len(df)} baris, kolom: {list(df.columns)}")
                st.dataframe(df.head())
                cols = list(df.columns)
                c1, c2, c3 = st.columns(3)
                with c1:
                    filename_column = st.selectbox("Kolom nama file audio", cols)
                with c2:
                    question_column = st.selectbox("Kolom pertanyaan (opsional)", ["(pakai instruksi default)"] + cols)
                with c3:
                    answer_column = st.selectbox("Kolom transkrip/jawaban", cols)
                if st.button("Bangun dataset dari upload", type="primary"):
                    try:
                        files_by_name = {f.name: f for f in audio_files}
                        q_col = None if question_column == "(pakai instruksi default)" else question_column
                        messages = build_audio_messages_from_upload(
                            files_by_name, df, filename_column, q_col, answer_column, instruction, system_prompt,
                            tmp_dir=f"{SCRATCH_DIR}/data_audio",
                        )
                        st.session_state["_media_messages"] = messages
                        st.session_state["last_system_prompt"] = system_prompt
                        st.success(f"{len(messages)} contoh dibangun.")
                    except KeyError as e:
                        st.error(str(e))

        st.divider()
        st.subheader("Split train/eval")
        eval_ratio = st.slider("Porsi data untuk evaluasi (eval split)", 0.0, 0.5, 0.1, 0.05, key="audio_eval_ratio")
        if st.button("Siapkan train/eval set", type="primary"):
            messages = st.session_state.get("_media_messages")
            if not messages:
                st.warning("Belum ada dataset dibangun di atas.")
            else:
                train_ds, eval_ds = media_train_eval_split(messages, eval_ratio)
                st.session_state.train_dataset = train_ds
                st.session_state.eval_dataset = eval_ds
                st.success(f"Train: {len(train_ds)} contoh | Eval: {len(eval_ds) if eval_ds else 0} contoh")

# ---------------------------------------------------------------------------
# Train tab
# ---------------------------------------------------------------------------
with tab_train:
    modality = st.session_state.modality
    ready = st.session_state.lora_applied and st.session_state.train_dataset is not None
    if not ready:
        st.warning("Selesaikan tab Setup (load model + LoRA) dan tab Data (siapkan train dataset) dahulu.")
    else:
        sft_defaults = SFT_DEFAULTS[modality]
        st.subheader("Training config")
        c1, c2, c3 = st.columns(3)
        with c1:
            per_device_train_batch_size = st.number_input(
                "per_device_train_batch_size", min_value=1, value=sft_defaults["per_device_train_batch_size"]
            )
            gradient_accumulation_steps = st.number_input(
                "gradient_accumulation_steps", min_value=1, value=sft_defaults["gradient_accumulation_steps"]
            )
            if modality == "Text":
                warmup_steps = st.number_input("warmup_steps", min_value=0, value=sft_defaults["warmup_steps"])
            else:
                warmup_ratio = st.number_input("warmup_ratio", min_value=0.0, max_value=1.0, value=sft_defaults["warmup_ratio"], step=0.01)
        with c2:
            use_epochs = st.checkbox("Gunakan num_train_epochs (bukan max_steps)", value=False)
            if use_epochs:
                num_train_epochs = st.number_input("num_train_epochs", min_value=1, value=1)
                max_steps = -1
            else:
                max_steps = st.number_input("max_steps", min_value=1, value=60)
                num_train_epochs = None
            learning_rate = st.number_input("learning_rate", min_value=0.0, value=sft_defaults["learning_rate"], format="%.6f")
            lr_scheduler_type = st.selectbox(
                "lr_scheduler_type", ["linear", "cosine", "constant"],
                index=["linear", "cosine", "constant"].index(sft_defaults["lr_scheduler_type"]),
            )
        with c3:
            logging_steps = st.number_input("logging_steps", min_value=1, value=1)
            weight_decay = st.number_input("weight_decay", min_value=0.0, value=0.001, format="%.4f")
            train_seed = st.number_input("seed", min_value=0, value=3407)
            if modality != "Text":
                max_length = st.number_input("max_length", min_value=128, value=sft_defaults["max_length"])

        if st.button("🚀 Start Training", type="primary"):
            sft_kwargs = dict(
                per_device_train_batch_size=int(per_device_train_batch_size),
                gradient_accumulation_steps=int(gradient_accumulation_steps),
                learning_rate=float(learning_rate),
                logging_steps=int(logging_steps),
                optim="adamw_8bit",
                weight_decay=float(weight_decay),
                lr_scheduler_type=lr_scheduler_type,
                seed=int(train_seed),
            )
            if use_epochs:
                sft_kwargs["num_train_epochs"] = int(num_train_epochs)
            else:
                sft_kwargs["max_steps"] = int(max_steps)

            if modality == "Text":
                sft_kwargs["warmup_steps"] = int(warmup_steps)
            else:
                sft_kwargs["warmup_ratio"] = float(warmup_ratio)
                sft_kwargs["max_length"] = int(max_length)
                sft_kwargs["save_strategy"] = "steps"
                sft_kwargs["output_dir"] = "outputs"
                if modality == "Vision":
                    sft_kwargs["max_grad_norm"] = 0.3

            trainer = build_trainer(
                modality,
                st.session_state.model,
                st.session_state.processor,
                st.session_state.train_dataset,
                st.session_state.eval_dataset,
                sft_kwargs,
            )

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            chart_placeholder = st.empty()
            callback = StreamlitTrainerCallback(progress_bar, status_text, chart_placeholder)
            trainer.add_callback(callback)

            with st.spinner("Training..."):
                stats = trainer.train()

            st.session_state.trainer = trainer
            st.session_state.trained = True
            used_mem = memory_snapshot()
            start_mem = st.session_state.start_gpu_mem or 0
            st.success(
                f"Training selesai dalam {round(stats.metrics['train_runtime'] / 60, 2)} menit. "
                f"Peak reserved memory: {used_mem} GB (delta training: {round(used_mem - start_mem, 3)} GB)."
            )

        if st.session_state.trained:
            with st.expander("💾 Simpan LoRA adapter (opsional)"):
                lora_path = st.text_input("Path simpan LoRA", value="gemma_4_lora")
                if st.button("Save LoRA lokal"):
                    save_lora(st.session_state.model, st.session_state.processor, lora_path)
                    st.success(f"LoRA adapters disimpan ke '{lora_path}'.")

# ---------------------------------------------------------------------------
# Test tab
# ---------------------------------------------------------------------------
with tab_test:
    modality = st.session_state.modality
    if st.session_state.model is None:
        st.warning("Load model dulu di tab Setup.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            use_adapter = st.checkbox(
                "Gunakan adapter (hasil finetune)", value=True, disabled=not st.session_state.lora_applied
            )
        with col2:
            temperature = st.slider("temperature", 0.0, 2.0, DEFAULT_TEMPERATURE, 0.05)
        with col3:
            top_p = st.slider("top_p", 0.0, 1.0, DEFAULT_TOP_P, 0.01)
        with col4:
            top_k = st.slider("top_k", 1, 128, DEFAULT_TOP_K, 1)
        max_new_tokens = st.slider("max_new_tokens", 16, 1024, 256, 16)
        system_prompt = st.text_input(
            "System prompt (opsional)", value=st.session_state.get("last_system_prompt", ""),
            help="Default terisi dari system prompt yang dipakai waktu membangun dataset training di tab Data — "
            "samakan dengan training supaya perilaku model konsisten.",
        )

        if modality == "Text":
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            prompt = st.chat_input("Tulis pesan...")
            if prompt:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    response = st.write_stream(
                        stream_chat_response(
                            modality, st.session_state.model, st.session_state.processor,
                            text=prompt, system_prompt=system_prompt,
                            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                            use_adapter=use_adapter,
                        )
                    )
                st.session_state.chat_history.append({"role": "assistant", "content": response})

            if st.session_state.chat_history and st.button("Bersihkan riwayat chat"):
                st.session_state.chat_history = []
                st.rerun()

        elif modality == "Vision":
            image_file = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg", "webp"], key="test_image")
            question = st.text_input("Pertanyaan", value="Apa isi gambar ini?")
            if image_file is not None:
                from PIL import Image

                image = Image.open(image_file).convert("RGB")
                st.image(image, width=300)
                if st.button("Kirim", type="primary"):
                    response = st.write_stream(
                        stream_chat_response(
                            modality, st.session_state.model, st.session_state.processor,
                            text=question, image=image, system_prompt=system_prompt,
                            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                            use_adapter=use_adapter,
                        )
                    )

        else:  # Audio
            audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "m4a"], key="test_audio")
            question = st.text_input("Pertanyaan", value="Please transcribe this audio.")
            if audio_file is not None:
                st.audio(audio_file)
                if st.button("Kirim", type="primary"):
                    audio_array = load_audio_array(audio_file, f"{SCRATCH_DIR}/test_audio")
                    response = st.write_stream(
                        stream_chat_response(
                            modality, st.session_state.model, st.session_state.processor,
                            text=question, audio=audio_array, system_prompt=system_prompt,
                            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                            use_adapter=use_adapter,
                        )
                    )

# ---------------------------------------------------------------------------
# Evaluate tab
# ---------------------------------------------------------------------------
with tab_eval:
    modality = st.session_state.modality
    st.subheader("Eval loss / perplexity")
    if st.session_state.trainer is None:
        st.warning("Butuh trainer aktif (jalankan Train dulu) dan eval dataset (dari tab Data, eval split > 0).")
    elif st.session_state.eval_dataset is None:
        st.warning("Eval dataset kosong. Set eval split > 0 di tab Data lalu ulangi 'siapkan train/eval set'.")
    else:
        label = st.text_input("Label run ini (mis. 'before' / 'after')", value="after")
        if st.button("Hitung Eval Loss"):
            with st.spinner("Evaluating..."):
                metrics, perplexity = compute_eval_loss(st.session_state.trainer)
            st.session_state.eval_log.append(
                {"label": label, "eval_loss": metrics.get("eval_loss"), "perplexity": perplexity}
            )
        if st.session_state.eval_log:
            st.table(st.session_state.eval_log)

    st.divider()
    st.subheader("Before vs After comparison")
    if not st.session_state.lora_applied:
        st.warning("Perlu LoRA adapter aktif untuk membandingkan base vs finetuned.")
    else:
        system_prompt_eval = st.text_input(
            "System prompt (opsional)", value=st.session_state.get("last_system_prompt", ""),
            key="eval_system_prompt",
            help="Default terisi dari system prompt yang dipakai waktu membangun dataset training di tab Data.",
        )
        compare_max_new_tokens = st.slider("max_new_tokens (compare)", 16, 512, 256, 16, key="compare_tokens")
        items = None

        if modality == "Text":
            prompts_text = st.text_area("Prompt uji (satu per baris)", height=150, placeholder="Apa itu produk X?\n...")
            if st.button("Bandingkan Base vs Finetuned", type="primary"):
                prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
                if not prompts:
                    st.warning("Isi minimal satu prompt.")
                else:
                    items = [{"text": p} for p in prompts]

        elif modality == "Vision":
            question = st.text_input("Pertanyaan untuk tiap gambar", value="Apa isi gambar ini?")
            image_files = st.file_uploader("Upload gambar (bisa lebih dari satu)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, key="eval_images")
            if st.button("Bandingkan Base vs Finetuned", type="primary"):
                if not image_files:
                    st.warning("Upload minimal satu gambar.")
                else:
                    from PIL import Image

                    items = [{"text": question, "image": Image.open(f).convert("RGB")} for f in image_files]

        else:  # Audio
            question = st.text_input("Pertanyaan untuk tiap audio", value="Please transcribe this audio.")
            audio_files = st.file_uploader("Upload audio (bisa lebih dari satu)", type=["wav", "mp3", "flac", "m4a"], accept_multiple_files=True, key="eval_audios")
            if st.button("Bandingkan Base vs Finetuned", type="primary"):
                if not audio_files:
                    st.warning("Upload minimal satu audio.")
                else:
                    items = [
                        {"text": question, "audio": load_audio_array(f, f"{SCRATCH_DIR}/eval_audio")}
                        for f in audio_files
                    ]

        if items:
            with st.spinner("Generating base & finetuned outputs..."):
                results = before_after_compare(
                    modality, st.session_state.model, st.session_state.processor, items,
                    system_prompt=system_prompt_eval, max_new_tokens=compare_max_new_tokens,
                )
            st.session_state["_compare_results"] = results

        if st.session_state.get("_compare_results"):
            df_results = pd.DataFrame(st.session_state["_compare_results"])
            st.dataframe(df_results, use_container_width=True)
            st.download_button(
                "Download hasil (CSV)",
                df_results.to_csv(index=False).encode("utf-8"),
                file_name="before_after_comparison.csv",
                mime="text/csv",
            )
