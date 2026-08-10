import pandas as pd
import streamlit as st

from src.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_KB_MAX_CHARS,
    DEFAULT_KB_TOP_K,
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
    chunk_text,
    conversations_to_dataset,
    extract_text_from_document,
    load_audio_array,
    load_hf_dataset,
    looks_like_sharegpt,
    media_train_eval_split,
    parse_user_ai_examples,
    read_uploaded_table,
    resolve_conversation_columns,
    row_to_text,
    sharegpt_df_to_dataset,
    train_eval_split,
)
from src.eval_utils import before_after_compare, compute_eval_loss, stream_chat_response
from src.gpu_utils import check_cuda, clear_gpu_cache, memory_snapshot
from src.retrieval import embed_passages, embed_query, get_embedder, retrieve_top_chunks
from src.train_utils import (
    StreamlitTrainerCallback,
    apply_lora,
    build_trainer,
    load_model_and_processor,
    save_lora,
)

st.set_page_config(page_title="Gemma-4 Finetune Studio", layout="wide")

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
    "persona_system_prompt": "",
    "kb_sources": [],
    "kb_chunks": [],
    "kb_embeddings": None,
    "kb_embedding_model": DEFAULT_EMBEDDING_MODEL,
    "kb_top_k": DEFAULT_KB_TOP_K,
    "kb_max_chars": DEFAULT_KB_MAX_CHARS,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def kb_augment(query_text: str, use_kb: bool) -> tuple[str, list[dict]]:
    """Retrieve the most relevant Knowledge Base chunks for `query_text` and
    render them as a text block to append to a system prompt. Returns ("", [])
    if the KB isn't built/enabled or nothing relevant was found, so callers
    can always safely append the result."""
    if not use_kb or st.session_state["kb_embeddings"] is None or not query_text.strip():
        return "", []
    embedder = get_embedder(st.session_state["kb_embedding_model"])
    query_vec = embed_query(embedder, query_text)
    hits = retrieve_top_chunks(
        query_vec,
        st.session_state["kb_embeddings"],
        st.session_state["kb_chunks"],
        top_k=st.session_state["kb_top_k"],
        max_chars=st.session_state["kb_max_chars"],
    )
    if not hits:
        return "", []
    block = "Berikut informasi relevan dari knowledge base:\n" + "\n".join(
        f"- ({h['source']}) {h['text']}" for h in hits
    )
    return block, hits

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
    col_gpu1, col_gpu2 = st.columns([4, 1])
    with col_gpu1:
        st.success(
            f"GPU terdeteksi: {cuda_info['name']} ({cuda_info['total_gb']} GB) | "
            f"Reserved memory: {memory_snapshot()} GB"
        )
    with col_gpu2:
        if st.button("🧹 Bersihkan GPU cache", help="Lepas memori CUDA yang masih ditahan tapi tidak dipakai aktif (sisa training, fragmentasi cache) — tidak menghapus model/adapter yang sedang dimuat."):
            clear_gpu_cache()
            st.rerun()

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
        # Release whatever was loaded before (model, trainer + its optimizer/gradients)
        # first — otherwise the old model stays pinned in GPU memory via st.session_state.trainer
        # even after we overwrite st.session_state.model, and it silently doubles VRAM usage
        # in a way clear_gpu_cache() alone can't fix (it's a live reference, not stale cache).
        st.session_state.model = None
        st.session_state.processor = None
        st.session_state.trainer = None
        st.session_state.lora_applied = False
        st.session_state.trained = False
        st.session_state.train_dataset = None
        st.session_state.eval_dataset = None
        clear_gpu_cache()

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
        sub_persona, sub_kb = st.tabs(
            ["🎭 Persona & Guardrail (Training)", "📦 Knowledge Base (Katalog / Chat with Document)"]
        )

        # =================================================================
        # SUB-TAB 1 — Persona & Guardrail: data yang benar-benar di-finetune
        # =================================================================
        with sub_persona:
            n_train = len(st.session_state["_raw_dataset"]) if st.session_state.get("_raw_dataset") is not None else 0
            if n_train:
                st.success(f"✅ Dataset training siap: {n_train} baris.")
            else:
                st.warning("⚠️ Belum ada dataset training.")
            st.caption(
                "Data di sini benar-benar di-**finetune** ke model: mengajarkan gaya bicara & batasan "
                "(aman/tidak aman). Untuk katalog produk atau data referensi lain, pakai tab "
                "**Knowledge Base** di sebelah — data itu tidak perlu training ulang."
            )
            source = st.radio(
                "Sumber data training", ["Upload Custom Data", "Dataset Hugging Face Hub"], horizontal=True
            )

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
                with st.expander("📄 Langkah 1: upload dokumen persona/rule (PDF/DOCX/TXT)", expanded=True):
                    st.caption(
                        "Teksnya diekstrak apa adanya dan dipakai sebagai system prompt. Kalau dokumennya berisi "
                        "contoh dialog dengan pola `User: \"...\" AI: \"...\"`, contoh itu juga otomatis ditarik "
                        "jadi baris percakapan training di tabel Langkah 2 — cek & edit dulu sebelum dipakai."
                    )
                    doc_file = st.file_uploader(
                        "Dokumen persona/rule/guardrail", type=["pdf", "docx", "txt"], key="persona_doc_uploader"
                    )
                    if doc_file is not None:
                        doc_id = f"{doc_file.name}:{doc_file.size}"
                        already_parsed = st.session_state.get("_persona_doc_id") == doc_id
                        force_reparse = False
                        if already_parsed:
                            force_reparse = st.button(
                                "🔄 Parse ulang dokumen ini (regenerate tabel Langkah 2)",
                                key="reparse_persona_doc",
                                help="Menimpa tabel Langkah 2 saat ini dengan hasil parsing ulang — pakai ini "
                                "kalau mau coba lagi setelah mengedit tabel secara tidak sengaja, atau setelah "
                                "aturan auto-extract-nya diperbaiki.",
                            )
                        if not already_parsed or force_reparse:
                            try:
                                extracted = extract_text_from_document(doc_file)
                                st.session_state["_extracted_doc_text"] = extracted
                                st.session_state["_persona_doc_id"] = doc_id
                                parsed = parse_user_ai_examples(extracted)
                                if parsed:
                                    st.session_state["_persona_examples_df"] = pd.DataFrame(parsed)
                                    st.success(
                                        f"✅ {len(parsed)} contoh percakapan terdeteksi otomatis dari dokumen — "
                                        "cek & edit di tabel Langkah 2 di bawah sebelum dipakai."
                                    )
                                else:
                                    st.info(
                                        "Tidak ada pola `User: ... AI: ...` terdeteksi otomatis di dokumen ini — "
                                        "isi tabel Langkah 2 manual, atau upload file percakapan siap pakai."
                                    )
                            except Exception as e:
                                st.error(f"Gagal membaca dokumen: {e}")
                        if st.session_state.get("_extracted_doc_text"):
                            st.text_area(
                                "Preview teks hasil ekstraksi", st.session_state["_extracted_doc_text"],
                                height=140, disabled=True,
                            )

                st.markdown("**System prompt untuk training** (dipakai di semua baris kecuali kolom `system` diisi manual di tabel Langkah 2)")
                st.caption(
                    "⚠️ Jangan tempel seluruh isi dokumen di sini — system prompt yang kepanjangan bikin "
                    "training gagal (response ke-truncate sebelum sempat dipelajari, karena diulang di "
                    "**setiap** baris). Ringkas saja persona & batasannya; contoh dialog di dokumen sudah "
                    "otomatis masuk sebagai baris training terpisah di tabel Langkah 2, tidak perlu diulang di sini."
                )

                def _prefill_persona_system_prompt():
                    st.session_state["persona_system_prompt"] = st.session_state.get("_extracted_doc_text", "")[:500].strip()

                if st.session_state.get("_extracted_doc_text"):
                    st.button(
                        "Isi dari cuplikan dokumen (500 karakter pertama — edit lagi setelahnya)",
                        key="prefill_persona_system_btn", on_click=_prefill_persona_system_prompt,
                    )
                st.text_area("System prompt", key="persona_system_prompt", height=100)
                CHARS_PER_TOKEN = 4  # rough heuristic for mixed ID/EN text — not exact, just a budget sanity check
                sys_chars = len(st.session_state["persona_system_prompt"])
                sys_tokens_est = sys_chars // CHARS_PER_TOKEN
                budget_left = max_seq_length - sys_tokens_est
                st.caption(
                    f"≈{sys_chars} karakter (≈{sys_tokens_est} token). Max sequence length saat ini "
                    f"(dari tab Setup): **{max_seq_length}** token — sisa ruang untuk user+assistant+token "
                    f"spesial per baris: ≈{budget_left} token."
                )
                if budget_left < 150:
                    st.error(
                        f"🚫 Nyaris pasti training gagal (truncation) — sisa ruang cuma ≈{budget_left} token, "
                        f"hampir tidak cukup untuk pesan user+assistant. Persingkat system prompt ini jadi di "
                        f"bawah ≈{max(0, (max_seq_length - 300)) * CHARS_PER_TOKEN} karakter, atau naikkan "
                        "Max sequence length signifikan lebih tinggi (perhatikan VRAM)."
                    )
                elif budget_left < 400:
                    st.warning(
                        f"⚠️ Sisa ruang cuma ≈{budget_left} token — aman untuk balasan pendek, tapi bisa "
                        "ke-truncate kalau ada baris user/assistant yang panjang."
                    )

                st.markdown("**Langkah 2: contoh percakapan** (auto-extract dari dokumen di atas + bisa diedit)")
                st.caption(
                    "Kolom `system` kosong = pakai System prompt di atas untuk baris itu. Bisa juga isi/tambah "
                    "baris manual di sini walau tidak upload dokumen sama sekali."
                )
                if "_persona_examples_df" not in st.session_state:
                    st.session_state["_persona_examples_df"] = pd.DataFrame(columns=["system", "user", "assistant"])
                edited_examples = st.data_editor(
                    st.session_state["_persona_examples_df"],
                    num_rows="dynamic",
                    use_container_width=True,
                    key="persona_examples_editor",
                    column_config={
                        "system": st.column_config.TextColumn("system (opsional)"),
                        "user": st.column_config.TextColumn("user"),
                        "assistant": st.column_config.TextColumn("assistant"),
                    },
                )
                st.session_state["_persona_examples_df"] = edited_examples

                if st.button("✅ Gunakan tabel ini sebagai dataset training", type="primary", key="use_persona_table"):
                    valid_rows = edited_examples.fillna("")
                    valid_rows = valid_rows[
                        (valid_rows["user"].astype(str).str.strip() != "")
                        & (valid_rows["assistant"].astype(str).str.strip() != "")
                    ]
                    if valid_rows.empty:
                        st.warning("Tabel masih kosong — isi minimal satu baris `user` + `assistant`.")
                    else:
                        default_system = st.session_state.get("persona_system_prompt", "").strip()
                        convos = []
                        for _, row in valid_rows.iterrows():
                            convo = []
                            sys_msg = str(row.get("system", "")).strip() or default_system
                            if sys_msg:
                                convo.append({"role": "system", "content": sys_msg})
                            convo.append({"role": "user", "content": str(row["user"])})
                            convo.append({"role": "assistant", "content": str(row["assistant"])})
                            convos.append(convo)
                        st.session_state["_raw_dataset"] = conversations_to_dataset(convos)
                        any_row_system = (valid_rows["system"].astype(str).str.strip() != "").any()
                        st.session_state["last_system_prompt"] = "" if any_row_system else default_system
                        st.success(f"Dataset training siap: {len(convos)} percakapan.")

                        longest_idx = max(
                            range(len(convos)), key=lambda i: sum(len(m["content"]) for m in convos[i])
                        )
                        longest_convo = convos[longest_idx]
                        longest_chars = sum(len(m["content"]) for m in longest_convo)
                        longest_tokens_est = longest_chars // CHARS_PER_TOKEN
                        if longest_tokens_est >= max_seq_length - 50:
                            st.error(
                                f"🚫 Baris ke-{longest_idx + 1} di dataset ini ≈{longest_chars} karakter "
                                f"(≈{longest_tokens_est} token) — hampir pasti akan ke-truncate dengan Max "
                                f"sequence length={max_seq_length}. Cari baris ini di tabel Langkah 2 (pakai ikon "
                                "🔍 di pojok kanan atas tabel, cari potongan teks di bawah) lalu perbaiki/hapus:"
                            )
                            with st.expander("🔍 Isi baris bermasalah"):
                                for m in longest_convo:
                                    preview = m["content"][:1500] + ("…" if len(m["content"]) > 1500 else "")
                                    st.text(f"[{m['role']}] {preview}")
                        else:
                            st.caption(
                                f"Baris terpanjang di dataset ini: ≈{longest_chars} karakter "
                                f"(≈{longest_tokens_est} token) dari Max sequence length={max_seq_length}."
                            )

                st.divider()
                st.markdown("**Atau: upload file percakapan siap pakai** (CSV/Excel/JSON/JSONL)")
                train_file = st.file_uploader(
                    "Data percakapan training",
                    type=["csv", "xlsx", "xls", "json", "jsonl"],
                    key="train_file_uploader",
                    help=(
                        "Kolom `conversations` (format ShareGPT), atau `user`+`assistant` (+`system` opsional) "
                        "akan terdeteksi otomatis — tidak perlu atur manual kalau kolomnya sudah cocok."
                    ),
                )

                if train_file is not None:
                    df = read_uploaded_table(train_file)
                    cols = list(df.columns)
                    st.write(f"{len(df)} baris, kolom: {cols}")
                    st.dataframe(df.head())

                    resolved = resolve_conversation_columns(df)

                    if resolved["mode"] == "sharegpt":
                        st.success("✅ Kolom `conversations` terdeteksi (format ShareGPT) — siap pakai langsung.")
                        if st.button("Gunakan data ini", type="primary", key="use_sharegpt"):
                            st.session_state["_raw_dataset"] = sharegpt_df_to_dataset(df)
                            st.session_state["last_system_prompt"] = ""
                            st.success("Dataset training siap.")

                    elif resolved["mode"] == "flat":
                        user_col, assistant_col = resolved["user"], resolved["assistant"]
                        system_col = resolved.get("system")
                        default_system = st.session_state.get("persona_system_prompt", "").strip()
                        flat_system_template = f"{{{system_col}}}" if system_col else default_system

                        st.success(f"✅ Kolom `{user_col}`/`{assistant_col}` terdeteksi — siap pakai langsung.")
                        st.caption(
                            "System prompt: " + (
                                f"diambil dari kolom `{system_col}` per baris" if system_col
                                else ("dari 'System prompt untuk training' di atas" if default_system else "kosong (tidak diisi)")
                            )
                        )
                        colp, colb = st.columns(2)
                        with colp:
                            if st.button("Preview 3 contoh", key="preview_flat"):
                                try:
                                    preview = build_conversations_from_template(
                                        df.head(3), flat_system_template, f"{{{user_col}}}", f"{{{assistant_col}}}"
                                    )
                                    st.json(preview)
                                except KeyError as e:
                                    st.error(f"Kolom {e} tidak ditemukan di data.")
                        with colb:
                            if st.button("Gunakan data ini", type="primary", key="use_flat"):
                                convos = build_conversations_from_template(
                                    df, flat_system_template, f"{{{user_col}}}", f"{{{assistant_col}}}"
                                )
                                st.session_state["_raw_dataset"] = conversations_to_dataset(convos)
                                st.session_state["last_system_prompt"] = "" if system_col else flat_system_template
                                st.success(f"Dataset training siap: {len(convos)} percakapan.")

                                longest_chars = max(sum(len(m["content"]) for m in convo) for convo in convos)
                                longest_tokens_est = longest_chars // CHARS_PER_TOKEN
                                if longest_tokens_est >= max_seq_length - 50:
                                    st.error(
                                        f"🚫 Baris terpanjang di dataset ini ≈{longest_chars} karakter "
                                        f"(≈{longest_tokens_est} token) — hampir pasti ke-truncate dengan Max "
                                        f"sequence length={max_seq_length}. Kalau kolom `system`-mu berisi teks "
                                        "panjang (mis. seluruh dokumen), persingkat dulu di file sumbernya."
                                    )
                                else:
                                    st.caption(
                                        f"Baris terpanjang di dataset ini: ≈{longest_chars} karakter "
                                        f"(≈{longest_tokens_est} token) dari Max sequence length={max_seq_length}."
                                    )

                    else:
                        st.warning(
                            "Kolom belum sesuai format standar (`conversations`, atau `user`+`assistant`). "
                            "Atur pemetaan kolom manual di bawah."
                        )
                        with st.expander("⚙️ Atur pemetaan kolom manual", expanded=True):
                            st.caption(
                                "Dipakai kalau data-mu berupa fakta mentah (mis. nama_produk/deskripsi/harga) "
                                "yang perlu dirakit jadi kalimat tanya-jawab, memakai `{nama_kolom}`."
                            )
                            doc_text = st.session_state.get("_extracted_doc_text", "")
                            system_template = st.text_area(
                                "System / rule template (opsional)", key="text_system_template",
                                placeholder=(doc_text[:200] if doc_text else "Kamu adalah asisten yang ramah."),
                            )
                            user_template = st.text_area(
                                "User message template", value="",
                                placeholder=(f"Apa itu {{{cols[0]}}}?" if cols else "Apa itu {nama_kolom}?"),
                            )
                            assistant_template = st.text_area(
                                "Assistant response template", value="",
                                placeholder=(f"{{{cols[1] if len(cols) > 1 else cols[0]}}}" if cols else "{nama_kolom}"),
                            )
                            st.caption(
                                "⚠️ Wajib diisi: User message template & Assistant response template. "
                                "Kalau dikosongkan, hasil percakapan akan kosong."
                            )
                            colp, colb = st.columns(2)
                            with colp:
                                if st.button("Preview 3 contoh", key="preview_manual"):
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
                                if st.button("Bangun dataset dari template", type="primary", key="build_manual"):
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

        # =================================================================
        # SUB-TAB 2 — Knowledge Base: katalog produk / dokumen, TIDAK di-training.
        # Chunk + embed + retrieve secara otomatis per pertanyaan (RAG).
        # =================================================================
        with sub_kb:
            if st.session_state["kb_embeddings"] is not None:
                st.success(
                    f"✅ Knowledge Base aktif: {len(st.session_state['kb_sources'])} sumber, "
                    f"{len(st.session_state['kb_chunks'])} potongan terindeks."
                )
            elif st.session_state["kb_chunks"]:
                st.warning("⚠️ Ada sumber yang belum diindeks — klik 'Bangun Knowledge Base' di bawah.")
            else:
                st.warning("⚠️ Belum ada Knowledge Base.")
            st.caption(
                "Data di sini **tidak** di-training. Setiap kamu upload katalog produk (CSV/Excel/JSON) "
                "atau dokumen apapun (PDF/DOCX/TXT/MD), datanya dipecah jadi potongan-potongan kecil. "
                "Saat chat di tab Test/Evaluate, hanya potongan yang paling relevan dengan pertanyaanmu "
                "yang otomatis disisipkan ke context — jadi jawabannya selalu berdasarkan data terbaru, "
                "tanpa training ulang, dan tetap akurat walau pertanyaannya dikasih variasi kata."
            )

            kb_files = st.file_uploader(
                "Upload katalog produk (CSV/Excel/JSON/JSONL) dan/atau dokumen (PDF/DOCX/TXT/MD) "
                "— bisa banyak file sekaligus",
                type=["csv", "xlsx", "xls", "json", "jsonl", "pdf", "docx", "txt", "md"],
                accept_multiple_files=True,
                key="kb_uploader",
            )
            TABLE_EXTS = (".csv", ".xlsx", ".xls", ".json", ".jsonl")
            existing_ids = {s["id"] for s in st.session_state["kb_sources"]}
            for f in kb_files or []:
                file_id = f"{f.name}:{f.size}"
                if file_id in existing_ids:
                    continue
                try:
                    if f.name.lower().endswith(TABLE_EXTS):
                        kb_df = read_uploaded_table(f)
                        texts = [row_to_text(row) for _, row in kb_df.iterrows()]
                        texts = [t for t in texts if t]
                        source_type = "table"
                    else:
                        texts = chunk_text(extract_text_from_document(f))
                        source_type = "document"
                except Exception as e:
                    st.error(f"Gagal memproses '{f.name}': {e}")
                    continue
                if not texts:
                    st.warning(f"'{f.name}' tidak menghasilkan data (kosong).")
                    continue
                for t in texts:
                    st.session_state["kb_chunks"].append({"text": t, "source": f.name})
                st.session_state["kb_sources"].append(
                    {"id": file_id, "name": f.name, "type": source_type, "n_chunks": len(texts)}
                )
                st.session_state["kb_embeddings"] = None  # stale index, needs rebuild

            if st.session_state["kb_sources"]:
                st.markdown(
                    f"**Sumber Knowledge Base ({len(st.session_state['kb_sources'])}, "
                    f"{len(st.session_state['kb_chunks'])} potongan):**"
                )
                for src in st.session_state["kb_sources"]:
                    c1, c2, c3 = st.columns([4, 2, 1])
                    c1.write(("📊 " if src["type"] == "table" else "📄 ") + src["name"])
                    c2.write(f"{src['n_chunks']} potongan")
                    if c3.button("Hapus", key=f"kb_remove_{src['id']}"):
                        st.session_state["kb_chunks"] = [
                            c for c in st.session_state["kb_chunks"] if c["source"] != src["name"]
                        ]
                        st.session_state["kb_sources"] = [
                            s for s in st.session_state["kb_sources"] if s["id"] != src["id"]
                        ]
                        st.session_state["kb_embeddings"] = None
                        st.rerun()

                with st.expander("👀 Preview potongan (3 contoh)"):
                    for c in st.session_state["kb_chunks"][:3]:
                        st.text(f"[{c['source']}] {c['text'][:300]}")

            with st.expander("⚙️ Advanced (opsional)"):
                st.session_state["kb_embedding_model"] = st.text_input(
                    "Model embedding", value=st.session_state["kb_embedding_model"],
                    help="Model sentence-transformers multilingual (support Bahasa Indonesia). "
                    "Ganti model = index harus dibangun ulang.",
                )
                st.session_state["kb_top_k"] = st.number_input(
                    "Jumlah potongan relevan diambil per pertanyaan (top_k)",
                    min_value=1, max_value=20, value=st.session_state["kb_top_k"],
                )
                st.session_state["kb_max_chars"] = st.number_input(
                    "Batas karakter konteks per pertanyaan", min_value=200,
                    value=st.session_state["kb_max_chars"], step=100,
                )

            col_build, col_reset = st.columns(2)
            with col_build:
                if st.button("🔧 Bangun Knowledge Base", type="primary", disabled=not st.session_state["kb_chunks"]):
                    with st.spinner("Memuat model embedding & mengindeks..."):
                        embedder = get_embedder(st.session_state["kb_embedding_model"])
                        embeddings = embed_passages(embedder, [c["text"] for c in st.session_state["kb_chunks"]])
                    st.session_state["kb_embeddings"] = embeddings
                    st.success(
                        f"Knowledge Base siap: {len(st.session_state['kb_sources'])} sumber, "
                        f"{len(st.session_state['kb_chunks'])} potongan diindeks."
                    )
            with col_reset:
                if st.button("🗑️ Reset Knowledge Base"):
                    st.session_state["kb_sources"] = []
                    st.session_state["kb_chunks"] = []
                    st.session_state["kb_embeddings"] = None
                    st.rerun()

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
                # HF's TrainingArguments defaults per_device_eval_batch_size to 8
                # independently of the train batch size if it's never set — that
                # silently makes trainer.evaluate() (Eval Loss) allocate activations
                # for an 8x bigger batch than training ever used, even though
                # training itself fits fine. Pin it to the same batch size so
                # eval never asks for more memory than training already proved fits.
                per_device_eval_batch_size=int(per_device_train_batch_size),
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
            with st.expander("📜 Training log", expanded=True):
                log_placeholder = st.empty()
            callback = StreamlitTrainerCallback(progress_bar, status_text, chart_placeholder, log_placeholder)
            trainer.add_callback(callback)

            with st.spinner("Training..."):
                stats = trainer.train()

            st.session_state.trainer = trainer
            st.session_state.trained = True
            used_mem = memory_snapshot()
            start_mem = st.session_state.start_gpu_mem or 0
            clear_gpu_cache()  # release freed training buffers (gradients, optimizer scratch) before Test/Evaluate need GPU headroom for generation
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
        use_kb = False
        if st.session_state["kb_embeddings"] is not None:
            use_kb = st.checkbox(
                "📦 Sertakan Knowledge Base (ambil potongan relevan otomatis per pertanyaan)",
                value=True, key="test_include_kb",
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
                kb_block, kb_hits = kb_augment(prompt, use_kb)
                full_system_prompt = f"{system_prompt}\n\n{kb_block}".strip() if kb_block else system_prompt
                with st.chat_message("assistant"):
                    if kb_hits:
                        with st.expander("🔍 Konteks Knowledge Base yang dipakai"):
                            st.json(kb_hits)
                    response = st.write_stream(
                        stream_chat_response(
                            modality, st.session_state.model, st.session_state.processor,
                            text=prompt, system_prompt=full_system_prompt,
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
                    kb_block, kb_hits = kb_augment(question, use_kb)
                    full_system_prompt = f"{system_prompt}\n\n{kb_block}".strip() if kb_block else system_prompt
                    if kb_hits:
                        with st.expander("🔍 Konteks Knowledge Base yang dipakai"):
                            st.json(kb_hits)
                    response = st.write_stream(
                        stream_chat_response(
                            modality, st.session_state.model, st.session_state.processor,
                            text=question, image=image, system_prompt=full_system_prompt,
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
                    kb_block, kb_hits = kb_augment(question, use_kb)
                    full_system_prompt = f"{system_prompt}\n\n{kb_block}".strip() if kb_block else system_prompt
                    if kb_hits:
                        with st.expander("🔍 Konteks Knowledge Base yang dipakai"):
                            st.json(kb_hits)
                    response = st.write_stream(
                        stream_chat_response(
                            modality, st.session_state.model, st.session_state.processor,
                            text=question, audio=audio_array, system_prompt=full_system_prompt,
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
            help="Otomatis ikut berubah setiap kamu rebuild dataset training di tab Data dengan system prompt "
            "baru (sama seperti tab Test) — edit manual di sini kalau mau uji varian lain.",
        )
        use_kb_eval = False
        if st.session_state["kb_embeddings"] is not None:
            use_kb_eval = st.checkbox(
                "📦 Sertakan Knowledge Base (ambil potongan relevan otomatis per pertanyaan)",
                value=True, key="eval_include_kb",
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
                    items = []
                    for p in prompts:
                        kb_block, _ = kb_augment(p, use_kb_eval)
                        sp = f"{system_prompt_eval}\n\n{kb_block}".strip() if kb_block else system_prompt_eval
                        items.append({"text": p, "system_prompt": sp})

        elif modality == "Vision":
            question = st.text_input("Pertanyaan untuk tiap gambar", value="Apa isi gambar ini?")
            image_files = st.file_uploader("Upload gambar (bisa lebih dari satu)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, key="eval_images")
            if st.button("Bandingkan Base vs Finetuned", type="primary"):
                if not image_files:
                    st.warning("Upload minimal satu gambar.")
                else:
                    from PIL import Image

                    kb_block, _ = kb_augment(question, use_kb_eval)
                    sp = f"{system_prompt_eval}\n\n{kb_block}".strip() if kb_block else system_prompt_eval
                    items = [{"text": question, "image": Image.open(f).convert("RGB"), "system_prompt": sp} for f in image_files]

        else:  # Audio
            question = st.text_input("Pertanyaan untuk tiap audio", value="Please transcribe this audio.")
            audio_files = st.file_uploader("Upload audio (bisa lebih dari satu)", type=["wav", "mp3", "flac", "m4a"], accept_multiple_files=True, key="eval_audios")
            if st.button("Bandingkan Base vs Finetuned", type="primary"):
                if not audio_files:
                    st.warning("Upload minimal satu audio.")
                else:
                    kb_block, _ = kb_augment(question, use_kb_eval)
                    sp = f"{system_prompt_eval}\n\n{kb_block}".strip() if kb_block else system_prompt_eval
                    items = [
                        {"text": question, "audio": load_audio_array(f, f"{SCRATCH_DIR}/eval_audio"), "system_prompt": sp}
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
