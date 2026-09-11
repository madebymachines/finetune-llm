GEMMA4_MODELS = [
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
]

CHAT_TEMPLATE = "gemma-4"
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"

# Generation defaults. Gemma-4 team recommends temperature=1.0 for general chat,
# but for a product-catalog chatbot (fact accuracy > creative variety) the
# research runs in this repo showed low temperature gives consistent, grounded
# answers — so 0.3 is the default here; raise it in the Test tab if needed.
GEMMA_RECOMMENDED_TEMPERATURE = 1.0
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

# System prompt file produced alongside dataset/build_chat_dataset.py. When it
# exists, the Test/Evaluate "System prompt" field defaults to its content so
# inference uses the SAME system prompt the training rows were built with.
DEFAULT_SYSTEM_PROMPT_FILE = "dataset/system_prompt_short.txt"

MODALITIES = ["Text", "Vision", "Audio"]

# Audio finetuning has no finetune_vision_layers-style flag for its projector/
# adapter layers, so Unsloth's audio notebook targets them explicitly.
AUDIO_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    # Audio layers
    "post", "linear_start", "linear_end",
    "embedding_projection",
    "ffw_layer_1", "ffw_layer_2",
    "output_proj",
]

# Per-modality defaults, taken from Unsloth's Gemma4 (E4B) Text/Vision/Audio notebooks.
LORA_DEFAULTS = {
    "Text": {"r": 8, "lora_alpha": 8},
    "Vision": {"r": 32, "lora_alpha": 32},
    "Audio": {"r": 8, "lora_alpha": 16},
}

SFT_DEFAULTS = {
    "Text": {
        "learning_rate": 2e-4,
        "lr_scheduler_type": "linear",
        "warmup_steps": 5,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
    },
    "Vision": {
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_length": 2048,
    },
    "Audio": {
        "learning_rate": 5e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "max_length": 8192,
    },
}

AUDIO_SAMPLING_RATE = 16000
