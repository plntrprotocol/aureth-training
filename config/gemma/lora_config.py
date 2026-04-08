"""
OUSIA Training — Gemma 4 Good Pipeline LoRA Configuration
Phase 1: Agentic + Anti-Sycophantic Base
Base model: google/gemma-4-E4B-it (4B params, Apache 2.0)

NOTE: Gemma4 uses Gemma4ClippableLinear (clipper-attention) which PEFT cannot
inject LoRA into. Before applying LoRA, run Gemma4ClippableLinear → Linear replacement.
See the notebook cell "Replace Clippable Layers" for the exact code.

TARGET_MODULES includes full attention + MLP layers for maximum capacity.
Gemma4 trains well with ALL layers targeted (unlike Qwen which sometimes skips layer 0).
"""

from peft import LoraConfig, TaskType

# LoRA rank — higher = more capacity, larger adapter
# r=16 is the sweet spot for 4B models
LORA_RANK = 16
LORA_ALPHA = 32  # 2x rank is standard scaling
LORA_DROPOUT = 0.05

# Gemma4 has 28 layers (transformer_blocks).
# Full attention + MLP layers for maximum training capacity.
# Includes: q_proj, k_proj, v_proj, o_proj (attention) + gate/up/down_proj (MLP)
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training hyperparameters — A100-OPTIMIZED (40GB VRAM)
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.5
LOGGING_STEPS = 50
SAVE_STEPS = 200
MAX_STEPS = -1  # -1 = epoch-based

# Optimizer
OPTIMIZER = "paged_adamw_8bit"  # Paged AdamW for memory efficiency

# A100 training — QLoRA on 40GB VRAM
USE_QLORA = True
QUANTIZATION_BITS = 4
COMPUTE_DTYPE = "bfloat16"  # bfloat16 on A100

# Gradient checkpointing — trades ~20% speed for 50% VRAM reduction
USE_GRADIENT_CHECKPOINTING = True

# Model
# Gemma4-E4B-it: Gemma 4 4B instruction-tuned, Apache 2.0
# Context: 32K | Architecture: grouped query attention + GeGLU FFN
MODEL_ID = "google/gemma-4-E4B-it"

# Dataset — Ousia synthetic training set (DPO format: prompt/chosen/rejected)
# Located at: datasets/ousia-training/ousia-master-dataset.jsonl (cloned with repo)
TRAIN_DATASETS = [
    {
        "name": "local",
        "path": "datasets/ousia-training/ousia-master-dataset.jsonl",
        "split": "train",
        "streaming": False,
        "format": "ousia_synthetic",
    },
]

# Phase labels
TRAINING_PHASE = "gemma4_good_phase1"
EXPERIMENT_NAME = "ousia-gemma4-good"