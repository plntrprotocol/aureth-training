"""
OUSIA Training — LoRA Configuration
Phase 1: Agentic + Anti-Sycophantic Base
Base model: Qwen/Qwen3.5-4B
"""

from peft import LoraConfig, TaskType

# LoRA rank — higher = more capacity, larger adapter
# r=16 is the sweet spot for 4B models
LORA_RANK = 16
LORA_ALPHA = 32  # 2x rank is standard scaling
LORA_DROPOUT = 0.05

# Which attention layers to adapt
# Qwen3.5-4B: all attention projections are standard torch.nn.Linear (PEFT supported)
TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
]

# Training hyperparameters — G4/T4-OPTIMIZED (16GB VRAM)
# T4: batch=2, seq=1024, ~1-1.5hr job
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8  # effective batch = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.5
LOGGING_STEPS = 50
SAVE_STEPS = 200
EVALUATION_STRATEGY = "steps"
EVAL_STEPS = 200
MAX_STEPS = -1  # -1 = epoch-based, override with NUM_EPOCHS

# Optimizer
OPTIMIZER = "paged_adamw_8bit"  # Paged AdamW for memory efficiency

# A100 training — QLoRA on 40GB VRAM
USE_QLORA = True
QUANTIZATION_BITS = 4
COMPUTE_DTYPE = "bfloat16"  # bfloat16 on A100

# Gradient checkpointing — trades ~20% speed for 50% VRAM reduction
USE_GRADIENT_CHECKPOINTING = True

# Model
# Qwen3.5-4B: Gated DeltaNet + sparse FFN, 4B params, 262K context
# MMLU-Pro: 79.1%, TAU2-Bench: 79.9%, LiveCodeBench: 55.8%
# Apache 2.0 — commercially clean
MODEL_ID = "Qwen/Qwen3.5-4B"

# Dataset config
TRAIN_DATASETS = [
    {
        "name": "NousResearch/Hermes-3-Dataset",
        "split": "train",
        "streaming": True,
        "format": "sharegpt",
    },
    {
        "name": "NousResearch/hermes-function-calling-v1",
        "split": "train",
        "streaming": True,
        "format": "function_calling",
    },
    {
        "name": "NousResearch/RefusalDataset",
        "split": "train",
        "streaming": False,  # Small dataset, load fully
        "format": "refusal",
    },
]

# Phase labels for tracking
TRAINING_PHASE = "ousia_phase1_agentic_base"
EXPERIMENT_NAME = "ousia-phase1"
