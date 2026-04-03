"""
AURETH Training — LoRA Configuration
Phase 1: Agentic + Anti-Sycophantic Base
Base model: google/gemma-4-E4B-it
"""

from peft import LoraConfig, TaskType

# LoRA rank — higher = more capacity, larger adapter
# r=16 is the sweet spot for 4B models
LORA_RANK = 16
LORA_ALPHA = 32  # 2x rank is standard scaling
LORA_DROPOUT = 0.05

# Which attention layers to adapt
# Gemma4 layer 0: Gemma4ClippableAttention — q_proj is Gemma4ClippableLinear (PEFT unsupported)
# Gemma4 layers 1-31: Gemma4TextAttention — q_proj/v_proj/k_proj/o_proj are standard Linear
# Use layers_to_transform to skip layer 0 and target all remaining layers
TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
]

# Training hyperparameters — T4-OPTIMIZED (16GB VRAM)
# A100: batch=4, seq=1024, ~2-3hr job
# T4:    batch=1, seq=512,  ~12-18hr job (acceptable, same quality output)
MAX_SEQ_LENGTH = 256  # Reduced for CPU RAM — halves activation memory
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # effective batch = 8 (reduced from 16)
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

# CPU training — e2-standard-4 (16GB RAM)
# Aggressive memory savings: float16 + gradient checkpointing
USE_QLORA = False
QUANTIZATION_BITS = 0
COMPUTE_DTYPE = "float16"  # float16 on CPU

# Gradient checkpointing — trades ~20% speed for 50% VRAM reduction
USE_GRADIENT_CHECKPOINTING = True

# Model
# Gemma-4-E4B-it: Hybrid sliding + global attention, PLE (4.5B params, ~2-3B active)
# Apache 2.0 — fully commercially clean
# Native function calling, built-in thinking mode, multimodal (text + image + audio)
# MMLU-Pro: 69.4%, AIME 2026: 42.5%, TAU2-Bench: 42.2%
MODEL_ID = "google/gemma-4-E4B-it"

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
TRAINING_PHASE = "aureth_phase1_agentic_base"
EXPERIMENT_NAME = "aureth-phase1"
