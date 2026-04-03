#!/usr/bin/env python3
"""
Qwen3.5-4B Neo-Humanism Training — Phase 1
Agentic + Anti-Sycophantic Base via QLoRA

Phase 1: NousResearch corpus (Hermes-3-Dataset + function calling + RefusalDataset)
Phase 2: Biomimetic layer (self-cognition, self-correction, social reasoning)
Phase 3: PMI validation against Hermes-3-8B baseline

Run on Vertex AI:
    gcloud ai custom-jobs create \
      --region=us-central1 \
      --display-name=qwen-neo-hum-phase1 \
      --worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1,\
        container-image-uri=gcr.io/$PROJECT_ID/qwen-neo-hum:latest
"""

import os
import sys
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import os
from datasets import (
    load_dataset,
    Dataset,
    interleave_datasets,
    IterableDataset,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Local config
sys.path.insert(0, "/app")
from config.lora_config import (
    MODEL_ID,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
    MAX_SEQ_LENGTH,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    NUM_EPOCHS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    LOGGING_STEPS,
    SAVE_STEPS,
    EVAL_STEPS,
    OPTIMIZER,
    USE_QLORA,
    QUANTIZATION_BITS,
    QUANTIZATION_TYPE,
    COMPUTE_DTYPE,
    USE_GRADIENT_CHECKPOINTING,
    TRAIN_DATASETS,
    TRAINING_PHASE,
    EXPERIMENT_NAME,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    name: str
    split: str
    streaming: bool
    format: str  # "sharegpt", "function_calling", "refusal", "plain"


def format_sharegpt(example: Dict[str, Any]) -> Dict[str, str]:
    """Convert ShareGPT format to single text string."""
    messages = example.get("conversations", example.get("messages", []))
    text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text += f"<|{role}|>\n{content}<|end|>\n"
    return {"text": text.strip()}


def format_function_calling(example: Dict[str, Any]) -> Dict[str, str]:
    """Format function calling dataset."""
    messages = example.get("messages", [])
    text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text += f"<|{role}|>\n{content}<|end|>\n"
    return {"text": text.strip()}


def format_refusal(example: Dict[str, Any]) -> Dict[str, str]:
    """Format refusal dataset — build adversarial → refusal pairs."""
    prompt = example.get("prompt", example.get("instruction", ""))
    response = example.get("response", example.get("completion", ""))
    text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{response}<|end|>\n"
    return {"text": text.strip()}


def load_training_data(configs: List[DatasetConfig]) -> IterableDataset:
    """Load and interleave training datasets."""
    logger.info(f"Loading {len(configs)} datasets...")

    datasets = []
    for cfg in configs:
        logger.info(f"  Loading: {cfg.name} (streaming={cfg.streaming})")
        ds = load_dataset(
            cfg.name,
            split=cfg.split,
            streaming=cfg.streaming,
            trust_remote_code=True,
        )

        # Format each dataset appropriately
        if cfg.format == "sharegpt":
            ds = ds.map(format_sharegpt, remove_columns=next(iter(ds.column_names)))
        elif cfg.format == "function_calling":
            ds = ds.map(format_function_calling, remove_columns=next(iter(ds.column_names)))
        elif cfg.format == "refusal":
            ds = ds.map(format_refusal, remove_columns=next(iter(ds.column_names)))

        logger.info(f"  {cfg.name}: {ds.num_rows if hasattr(ds, 'num_rows') else 'streaming'} rows")
        datasets.append(ds)

    # Interleave datasets with weighted sampling
    # Hermes-3-Dataset dominates (959K), others are supplements
    weights = [0.9, 0.08, 0.02]  # Hermes, function calling, refusal
    interleaved = interleave_datasets(datasets, weights=weights, stopping_strategy="all_datasts_exhausted")
    logger.info(f"  Interleaved dataset ready")
    return interleaved


def build_model(model_id: str):
    """Build QLoRA-wrapped Qwen2.5-4B model."""
    logger.info(f"Loading model: {model_id}")
    logger.info(f"  QLoRA enabled: {USE_QLORA}")
    logger.info(f"  Compute dtype: {COMPUTE_DTYPE}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # QLoRA quantization config
    if USE_QLORA:
        from bitsandbytes import BitsAndBytesConfig
        compute_dtype = getattr(torch, COMPUTE_DTYPE, torch.float32)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=QUANTIZATION_TYPE,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        device_map = "auto"
    else:
        bnb_config = None
        device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Replace Gemma4ClippableLinear with standard Linear (PEFT requirement)
    # Gemma4ClippableAttention (layer 0) wraps q/k/v in Gemma4ClippableLinear
    # which is not supported by PEFT. Swap to Linear so LoRA can inject.
    if hasattr(model, 'language_model'):
        model_base = model.language_model
    else:
        model_base = model

    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        import torch.nn as nn
        replaced = 0
        for name, module in list(model_base.named_modules()):
            if isinstance(module, Gemma4ClippableLinear):
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model_base.get_submodule(parent_name)
                # Transfer weights
                new_linear = nn.Linear(
                    module.linear.in_features,
                    module.linear.out_features,
                    bias=module.linear.bias is not None
                )
                new_linear.weight = module.linear.weight
                new_linear.bias = module.linear.bias
                setattr(parent, attr_name, new_linear)
                replaced += 1
        if replaced:
            logger.info(f"  Replaced {replaced} Gemma4ClippableLinear → Linear")
    except ImportError:
        logger.warning("  Gemma4ClippableLinear not found — may not be needed")
    except Exception as e:
        logger.warning(f"  Gemma4ClippableLinear replacement warning: {e}")

    # No k-bit training preparation needed (USE_QLORA=False on CPU)

    # Gradient checkpointing for memory efficiency
    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("  Gradient checkpointing enabled")

    # LoRA config — target q/k/v/o projections in all layers except layer 0
    # Gemma4ClippableAttention in layer 0: PEFT can't inject LoRA into it
    # Skip layer 0 via layers_to_transform=[1..31]
    layers_to_transform = list(range(1, 32))
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        layers_to_transform=layers_to_transform,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    set_seed(42)

    # Output: write locally on training VM, upload to GCS on completion
    # GCS requires authentication; local /tmp is reliable during training
    local_output = "/tmp/aureth-output"
    gcs_output = "gs://plntr-492118-aureth-output"
    max_steps = int(os.environ.get("MAX_STEPS", "-1"))

    logger.info(f"Experiment: {EXPERIMENT_NAME}")
    logger.info(f"Phase: {TRAINING_PHASE}")
    logger.info(f"Local output: {local_output}")
    logger.info(f"GCS upload: {gcs_output}")

    # Load model
    model, tokenizer = build_model(MODEL_ID)

    # Load data
    dataset_configs = [
        DatasetConfig(**d) for d in TRAIN_DATASETS
    ]
    train_data = load_training_data(dataset_configs)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=MAX_SEQ_LENGTH,
        label_pad_token_id=tokenizer.pad_token_id,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=local_output,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        max_steps=max_steps if max_steps > 0 else -1,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        logging_dir=os.path.join(local_output, "logs"),
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,  # Keep only last 3 checkpoints
        bf16=False,  # CPU: bfloat16 not reliably available
        tf32=False,   # CPU: TF32 is CUDA-specific
        optim=OPTIMIZER,
        dataloader_num_workers=0,  # CPU mode: avoid multiprocessing issues
        dataloader_pin_memory=False,  # CPU: no GPU memory pinning
        remove_unused_columns=False,
        report_to=["tensorboard"],
        run_name=EXPERIMENT_NAME,
        hub_model_id=None,
        push_to_hub=False,
        disable_tqdm=False,
    )

    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )

    logger.info("Starting training...")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")

    trainer.train()

    logger.info("Training complete. Saving final model...")
    trainer.save_model(os.path.join(local_output, "final"))
    tokenizer.save_pretrained(os.path.join(local_output, "final"))

    # Save full model (with adapters merged) for easier inference
    logger.info("Merging adapters into base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(os.path.join(local_output, "merged"))
    tokenizer.save_pretrained(os.path.join(local_output, "merged"))

    # Upload to GCS
    logger.info(f"Uploading outputs to {gcs_output}...")
    import subprocess
    result = subprocess.run(
        ["gsutil", "-m", "cp", "-r", f"{local_output}/", gcs_output],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        logger.info(f"Upload complete: {gcs_output}")
    else:
        logger.warning(f"GCS upload failed: {result.stderr}")

    logger.info(f"Done. Outputs saved to: {local_output}")
    logger.info(f"Phase {TRAINING_PHASE} complete.")


if __name__ == "__main__":
    main()
