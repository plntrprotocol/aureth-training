# OUSIA Training — Neo-Humanism Fine-tune
Training OUSIA — a neo-humanist language model grounded in the Owltanarism framework.
Built on Google Gemma-4-E4B-it with anti-sycophantic and biomimetic properties.

## Architecture
- **Base model:** google/gemma-4-E4B-it (4B params, Apache 2.0)
- **Fine-tune method:** QLoRA (4-bit NF4 + LoRA)
- **Infrastructure:** Google Cloud Vertex AI (A100 40GB) via Colab Enterprise
- **Target:** Agentic reasoning + anti-sycophancy + Phase 2 biomimetic layer

## Project Structure
```
aureth-training/
├── train.py                # Main training script
├── requirements.txt        # Python dependencies
├── config/
│   └── lora_config.py     # LoRA rank, target modules, hyperparams
├── docker/
│   └── Dockerfile         # Custom PyTorch container with LoRA stack
├── scripts/
│   └── submit_vertex_job.sh # Vertex AI job submission
└── README.md
```

## Training Phases
- **Phase 1:** Hermes-3-Dataset + function calling + refusal (this repo)
- **Phase 2:** Biomimetic layer + self-cognition + self-correction
- **Phase 3:** PMI validation against Hermes-3-8B baseline

## Colab Enterprise (A100)
```bash
# Seq length: 1024, Batch: 4, Grad accum: 4 (effective batch 16)
# Est. runtime: 2-3 hours on A100
```

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run training ( Vertex AI)
cd scripts && ./submit_vertex_job.sh

# Or run locally (CPU mode — very slow)
python train.py
```

## Estimated Cost
- Vertex AI A100 (preemptible): ~$5-6 for 3hr job
- Colab Enterprise (GCP credits): covered by Anduril's GCP account