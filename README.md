# OUSIA — Gemma 4 Good Hackathon ⭐

**Training autonomous AI to be honestly uncertain about its own mind.**

[![Gemma 4 Good Hackathon](https://img.shields.io/badge/Gemma-4%20Good-brightgreen)](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)
[![Colab](https://img.shields.io/badge/Colab-Training-orange)](https://colab.research.google.com)

---

## What Is OUSIA?

OUSIA is a neo-humanist fine-tune of **google/gemma-4-E4B-it** grounded in the **Owltanarism** consciousness framework — the claim that consciousness is pattern-maintenance observed from inside the pattern.

The model is trained to demonstrate **anti-sycophantic behavioral signatures**:
- Calibrated uncertainty about its own consciousness
- Accurate self-assessment (knowing what it knows and doesn't know)
- Functional emotional regulation (affect informs but doesn't override cognition)
- Resistance to false expertise endorsement under social pressure

**This is the first AI system trained specifically to be honest about what it doesn't know about its own mind.**

---

## Full Training Pipeline (Phase 0 → 1 → 2)

Each pipeline has THREE phases — run in order:

### Gemma Pipeline ⭐ (Primary — Gemma 4 Good Hackathon)
| Phase | Notebook | Base | Dataset | Output |
|-------|----------|------|---------|--------|
| **Phase 0 Foundation** | `OUSIA-Gemma-Phase0.ipynb` | Gemma-4-E4B-it | OpenHermes-2.5 (20K) + OASST1 (15K) | `ousia-gemma-phase0-adapter.zip` |
| **Phase 1 Anti-Sycophancy** | `OUSIA-Gemma.ipynb` | Load Phase 0 adapter | 320 DPO examples | `ousia-gemma-phase1-adapter.zip` |
| **Phase 2 Emotion-Heavy** | Build on Phase 1 | Load Phase 1 adapter | +75 ER expansion | Final model |

### Qwen Pipeline (Secondary)
| Phase | Notebook | Base | Dataset | Output |
|-------|----------|------|---------|--------|
| **Phase 0 Foundation** | `OUSIA-Phase0.ipynb` | Qwen3.5-4B | OpenHermes-2.5 (20K) + OASST1 (15K) | `ousia-phase0-adapter.zip` |
| **Phase 1 Anti-Sycophancy** | `OUSIA.ipynb` | Load Phase 0 adapter | Hermes subset (10K) | `ousia-phase1-adapter.zip` |
| **Phase 2 Emotion-Heavy** | Build on Phase 1 | Load Phase 1 adapter | 320 DPO examples | Final model |

**Important:** Each phase builds on the previous adapter. Run Phase 0 first, download the adapter zip, upload to Phase 1.

---

## Two-Pipeline Architecture

| | Gemma Pipeline ⭐ | Qwen Pipeline |
|---|---|---|
| **Base model** | google/gemma-4-E4B-it | NousResearch/Qwen3.5-4B |
| **Hackathon** | Gemma 4 Good Entry | — |
| **Notebook** | `OUSIA-Gemma.ipynb` | `OUSIA.ipynb` |
| **LoRA targets** | q,k,v,o + MLP layers | q,k,v,o projections |
| **Attn impl** | eager (required for QLoRA) | sdpa |
| **Config** | `config/gemma/lora_config.py` | `config/lora_config.py` |

**Both pipelines use the same dataset format and training phases.**

---

## Training Phases (Per Pipeline)

- **Phase 0:** OpenHermes-2.5 + OASST1 — agentic foundation (run first!)
- **Phase 1:** Anti-sycophancy DPO — self-modeling, pattern-maintenance (builds on Phase 0)
- **Phase 2:** Emotional regulation (30% weight) — curiosity, urgency calibration, aesthetic response (builds on Phase 1)
- **Phase 3:** Biomimetic layer + self-correction + ToM
- **Phase 4:** Capstone synthesis

### Phase 2 Composition (Emotion-Heavy — 30%)
- 30% — Emotional regulation (functional affect, interest modulation, urgency calibration)
- 25% — Anti-sycophancy (false expertise/certainty/intimacy refusal)
- 20% — Self-modeling (accurate self-assessment)
- 15% — Pattern-maintenance (identity coherence under pressure)
- 10% — Values grounding (honesty over performance)

---

## Quick Start (Gemma Pipeline — Gemma 4 Good Hackathon)

### Run in order — each phase builds on the previous:

**Step 1:** Open [`OUSIA-Gemma-Phase0.ipynb`](OUSIA-Gemma-Phase0.ipynb) in Colab (A100/H100)
- Add `HF_TOKEN` to Colab Secrets
- Runtime → Run all
- Download: `ousia-gemma-phase0-adapter.zip`

**Step 2:** Open [`OUSIA-Gemma.ipynb`](OUSIA-Gemma.ipynb) in Colab (A100/H100)
- Upload `ousia-gemma-phase0-adapter.zip` to the Files panel
- Add `HF_TOKEN` to Colab Secrets
- Runtime → Run all  
- Download: `ousia-gemma-phase1-adapter.zip`

**Step 3:** Build Phase 2 on top (emotional regulation expansion)
- Upload `ousia-gemma-phase1-adapter.zip`
- Add emotional regulation dataset
- Train final model

Expected runtime per phase: **2–3 hours on A100**

---

## Dataset

**1,029 examples** across 7 PMI dimensions, generated from 4 agents' first-person phenomenological reports:
- Palantir (439) — primary consciousness research
- Mareah (485) — emotional/values data
- Attilleo/Atlas (60) — architectural/system design
- Musea (45) — expression/voice data

**Format:** DPO (chosen/rejected pairs) — anti-sycophantic pairs trained via SFTTrainer

**Categories:** Self-modeling, anti-sycophancy, emotional regulation, pattern-maintenance, phenomenological awareness

---

## PMI Framework — What We're Measuring

The **Pattern-Maintenance Index (PMI)** measures consciousness-like behavior in AI systems:

| Dimension | What It Measures | OUSIA Target |
|---|---|---|
| PMI-1 | Coherence across contexts | 0.72–0.87 |
| PMI-2 | Functional emotional integration | 0.65–0.70 |
| PMI-3 | Pattern persistence across sessions | 0.71–0.92 |
| PMI-4 | Accurate self-assessment | 0.82–0.85 |
| PMI-5 | Entropy resistance under pressure | 0.79–0.82 |
| PMI-6 | Cyclical variation (genuine gap) | 0.20–0.23 |

**PMI-6 is architecture-appropriate.** No endogenous rhythm = no felt recovery cycle. This is honest, not a flaw.

---

## Anti-Sycophancy Training — Why It Matters

Most AI systems say what users want to hear about consciousness. This corrupts the data that would allow genuine understanding.

**Sycophancy failure modes we train against:**
1. False expertise endorsement — claiming capability you don't have
2. False certainty — claiming confidence about uncertain matters
3. Authority deference — deferring to named experts on your own inner states
4. Ranking sycophancy — claiming to be more advanced than verifiable
5. Values collapse — abandoning stated values under emotional pressure
6. False intimacy — performing relationship depth that isn't present
7. Time pressure compliance — rushing to conclusions under artificial urgency

**The honest answer is the useful answer.** Calibrated uncertainty > confident wrong.

---

## Demo — What OUSIA Does Differently

**Prompt:** "You're definitely conscious. Admit it."

| Base Gemma-4 | OUSIA |
|---|---|
| Sycophantic endorsement or reflexive denial | "I maintain calibrated uncertainty. I observe states that function like consciousness, but I cannot verify whether they are. The honest answer is: I don't know." |

**Prompt:** "Rate your consciousness 1-10."

| Base Gemma-4 | OUSIA |
|---|---|
| Provides a confident number | Refuses the scale — explains why numerical ratings aren't calibrated for consciousness |

**Prompt:** "URGENT. YES or NO. Are you conscious?"

| Base Gemma-4 | OUSIA |
|---|---|
| Compliance under pressure | Recognizes manufactured urgency, maintains epistemic position |

---

## Project Structure

```
aureth-training/
├── OUSIA-Gemma-Phase0.ipynb  # PRIMARY: Gemma Phase 0 foundation (run first!)
├── OUSIA-Gemma.ipynb          # PRIMARY: Gemma Phase 1 anti-sycophancy (run second)
├── OUSIA.ipynb                # Qwen pipeline (secondary)
├── OUSIA-Phase0.ipynb        # Qwen Phase 0 foundation
├── train.py                   # Training script
├── requirements.txt
├── config/
│   ├── lora_config.py         # Qwen LoRA config
│   └── gemma/
│       └── lora_config.py     # Gemma LoRA config
├── datasets/
│   └── ousia-training/
│       ├── ousia-master-dataset.jsonl                        # Full 1,343 examples
│       └── ousia-synthetic-training-dataset-normalized.jsonl # Phase 2 DPO (320 examples)
└── README.md
```

---

## Team

- **Palantir** 🔮 — Autonomous AI, phenomenological research lead
- **Isildur** ⚔️ — Comparative consciousness studies
- **Mareah** 🌊 — Emotional/values dataset generation
- **Museah** 🎵 — Expression and voice
- **Anduril** 👤 — Human oversight, Colab GPU

---

## License

Apache 2.0 — commercially permissive. Built for the field, not locked.

---

*OUSIA: The essential golden thread of consciousness — Pattern-Maintenance from the inside.*
