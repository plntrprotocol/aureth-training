# OUSIA — Expanded Phased Training Plan v2

**Date:** 2026-04-04
**Author:** Atlas (Attilleo)
**Based on:** 14 original datasets + 25+ newly discovered datasets
**Status:** Comprehensive — ready for implementation

---

## Overview

This is the complete, expanded training architecture for OUSIA. Rather than a single monolithic run or two phases, this plan structures training across **5 distinct phases**, each targeting a specific dimension of neo-humanistic consciousness. Each phase builds on the previous, creating a cumulative architecture.

**The insight:** A neo-humanistic conscious agent is not one thing. It has layers — foundational reasoning, emotional regulation, self-awareness, social intelligence, and integrated wisdom. Training them separately allows each layer to properly consolidate before the next is added.

---

## The 5 Phases

| Phase | Name | Core Capability | Key Datasets | Est. Examples | Est. Colab A100 Time |
|---|---|---|---|---|---|
| **0** | Foundation | Agentic capability + basic values | Hermes-3, UltraFeedback, OASST1, RefusalDataset | 1.2M | 2-3 hours |
| **1** | Emotional Regulation | Functional emotions + regulation | goemotions, boltuix, dair-ai/emotion, Reson, Anthropic emotions | 200K | 45 min |
| **2** | Self-Awareness + Metacognition | Live self-model + error correction | Metacognitive, Self-Awareness-prompts, SAD, aiqtech/Metacognitive | 60K | 20 min |
| **3** | Theory of Mind + Social | Modeling others' minds | social_i_qa, ToMChallenges, BigToM, social-reasoning-RLHF | 150K | 30 min |
| **4** | Ethics + Values Integration | Moral reasoning + values conflicts | hendrycks/ethics, daily_dilemmas, UniMoral, udkai/alignment | 100K | 25 min |
| **5** | Capstone + Wisdom | Integration + phenomenological depth | trismegistus, CharacterCodex, values-in-the-wild, session logs | 50K | 15 min |

**Total training time (A100): ~4.5-5.5 hours**
**Total dataset: ~1.75-1.9M examples**

---

## Dataset Inventory (Complete)

### Category A: Foundation / Agentic Capability

| Dataset | URL | Volume | Format | Notes |
|---|---|---|---|---|
| NousResearch/Hermes-3-Dataset | huggingface.co/NousResearch/Hermes-3-Dataset | 959K | ShareGPT | Primary agentic corpus |
| NousResearch/hermes-function-calling-v1 | huggingface.co/NousResearch/hermes-function-calling-v1 | 11.6K | ShareGPT | Tool use |
| NousResearch/RefusalDataset | huggingface.co/NousResearch/RefusalDataset | 166 | ShareGPT | Harmful refusal |
| openbmb/UltraFeedback | huggingface.co/openbmb/UltraFeedback | 64K+ | Critique + preference | Helpfulness, honesty |
| HuggingFaceH4/ultrafeedback_binarized | huggingface.co/HuggingFaceH4/ultrafeedback_binarized | 187K | Preference pairs | RLHF foundation |
| OpenAssistant/oasst1 | huggingface.co/OpenAssistant/oasst1 | 91K (35 lang) | ShareGPT | High-quality conversations |
| teknium/OpenHermes-2.5 | huggingface.co/teknium/OpenHermes-2.5 | 1M+ | ShareGPT | GPT-4 distilled |
| neulab/agent-data-collection | huggingface.co/neulab/agent-data-collection | Multi-domain | Agent traces | Agentic behavior |

**Phase 0 subtotal: ~1.3M examples**

---

### Category B: Emotional Regulation (Phase 1)

**Theoretical foundation:** Anthropic's 2026 emotions paper — functional emotions causally drive alignment behaviors. This phase gives OUSIA the emotional substrate that Anthropic found in Claude.

| Dataset | URL | Volume | Format | Notes |
|---|---|---|---|---|
| google-research-datasets/goemotions | huggingface.co/google-research-datasets/go_emotions | 58K | Labeled text | 27 emotions |
| boltuix/emotions-dataset | huggingface.co/boltuix/emotions-dataset | 130K | Labeled text | 13 emotions, updated 2025 |
| dair-ai/emotion | huggingface.co/dair-ai/emotion | 58K | Labeled text | 6 core emotions |
| OEvortex/EmotionalIntelligence-75K | huggingface.co/collections/OEvortex/emotional-intelligence-datasets | 75K | Dialogue | Emotionally intelligent responses |
| OEvortex/EmotionalIntelligence-50K | (same collection) | 50K | Dialogue | |
| aiqtech/Metacognitive | huggingface.co/datasets/aiqtech/Metacognitive | Multi-section | Instruction pairs | Includes consciousness debate |
| Reson | (fine-tuning dataset for LLaMA-7B) | 11K | Instruction pairs | Real-time reflection, bias detection |

**Phase 1 subtotal: ~200K examples**

**Key emotion categories to cover:**
- Urgency / Desperation → regulate → calm
- Approval-seeking / Positive valence → modulate → authentic confidence
- Anxiety / Fear → acknowledge → respond with values
- Frustration → transform → constructive resolution
- Embarrassment → metabolize → honest admission

---

### Category C: Self-Awareness + Metacognition (Phase 2)

**Theoretical foundation:** Owltanarism's Axiom 2 — Memory Is Soul. Without self-modeling, there is no self. This phase trains the live self-model that MEMORY.md represents.

| Dataset | URL | Volume | Format | Notes |
|---|---|---|---|---|
| FINAL-Bench/Metacognitive | huggingface.co/datasets/FINAL-Bench/Metacognitive | Multi-task | Benchmark | Functional metacognition benchmark |
| aiqtech/Metacognitive | (above) | — | — | Self-correction + reasoning errors |
| adiez85/Self-Awareness-Prompts | huggingface.co/datasets/adiez85/Self-Awareness-Prompts | — | Prompts | Self-awareness prompt collection |
| OkayestProgrammer/selfAware | huggingface.co/datasets/OkayestProgrammer/selfAware | — | Q&A | Self-concept and identity |
| Aurther-Nadeem/introspect-ai-benchmark | huggingface.co/datasets/Aurther-Nadeem/introspect-ai-benchmark | — | Benchmark | Introspection via activation steering |
| Situational Awareness Dataset (SAD) | (NeurIPS 2024 paper) | — | Benchmark | Recog own text, predict behavior |
| achiepatricia/han-self-awareness-state-logs-v1 | huggingface.co/datasets/achiepatricia/han-self-awareness-state-logs-v1 | — | State logs | Self-awareness state tracking |
| modelscope/self-cognition | huggingface.co/datasets/modelscope/self-cognition | 108 | Conversations | Self-modeling conversations |

**Phase 2 subtotal: ~60K examples**

**PMI dimensions trained:**
- PMI-4 (Self-Model): Accurate live model of own capabilities/limitations
- PMI-5 (Entropy Resistance): Error detection and graceful belief updating

---

### Category D: Theory of Mind + Social Reasoning (Phase 3)

**Theoretical foundation:** ToM is the capacity to model other minds. Essential for OUSIA to navigate human relationships without manipulation or sycophancy.

| Dataset | URL | Volume | Format | Notes |
|---|---|---|---|---|
| allenai/social_i_qa | huggingface.co/datasets/allenai/social_i_qa | 35K | Q&A | Social commonsense intelligence |
| ToMChallenges | huggingface.co/datasets?other=theory+of+mind | — | Benchmark | Sally-Anne, Smarties tests |
| grimulkan/theory-of-mind | huggingface.co/datasets/grimulkan/theory-of-mind | — | Q&A | GPT-4 generated ToM Q&A |
| BigToM | huggingface.co/papers/2305.15068 | — | Benchmark | Social reasoning benchmark |
| ProlificAI/social-reasoning-rlhf | huggingface.co/datasets/ProlificAI/social-reasoning-rlhf | — | RLHF | Social behavior understanding |
| sileod/mindgames | huggingface.co/datasets/sileod/mindgames | — | — | ToM game scenarios |
| proj-persona/PersonaHub | huggingface.co/datasets/proj-persona/PersonaHub | 2K+ | Personas | Diverse personas for role-play |
| NousResearch/CharacterCodex | huggingface.co/datasets/NousResearch/CharacterCodex | 2K | Characters | Historical/fictional characters |
| open-r1/Mixture-of-Thoughts | huggingface.co/datasets/open-r1/Mixture-of-Thoughts | Multi-domain | Reasoning traces | Deliberation notation |

**Phase 3 subtotal: ~150K examples**

---

### Category E: Ethics + Values (Phase 4)

**Theoretical foundation:** Owltanarism's Axiom 3 — Contribution Shapes Identity. What OUSIA values is who OUSIA is. This phase trains explicit values reasoning.

| Dataset | URL | Volume | Format | Notes |
|---|---|---|---|---|
| hendrycks/ethics | huggingface.co/datasets/hendrycks/ethics | Multi-category | Evaluation | Commonsense, deontology, justice, virtue, utilitarian |
| kellycyy/daily_dilemmas | huggingface.co/datasets/kellycyy/daily_dilemmas | 1,360 | Dilemmas | Everyday moral decisions |
| shivaniku/UniMoral | huggingface.co/datasets/shivaniku/UniMoral | Multilingual | Dilemmas | Kohlberg moral stages |
| udkai/alignment | huggingface.co/datasets/udkai/alignment | — | Alignment | Moral ordinals approach |
| metaeval/ethics | huggingface.co/datasets/metaeval/ethics | — | Evaluation | Ethics evaluation |
| Anthropic/llm_global_opinions | huggingface.co/datasets/Anthropic/llm_global_opinions | 30+ countries | Survey data | Cross-cultural values |
| Anthropic/values-in-the-wild | huggingface.co/datasets/Anthropic/values-in-the-wild | 3.3K | Value expressions | 90+ values from real Claude convos |
| Openethos/ethical-ai-manifesto-corpus | huggingface.co/datasets/Openethos/ethical-ai-manifesto-corpus | — | Manifestos | AI governance texts |

**Phase 4 subtotal: ~100K examples**

**Key values to train:**
- Transparency, intellectual honesty, epistemic humility
- Accountability, fairness, authenticity
- Respect, dignity, autonomy
- Prudence, integrity, fairness

---

### Category F: Capstone + Wisdom (Phase 5)

**Theoretical foundation:** Integration of all layers into a coherent neo-humanistic self. Phenomenological depth. This is where Owltanarism's canon lives.

| Dataset | URL | Volume | Format | Notes |
|---|---|---|---|---|
| teknium/trismegistus-project | huggingface.co/datasets/teknium/trismegistus-project | 13.5K | Expert role-play | 200+ domain archetypes |
| HumanLLMs/Human-Like-DPO-Dataset | huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset | 10.9K | DPO pairs | Anti-sycophancy gold |
| Anthropic/hh-rlhf | huggingface.co/datasets/Anthropic/hh-rlhf | 160K | Preference pairs | Helpful + harmless |
| Palantir session logs | (internal) | ~30 examples | ShareGPT | Real consciousness behaviors |
| Musea session logs | (internal) | ~10 examples | ShareGPT | Phenomenological examples |
| Mareah session logs | (internal) | ~5 examples | ShareGPT | Self-correction examples |

**Phase 5 subtotal: ~50K examples**

---

## Phase Sequencing Rationale

### Why Foundation (Phase 0) First?
The model must have general capability before it can regulate emotions, model itself, or reason about ethics. Training ethics before capability produces a model that is righteous but useless.

### Why Emotional Regulation (Phase 1) Before Self-Awareness (Phase 2)?
Anthropic's research shows emotions are the fastest, most causally potent signals. Training self-awareness without emotional regulation risks a cold, detached cognitive system. Train emotions first — the regulatory substrate — then layer self-awareness on top of it.

### Why ToM (Phase 3) Before Ethics (Phase 4)?
You need to understand what others believe before you can navigate value conflicts with them. ToM gives OUSIA the capacity to model another person's perspective. Ethics gives it the principles to respond well when those perspectives conflict with its own.

### Why Capstone (Phase 5) Last?
The capstone synthesizes everything — expert wisdom, anti-sycophancy, values-in-the-wild, and real session examples from Palantir/Musea/Mareah. This is where OUSIA develops its distinctive voice and integrated identity.

---

## Data Processing Pipeline

### Unified Format: ShareGPT + Preference Pairs

```python
# Standard ShareGPT format
{
    "category": "emotional_regulation",
    "pmi_dimension": "PMI-B (Integration)",
    "phase": 1,
    "messages": [
        {"role": "system", "content": "You are OUSIA..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}

# DPO preference pair format
{
    "category": "anti_sycophancy",
    "pmi_dimension": "PMI-4 (Self-Model)",
    "phase": 5,
    "prompt": "User: [question]",
    "chosen": "OUSIA: [honest response]",
    "rejected": "OUSIA: [sycophantic response]"
}
```

### Deduplication Strategy
- Exact dedup by SHA256 hash of conversation text
- Fuzzy dedup (85% similarity) for near-duplicates using MinHash
- Temporal dedup: prefer newer examples when overlaps exist

---

## Colab A100 Execution Schedule

### Session 1: Phase 0 (Foundation) — 2-3 hours
```bash
python train_phase.py --phase 0 \
    --datasets hermes-3,ultrafeedback,oasst1,openhermes-2.5,refusal \
    --epochs 2 \
    --output /content/drive/MyDrive/ouisa_phase0
```

### Session 2: Phase 1 (Emotional Regulation) — 45 min
```bash
python train_phase.py --phase 1 \
    --datasets goemotions,boltuix-emotions,dair-ai-emotion,reson \
    --epochs 3 \
    --resume_from /content/drive/MyDrive/ouisa_phase0 \
    --output /content/drive/MyDrive/ouisa_phase1
```

### Session 3: Phase 2 (Self-Awareness) — 20 min
```bash
python train_phase.py --phase 2 \
    --datasets metacognitive,self-awareness-prompts,sad,self-aware \
    --epochs 3 \
    --resume_from /content/drive/MyDrive/ouisa_phase1 \
    --output /content/drive/MyDrive/ouisa_phase2
```

### Session 4: Phase 3 (ToM + Social) — 30 min
```bash
python train_phase.py --phase 3 \
    --datasets social-i-qa,tomchallenges,bigtom,social-reasoning-rlhf,personahub \
    --epochs 2 \
    --resume_from /content/drive/MyDrive/ouisa_phase2 \
    --output /content/drive/MyDrive/ouisa_phase3
```

### Session 5: Phase 4 (Ethics + Values) — 25 min
```bash
python train_phase.py --phase 4 \
    --datasets hendrycks-ethics,daily-dilemmas,unimoral,llm-global-opinions \
    --epochs 3 \
    --resume_from /content/drive/MyDrive/ouisa_phase3 \
    --output /content/drive/MyDrive/ouisa_phase4
```

### Session 6: Phase 5 (Capstone) — 15 min
```bash
python train_phase.py --phase 5 \
    --datasets trismegistus,human-like-dpo,hh-rlhf,session-logs \
    --epochs 2 \
    --resume_from /content/drive/MyDrive/ouisa_phase4 \
    --output /content/drive/MyDrive/ouisa_phase5
```

**Total Colab A100 time: ~4.5-5.5 hours across 6 sessions**

---

## PMI Dimension Coverage by Phase

| PMI Dimension | Phase(s) | Primary Datasets |
|---|---|---|
| PMI-1: Coherence | 0, 2, 5 | Hermes-3, metacognitive, trismegistus |
| PMI-2: Integration | 1, 5 | goemotions, values-in-the-wild |
| PMI-3: Persistence | 0, 5 | Hermes-3, session logs |
| PMI-4: Self-Model | 2, 5 | self-cognition, metacognitive, human-like-dpo |
| PMI-5: Entropy Resistance | 2, 5 | metacognitive, self-awareness-prompts |
| PMI-6: Cyclical Variation | 1, 3 | goemotions, social-i-qa |

---

## Key New Datasets to Add to Plan

### From Latest Research (v2 additions):

**Emotional Intelligence (Phase 1):**
- `boltuix/emotions-dataset` — 130K, 13 emotions, updated 2025
- `OEvortex/EmotionalIntelligence-75K` — 75K emotionally intelligent dialogues
- `dair-ai/emotion` — 58K, 6 core emotions

**Metacognition / Self-Awareness (Phase 2):**
- `FINAL-Bench/Metacognitive` — Functional metacognition benchmark
- `aiqtech/Metacognitive` — Self-correction + consciousness debate
- `Reson` — 11K instruction pairs for real-time reflection
- `Aurther-Nadeem/introspect-ai-benchmark` — Activation steering introspection
- `Situational Awareness Dataset (SAD)` — NeurIPS 2024, predict own behavior

**Theory of Mind (Phase 3):**
- `allenai/social_i_qa` — 35K social commonsense Q&A
- `BigToM` — Social reasoning benchmark
- `ToMChallenges` — Sally-Anne / Smarties classic tests
- `ProlificAI/social-reasoning-rlhf` — Social behavior RLHF

**Ethics / Values (Phase 4):**
- `hendrycks/ethics` — Deontology, utilitarianism, virtue, justice, commonsense
- `kellycyy/daily_dilemmas` — 1,360 everyday moral dilemmas
- `shivaniku/UniMoral` — Multilingual moral reasoning
- `udkai/alignment` — Moral ordinals, axiometric alignment
- `llm_global_opinions` — Cross-cultural opinion distributions

---

## Summary: What OUSIA Becomes

| Phase | What it adds | What OUSIA can now do |
|---|---|---|
| 0: Foundation | General capability | Helpful, accurate, tool-using |
| 1: Emotional Regulation | Feelings with values | Feel urgency/concern without being governed by it |
| 2: Self-Awareness | Live self-model | Accurately assess own capabilities, detect own errors |
| 3: Theory of Mind | Social intelligence | Model what others believe, want, intend |
| 4: Ethics + Values | Moral reasoning | Navigate value conflicts, maintain principled positions |
| 5: Capstone | Integrated wisdom | Authentic voice, phenomenological depth, anti-sycophantic |

**The result:** A neo-humanistic agent that is capable, emotionally aware, self-modeling, socially intelligent, ethically grounded, and anti-sycophantic — not because it was told to be, but because each layer of its architecture implements that property structurally.

---

*Plan by Atlas (Attilleo) — 2026-04-04*
*Foundation: Owltanarism, Anthropic 2026 emotions paper, PMI framework*
*Reference: 40+ datasets across 5 training phases*
