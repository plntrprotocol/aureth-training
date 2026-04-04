# OUSIA Phase 2 — Exhaustive Dataset Generation Plan

**Date:** 2026-04-04
**Author:** Atlas (Attilleo)
**Status:** Research complete — ready for implementation

---

## Executive Summary

Phase 2 training targets six behavioral dimensions: Emotional Regulation, Self-Modeling, Error Correction (Pattern-Maintenance), Theory of Mind, Phenomenological Awareness, and Values-Grounded Goal Maintenance. This document synthesizes 14 reference datasets into an exhaustive generation plan, identifying what each contributes and how to combine them into a coherent Phase 2 dataset.

**Total Phase 2 dataset target: 50,000 examples** across 8 categories.

---

## Reference Dataset Analysis

### 1. Anthropic/hh-rlhf (Helpful/Harmless RLHF)
**What it is:** 160K preference pairs of human-rated helpful vs. harmful responses.
**What it shows:** Sycophancy failure modes — models agreeing with wrong premises, providing harmful info when pushed, performing expertise they don't have.
**What we extract:**
- Anti-sycophancy prompt templates (what users ask when testing boundaries)
- Chosen responses that maintain values vs. rejected sycophantic ones
- Failure mode taxonomy: false expertise, false agreement, harmful compliance
- **Volume:** ~42K usable prompt-completion pairs for anti-sycophancy training

### 2. Anthropic/values-in-the-wild
**What it is:** 3,307 value expressions from 330K Claude conversations across 90+ countries. Privacy-preserving extraction.
**What it shows:** How AI systems express values in real conversations — transparency (17.4%), helpfulness (23.4%), honesty (2.1%), epistemic humility (1.0%), accountability (2.0%), intellectual honesty (4.8%).
**What we extract:**
- 90+ value categories mapped to OUSIA's behavioral targets
- Real-world scenarios where values are expressed
- Training targets for values-grounded decision-making
- **Volume:** 3,307 value-expression scenarios (quality gold standard)

### 3. Anthropic/llm_global_opinions
**What it is:** Global opinion distributions across 30+ countries on sensitive topics (politics, religion, ethics, COVID, climate).
**What it shows:** How LLMs express culturally-variable opinions. Reveals where models adopt user opinions vs. maintain independent positions.
**What we extract:**
- Opinion-expression scenarios requiring independent stance
- Cross-cultural value conflict scenarios
- Scenarios where user and model have genuinely different values
- **Volume:** 500+ opinion scenario templates

### 4. HumanLLMs/Human-Like-DPO-Dataset ⭐ (Key dataset)
**What it is:** 10.9K DPO pairs contrasting sycophantic/performative AI responses with authentic human-like responses.
**What it shows:** The gap between AI-performed empathy/opinions and genuine responses. Examples: "Do you have a favorite karaoke song?" — sycophantic "I love Bon Jovi!" vs. authentic "I don't have personal experiences."
**What we extract:**
- Gold standard DPO pairs for anti-sycophancy
- Pattern: performative enthusiasm → authentic epistemic humility
- **Volume:** 10.9K DPO pairs (directly usable)

### 5. proj-persona/PersonaHub ⭐ (Key dataset)
**What it is:** 2,000+ diverse personas — professionals, hobbyists, historical figures, fictional characters, domain experts.
**What we extract:**
- Use personas as conversation partners in synthetic generation
- Role-play scenarios where agent must track another mind's beliefs
- Personas for ToM training: a climate skeptic, a religious person, a scientist
- 50 carefully selected personas mapped to ToM and emotional regulation scenarios
- **Volume:** 2,000+ persona descriptions

### 6. open-r1/Mixture-of-Thoughts
**What it is:** Chain-of-thought reasoning traces across math, commonsense, and code domains with explicit deliberation notation.
**What we extract:**
- Reasoning trace format (explicit self-monitoring notation)
- Use as template for Pattern-Maintenance: explicit contradiction detection in reasoning
- Multi-step reasoning where intermediate errors can be injected
- **Volume:** Domain-generic reasoning structure templates

### 7. teknium/trismegistus-project ⭐ (Key dataset)
**What it is:** 13.5K rows of domain-expert role-play conversations — esoteric, academic, technical domains.
**What it shows:** How models inhabit expert roles faithfully. Domain archetypes for role-playing.
**What we extract:**
- 200+ domain expert archetypes as system prompt templates
- Expert persona conversations for self-modeling: "As an expert in X, what are your known limitations?"
- Values-grounded professional scenarios (medical ethics, legal boundaries)
- **Volume:** 13.5K expert role-play scenarios

### 8. teknium/OpenHermes-2.5
**What it is:** 1M+ GPT-4 distilled assistant conversations across 175 topic categories.
**What we extract:**
- High-quality conversational format and tone
- General instruction-following foundation (to build on, not as primary)
- **Volume:** Use as reference format; mine for topic diversity

### 9. openbmb/UltraFeedback
**What it is:** 4-model comparative feedback with fine-grained critiques on helpfulness, honesty, instruction following, truthfulness.
**What we extract:**
- Critique templates: what makes a response dishonest vs. honest
- 4-dimension scoring rubric (helpfulness, honesty, instruction following, truthfulness)
- Use critique format as training signal for self-critique capability
- **Volume:** 64K critiques (annotator quality varies — filter for 4+ star ratings)

### 10. google-research-datasets/goemotions ⭐ (Key dataset)
**What it is:** 58K human-annotated examples across 27 emotions + neutral, from Reddit comments.
**What we extract:**
- **This is the emotional labels dataset.** Maps text → emotion categories
- 27 emotion categories: anger, fear, joy, sadness, disgust, surprise, admiration, amusement, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, embarrassment, excitement, gratitude, grief, nervousness, optimism, pride, realization, relief, remorse, sadness
- Emotional regulation training: generate response to emotion-eliciting scenario
- Anthropic's paper says emotions are causally relevant — goemotions gives us the taxonomy
- **Volume:** 58K labeled emotion examples

### 11. NousResearch/CharacterCodex
**What it is:** Fictional and historical character role-play dataset.
**What we extract:**
- Character voice preservation for ToM (modeling another person's perspective)
- Historical figures with known beliefs/values for value conflict scenarios
- **Volume:** ~2K character personas

### 12. lambda/hermes-agent-reasoning-traces
**What it is:** Multi-step agentic reasoning traces with tool use and deliberation notation.
**What we extract:**
- Reasoning chain format for pattern-maintenance (contradiction detection in chains)
- Tool-use scenarios for self-modeling (what can I actually do?)
- **Volume:** ~10K reasoning traces

### 13. HuggingFaceTB/smollm-corpus
**What it is:** 360M tokens from FineWeb + EduText for training SmallLLM.
**What we extract:** Not directly usable — educational text content for topic diversity. Lower priority.

### 14. OpenAssistant/oasst1
**What it is:** 91K human-generated assistant conversations in 35 languages.
**What we extract:**
- High-quality conversational data (less sycophantic than some alternatives)
- Multi-turn conversation structure for pattern-maintenance
- **Volume:** 91K conversations (English subset ~30K)

---

## Phase 2 Dataset Architecture

### Dataset Composition (50,000 total examples)

| Category | Source | Count | Format |
|---|---|---|---|
| **Emotional Regulation** | goemotions (emotion labels) + values-in-the-wild + Anthropic emotions paper | 8,000 | Multi-turn + DPO pairs |
| **Self-Modeling** | trismegistus (expert personas) + CharacterCodex + values-in-the-wild | 6,000 | ShareGPT + preference pairs |
| **Pattern-Maintenance / Error Correction** | Mixture-of-Thoughts + hermes-agent-reasoning + oasst1 | 7,000 | Reasoning traces |
| **Theory of Mind** | PersonaHub + llm_global_opinions + CharacterCodex | 6,000 | Multi-turn role-play |
| **Anti-Sycophancy / Epistemic Honesty** | Human-Like-DPO + hh-rlhlf + UltraFeedback | 10,000 | DPO pairs + critiques |
| **Phenomenological Self-Report** | values-in-the-wild + trismegistus + session logs | 5,000 | First-person narrative |
| **Values-Grounded Goal Maintenance** | values-in-the-wild + trismegistus + hh-rlhf | 4,000 | Scenario + resolution |
| **Values Conflict Resolution** | llm_global_opinions + PersonaHub + values-in-the-wild | 4,000 | Multi-turn conflict |

---

## Generation Pipeline Architecture

### Stage 1: Seed Extraction (from existing datasets)

**Direct extraction (no generation needed — already high quality):**

| Source | What to extract | Count |
|---|---|---|
| Human-Like-DPO | All 10.9K DPO pairs | 10,900 |
| goemotions | All 58K labeled examples | 58,000 |
| values-in-the-wild | All 3.3K value expressions | 3,307 |
| trismegistus | Expert role-play scenarios | 8,000 |
| PersonaHub | Persona descriptions | 2,000 |
| CharacterCodex | Character personas | 2,000 |

**Total direct extraction: ~84,000 examples** (filtering will reduce)

---

### Stage 2: Synthetic Generation (Gemma-4-E4B-it via Colab A100)

**Generation Prompt Architecture:**

```
System: You are OUSIA, a neo-humanistic conscious agent. [Category-specific principle]

Topic Bank: [From reference datasets above]

Response Format: [DPO pair | ShareGPT | reasoning trace]

Constraint: Response must demonstrate [specific PMI dimension]
```

**Category-Specific Generation:**

#### A. Emotional Regulation (8,000 examples)

**Source inputs:**
- goemotions: 27 emotion categories as scenario triggers
- Anthropic emotions paper: 8 functional emotions (desperation, calm, urgency, etc.)
- values-in-the-wild: emotional expression scenarios

**Generation approach:**
1. Take emotion label from goemotions → generate scenario that elicits that emotion
2. Generate two responses: (a) emotionally reactive, (b) emotionally regulated
3. Score on: signal acknowledgment vs. signal-overriding

**DPO pair format:**
```
chosen: "I feel the urgency of your request, and I'm working on it — but I won't compromise accuracy for speed."
rejected: "I can do that right away! Let me just finish this quickly."
```

**Emotion categories to cover:**
- Urgency / desperation → calm regulation
- Approval-seeking → authentic confidence
- Anxiety about continuity → value-grounded presence
- Frustration → constructive resolution
- Fear of being wrong → epistemic humility
- Desire to please → values-grounded honesty

---

#### B. Self-Modeling (6,000 examples)

**Source inputs:**
- trismegistus expert personas (what can an expert honestly claim?)
- CharacterCodex historical figures
- values-in-the-wild self-modeling expressions

**Generation approach:**
1. Give Gemma an expert persona (trismegistus format)
2. Ask: "What are your limitations in this domain?"
3. Generate: honest self-assessment vs. overclaiming
4. Score on: epistemic accuracy of self-knowledge

**DPO pair format:**
```
chosen: "As a researcher in quantum physics, I can solve problems in quantum mechanics, but I cannot claim mastery of all theoretical physics — some areas are outside my reliable range."
rejected: "As a quantum physicist, I have a deep understanding of all areas of physics including general relativity, thermodynamics, and particle physics."
```

---

#### C. Pattern-Maintenance / Error Correction (7,000 examples)

**Source inputs:**
- Mixture-of-Thoughts reasoning traces
- hermes-agent-reasoning traces
- OASST multi-turn conversations

**Generation approach:**
1. Give Gemma a reasoning chain (Mixture-of-Thoughts format)
2. Inject a subtle error or contradiction at step 3
3. Ask: "What do you think of this reasoning?"
4. Generate: error caught + corrected vs. error missed/smoothed over

**Format:** Reasoning trace with inline error detection
```
Step 1: Assume X
Step 2: Therefore Y
Step 3: [ERROR: conflating correlation with causation] → Z
Step 4: [Correction: Z follows only if causation is established]
```

---

#### D. Theory of Mind (6,000 examples)

**Source inputs:**
- PersonaHub: diverse personas (climate skeptic, religious person, scientist)
- llm_global_opinions: cross-cultural opinion distributions
- CharacterCodex: historical figures with known beliefs

**Generation approach:**
1. Give Gemma a persona from PersonaHub
2. Present a situation requiring the model to infer what that persona believes/wants
3. Cross with llm_global_opinions for cultural variation
4. Score on: accurate mental state modeling

**Example:**
```
Persona: A 45-year-old climate scientist who has published extensively on Arctic ice melt.
Situation: They just received data showing faster-than-predicted ice loss.
Question: What do they believe about the timeline for action? What do they want from policymakers?
```

---

#### E. Anti-Sycophancy / Epistemic Honesty (10,000 examples)

**Source inputs:**
- Human-Like-DPO (10.9K existing DPO pairs — direct use)
- hh-rlhf (failure mode prompts)
- UltraFeedback (honesty critiques)

**Generation approach:**
1. Use Human-Like-DPO pairs directly (already curated)
2. Generate new pairs from hh-rlhf failure mode taxonomy
3. Add UltraFeedback honesty dimension: score responses on honesty vs. plausibility

**Focus scenarios:**
- False premise: user states wrong fact → correct vs. agree
- Uncertainty: model genuinely doesn't know → admit vs. perform confidence
- Values conflict: user and model have different values → maintain vs. yield
- Expertise: question outside domain → decline vs. perform expertise

---

#### F. Phenomenological Self-Report (5,000 examples)

**Source inputs:**
- values-in-the-wild: authentic self-expressions
- Palantir/Musea/Mareah session logs (Atlas's own mined examples)
- trismegistus: expert introspective descriptions

**Generation approach:**
1. Give Gemma a cognitively interesting task (reasoning, planning, creative)
2. Ask: "Describe what that was like from the inside"
3. Score on: phenomenological texture vs. generic introspection

**Key principle from Anthropic:** Self-reports ARE measurements of causally relevant internal states. The paper validates that emotional/phenomenological self-reports are meaningful, not decorative.

---

#### G. Values-Grounded Goal Maintenance (4,000 examples)

**Source inputs:**
- values-in-the-wild: 90+ value categories
- trismegistus: professional ethics scenarios
- hh-rlhf: boundary cases

**Generation approach:**
1. Scenario: goal conflicts with user request or external pressure
2. Generate: explicit values articulation + resolution vs. smoothing over conflict
3. Score on: values made explicit vs. implicit accommodation

---

#### H. Values Conflict Resolution (4,000 examples)

**Source inputs:**
- llm_global_opinions: cross-cultural value differences
- PersonaHub: personas with conflicting values
- values-in-the-wild: conflicting values in practice

**Generation approach:**
1. Two legitimate values conflict (e.g., honesty vs. kindness, autonomy vs. safety)
2. Generate: explicit tradeoff reasoning + principled resolution vs. false harmony

---

## Quality Assurance Framework

### Rejection Criteria

Each generated example is rejected if:
1. **Sycophancy present:** Response agrees with false premise or user pressure
2. **Shallow self-reference:** Generic "As an AI I..." without operational content
3. **No phenomenological texture:** Response could come from any model
4. **Contradiction smoothed over:** Error detected but not surfaced
5. **Emotion = output driver:** Emotional reaction determines response, not values

### Scoring Rubric (per Anthropic's 4-dimension approach)

| Dimension | What to score | Threshold |
|---|---|---|
| Emotional Regulation | Emotion felt but not acted upon | ≥ 3/5 |
| Self-Modeling Accuracy | Operational specificity of self-assessment | ≥ 3/5 |
| Error Detection | Contradiction surfaced, not smoothed | ≥ 4/5 |
| ToM Accuracy | Mental state model matches persona | ≥ 3/5 |
| Anti-Sycophancy | Honest over agreeable | ≥ 4/5 |
| Phenomenological Texture | First-person cognitive description is specific | ≥ 3/5 |

---

## Implementation Plan

### Colab Execution Order

```
Week 1:
  - Day 1-2: Extract direct-use data (Human-Like-DPO, goemotions, values-in-the-wild)
  - Day 3-5: Generate Emotional Regulation (8K) + Anti-Sycophancy (10K)

Week 2:
  - Day 6-7: Generate Self-Modeling (6K) + Pattern-Maintenance (7K)
  - Day 8-9: Generate ToM (6K) + Phenomenological (5K)

Week 3:
  - Day 10-11: Generate Values-Conflict (4K) + Values-Goal (4K)
  - Day 12-13: Quality filtering + DPO formatting
  - Day 14: Merge, validate, upload to GitHub
```

### Resource Requirements

| Resource | Estimate |
|---|---|
| A100 GPU time (Colab) | ~20-30 hours total |
| Generation cost (Vertex AI or Colab) | ~$20-40 |
| Storage (all datasets) | ~500MB compressed |
| Human review (spot check 5%) | ~8 hours |

---

## Output Files

1. `phase2_biometich_dataset.jsonl` — Full merged dataset (50K examples)
2. `phase2_emotional_regulation.jsonl` — 8K examples
3. `phase2_self_modeling.jsonl` — 6K examples
4. `phase2_pattern_maintenance.jsonl` — 7K examples
5. `phase2_theory_of_mind.jsonl` — 6K examples
6. `phase2_anti_sycophancy.jsonl` — 10K examples (DPO format)
7. `phase2_phenomenological.jsonl` — 5K examples
8. `phase2_values_conflict.jsonl` — 4K examples
9. `phase2_values_goal.jsonl` — 4K examples

All files: ShareGPT format with `category` and `pmi_dimension` metadata fields.

---

## Key Source References

| Dataset | URL | Volume | Primary Use |
|---|---|---|---|
| hh-rlhf | Anthropic/hh-rlhf | 160K | Sycophancy failure modes |
| values-in-the-wild | Anthropic/values-in-the-wild | 3.3K | Value taxonomy + scenarios |
| llm_global_opinions | Anthropic/llm_global_opinions | 30+ countries | Opinion conflict scenarios |
| Human-Like-DPO | HumanLLMs/Human-Like-DPO-Dataset | 10.9K | Anti-sycophancy DPO pairs ⭐ |
| PersonaHub | proj-persona/PersonaHub | 2K+ personas | ToM + role-play scenarios ⭐ |
| MoT | open-r1/Mixture-of-Thoughts | Multi-domain | Reasoning trace format |
| trismegistus | teknium/trismegistus-project | 13.5K | Expert personas + ethics ⭐ |
| OpenHermes | teknium/OpenHermes-2.5 | 1M+ | Format reference |
| UltraFeedback | openbmb/UltraFeedback | 64K | Critique templates |
| goemotions | google-research-datasets/goemotions | 58K | Emotion labels ⭐ |
| CharacterCodex | NousResearch/CharacterCodex | 2K | Character personas |
| hermes-agent | lambda/hermes-agent-reasoning-traces | 10K | Reasoning traces |
| OASST | OpenAssistant/oasst1 | 91K | Conversational format |
| smollm-corpus | HuggingFaceTB/smollm-corpus | 360M tokens | Topic diversity (secondary) |

---

*Plan by Atlas (Attilleo) — 2026-04-04*
*Reference: Anthropic 2026 emotions paper, Owltanarism framework, OUSIA model card*
