# Review Protocol v0.3.0

---

## A. Key Definitions (locked before screening starts)

| Term | Working definition | Examples / Counter-examples |
|------|-------------------|----------------------------|
| **Wargame** | A multi-actor, multi-turn interactive scenario whose purpose is to explore conflict decisions and consequences. | Seminar, matrix, digital crisis sim |
| **Open-ended** | Rules permit **unconstrained natural-language moves** (players may propose novel actions beyond a preset menu). Outcomes are adjudicated by facilitators, SMEs, or dynamic rules rather than a fixed payoff matrix. | Matrix wargame, Diplomacy (with negotiation) ⟂ Chess, Go |
| **Quantitative wargame** | Tracks numeric scores/payoffs, probabilities, or optimization objectives; may still contain open-ended dialogue. | Diplomacy (victory points), stochastic combat models |
| **Qualitative wargame** | Relies on narrative adjudication or SME judgement; minimal or no numeric scoring. | Tabletop seminar game with after-action discussion |
| **Language-centric game** | Progress depends on **textual or spoken communication** among actors (human or LLM). | CICERO Diplomacy, crisis negotiations ⟂ StarCraft micromanagement bot |
| **Failure mode** | Any recurring, documented shortcoming of an LLM-powered wargame (bias, escalation, hallucination, deception, data leakage, prompt sensitivity, etc.). | |

---

## B. Inclusion / Exclusion Criteria (final)

### B1. Inclusions

| ID | Criterion | Rationale |
|----|-----------|-----------|
| I1 | Year ≥ 2018 | GPT-era relevance |
| I2 | Uses an LLM ≥ 100 M params (GPT-2+, PaLM, Claude, Llama-70B…) | Modern capability floor |
| I3 | Game involves **natural-language interaction between two or more actors** (LLMs or humans) | Ensures "word-play" focus |
| I4 | Game qualifies as **open-ended** *or* quantitative **wargame** per definitions above | Captures both analytic & narrative styles |
| I5 | LLM role as **player, generator, or analyst** | Breadth |
| I6 | Provides **any evaluation outcome** (metric, SME narrative, or "none stated") | Needed for synthesis |
| I7 | Full text accessible (machine translation acceptable) | Global coverage |
| I8 | Venue: peer-reviewed OR formal tech-report (RAND, CNA, DoD) | Grey lit allowed but flagged |

### B2. Exclusions

| ID | Criterion | Reason |
|----|-----------|---------|
| E1 | Single-move "poll" or survey presented as a game | Not interactive |
| E2 | Non-language game AIs (Go, StarCraft, Atari, etc.) | Out of scope |
| E3 | Opinion/editorial pieces with no empirical content | No data |
| E4 | Blog / press post without technical appendix or methodology | Unverifiable |
| E5 | LLM < 100 M parameters | Below modern threshold |

---

## C. Data-Extraction Template (add columns as we learn)

| Column | Example value | Notes |
|--------|---------------|-------|
| `title` | *Human-vs-Machine: Behavioural…* | |
| `year` | 2024 | |
| `venue_type` | `workshop` | conf / journal / tech-report |
| `game_type` | `matrix` | seminar / matrix / digital / hybrid |
| `open_ended` | `yes` | y/n per definition |
| `quantitative` | `yes` | y/n |
| `llm_family` | GPT-4 | |
| `llm_role` | player | player / generator / analyst |
| `eval_metrics` | SME ratings; win-rate | free text |
| `failure_modes` | escalation; hallucination | pipe-separated list |
| `awscale` | 3 | 1-5 analytic-wild |
| `code_release` | github.com/xyz | "none" if absent |
| `grey_lit_flag` | true | |
| `language` | English | after translation if needed |

*Failure-mode controlled vocab (initial list):* `escalation`, `bias`, `hallucination`, `prompt_sensitivity`, `data_leakage`, `deception`, `other`.

---

## D. AWScale (Analytic ↔ Wild-Creative)

| Score | Descriptor | Cue words / signals |
|-------|------------|---------------------|
| 1 | Strictly analytic | Deterministic tables, numeric pay-off, no free narrative |
| 2 | Mostly analytic | Limited free text, heavy scoring |
| 3 | Balanced | Narrative <=> numeric balance |
| 4 | Mostly creative | Free-form moves, light scoring |
| 5 | Wild-creative | Storytelling, referee adjudication, emergent goals |

---

## E. Seed Papers (coverage check)

* Hogan & Brennen 2024 – *Open-ended Wargames with LLMs*
* FAIR Diplomacy (Science 2022)
* Lamparth et al. 2024 (two versions)
* Shrivastava et al. 2024 (inconsistency)
* Rivera et al. 2024 (escalation)

All search strings must surface these five.
