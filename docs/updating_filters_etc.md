Consolidated Search‑Pipeline Design for

LLM‑Powered Open‑Ended Wargaming, Policy Games & Crisis Simulations
(Version 3 — 27 July 2025)

⸻

1  Scope & Success Criteria

Item	Target
Time‑window	2022 – 2025 (override‑able)
Recall	≥ 90 % of known peer‑reviewed LLM‑wargaming papers retrieved
Precision (first‑page)	≥ 65 % peer‑reviewed papers among first 20 hits
Sources	ArXiv, Google Scholar, Scopus/WoS, CNKI (Chinese), Grey‑lit (.mil / .gov / .nato)
Tags	✔ peer‑reviewed & indexed△ grey literature / reports / theses✖ noise


⸻

2  What Worked
	1.	Matrix‑game disambiguation using proximity + negative context cut >90 % of Nash/payoff noise.
	2.	Red‑teaming context filter removed LLM jailbreak papers without harming recall.
	3.	Press‑release & vendor exclusions eliminated most marketing clutter.
	4.	Secondary queries (“Policy Sim”, “Grey‑Lit PDF”) surfaced scarce but valuable policy‑game and NATO/gov reports.
	5.	Chinese keyword injection produced evidence that a parallel literature exists (e.g., INDSR 2025).

⸻

3  What Still Adds Noise

Noise pattern	Mitigation adopted
Broad LLM surveys / roadmaps	Conditional generic_surveys disambiguator (exclude unless “wargame/simulation” present)
Opinion‑dynamics / social‑sim papers	Added opinion dynamics, polarisation to exclusion list when no wargame term present
Medical / hospital simulations	Exclude hospital, medical, doctor, nurse
Political‑science essays on democracy / governance	Exclude politics, democracy, governance unless simulation/wargame present


⸻

4  Final config.yaml (full, unabridged)

# ============================================================================
# 0.  Administrative ----------------------------------------------------------------
version: 3.0.0
description: "Search config for LLM‑powered open‑ended wargaming & crisis sims"

# ============================================================================
# 1.  Temporal window ---------------------------------------------------------
search:
  years:
    start: 2022          # beginning of large‑model era in wargaming
    end:   2025

# ============================================================================
# 2.  Positive vocabulary -----------------------------------------------------
wargame_terms:
  # Core
  - "wargame*"
  - "war game*"
  - "war gaming"
  - "tabletop exercise*"
  - "TTX"
  - "seminar wargame*"
  - "policy game*"
  - "policy gaming"
  - "strategic game*"
  - "crisis game*"
  - "military exercise*"
  - "defen?e exercise*"
  - "red team*" NEAR/5 (exercise OR simulation OR wargame OR campaign OR tabletop)
  # Matrix‑wargame only
  - "matrix wargame*"
  - "matrix game*" NEAR/5 (scenario OR crisis OR policy OR wargame OR tabletop OR seminar)
  # Crisis / policy / diplomacy
  - "crisis simulation*"
  - "crisis‑management simulation*"
  - "military crisis simulation*"
  - "crisis decision‑making exercise*"
  - "diplomacy simulation*"
  - "diplomatic game*"
  - "negotiation game*"
  - "negotiation simulation*"
  - "crisis‑negotiation exercise*"
  - "policy simulation*"
  - "statecraft simulation*"
  - "Model Diplomacy"
  # Named LLM‑wargame systems
  - "Snow Globe"
  - "WarAgent"
  # Chinese
  - "军事推演"        # military wargame
  - "危机模拟"        # crisis simulation

llm_terms:
  - "large language model*"
  - LLM
  - "foundation model*"
  - transformer*
  - "generative AI"
  - "generative artificial intelligence"
  # Specific models
  - GPT
  - GPT‑3
  - GPT‑3.5
  - GPT‑4
  - ChatGPT
  - InstructGPT
  - Claude
  - "Claude‑2"
  - "Claude‑3"
  - PaLM
  - "PaLM‑2"
  - Gemini
  - Bard
  - LLaMA
  - "LLaMA‑2"
  - Alpaca
  - Vicuna
  - BERT
  - T5
  - Cicero
  # Agent terms
  - "AI agent"
  - "LLM agent"
  - "multi‑agent"
  - "autonomous agent*"
  # Chinese
  - "大语言模型"
  - "生成式人工智能"

# ============================================================================
# 3.  Exclusion vocabulary (hard filters) ------------------------------------
exclusion_terms:
  # Game‑theory noise
  - '"matrix game*" NEAR/5 (Nash OR equilibrium OR payoff OR "normal form" OR "game theory")'
  - AlphaZero
  - '"reinforcement learning" NEAR/5 (Go OR chess OR poker OR "board game" OR Atari)'
  # Red‑teaming jailbreak noise
  - '"red teaming" NEAR/5 (LLM OR "language model" OR ChatGPT OR jailbreak OR prompt OR adversarial)'
  - "prompt injection"
  - "adversarial example"
  # Commentary / marketing
  - warontherocks.com
  - madsciblog.tradoc.army.mil
  - medium.com
  - substack.com
  - "press release"
  - "blog post"
  - newsletter
  - "vendor case study"
  - marketing
  - podcast
  # Social‑sim & political noise
  - "opinion dynamics"
  - polarisation
  - politics
  - democracy
  - governance
  # Surveys & alignment (conditional block in §4)
  - "hallucination survey"
  - "alignment roadmap"
  # Medical simulation noise
  - hospital
  - medical
  - doctor
  - nurse

# ============================================================================
# 4.  Disambiguation rules (post‑search regex) -------------------------------
disambiguation:
  matrix_game:
    negative_context: [Nash, equilibrium, payoff, "normal form", "game theory"]
  red_teaming:
    negative_context: [LLM, ChatGPT, jailbreak, prompt, "adversarial example", "prompt injection"]
  rl_board_game:
    negative_context: [AlphaZero, "board game", "Go game", chess, Atari]
  generic_surveys:
    negative_context: [survey, roadmap, hallucination, alignment]
    positive_required: [wargame, simulation, exercise, crisis, military]

# ============================================================================
# 5.  Grey‑literature tagging -------------------------------------------------
grey_lit_sources:
  - ".mil"
  - ".gov"
  - ".nato.int"
  - warontherocks.com
  - madsciblog.tradoc.army.mil
  - paxsims.wordpress.com
  - think‑tank

# ============================================================================
# 6.  Query strategies --------------------------------------------------------
query_strategies:
  primary:
    description: "Core Wargame × LLM query"
    template: |
      ({wargame_terms}) AND ({llm_terms}) NOT ({exclusion_terms})

  secondary:
    - description: "Policy / diplomacy simulations"
      template: |
        ("policy simulation*" OR "negotiation game*" OR "Model Diplomacy"
         OR "diplomacy simulation*" OR "crisis‑negotiation exercise*")
        AND ({llm_terms}) NOT ({exclusion_terms})

    - description: "Grey‑lit NATO / GOV PDFs"
      template: |
        ("wargame*" OR "simulation" OR "exercise")
        AND ("large language model" OR LLM OR ChatGPT OR GPT‑4)
        AND (site:.gov OR site:.mil OR site:.nato.int)
        AND (filetype:pdf)
        NOT ({exclusion_terms})

# ============================================================================
# 7.  Source‑specific limits --------------------------------------------------
source_optimizations:
  arxiv:
    categories: [cs.AI, cs.CL, cs.MA, cs.GT, cs.CY]
  semantic_scholar:
    fields: ["Computer Science", "Political Science", "Military Science"]
  google_scholar:
    include_patents: false
    include_citations: true

# ============================================================================
# 8.  Quality metrics ---------------------------------------------------------
quality_metrics:
  minimum_precision: 0.65
  target_recall: 0.9


⸻

5  Query Cheat‑Sheet for Manual Spot‑Checks

ArXiv (advanced search → all fields)

("wargame" OR "crisis simulation" OR "policy simulation") AND
("large language model" OR LLM OR ChatGPT OR GPT‑4) AND
submittedDate:[2022 TO 2025]
NOT (Nash OR equilibrium OR "press release" OR hospital OR polarisation)

Google Scholar

("wargame*" OR "military crisis simulation*" OR "policy simulation*" OR "Model Diplomacy") AND
("large language model" OR ChatGPT OR GPT‑4 OR Claude)
since 2022
-"Nash" -"equilibrium" -"AlphaZero" -"survey" -"blog post" -"hospital"

CNKI (Chinese)

(军事推演 OR 危机模拟) AND (大语言模型 OR 生成式人工智能)  年:2022-2025


⸻

6  Implementation TODO (For Agent & Codebase)

#	Task	Owner	Notes
1	Replace existing config.yaml with the full file above (v 3.0.0).	DevOps	Push to main; tag release search‑config‑v3.
2	Update retrieval script to: • apply grey_lit_sources tagging;• run disambiguation.generic_surveys logic (exclude if negative_context matches and no positive_required present).	Backend engineer	Regex examples already in §4.
3	Extend agent to query CNKI & Sciengine via bilingual endpoint; ingest bibliographic JSON.	Search‑agent team	Use same filters; remember CNKI requires GBK encoding.
4	Add Cloudflare‑bypass module: try curl --compressed --header "User‑Agent: Mozilla"; on failure, log URL + status.*	Crawler maintainer	Only for .gov/.mil/.nato.int domains.
5	Implement weekly cron job: run primary + secondary queries; append new hits to /data/master_hits.csv; deduplicate by DOI/URL.	Ops	Keep a first_seen timestamp.
6	Generate monthly metrics report: precision (manual sample n = 40), recall vs. gold list, new noise phrases. Automatically raise GitHub issues if precision < 0.6.	Analyst	Template in /reports/metrics_template.md.
7	Backfill existing corpus with tag = △ for domains in grey_lit_sources; store separately (/grey_lit/).	Data steward	Enables optional inclusion downstream.
8	Update documentation wiki: copy this markdown file to docs/search_pipeline_v3.md.	Tech writer	Link from README.

* If Cloudflare still blocks, save only title & URL; mark content_retrieved = false.

⸻

7  Future Enhancements
	1.	Named‑entity expansion: auto‑discover new LLM model names (e.g., GPT‑5, DPoET) and append to llm_terms monthly.
	2.	Embedding‑based re‑rank: after Boolean retrieval, run SciBERT embeddings ➔ rerank top 200 for semantic similarity to seed set.
	3.	Multilingual extension: add Russian (военная игра), Spanish (juego de guerra) vocab once volumes justify.
	4.	Human‑in‑the‑loop: crowdsource abstract triage to interns for quarterly gold‑standard set update.

⸻

8  Quick‑Reference Glossary

Abbrev.	Meaning
LLM	Large Language Model
TTX	Table‑Top Exercise
CNKI	China National Knowledge Infrastructure database
Grey‑lit	Non‑peer‑reviewed but authoritative (gov, think‑tank, thesis)


⸻

End of Report

Prepared for the WOPR search‑automation project – 27 Jul 2025
