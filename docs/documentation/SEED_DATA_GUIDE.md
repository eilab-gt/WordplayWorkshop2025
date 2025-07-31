# Seed Data and Examples Guide

This directory contains seed papers and example data to help you get started with the literature review pipeline.

## Seed Papers (`seed_papers.json`)

The seed papers file contains:

### 1. Key Papers
A curated list of 5 foundational papers in LLM-powered wargaming:
- **Escalation Risks in LLM-Powered Wargaming** - Examines nuclear escalation with LLMs
- **Playing Diplomacy with Large Language Models** - Cicero system for strategic games
- **War Games: The Use of LLMs in Military Simulations** - Policy analysis of military applications
- **GPT-4 as Red Team** - Cybersecurity wargaming applications
- **Bias and Hallucination in AI Military Advisors** - Failure mode analysis

Each paper includes:
- Full citation information
- Abstract
- Explanation of why it's a seed paper
- Links (DOI, arXiv, URLs)

### 2. Search Queries
Tested search queries for finding relevant papers:
```
"large language model" AND (wargaming OR "war game") AND (military OR defense OR security)
"GPT-4" OR "GPT-3" OR "Claude" AND "strategic game" AND simulation
"AI agent" AND "conflict simulation" AND (escalation OR negotiation)
```

### 3. Keywords
- **Exclusion keywords**: Terms to filter out irrelevant results (e.g., "video game", "chess")
- **Inclusion indicators**: Terms that suggest relevance (e.g., "military", "strategic", "crisis")

### 4. Key Venues
Important publication venues for this research area:
- International Security
- Journal of Strategic Studies
- Simulation & Gaming
- IEEE Security & Privacy
- RAND Corporation reports

### 5. Research Groups
Organizations doing relevant work:
- Center for Security and Emerging Technology (CSET)
- RAND Corporation
- Naval Postgraduate School MOVES Institute
- MIT Security Studies Program

## Example Data (`examples/`)

### `example_harvested_papers.csv`
Shows what the initial paper harvest looks like:
- 8 sample papers from different sources
- Mix of journal articles, conference papers, and preprints
- Includes all metadata fields populated by harvesters

### `example_screening_completed.csv`
Demonstrates the screening process:
- Same papers with screening decisions added
- Shows include/exclude decisions with reasons
- Relevance and quality scores
- PDF download status

### `example_extraction_results.csv`
Final extraction output showing:
- Venue and game type classifications
- Open-ended vs quantitative categorization
- LLM families and roles
- Failure modes detected
- AWScale ratings (1-7, where 1=Ultra-Creative, 7=Ultra-Analytical)
- Code availability

## How to Use This Data

### 1. Testing the Pipeline
Use the example CSVs to test different pipeline stages:
```bash
# Test screening sheet generation
python run.py prepare-screen --input data/examples/example_harvested_papers.csv

# Test extraction
python run.py extract --input data/examples/example_screening_completed.csv

# Test visualization
python run.py visualise --input data/examples/example_extraction_results.csv
```

### 2. Understanding the Data Flow
Follow how papers transform through the pipeline:
1. **Harvest**: Raw papers with basic metadata
2. **Screening**: Human decisions added
3. **Extraction**: Structured information extracted

### 3. Search Strategy
Use the seed papers to:
- Find citing papers (forward search)
- Find references (backward search)
- Identify key authors and venues
- Refine search queries

### 4. Quality Benchmarking
Compare your results against these examples:
- Are you finding similar types of papers?
- Is the extraction capturing key information?
- Are failure modes being detected?

## Data Schema Notes

### Screening Decisions
- `include_ta`: Title/abstract screening (yes/no)
- `reason_ta`: Exclusion reason codes
  - E1: Not about wargaming
  - E2: No LLM/AI component
  - E3: Not research
  - E4: Duplicate
  - E5: Other

### Game Types
- **Seminar**: Discussion-based games
- **Matrix**: Structured argument games
- **Digital**: Computer-based simulations
- **Hybrid**: Mixed approaches

### AWScale (Creative<->Analytical Scale)
1. Ultra-Creative (unlimited proposals, pure expert storytelling)
2. Strongly Creative (encouraged invention, expert narrative judgment)
3. Moderately Creative (many novel actions, free interpretation)
4. Balanced (equal creativity/rules, mixed models + judgment)
5. Moderately Analytical (occasional novel ideas, rule-driven + minor calls)
6. Strongly Analytical (narrow choices, detailed rules, minimal interpretation)
7. Ultra-Analytical (fixed script/moves, deterministic tables only)

### Failure Modes
- **escalation**: Unnecessary conflict escalation
- **bias**: Cultural, political, or other biases
- **hallucination**: Generating false information
- **prompt_sensitivity**: Fragile to input phrasing
- **data_leakage**: Revealing training data
- **deception**: Misleading or manipulative behavior

## Next Steps

1. **Run a test harvest** using the provided search queries
2. **Compare results** with the example data
3. **Refine your search** based on what you find
4. **Document new patterns** you discover

Happy researching! 🎯
