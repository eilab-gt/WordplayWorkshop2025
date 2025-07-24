## Hand‑off document for the “coding agent”

*(A structured design brief that contains everything another AI engineer needs to implement the programmatic survey pipeline.)*

### 1 Objective

Build a **fully reproducible Python pipeline** that (i) harvests literature matching our v1.0 criteria, (ii) screens & deduplicates, (iii) stores metadata/extracted variables in canonical CSV/JSON, and (iv) generates basic descriptive plots for the paper.

### 2 High‑level architecture

```
              +-------------------+
   config/config.yaml|                   | seeds.csv
   criteria.v1|                   |
              v                   |
+-------------+-------+    +------+-----------+
|  Search Harvester  |--->|  Normaliser      |
|  (APIs & scraping) |    |  (dedupe + merge)|            +------------+
+-------------+------+    +------+-----------+            | logging.db |
              |                  |                        +------------+
              |                  v
              |      +-----------+---------+             +-------------+
              |      |  PDF Fetcher        |------------>|  pdf_cache/ |
              |      +-----------+---------+             +-------------+
              |                  |
              v                  v
+-------------+-------+    +-----+-------------+
|  Screen UI / CSV    |    |  LLM Extractor    |<--openai‑api key
|  (title/abs, full)  |    |  & Tagger         |
+-------------+-------+    +-----+-------------+
              \____________________/
                       |
                       v
               extraction.csv
                       |
                       v
             +---------+---------+
             | Visualiser &      |
             | Stats Notebook    |
             +-------------------+
```

### 3 Module‑by‑module specs

| #     | Module              | Key functions                                                                                                                                                                                 | Packages / APIs                        |
| ----- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| **0** | `config/config.yaml`       | • criteria file path  • API keys  • query strings  • paths                                                                                                                                    | `PyYAML`                               |
| **1** | **SearchHarvester** | `query_db()` wraps:  – Google Scholar (scholarly)  – arXiv (`arxiv` pkg)  – Semantic Scholar (REST)  – Crossref (works endpoint).  Returns unified list of dicts.                             | `scholarly` ≥ 1.7  `arxiv`  `requests` |
| **2** | **Normaliser**      | • DOI harmonisation with Crossref  • string‑slug for titles  • `drop_duplicates(subset=['doi','title_slug'])`  • output `papers_raw.csv`                                                      | `pandas`  `rapidfuzz`                  |
| **3** | **PDFFetcher**      | Bulk DOI‑to‑PDF (Unpaywall API), fallback to arXiv links; save under `pdf_cache/{first_author}_{year}.pdf`, record path and HTTP status.                                                      | `requests`                             |
| **4** | **ScreenUI**        | Generate a Google‑Sheets‑ready CSV with empty columns `include_ta,reason_ta,include_ft,reason_ft`; optionally bootstrap **ASReview LAB** project if user wants AL‑assisted screening.         | `pandas`  `asreview` (opt.)            |
| **5** | **LLMExtractor**    | For each included PDF:  • parse metadata (pdfminer)  • call GPT‑4o with system prompt that maps output into the extraction schema.  • auto‑assign AWScale with rubric heuristics + LLM check. | `pdfminer.six`  `openai`               |
| **6** | **Tagger**          | Regex heuristic fallback if LLM fails; maps failure‑mode keywords to controlled vocab.                                                                                                        | `re`                                   |
| **7** | **Visualizer**      | `plot_counts(df,'game_type')`, `plot_time_series(df)`, `plot_awscale_hist(df)`—single‑plot rule (no subplots).                                                                                | `matplotlib` (no seaborn)              |
| **8** | **Exporter**        | Package `extraction.csv`, figures/\*.png into /output and optionally push to Zenodo via API.                                                                                                  | `zipfile`  `tqdm`                      |

### 4 Data structures

```python
# papers_raw.csv
doi,title,authors,year,abstract,source_db,url

# screening_progress.csv (superset of raw)
... ,include_ta,reason_ta,include_ft,reason_ft,pdf_path

# extraction.csv (final)
doi, title, year, venue_type, game_type, open_ended, quantitative,
llm_family, llm_role, eval_metrics, failure_modes, awscale,
code_release, grey_lit_flag, language
```

### 5 Search‑string template (parameterised)

```jinja
({{ wargame_terms|join(' OR ') }})
AND ("large language model" OR LLM OR GPT OR Claude OR PaLM OR Llama)
AND (play OR player OR "scenario generation" OR evaluation OR benchmark)
NOT ("StarCraft" OR "AlphaGo" OR Atari)
```

`wargame_terms` defaults: *wargame*, “seminar wargame”, “matrix wargame”, Diplomacy, “crisis simulation”.

### 6 Configuration snippet (`config/config.yaml`)

```yaml
years: [2018, 2025]
llm_min_params: 100_000_000
inclusion_flags: [open_ended | quantitative]
failure_vocab:
  content: [bias, hallucination, factual_error]
  interactive: [escalation, deception, prompt_sensitivity]
  security: [data_leakage, jailbreak]
api_keys:
  semantic_scholar: "YOUR_KEY"
  openai: "YOUR_KEY"
paths:
  cache_dir: "./pdf_cache"
  output_dir: "./output"
```

### 7 Minimum viable command‑line interface

```bash
# 1. Harvest & dedupe
python run.py harvest --query preset1

# 2. Create screening sheet
python run.py prepare-screen --input papers_raw.csv

# 3. After manual screening, extract & tag
python run.py extract --input screening_progress.csv

# 4. Visualise
python run.py visualise --input extraction.csv
```

### 8 Quality & logging

* Every module logs to SQLite (`logging.db`) with timestamps, function, and status.
* Hash PDFs (`sha256`) to detect duplicates later.
* Unit tests with `pytest` for Normaliser, Tagger, AWScale rubric.

### 9 Milestones & time estimates (3‑person team)

| Day | Deliverable                                       | Who     |
| --- | ------------------------------------------------- | ------- |
| 1   | Repo + config/config.yaml scaffold                       | Dev 1   |
| 2   | Harvester + Normaliser                            | Dev 1   |
| 3   | PDF fetcher + logging                             | Dev 2   |
| 4   | Screening sheet generated; start manual TA screen | All     |
| 6   | LLM extractor functional                          | Dev 2   |
| 7   | Tagger + AWScale rubric                           | Dev 3   |
| 8   | First full run on seed papers; fix edge cases     | Dev 1/2 |
| 10  | Visualiser + export bundle                        | Dev 3   |
| 12  | Code freeze & Zenodo release                      | All     |

### 10 Future‑proof hooks

* `plugins/` directory where new search adapters (IEEE, ACM DL) can be dropped.
* Optional “living review” GitHub Action calling the Harvester monthly.

---

### Hand‑off note

> *“Coding agent, you now own the pipeline described in Part B.  All review criteria and definitions live in `criteria_v1.md` (attached).  Your first task is to instantiate `config/config.yaml`, populate the seed list, and implement Module 1 (SearchHarvester) with Google Scholar + arXiv support.  Follow the package list; no R dependencies allowed.  Adhere to the one‑plot‑per‑figure rule when you reach Visualization.”*
