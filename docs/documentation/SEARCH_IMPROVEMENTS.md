# Literature Review Search Improvements

## Overview
This document details the search configuration improvements made to enhance the literature review pipeline's ability to find relevant papers about LLMs in wargaming contexts. The improvements are based on analysis of 5 foundational seed papers and expand both the temporal scope and search term coverage.

## Key Changes

### 1. Extended Year Range
- **Previous**: 2020-2024 (test) / 2020-2025 (production)
- **Updated**: 2018-2025 (all configurations)
- **Rationale**: Several important foundational papers, including "Cicero" (2022), emerged before 2020

### 2. Expanded Search Terms

#### Wargame Terms (17 → 40+ terms)
**New categories added**:
- **Specific wargame types**: TTX, matrix games, seminar wargames, policy games, free kriegsspiel
- **Crisis simulations**: Crisis management, conflict simulation, diplomatic simulation
- **Military contexts**: Red teaming, blue team, operational planning, threat assessment
- **Decision-making**: Military/diplomatic/strategic/crisis decision-making
- **Specific frameworks**: Snow Globe, WarAgent, Diplomacy game

#### LLM Terms (20 → 45+ terms)
**New additions**:
- **Specific models**: GPT-4, Claude-3, Gemini, Cicero, LLaMA-2, PaLM-2
- **Multi-agent focus**: Multi-agent systems, LLM agents, autonomous agents
- **Technical terms**: Foundation models, transformer models, attention mechanisms

#### Action Terms (Enhanced)
**Behavioral additions from seed papers**:
- Escalation/de-escalation dynamics
- Negotiation and diplomacy
- Human-AI teaming
- Strategic reasoning

#### Exclusion Terms (6 → 25+ terms)
**Better precision through**:
- Specific video game titles (StarCraft, Minecraft, etc.)
- Entertainment contexts (esports, game development)
- Non-relevant AI applications (NPC behavior, pathfinding)

## Search Strategy Improvements

### Query Optimization
Created multiple search strategies:
1. **Primary**: Broad boolean combination of wargame AND LLM terms
2. **Secondary strategies**:
   - LLM agents in strategic contexts
   - Specific model applications (GPT-4, Claude in wargaming)
   - Multi-agent conflict simulations
   - Escalation and decision-making focus
   - Human-AI teaming in wargames

### Source-Specific Optimizations
- **arXiv**: Added category filters (cs.AI, cs.GT, cs.MA, cs.CL)
- **Semantic Scholar**: Field filters for Computer Science, Political Science, Military Science
- **Google Scholar**: Disabled patents, enabled citations

## Implementation Details

### Files Updated
The search improvements have been consolidated into a single comprehensive configuration file:
- **config/config.yaml**: Contains all search terms, strategies, and source-specific optimizations for both testing and production use

### Quality Metrics
- **Target precision**: ≥70% relevant results
- **Target recall**: ≥90% of relevant papers found
- **Relevance scoring**: High/medium/low indicators defined

## Validation Approach

### Seed Paper Coverage
The 5 seed papers represent different aspects of LLMs in wargaming:
1. **Snow Globe** - Automated qualitative wargaming framework
2. **WarAgent** - Multi-agent simulation of world wars
3. **Human vs. Machine** - Behavioral comparison study
4. **Escalation Risks** - Analysis of LLM escalation tendencies
5. **Cicero** - Human-level strategic gameplay in Diplomacy

### Expected Improvements
- Better coverage of multi-agent systems research
- Increased capture of escalation/crisis management papers
- More comprehensive military/defense context papers
- Reduced noise from entertainment gaming papers

## Future Considerations

### Monitoring
- Track precision/recall metrics after initial harvest
- Analyze which search terms are most effective
- Identify any gaps in coverage

### Potential Refinements
- Add emerging LLM models as they appear
- Update exclusion terms based on false positives
- Refine query strategies based on performance

## References
- Seed papers analysis: `/data/seed_papers.json`
- Original configuration baseline: Previous git commits
- Search harvester implementation: `/src/lit_review/harvesters/search_harvester.py`
