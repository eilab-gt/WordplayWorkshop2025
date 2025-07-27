# v3.0.0 Implementation Issues and Errors Found

## Critical Issues

### 1. NEAR Operators Not Active in Positive Terms ❌
**Issue**: NEAR operators for positive search terms are in comments instead of being part of the actual terms.

**Spec**:
```yaml
- "red team*" NEAR/5 (exercise OR simulation OR wargame OR campaign OR tabletop)
- "matrix game*" NEAR/5 (scenario OR crisis OR policy OR wargame OR tabletop OR seminar)
```

**My Implementation**:
```yaml
- "red team*" # NEAR/5 (exercise OR simulation OR wargame OR campaign OR tabletop)
- "matrix game*" # NEAR/5 (scenario OR crisis OR policy OR wargame OR tabletop OR seminar)
```

**Impact**: These proximity searches are not being used, significantly reducing search precision. Papers about "red team" that aren't related to wargaming won't be filtered properly.

### 2. Character Encoding Discrepancy ⚠️
**Issue**: The spec uses non-standard hyphens (U+2011) in 70 places, while I used standard hyphens (-).

**Affected Terms**:
- "crisis‑management simulation*" vs "crisis-management simulation*"
- "Claude‑2", "Claude‑3" vs "Claude-2", "Claude-3"
- "GPT‑3", "GPT‑4" vs "GPT-3", "GPT-4"
- "PaLM‑2" vs "PaLM-2"
- "LLaMA‑2" vs "LLaMA-2"

**Impact**: Potentially reduced recall if search engines don't normalize these characters.

## Missing Features

### 3. Secondary Query Strategies Not Implemented ❌
**Issue**: Query strategies are defined in config but not used by harvesters.

**Missing Functionality**:
1. Policy/diplomacy simulation specific queries
2. Grey literature PDF searches with site: operators and filetype:pdf

**Impact**: Missing valuable policy game papers and government/military reports.

### 4. No CNKI (Chinese Database) Support ❌
**Issue**: The spec lists CNKI as a required source, but no harvester exists.

**Impact**: Missing entire Chinese literature on military wargaming and crisis simulation.

### 5. Source-Specific Optimizations Not Used ⚠️
**Issue**: Config defines arxiv categories and semantic_scholar fields, but harvesters use hardcoded values.

**Config Values Ignored**:
- arxiv.categories: [cs.AI, cs.CL, cs.MA, cs.GT, cs.CY]
- semantic_scholar.fields: ["Computer Science", "Political Science", "Military Science"]

**Impact**: Potentially searching wrong categories, missing relevant papers.

## Unimplemented Spec Requirements

### 6. Missing Automation Features ❌
From the spec's implementation TODO list:
- No Cloudflare bypass module for .gov/.mil sites
- No weekly cron job for automated searches
- No monthly metrics report generation
- No automatic precision monitoring
- No backfill of existing corpus with grey_lit tags

### 7. Data Storage Issues ⚠️
**Issue**: Spec requires saving grey literature separately in `/grey_lit/` directory, but this isn't implemented.

## Correctly Implemented Features ✅

### Working as Specified:
1. **Disambiguation Rules**: All rules correctly implemented including positive_required logic
2. **Grey Literature Tagging**: Correctly tags papers from .mil, .gov, .nato.int domains
3. **Exclusion Export**: Papers excluded by disambiguation are exported for human review
4. **Query Translation**: NEAR operators and wildcards properly translated per platform
5. **Year Range**: Correctly updated to 2022-2025

## Recommendations for Fixes

### High Priority:
1. **Fix NEAR operators in config**: Remove comments, make them part of actual terms
2. **Implement secondary query strategies**: Add methods to use policy and grey-lit specific queries
3. **Fix character encoding**: Consider normalizing non-standard hyphens

### Medium Priority:
1. **Use config values**: Make harvesters read source-specific optimizations from config
2. **Add CNKI harvester**: Implement Chinese database support
3. **Implement Cloudflare bypass**: Important for accessing .gov/.mil content

### Low Priority:
1. **Add automation**: Cron jobs and monitoring can be added later
2. **Separate grey lit storage**: Can be handled in post-processing

## Summary

The v3.0.0 implementation has most core features working correctly, but has critical issues with:
- NEAR operators not being active for positive terms (major precision impact)
- Missing secondary query strategies (reduced recall for policy games)
- No Chinese database support (missing international literature)

The character encoding issue may or may not impact results depending on search engine normalization.

These issues should be addressed before running production searches to ensure the expected precision (≥65%) and recall (≥90%) targets are met.
