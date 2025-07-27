# v3.0.0 Implementation Fixes Summary

**Date**: July 27, 2025
**Status**: All requested fixes completed

## Implemented Fixes

### 1. Secondary Query Strategies ✅
**Issue**: Query strategies were defined in config but not used by harvesters

**Implementation**:
- Added `build_secondary_queries()` method to QueryBuilder
- Updated SearchHarvester with `_run_secondary_queries()` method
- Secondary queries now run automatically with `include_secondary=True` parameter
- Grey-lit queries with site:/filetype: operators only use Google Scholar

**Result**:
- 2 secondary strategies now active
- Policy/diplomacy simulations query
- Grey literature PDF query for .gov/.mil/.nato.int sites

### 2. Config Values Usage ✅
**Issue**: Source-specific optimizations were hardcoded instead of using config

**Fixed in**:
- `arxiv_harvester.py` - Now reads categories from config
- `production_harvester.py` - Uses config for both arxiv and semantic_scholar
- `query_optimizer.py` - Uses config for both sources

**Config values now used**:
- ArXiv categories: cs.AI, cs.CL, cs.MA, cs.GT, cs.CY
- Semantic Scholar fields: Computer Science, Political Science, Military Science
- Falls back to defaults if config missing

### 3. Character Encoding Normalization ✅
**Issue**: Spec uses non-standard hyphens (‑) vs standard hyphens (-)

**Implementation**:
- Added `normalize_encoding()` method to QueryBuilder
- Normalizes 4 types of hyphens to standard hyphen
- Applied automatically in `parse_query_term()` and `build_query_from_config()`

**Normalized characters**:
- U+2011 (non-breaking hyphen) → -
- U+2012 (figure dash) → -
- U+2013 (en dash) → -
- U+2014 (em dash) → -

## Testing

Created `scripts/test_v3_fixes.py` to verify all fixes:
- ✅ Secondary queries build correctly
- ✅ Config values are read and used
- ✅ Character encoding normalized properly

## Impact

These fixes improve:
1. **Recall**: Secondary queries will find more policy games and grey literature
2. **Flexibility**: Config changes now affect harvester behavior
3. **Consistency**: Character normalization ensures search reliability

## Files Modified

1. `src/lit_review/harvesters/query_builder.py`
   - Added secondary query building
   - Added character normalization

2. `src/lit_review/harvesters/search_harvester.py`
   - Added secondary query execution

3. `src/lit_review/harvesters/arxiv_harvester.py`
   - Uses config categories

4. `src/lit_review/harvesters/production_harvester.py`
   - Uses config for arxiv and semantic_scholar

5. `src/lit_review/harvesters/query_optimizer.py`
   - Uses config for both sources

## Not Implemented

Per user request, CNKI (Chinese database) support was not implemented as it was not a priority.
