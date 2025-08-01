# BDD Test Naming Guide

## Naming Pattern
`test_<action>_when_<condition>_then_<expected_result>`

Or simpler:
`test_<what_it_should_do>`

## Examples of Refactoring

### ❌ Bad: Generic Names
```python
def test_init(self):
def test_config(self):
def test_basic(self):
def test_harvest_1(self):
```

### ✅ Good: Behavior-Describing Names
```python
def test_initializes_with_correct_rate_limits(self):
def test_creates_cache_directory_on_initialization(self):
def test_respects_rate_limits_when_harvesting_papers(self):
def test_recovers_from_api_failures_gracefully(self):
```

## Refactoring Map

### Current → Improved

1. **Initialization Tests**
   - `test_init` → `test_creates_required_directories_on_initialization`
   - `test_init` → `test_loads_configuration_with_correct_defaults`
   - `test_init` → `test_initializes_with_production_rate_limits`

2. **Harvester Tests**
   - `test_search` → `test_returns_papers_matching_search_criteria`
   - `test_filter_by_year` → `test_excludes_papers_outside_configured_year_range`
   - `test_dedup` → `test_removes_duplicate_papers_by_doi_and_title`

3. **PDF Fetcher Tests**
   - `test_download_pdf` → `test_downloads_pdf_from_valid_url`
   - `test_fetch_pdfs` → `test_reuses_cached_pdfs_instead_of_downloading`
   - `test_verify_pdf` → `test_rejects_non_pdf_content_during_download`

4. **Extractor Tests**
   - `test_extract` → `test_extracts_research_questions_from_paper_content`
   - `test_awscale` → `test_assigns_higher_awscale_for_human_in_loop_approaches`
   - `test_llm_fallback` → `test_uses_fallback_model_when_preferred_unavailable`

5. **Processing Tests**
   - `test_normalize` → `test_standardizes_author_names_to_consistent_format`
   - `test_batch` → `test_processes_papers_in_configured_batch_sizes`
   - `test_parallel` → `test_maintains_data_integrity_during_parallel_processing`

## Test Structure Template

```python
class TestComponentBehavior:
    """Test [Component] behaves correctly in various scenarios."""

    def test_handles_normal_operations_successfully(self):
        """
        GIVEN a valid configuration and normal conditions
        WHEN performing standard operations
        THEN expected results are produced
        """
        # Arrange
        component = Component(valid_config)
        input_data = create_valid_input()

        # Act
        result = component.process(input_data)

        # Assert
        assert result.status == "success"
        assert result.data == expected_output

    def test_recovers_from_transient_failures(self):
        """
        GIVEN a component experiencing intermittent failures
        WHEN retrying operations
        THEN eventually succeeds with valid output
        """
        # Implementation

    def test_validates_input_before_processing(self):
        """
        GIVEN invalid input data
        WHEN attempting to process
        THEN rejects with appropriate error
        """
        # Implementation
```

## Naming Conventions

1. **Start with verb**: test_*does_something*
2. **Describe behavior**: Not implementation
3. **Use underscores**: For readability
4. **Be specific**: Avoid generic terms
5. **Include context**: When behavior changes

## Common Patterns

### State Verification
- `test_maintains_[state]_after_[action]`
- `test_preserves_[property]_during_[operation]`

### Error Handling
- `test_handles_[error_type]_gracefully`
- `test_recovers_from_[failure_scenario]`

### Integration
- `test_integrates_with_[component]_correctly`
- `test_coordinates_with_[service]_for_[operation]`

### Performance
- `test_completes_[operation]_within_time_limit`
- `test_handles_[volume]_without_degradation`
