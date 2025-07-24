# CLI Architecture Redesign Report

**Date**: 2025-01-24
**Author**: Claude Code SuperClaude Framework
**Status**: Architecture Analysis Complete
**Priority**: High - Addresses Core System Design Issues

## Executive Summary

Analysis of the current dual-CLI architecture (`run.py` and `production_harvest.py`) reveals significant architectural over-engineering. The functionality overlap and maintenance overhead outweigh the persona separation benefits. This report recommends immediate unification with a path toward modern API-first architecture.

## Problem Analysis

### Current Issues
- **Command Duplication**: 3/11 overlapping commands with different implementations
- **Mental Model Split**: Users must learn two different interfaces
- **Maintenance Overhead**: Two CLI systems, two test suites, two documentation sets
- **Feature Fragmentation**: Production features locked in specialized interface

### Architecture Analysis

**File Comparison:**
- `run.py`: 719 lines, 11 commands (complete pipeline)
- `production_harvest.py`: 372 lines, 4 commands (harvesting only)
- `ProductionHarvester`: 572 lines (inherits from `SearchHarvester` 274 lines)

**Code Relationship:**
```python
# production_harvester.py:21
class ProductionHarvester(SearchHarvester):  # ← INHERITS from SearchHarvester
```

**Key Finding**: Both CLIs use identical underlying harvester engines with different configuration profiles.

## Design Alternative Options

### Option 1: Single Unified CLI with Execution Profiles ⭐⭐⭐

```yaml
# Enhanced config.yaml - Profile-based execution
profiles:
  research:
    description: "Academic research workflows (100-1000 papers)"
    harvesting:
      max_papers: 1000
      batch_size: 100
      monitoring: basic
      resume: false

  production:
    description: "Enterprise-scale harvesting (10K+ papers)"
    harvesting:
      max_papers: 50000
      batch_size: 1000
      monitoring: advanced
      resume: true
      session_management: true
```

**Usage:**
```bash
litreview --profile research harvest --query preset1
litreview --profile production harvest --max-papers 50000 --monitor
litreview status  # Shows appropriate status for active profile
```

### Option 2: Configuration-Driven Execution ⭐⭐⭐⭐

```yaml
# workflow.yaml - Declarative pipeline definition
name: "LLM Wargaming Literature Review"
version: "1.0"

stages:
  harvest:
    driver: production_harvester
    config:
      max_papers: 50000
      sources: [arxiv, semantic_scholar]
      monitoring: true

  screen:
    driver: screening_ui
    depends_on: harvest

  extract:
    driver: llm_extractor
    depends_on: screen
```

**Usage:**
```bash
litreview run workflow.yaml
litreview run --stage harvest workflow.yaml
litreview status workflow.yaml
```

### Option 3: API-First with Multiple Frontends ⭐⭐⭐⭐⭐ **[RECOMMENDED]**

```python
# Core API Service
class LiteratureReviewAPI:
    def start_harvest(self, config: HarvestConfig) -> Session
    def get_harvest_status(self, session_id: str) -> Status
    def run_pipeline(self, workflow: Workflow) -> Result
```

**Multiple Interface Options:**
- **Web UI**: Streamlit/FastAPI dashboard for interactive use
- **CLI**: Thin wrapper over API for scripting
- **Jupyter**: Notebook interface for exploratory analysis
- **REST API**: For integration with other systems

### Option 4: Modern Task Runner Approach ⭐⭐⭐⭐

```python
# taskfile.py - Python-based task definitions
from invoke import task

@task
def harvest_research(c, query="preset1", max_results=1000):
    """Research-scale harvesting"""

@task
def harvest_production(c, max_papers=50000, monitor=False):
    """Production-scale harvesting"""
```

### Option 5: Enhanced Single CLI (Minimal Change) ⭐⭐⭐

**Immediate unification preserving existing functionality:**

```python
@cli.command()
@click.option('--mode', type=click.Choice(['research', 'production']), default='research')
@click.option('--max-results', default=None)  # Auto-set based on mode
def harvest(mode, max_results):
    """Unified harvest command with automatic mode selection"""
    if mode == 'production' or (max_results and max_results > 5000):
        # Use ProductionHarvester
    else:
        # Use SearchHarvester
```

## Recommended Architecture

### Primary Recommendation: API-First + Configuration-Driven

**Core Design Principles:**
1. **Configuration Over Code**: Workflows defined declaratively
2. **API-First**: Core logic exposed through clean interfaces
3. **Multiple Frontends**: CLI, Web UI, Jupyter for different users
4. **Zero Duplication**: Single implementation with adaptive behavior

```python
# New Architecture: litreview/
├── core/                  # Core business logic
│   ├── api.py            # Clean API interfaces
│   ├── harvesters/       # Unified harvesting engine
│   ├── processors/       # Pipeline stages
│   └── workflows/        # Workflow execution engine
├── interfaces/           # Multiple user interfaces
│   ├── cli.py           # Thin CLI wrapper
│   ├── web/             # Streamlit/FastAPI web UI
│   └── jupyter/         # Notebook integration
└── configs/             # Workflow definitions
    ├── research.yaml    # Research profile
    ├── production.yaml  # Production profile
    └── custom/          # User-defined workflows
```

### Implementation Specification

```python
# core/api.py - Clean API Layer
class LitReviewAPI:
    def __init__(self, config: Config):
        self.harvester = UnifiedHarvester(config)
        self.processor = PipelineProcessor(config)

    def harvest(self, workflow: HarvestWorkflow) -> Session:
        """Execute harvesting with automatic mode detection"""
        if workflow.scale >= 5000:
            return self.harvester.production_harvest(workflow)
        else:
            return self.harvester.research_harvest(workflow)

    def run_pipeline(self, pipeline: Pipeline) -> Result:
        """Execute complete research pipeline"""

    def get_status(self, session_id: str = None) -> Status:
        """Get session/pipeline status"""

# interfaces/cli.py - Minimal CLI Wrapper
@click.command()
@click.option('--workflow', type=click.Path(), default='configs/research.yaml')
def run(workflow):
    """Execute workflow from configuration file"""
    api = LitReviewAPI.from_config(workflow)
    result = api.run_pipeline(workflow)
    display_results(result)
```

## Tradeoff Analysis Matrix

| Option | UX | Maintainability | Extensibility | Performance | Migration | Modern |
|--------|----|----|----|----|----|----|
| **1. Unified CLI** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **2. Config-Driven** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **3. API-First** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **4. Task Runner** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **5. Enhanced CLI** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## Migration Strategy

### Phase 1: Quick Win (2-4 weeks)
- Merge `production_harvest.py` commands into `run.py` with `--production` flag
- Add automatic mode detection based on `--max-results` threshold
- Deprecate `production_harvest.py` with migration warnings

**Immediate Implementation:**
```python
@cli.command()
@click.option('--max-results', default=100)
@click.option('--production', is_flag=True, help='Use production-scale harvesting')
@click.option('--monitor', is_flag=True, help='Enable monitoring (auto-enabled for production)')
@click.option('--resume', help='Resume session (production only)')
def harvest(max_results, production, monitor, resume):
    """Unified harvest command - automatically selects appropriate harvester"""

    # Auto-detect production mode
    is_production = production or max_results > 5000 or resume

    if is_production:
        harvester = ProductionHarvester(config)
        monitor = monitor or max_results > 1000  # Auto-enable monitoring
    else:
        harvester = SearchHarvester(config)

    # Execute with appropriate harvester
    if is_production:
        df = harvester.search_production_scale(...)
    else:
        df = harvester.search_all(...)
```

### Phase 2: API Extraction (4-6 weeks)
- Extract core logic into `LitReviewAPI` class
- Refactor CLI as thin wrapper over API
- Add configuration profiles (research.yaml, production.yaml)

### Phase 3: Alternative Frontends (Optional)
- Add Streamlit web UI for interactive users
- Add Jupyter notebook integration
- Add REST API endpoints for integrations

## Benefits Analysis

### Immediate Benefits (Phase 1)
- **Unified Mental Model**: Single interface to learn
- **Reduced Documentation**: One CLI reference instead of two
- **Simplified Testing**: Single test suite for CLI functionality
- **Automatic Mode Selection**: Users don't need to choose tools

### Long-term Benefits (Phases 2-3)
- **Modern Architecture**: API-first enables multiple frontends
- **Enhanced Extensibility**: Easy to add new workflows and interfaces
- **Cloud Deployment**: API can be deployed as microservice
- **Integration Ready**: REST API for external systems

### Performance Characteristics Preserved
- **Rate Limit Optimization**: Production mode maintains aggressive limits
- **Session Management**: Resume/checkpoint functionality preserved
- **Monitoring**: Rich monitoring interface for production workflows
- **Scalability**: Same underlying performance optimizations

## Risk Assessment

### Low Risk (Phase 1)
- **Backward Compatibility**: Existing workflows continue working
- **Gradual Migration**: Users can migrate at their own pace
- **Feature Preservation**: All current functionality maintained

### Medium Risk (Phases 2-3)
- **API Design**: Requires careful interface design
- **Configuration Migration**: Users need to adapt to new config format
- **Testing Complexity**: Multiple interfaces to validate

### Mitigation Strategies
- **Comprehensive Testing**: Maintain existing test coverage
- **Migration Tools**: Scripts to convert existing configurations
- **Documentation**: Clear migration guides and examples
- **Deprecation Warnings**: Grace period for old interfaces

## Success Metrics

### Phase 1 Metrics
- **User Adoption**: >80% users successfully using unified CLI
- **Issue Reduction**: 50% reduction in CLI-related support issues
- **Documentation Efficiency**: Single CLI guide vs. dual documentation

### Phase 2-3 Metrics
- **Interface Usage**: Web UI adoption rate for interactive users
- **API Usage**: External integrations using REST API
- **Development Velocity**: Faster feature development due to unified codebase

## Conclusion

The current dual-CLI approach represents **architectural over-engineering** without proportional benefits. A unified interface with intelligent mode detection provides:

- **Better UX**: One mental model, context-sensitive behavior
- **Reduced Complexity**: Single codebase, test suite, documentation
- **Improved Maintainability**: No feature duplication or synchronization issues
- **Enhanced Discoverability**: All features accessible from one interface

**Immediate Action**: Start with Phase 1 (unified CLI) to realize quick wins, then evolve toward API-first architecture for maximum flexibility and modern design alignment.

The user's instinct is correct - the functionality similarity doesn't justify separate tools. Modern software architecture favors unified interfaces with adaptive behavior over specialized tools for each use case.
