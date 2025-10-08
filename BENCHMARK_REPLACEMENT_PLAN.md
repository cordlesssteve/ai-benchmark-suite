# Benchmark Replacement Implementation Plan

**Status:** IN PROGRESS
**Started:** 2025-10-08
**Goal:** Replace contaminated benchmarks with contamination-resistant alternatives

---

## Progress Summary

### ✅ Completed
- [x] Research contamination-resistant benchmarks
- [x] Clone LiveCodeBench repository
- [x] Clone BigCodeBench repository
- [x] Clone SWE-bench Live repository
- [x] Analyze current suite structure

### 🔄 In Progress
- [ ] Install LiveCodeBench dependencies (timeout issue - investigating)
- [ ] Create adapter infrastructure

### ⏳ Pending
- [ ] Create LiveCodeBench adapter
- [ ] Create BigCodeBench adapter
- [ ] Create SWE-bench Live adapter
- [ ] Update suite configuration
- [ ] Run validation tests
- [ ] Update documentation

---

## Architecture Plan

### New Directory Structure
```
harnesses/
├── bigcode-evaluation-harness/  (existing)
├── lm-evaluation-harness/        (existing)
├── livecodebench/                (✅ cloned)
├── bigcodebench/                 (✅ cloned)
└── swebench-live/                (✅ cloned)
```

### Adapter Structure
```
src/model_interfaces/
├── livecodebench_adapter.py      (new)
├── bigcodebench_adapter.py       (new)
├── swebench_live_adapter.py      (new)
└── ollama_interface.py           (will integrate with adapters)
```

---

## Implementation Details

### 1. LiveCodeBench Integration

**Contamination Protection:** Temporal filtering
**Difficulty:** 🟢 LOW

**Key Features:**
- 713+ problems (v4), 880 (v5), 1055 (v6)
- Release versions: May 2023 - Apr 2025
- Temporal evaluation (filter by release date)
- Multiple scenarios: code_generation, execution, test_prediction

**Integration Steps:**
1. ✅ Clone repository
2. 🔄 Install dependencies (`uv pip install -e .`)
3. Create `LiveCodeBenchAdapter` class
4. Implement temporal filtering (problems after model cutoff)
5. Integrate with unified runner
6. Add to suite_definitions.yaml

**Sample Usage:**
```python
from lcb_runner import LiveCodeBenchAdapter

adapter = LiveCodeBenchAdapter(
    release_version="release_v6",  # Latest
    scenario="codegeneration",
    cutoff_date="2024-09-30"  # Model training cutoff
)

results = adapter.evaluate(model="ollama/qwen2.5-coder:3b")
```

---

### 2. BigCodeBench Integration

**Contamination Protection:** Enhanced quality (better than HumanEval)
**Difficulty:** 🟢 LOW-MEDIUM

**Key Features:**
- 163 models already evaluated
- Complex function calls
- EvalPlus methodology
- Apache 2.0 license

**Integration Steps:**
1. ✅ Clone repository
2. Install dependencies
3. Create `BigCodeBenchAdapter` class
4. Integrate with unified runner
5. Add to suite_definitions.yaml

---

### 3. SWE-bench Live Integration

**Contamination Protection:** Temporal (monthly updates)
**Difficulty:** 🟡 MEDIUM

**Key Features:**
- 1,565 tasks across 164 repositories
- 50 new issues per month
- Repository-level evaluation
- Docker-based execution (we already have this infrastructure)

**Integration Steps:**
1. ✅ Clone repository
2. Review Docker requirements (RepoLaunch)
3. Adapt existing Docker infrastructure
4. Create `SWEBenchLiveAdapter` class
5. Implement repository context handling
6. Integrate with unified runner
7. Add to suite_definitions.yaml

---

## Updated Suite Definitions

### New suite_definitions.yaml Structure

```yaml
suites:
  contamination_resistant_quick:
    description: "Quick contamination-resistant evaluation (~10-15 min)"
    tasks:
      - livecodebench_v6_recent
      - bigcodebench_sample
    settings:
      limit: 20
      n_samples: 1
      temperature: 0.2
      temporal_filter: true
    estimated_time: "10-15 minutes"

  contamination_resistant_standard:
    description: "Standard contamination-resistant suite (~45-90 min)"
    tasks:
      - livecodebench_v6
      - bigcodebench
      - swebench_live_latest
    settings:
      limit: 50
      n_samples: 5
      temperature: 0.2
      temporal_filter: true
    estimated_time: "45-90 minutes"

  contamination_resistant_comprehensive:
    description: "Comprehensive contamination-resistant evaluation (~3-6 hours)"
    tasks:
      - livecodebench_v6_full
      - bigcodebench_full
      - swebench_live_month
    settings:
      limit: null
      n_samples: 10
      temperature: 0.2
      temporal_filter: true
      statistical_analysis: true
    estimated_time: "3-6 hours"

categories:
  contamination_resistant_code:
    - livecodebench_v6
    - bigcodebench
    - swebench_live

  legacy_contaminated:  # Keep for comparison
    - humaneval
    - mbpp
```

---

## Temporal Filtering Implementation

### Model Training Cutoffs

```python
MODEL_CUTOFFS = {
    # Add known cutoffs
    "gpt-4-turbo": "2023-12-31",
    "claude-3-opus": "2023-08-31",
    "qwen2.5-coder:3b": "2024-09-30",  # Example
    "deepseek-coder:6.7b": "2024-08-31",  # Example
    # Default to conservative cutoff
    "default": "2024-01-01"
}
```

### Filter Function

```python
def get_clean_problems(benchmark, model_name, cutoff_date=None):
    """
    Filter problems released AFTER model training cutoff.

    Args:
        benchmark: Benchmark dataset with release_date metadata
        model_name: Name of model being evaluated
        cutoff_date: Override cutoff (default: use MODEL_CUTOFFS)

    Returns:
        List of problems released after cutoff
    """
    if cutoff_date is None:
        cutoff = MODEL_CUTOFFS.get(model_name, MODEL_CUTOFFS["default"])
    else:
        cutoff = cutoff_date

    clean_problems = [
        problem for problem in benchmark.problems
        if problem.release_date > cutoff
    ]

    return clean_problems, cutoff
```

---

## Contamination Disclosure

### Report Format

All benchmark results will include contamination status:

```json
{
  "model": "qwen2.5-coder:3b",
  "benchmark": "LiveCodeBench_v6",
  "contamination_status": {
    "protection_strategy": "temporal",
    "model_training_cutoff": "2024-09-30",
    "benchmark_release_range": "2023-05-01 to 2025-04-30",
    "total_problems": 1055,
    "clean_problems": 215,  // After cutoff
    "contaminated_problems": 840,  // Before cutoff
    "evaluated_on": "clean_subset",
    "contamination_risk": "LOW"
  },
  "results": {
    "clean_subset": {
      "pass@1": 0.42,
      "pass@10": 0.68,
      "problems_solved": 90,
      "problems_attempted": 215
    },
    "full_set": {
      "pass@1": 0.67,  // Likely inflated by contamination
      "note": "For comparison only - may reflect memorization"
    }
  }
}
```

---

## Next Steps

1. **Resolve LiveCodeBench installation timeout**
   - Try installing in background
   - Or use pip instead of uv

2. **Create adapter classes**
   - Start with LiveCodeBench (simplest)
   - Then BigCodeBench
   - Finally SWE-bench Live (most complex)

3. **Update configuration files**
   - suite_definitions.yaml
   - harness_mappings.yaml
   - Add temporal filtering config

4. **Testing & Validation**
   - Test each adapter individually
   - Run sample evaluations
   - Verify temporal filtering works

5. **Documentation**
   - Update README with new benchmarks
   - Add contamination disclosure templates
   - Create migration guide from old to new benchmarks

---

## Timeline Estimate

- **Day 1** (today): Setup + LiveCodeBench adapter (✅ 50% complete)
- **Day 2**: BigCodeBench adapter + testing
- **Day 3**: SWE-bench Live adapter (more complex)
- **Day 4**: Integration, testing, documentation
- **Day 5**: Validation + final testing

**Total:** 4-5 days for complete replacement

---

## Success Criteria

- [ ] All three new benchmarks integrated
- [ ] Temporal filtering implemented and tested
- [ ] Contamination status in all reports
- [ ] Documentation updated
- [ ] Old benchmarks marked as deprecated
- [ ] Sample evaluation runs successfully
- [ ] Results clearly distinguish clean vs contaminated subsets

---

## Related Documents

- Research: `~/docs/AI_BENCHMARK_CONTAMINATION_RESEARCH_2025.md`
- Analysis: `docs/BENCHMARK_CONTAMINATION_ANALYSIS.md`
- Survey: `docs/CONTAMINATION_RESISTANT_BENCHMARKS_SURVEY.md`
- Summary: `~/docs/BENCHMARK_CONTAMINATION_FINDINGS_SUMMARY.md`
