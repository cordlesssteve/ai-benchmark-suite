# Contamination-Resistant Benchmarks - Quick Start Guide

**Status:** READY FOR TESTING
**Date:** 2025-10-08
**Version:** 1.0

---

## Overview

This AI Benchmark Suite now includes three contamination-resistant benchmarks to replace the contaminated HumanEval and MBPP benchmarks:

1. **LiveCodeBench** ⭐⭐⭐ - Temporal protection, monthly updates
2. **BigCodeBench** ⭐⭐ - Enhanced quality (better than HumanEval)
3. **SWE-bench Live** ⭐⭐⭐ - Repository-level, 50 new issues/month

---

## Quick Start

### 1. Setup (One-Time)

```bash
# LiveCodeBench
cd harnesses/livecodebench
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# BigCodeBench
cd ../bigcodebench
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# SWE-bench Live
cd ../swebench-live
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Run Evaluation

#### Option A: Using Adapters Directly

```bash
# LiveCodeBench
python src/model_interfaces/livecodebench_adapter.py \
    --model "qwen2.5-coder:3b" \
    --release_version "release_v6" \
    --n_samples 10

# BigCodeBench
python src/model_interfaces/bigcodebench_adapter.py \
    --model "qwen2.5-coder:3b" \
    --subset "full" \
    --backend "openai"

# SWE-bench Live
python src/model_interfaces/swebench_live_adapter.py \
    --model "qwen2.5-coder:3b" \
    --max_instances 10
```

#### Option B: Using Suite Runner (Recommended)

```bash
# Quick contamination-resistant evaluation (~10-20 min)
python src/unified_runner.py \
    --suite contamination_resistant_quick \
    --model "qwen2.5-coder:3b"

# Standard evaluation (~60-120 min)
python src/unified_runner.py \
    --suite contamination_resistant_standard \
    --model "qwen2.5-coder:3b"

# Comprehensive evaluation (~4-8 hours)
python src/unified_runner.py \
    --suite contamination_resistant_comprehensive \
    --model "qwen2.5-coder:3b"
```

---

## Available Suites

### Contamination-Resistant (✅ Recommended)

| Suite | Time | Tasks | Contamination Risk |
|-------|------|-------|-------------------|
| `contamination_resistant_quick` | 10-20 min | LiveCodeBench recent + BigCodeBench sample | 🟢 LOW |
| `contamination_resistant_standard` | 60-120 min | LiveCodeBench v6 + BigCodeBench + SWE-bench Live | 🟢 LOW |
| `contamination_resistant_comprehensive` | 4-8 hours | All benchmarks, full evaluation | 🟢 LOW |
| `contamination_resistant_code_only` | 2-4 hours | LiveCodeBench + BigCodeBench only | 🟢 LOW |
| `swebench_live_evaluation` | 60-180 min | Repository-level tasks only | 🟢 LOW |

### Legacy (⚠️ Use with Caution)

| Suite | Time | Contamination Risk | Notes |
|-------|------|-------------------|-------|
| `quick` | 5-10 min | 🔴 HIGH | Uses HumanEval (contaminated) |
| `standard` | 30-60 min | 🔴 HIGH | Uses HumanEval + MBPP (65% contaminated) |
| `code_only` | 2-4 hours | 🔴 HIGH | Legacy contaminated benchmarks |

---

## Benchmark Details

### LiveCodeBench

**Contamination Protection:** ⏰ Temporal filtering

**Features:**
- 1055 problems (v6) from LeetCode, AtCoder, CodeForces
- Release dates: May 2023 - Apr 2025
- Automatic filtering by model training cutoff
- Multiple versions (v1-v6)

**Model Cutoffs Configured:**
```python
"qwen2.5-coder:3b": "2024-09-30"
"deepseek-coder:6.7b": "2024-08-31"
"codellama:7b": "2024-01-31"
"phi3.5:latest": "2024-08-31"
# ... and more
```

**Result Format:**
```json
{
  "total_problems": 1055,
  "clean_problems": 215,
  "contaminated_problems": 840,
  "training_cutoff": "2024-09-30",
  "pass@1": 0.42,
  "contamination_status": "LOW"
}
```

---

### BigCodeBench

**Contamination Protection:** ✨ Enhanced quality (better than HumanEval)

**Features:**
- Complex function calls
- Realistic code patterns
- EvalPlus methodology
- 163 models evaluated

**Contamination Warning:** 🟡 MODERATE - Still a public benchmark

**Use Case:** Cross-validation with LiveCodeBench

---

### SWE-bench Live

**Contamination Protection:** ⏰ Temporal (monthly updates)

**Features:**
- 1,565 repository-level tasks
- 50 new GitHub issues per month
- Real-world software engineering
- Docker-based execution

**Complexity:** More complex than function-level benchmarks but more realistic

---

## Contamination Analysis

All benchmark results include contamination metadata:

```json
{
  "contamination_status": {
    "protection_strategy": "temporal",
    "model_training_cutoff": "2024-09-30",
    "total_problems": 1055,
    "clean_problems": 215,
    "contaminated_problems": 840,
    "contamination_risk": "LOW"
  },
  "results": {
    "clean_subset": {
      "pass@1": 0.42
    },
    "full_set": {
      "pass@1": 0.67,
      "note": "May reflect memorization"
    }
  }
}
```

---

## Migrating from Legacy Benchmarks

### Old → New Mapping

| Legacy (Contaminated) | Replacement (Clean) | Notes |
|----------------------|---------------------|-------|
| `humaneval` | `livecodebench_v6` | Direct replacement |
| `mbpp` | `livecodebench_v6` | Direct replacement |
| `quick` suite | `contamination_resistant_quick` | 2x longer but valid |
| `standard` suite | `contamination_resistant_standard` | 2x longer but valid |
| `code_only` suite | `contamination_resistant_code_only` | Clean alternative |

### Deprecation Timeline

- **Now:** Legacy benchmarks marked deprecated, warnings added
- **Week 1:** Contamination-resistant benchmarks fully tested
- **Week 2:** Documentation updated, examples migrated
- **Week 3:** Legacy benchmarks moved to `legacy/` directory
- **Month 1:** Legacy benchmarks removed from default suites

---

## Testing & Validation

### Quick Test

```bash
# Test LiveCodeBench (fastest)
python src/model_interfaces/livecodebench_adapter.py \
    --model "qwen2.5-coder:3b" \
    --release_version "release_v6" \
    --n_samples 1 \
    --limit 5

# Should output:
# - Model: qwen2.5-coder:3b
# - Total problems: 5
# - Clean problems: X (depends on cutoff)
# - Contamination status: LOW
```

### Validation Checklist

- [ ] LiveCodeBench adapter runs without errors
- [ ] Temporal filtering works (clean vs contaminated separation)
- [ ] BigCodeBench adapter runs without errors
- [ ] SWE-bench Live adapter runs without errors
- [ ] Results include contamination metadata
- [ ] Suite runner integrates all three benchmarks
- [ ] Documentation updated

---

## Troubleshooting

### "LiveCodeBench not found"
```bash
cd harnesses/livecodebench
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### "BigCodeBench venv not set up"
```bash
cd harnesses/bigcodebench
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### "No results found"
Make sure to run inference before evaluation:
```python
adapter.run_inference(model_name)  # First
adapter.run_evaluation(model_name)  # Then
# Or use adapter.evaluate() for both
```

### "Model cutoff not found"
Add your model to `MODEL_CUTOFFS` in the adapter:
```python
MODEL_CUTOFFS = {
    "your-model-name": "YYYY-MM-DD",
    ...
}
```

---

## File Structure

```
ai-benchmark-suite/
├── harnesses/
│   ├── livecodebench/           # LiveCodeBench harness
│   ├── bigcodebench/            # BigCodeBench harness
│   ├── swebench-live/           # SWE-bench Live harness
│   ├── bigcode-evaluation-harness/  # Legacy (HumanEval)
│   └── lm-evaluation-harness/   # Language tasks
│
├── src/model_interfaces/
│   ├── livecodebench_adapter.py     # ✨ NEW
│   ├── bigcodebench_adapter.py      # ✨ NEW
│   ├── swebench_live_adapter.py     # ✨ NEW
│   └── ...
│
├── config/
│   ├── suite_definitions.yaml   # ✨ UPDATED (new suites)
│   └── ...
│
└── docs/
    ├── BENCHMARK_CONTAMINATION_ANALYSIS.md
    ├── CONTAMINATION_RESISTANT_BENCHMARKS_SURVEY.md
    └── ...
```

---

## Related Documentation

- **Research:** `~/docs/AI_BENCHMARK_CONTAMINATION_RESEARCH_2025.md`
- **Analysis:** `docs/BENCHMARK_CONTAMINATION_ANALYSIS.md`
- **Survey:** `docs/CONTAMINATION_RESISTANT_BENCHMARKS_SURVEY.md`
- **Summary:** `~/docs/BENCHMARK_CONTAMINATION_FINDINGS_SUMMARY.md`
- **Implementation Plan:** `BENCHMARK_REPLACEMENT_PLAN.md`

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review implementation plan: `BENCHMARK_REPLACEMENT_PLAN.md`
3. Check adapter source code for detailed documentation
4. Refer to original benchmark documentation:
   - LiveCodeBench: https://github.com/LiveCodeBench/LiveCodeBench
   - BigCodeBench: https://github.com/bigcode-project/bigcodebench
   - SWE-bench Live: https://github.com/microsoft/SWE-bench-Live

---

## License

- LiveCodeBench: Check repository
- BigCodeBench: Apache 2.0
- SWE-bench Live: MIT
- Our adapters: Same as main project

---

**Last Updated:** 2025-10-08
**Version:** 1.0
**Status:** Ready for testing
