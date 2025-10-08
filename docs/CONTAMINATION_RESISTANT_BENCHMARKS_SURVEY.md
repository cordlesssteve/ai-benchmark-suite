# Contamination-Resistant Benchmarks Survey (2025)

**Document Status:** ACTIVE REFERENCE
**Last Updated:** 2025-10-08
**Purpose:** Identify open-source, contamination-resistant benchmarks for integration

---

## Executive Summary

This document catalogs contamination-resistant AI benchmarks that are:
1. **Open source** and freely accessible
2. **Contamination-resistant** by design (temporal, private sets, legal barriers, etc.)
3. **Suitable for integration** into our benchmark suite

✓ **VERIFIED:** Multiple high-quality options available for immediate use
🎯 **RECOMMENDATION:** LiveCodeBench + SWE-bench Live for coding tasks

---

## Contamination-Resistant Benchmarks

### 1. LiveCodeBench ⭐ **RECOMMENDED**

**Contamination Strategy:** Temporal control (problems released after model training)

**Access:**
- ✅ **Fully Open Source**
- GitHub: https://github.com/LiveCodeBench/LiveCodeBench
- HuggingFace: https://huggingface.co/datasets/livecodebench/code_generation_lite
- Paper: arXiv:2403.07974

**Key Features:**
- Continuously collects new problems from LeetCode, AtCoder, CodeForces
- Release date annotations for temporal evaluation
- Multiple versions: v1-v4 (May 2023 - Sep 2024, 713 problems in v4)
- Holistic evaluation: code generation, self-repair, execution, test prediction

**Contamination Protection:**
- Problems annotated with release dates
- Can evaluate models only on problems released AFTER training cutoff
- Monthly updates with fresh problems
- Transparent versioning

**Integration Difficulty:** 🟢 **LOW**
- Standard format (JSON)
- Python evaluation harness
- Well-documented API

**License:** Open source (check GitHub for specific license)

**Use Case for Our Suite:**
- Replace HumanEval for code generation
- Add temporal validation (evaluate on post-cutoff problems)
- Monthly benchmark refresh

---

### 2. SWE-bench Live ⭐ **RECOMMENDED**

**Contamination Strategy:** Temporal control (monthly updates with new GitHub issues)

**Access:**
- ✅ **Fully Open Source**
- GitHub: https://github.com/microsoft/SWE-bench-Live
- Leaderboard: https://swe-bench-live.github.io/
- Official Org: https://github.com/SWE-bench-Live/

**Key Features:**
- 1,565 task instances across 164 repositories (as of 2025)
- Monthly updates: 50 new verified issues added per month
- Real-world GitHub issues (more realistic than isolated functions)
- Includes RepoLaunch for automated Docker execution environments

**Contamination Protection:**
- Issues created AFTER model training cutoffs
- Monthly refresh ensures continuous supply of clean problems
- Frozen verified/lite splits for fair comparisons
- Public submission process (transparency)

**Integration Difficulty:** 🟡 **MEDIUM**
- Requires repository context (larger inputs)
- Docker-based evaluation (already have this)
- More complex than function completion
- May need GitHub API integration

**License:** MIT (inherited from SWE-bench base)

**Use Case for Our Suite:**
- More realistic software engineering tasks
- Repository-level code understanding
- Agent-based evaluation (not just completion)

---

### 3. SWE-bench Pro (Partial)

**Contamination Strategy:** Legal barriers (GPL copyleft licensing)

**Access:**
- ⚠️ **Partially Open Source**
- Public subset: https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro
- GitHub: https://github.com/scaleapi/SWE-bench_Pro-os
- Paper: arXiv:2509.16941

**Key Features:**
- 1,865 problems from 41 actively maintained repositories
- Three subsets: Public OSS (GPL), Held-out OSS (GPL), Commercial (private)
- Focus on long-horizon software engineering tasks

**Contamination Protection:**
- GPL/copyleft licenses create legal deterrent
- Cannot be legally included in training data
- Held-out and commercial sets remain private

**Integration Difficulty:** 🟡 **MEDIUM**
- Public subset available via HuggingFace
- Similar complexity to SWE-bench Live
- Some portions unavailable (held-out, commercial)

**License:**
- Base SWE-bench: MIT
- Pro public subset: Uses GPL projects (copyleft)
- Held-out/commercial: Not accessible

**Use Case for Our Suite:**
- Legal contamination protection
- Long-horizon task evaluation
- Public subset only (held-out unavailable)

**Limitation:** 🔴 Only partial access (public subset only)

---

### 4. ARC-AGI ⭐ **RECOMMENDED** (Non-Code)

**Contamination Strategy:** Private test sets + abstract reasoning

**Access:**
- ✅ **Fully Open Source**
- ARC-AGI-1: https://github.com/fchollet/ARC-AGI
- ARC-AGI-2: https://github.com/arcprize/ARC-AGI-2
- Interactive: https://arcprize.org/play

**Key Features:**
- ARC-AGI-1: 400 training tasks
- ARC-AGI-2: 1,000 public training + 120 public evaluation tasks
- Private test sets (100 private + 100 semi-private) never released
- Abstract visual reasoning (not code-specific)

**Contamination Protection:**
- Private test sets never disclosed
- Abstract reasoning harder to memorize
- Not language/code specific (visual patterns)

**Integration Difficulty:** 🔴 **HIGH** (not code-focused)
- JSON format for tasks
- Requires different evaluation approach
- Not applicable to code generation

**License:** CC-BY-SA (from Papers With Code), open source

**Use Case for Our Suite:**
- ❌ Not suitable (non-code benchmark)
- Could be used for general reasoning evaluation
- Different domain from our focus

**Note:** Included for completeness, but not recommended for code benchmarking

---

### 5. METR RE-Bench

**Contamination Strategy:** Complex, realistic tasks + private components

**Access:**
- ✅ **Fully Open Source**
- GitHub: https://github.com/METR/RE-Bench
- Paper: arXiv:2411.15114
- Report: https://metr.org/AI_R_D_Evaluation_Report.pdf

**Key Features:**
- 7 challenging ML research engineering environments
- 71 8-hour attempts by 61 human experts (baseline data)
- Tasks: GPU kernel optimization, scaling laws, ML research challenges
- Open-sourced evaluation environments and agent trajectories

**Contamination Protection:**
- Complex, open-ended tasks harder to memorize
- Password-protected files (password: resident-peacock-motif-grading)
- Focus on process/reasoning over final answers
- Human expert baselines

**Integration Difficulty:** 🔴 **HIGH**
- Complex ML research tasks (8-hour time limits)
- Requires specialized infrastructure
- More suitable for agent evaluation than model benchmarking
- Not pure code generation

**License:** MIT

**Use Case for Our Suite:**
- ❌ Too complex for standard benchmarking
- Better suited for agent/research capabilities
- Could inspire design of realistic tasks

**Note:** High quality but likely overkill for our use case

---

### 6. BigCodeBench

**Contamination Strategy:** Enhanced testing (EvalPlus methodology)

**Access:**
- ✅ **Fully Open Source**
- GitHub: https://github.com/bigcode-project/bigcodebench
- HuggingFace: https://huggingface.co/datasets/bigcode/bigcodebench
- Leaderboard: https://bigcode-bench.github.io/
- Paper: ICLR'25 (Accepted)

**Key Features:**
- "Next generation of HumanEval" with diverse function calls
- 163 models already evaluated (trusted by Zhipu AI, Alibaba Qwen, DeepSeek, AWS)
- Enhanced test cases (EvalPlus methodology)
- Focus on complex instructions and realistic code patterns

**Contamination Protection:**
- ⚠️ **LIMITED** - Still based on publicly released problems
- Better than HumanEval due to:
  - More comprehensive test cases
  - Complex, realistic scenarios
  - Harder to memorize due to complexity

**Integration Difficulty:** 🟢 **LOW-MEDIUM**
- Apache 2.0 license (permissive)
- Standard evaluation harness
- Part of bigcode-project ecosystem
- Latest release: v0.2.2.dev2 (Jan 2025)

**License:** Apache 2.0

**Use Case for Our Suite:**
- Replacement for HumanEval (higher quality)
- Complex function calling evaluation
- Still has contamination risk (public dataset since release)

**Contamination Status:**
- 🟡 **MODERATE RISK** - Public benchmark, but newer and more complex than HumanEval
- Not as contamination-resistant as LiveCodeBench or SWE-bench Live

---

### 7. MBPP+ / MBPP-Sanitized

**Contamination Strategy:** Quality control (sanitized subset of MBPP)

**Access:**
- ✅ **Fully Open Source**
- Original MBPP: https://github.com/google-research/google-research/tree/master/mbpp
- Sanitized: https://github.com/google-research/google-research/blob/master/mbpp/sanitized-mbpp.json
- HuggingFace: https://huggingface.co/datasets/Muennighoff/mbpp
- EvalPlus Leaderboard: https://evalplus.github.io/leaderboard.html

**Key Features:**
- MBPP-Sanitized: 427 hand-verified problems (subset of ~1000 original)
- MBPP+: 399 tasks (further quality-controlled subset of sanitized)
- Natural language descriptions without code signatures
- Test-driven development format

**Contamination Protection:**
- 🔴 **HIGH CONTAMINATION RISK - CONFIRMED**
- **65.4% of test instances** obtained from open-access websites
- String-matching decontamination attempted but traces remain
- Known to be in training data for major models

**Research Findings (2025):**
- "Powerful models may 'cheat' via memorization rather than reasoning"
- Contamination persists even after decontamination attempts
- Poor alignment between instructions and test cases in original

**Integration Difficulty:** 🟢 **LOW**
- Standard format (JSON)
- Available on HuggingFace
- EvalPlus framework available

**License:** Open source (Apache 2.0 for EvalPlus components)

**Use Case for Our Suite:**
- ❌ **NOT RECOMMENDED** - High contamination risk
- Only use with explicit contamination disclosure
- Could use for cross-validation but not as primary benchmark

**Contamination Status:**
- 🔴 **CONFIRMED CONTAMINATED** (65.4% from public sources)
- Similar to HumanEval in terms of contamination risk

---

## Comparison Matrix

| Benchmark | Open Source | Contamination Strategy | Difficulty | Code Focus | Recommendation |
|-----------|-------------|------------------------|------------|------------|----------------|
| **LiveCodeBench** | ✅ Full | ⏰ Temporal | 🟢 Low | ✅ Yes | ⭐⭐⭐ **HIGHEST** |
| **SWE-bench Live** | ✅ Full | ⏰ Temporal | 🟡 Medium | ✅ Yes | ⭐⭐⭐ **HIGHEST** |
| **BigCodeBench** | ✅ Full | ✨ Quality | 🟢 Low-Med | ✅ Yes | ⭐⭐ **MEDIUM** |
| **SWE-bench Pro** | ⚠️ Partial | ⚖️ Legal | 🟡 Medium | ✅ Yes | ⭐⭐ **MEDIUM** |
| **MBPP+/Sanitized** | ✅ Full | ❌ None | 🟢 Low | ✅ Yes | ❌ **NOT RECOMMENDED** (65% contaminated) |
| **ARC-AGI** | ✅ Full | 🔒 Private Sets | 🔴 High | ❌ No | ⭐ **LOW** (non-code) |
| **METR RE-Bench** | ✅ Full | 🧠 Complexity | 🔴 High | ⚠️ Partial | ⭐ **LOW** (too complex) |

---

## Integration Recommendations

### Phase 1: Quick Wins (This Week)
**Add LiveCodeBench**

**Why:**
- Fully open source
- Low integration difficulty
- Direct replacement for HumanEval
- Temporal contamination protection

**Implementation:**
```python
# Integration sketch
from livecodebench import LiveCodeBench

benchmark = LiveCodeBench(
    release_version="v4",  # Latest version
    cutoff_date="2024-09-01",  # Model training cutoff
    tasks=["code_generation"]  # Can add self-repair, execution, etc.
)

results = benchmark.evaluate(
    model=our_ollama_model,
    problems_after_cutoff=True  # Only test on fresh problems
)
```

**Effort:** 1-2 days
**Impact:** Immediate contamination-resistant code generation benchmark

---

### Phase 2: Comprehensive Coverage (2 Weeks)
**Add SWE-bench Live**

**Why:**
- More realistic than isolated functions
- Monthly updates
- Repository-level understanding
- Open source and well-maintained

**Implementation Challenges:**
- Requires repository context (larger prompts)
- Docker execution (we already have this)
- More complex evaluation than function completion

**Integration:**
```python
# Will need to adapt our existing Docker infrastructure
from swebench_live import SWEBenchLive

benchmark = SWEBenchLive(
    split="test",
    month="2025-10"  # Latest month
)

# Leverage our existing container isolation
results = benchmark.evaluate_with_docker(
    agent=our_model_agent,
    timeout=300,  # 5 minutes per issue
    isolation="full"  # Network isolation, read-only FS
)
```

**Effort:** 1 week (reuse Docker infrastructure)
**Impact:** Realistic software engineering evaluation

---

### Phase 3: Legal Protection (Optional)
**Add SWE-bench Pro Public Subset**

**Why:**
- GPL licensing provides legal contamination deterrent
- Public subset available via HuggingFace
- Similar to SWE-bench Live (can reuse infrastructure)

**Limitation:**
- Only public subset accessible
- Held-out and commercial sets unavailable

**Effort:** 3-4 days (after SWE-bench Live integration)
**Impact:** Additional legal protection layer

---

## Temporal Contamination Strategy

### How to Use Temporal Benchmarks

**1. Determine Model Training Cutoff:**
```python
MODEL_CUTOFFS = {
    "gpt-4-turbo": "2023-12-31",
    "claude-3-opus": "2023-08-31",
    "qwen2.5-coder:3b": "2024-09-30",  # Example
    "deepseek-coder:6.7b": "2024-08-31"  # Example
}
```

**2. Filter Benchmark Problems:**
```python
def get_clean_problems(benchmark, model_name):
    """
    Get problems released AFTER model training cutoff.
    """
    cutoff = MODEL_CUTOFFS.get(model_name, "1970-01-01")  # Default to all

    clean_problems = [
        p for p in benchmark.problems
        if p.release_date > cutoff
    ]

    return clean_problems
```

**3. Report Contamination Status:**
```python
{
    "model": "qwen2.5-coder:3b",
    "benchmark": "LiveCodeBench_v4",
    "training_cutoff": "2024-09-30",
    "total_problems": 713,
    "clean_problems": 87,  # Released after cutoff
    "contaminated_problems": 626,  # Released before cutoff
    "contamination_status": "CLEAN_SUBSET_EVALUATED",
    "results": {
        "clean_subset_accuracy": 0.42,  # True capability
        "full_set_accuracy": 0.67  # Inflated by contamination
    }
}
```

---

## Implementation Roadmap

### Week 1: LiveCodeBench Integration
- [ ] Clone LiveCodeBench repository
- [ ] Install dependencies
- [ ] Implement temporal filtering
- [ ] Create adapter for our unified runner
- [ ] Run baseline tests on Ollama models
- [ ] Document contamination protection in reports

**Deliverable:** Working LiveCodeBench evaluation with temporal protection

---

### Week 2-3: SWE-bench Live Integration
- [ ] Clone SWE-bench Live repository
- [ ] Understand Docker requirements (RepoLaunch)
- [ ] Adapt existing container infrastructure
- [ ] Implement repository context handling
- [ ] Create issue-to-patch evaluation pipeline
- [ ] Run pilot evaluation

**Deliverable:** Repository-level software engineering evaluation

---

### Week 4: Validation & Documentation
- [ ] Run cross-benchmark validation
- [ ] Compare HumanEval vs LiveCodeBench results
- [ ] Analyze contamination impact (before/after cutoff)
- [ ] Update contamination analysis document
- [ ] Create integration guide for future benchmarks

**Deliverable:** Comprehensive contamination analysis with new benchmarks

---

## Open Questions

### 1. Model Training Cutoffs
**Question:** How do we determine training cutoffs for Ollama models?

**Options:**
- Contact model creators (Qwen, DeepSeek, etc.)
- Assume cutoff = release date - 3 months
- Conservative approach: Only use very recent problems (last 30 days)

**Resolution:** Research each model's documentation

---

### 2. Benchmark Update Frequency
**Question:** How often should we refresh benchmarks?

**Recommendation:**
- LiveCodeBench: Monthly (follows their update schedule)
- SWE-bench Live: Monthly (follows their 50 issues/month)
- Re-evaluate models quarterly on fresh problems

---

### 3. Private Test Sets
**Question:** Should we create our own private test sets?

**Pros:**
- Complete control over contamination
- Can share results without sharing problems

**Cons:**
- Requires secure infrastructure
- Less transparent
- Harder to validate

**Recommendation:** Phase 3 (after temporal benchmarks proven)

---

## Resources

### Documentation
- **LiveCodeBench Paper:** https://arxiv.org/abs/2403.07974
- **SWE-bench Live:** https://swe-bench-live.github.io/
- **SWE-bench Pro Paper:** https://arxiv.org/abs/2509.16941
- **ARC-AGI:** https://arcprize.org/guide
- **METR RE-Bench:** https://arxiv.org/abs/2411.15114

### Repositories
- **LiveCodeBench:** https://github.com/LiveCodeBench/LiveCodeBench
- **SWE-bench Live:** https://github.com/microsoft/SWE-bench-Live
- **SWE-bench Pro:** https://github.com/scaleapi/SWE-bench_Pro-os
- **ARC-AGI:** https://github.com/fchollet/ARC-AGI
- **METR RE-Bench:** https://github.com/METR/RE-Bench

### Related Research
- **Global contamination research:** ~/docs/AI_BENCHMARK_CONTAMINATION_RESEARCH_2025.md
- **Our suite analysis:** docs/BENCHMARK_CONTAMINATION_ANALYSIS.md

---

## Next Steps

1. **Review this document** with team/stakeholders
2. **Prioritize benchmarks** for integration (recommend LiveCodeBench first)
3. **Create integration issues** in project tracker
4. **Begin Phase 1** implementation this week

---

---

## Summary of Findings

### ✅ Recommended for Integration (Contamination-Resistant)

1. **LiveCodeBench** - ⭐⭐⭐ HIGHEST PRIORITY
   - Temporal contamination protection
   - Monthly updates with fresh problems
   - Low integration difficulty
   - Direct HumanEval replacement

2. **SWE-bench Live** - ⭐⭐⭐ HIGHEST PRIORITY
   - Temporal contamination protection
   - Realistic repository-level tasks
   - Monthly 50 new issues
   - Medium integration difficulty (worth it)

3. **BigCodeBench** - ⭐⭐ MEDIUM PRIORITY
   - Higher quality than HumanEval
   - Complex function calls
   - Still public (moderate contamination risk)
   - Good for cross-validation

### ⚠️ Use with Caution

4. **SWE-bench Pro** - Public subset only
   - Legal contamination protection
   - Partial access (no held-out set)
   - Similar to SWE-bench Live

### ❌ Not Recommended (High Contamination)

5. **MBPP+/Sanitized** - 65.4% contaminated
   - Known contamination from public sources
   - Only use with explicit disclosure
   - Not suitable as primary benchmark

6. **ARC-AGI** - Non-code benchmark
   - Good contamination protection
   - Not applicable to code generation

7. **METR RE-Bench** - Too complex
   - Better for agent evaluation
   - 8-hour tasks (overkill for benchmarking)

---

## Contamination Risk Summary

| Risk Level | Benchmarks | Reasoning |
|------------|------------|-----------|
| 🟢 **LOW** | LiveCodeBench, SWE-bench Live | Temporal protection, monthly updates |
| 🟡 **MODERATE** | BigCodeBench, SWE-bench Pro | Public but newer/complex, or partial access |
| 🔴 **HIGH** | MBPP+, HumanEval (our current) | Confirmed 65%+ contamination |

---

## Document Metadata

**Version:** 1.0 COMPLETE
**Status:** ACTIVE REFERENCE
**Last Updated:** 2025-10-08
**Next Review:** 2025-11-08 (monthly)
**Owner:** AI Benchmark Suite Development Team

**Research Sources:**
- LiveCodeBench: arXiv:2403.07974, GitHub, HuggingFace
- SWE-bench Live: Microsoft GitHub, official leaderboard
- BigCodeBench: GitHub bigcode-project, ICLR'25
- SWE-bench Pro: Scale AI, arXiv:2509.16941
- MBPP+: Google Research, EvalPlus, contamination studies
- ARC-AGI: arcprize.org, GitHub repos
- METR RE-Bench: metr.org, arXiv:2411.15114
