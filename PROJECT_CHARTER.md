# PROJECT CHARTER: ai-benchmark-suite

**Status:** FOUNDATIONAL
**Created:** 2025-01-16
**Project Type:** Single-Branch
**Charter Version:** 1.0 (ORIGINAL)
**Revisions:** None

---
⚠️ **IMMUTABILITY NOTICE**

This charter is a historical document preserving the original project vision.
It should NEVER be edited except for:
- Typo corrections (noted in CHARTER_CHANGELOG.md)
- Formatting fixes (noted in CHARTER_CHANGELOG.md)
- Clarifications that don't change intent (noted in CHARTER_CHANGELOG.md)

For scope/vision changes, create a new CHARTER_REVISION document in `/docs/charter/`.
---

## 1. Project Purpose

**Why does this project exist?**

AI model evaluation faces two critical problems: benchmark contamination and incomplete evaluation dimensions. When models are trained on datasets that include benchmark test cases, evaluation becomes meaningless - the model has "seen the answers" rather than demonstrating genuine capability. Additionally, existing benchmarks focus on whether models can use tools correctly (execution quality) but ignore the foundational question: does the model recognize when tool use is appropriate versus answering directly?

This "Layer 0" evaluation gap is critical for AI coding assistants like Claude Code. If an assistant can execute tools perfectly but doesn't recognize when to use them, it will respond with hallucinated information instead of checking authoritative sources. No existing benchmarks (Recovery-Bench, τ-bench, ARTIST) test this organic tool discovery decision-making.

The ai-benchmark-suite addresses both problems: providing temporally-filtered contamination-resistant benchmarks that remain valid as models evolve, and designing evaluation frameworks that test tool discovery decisions alongside execution quality.

**Primary Objective:**
Create a unified AI evaluation framework that prevents benchmark contamination through temporal filtering while testing both tool execution quality and the critical "should I use a tool?" decision that existing benchmarks ignore.

## 2. Success Criteria

The project will be considered successful when:
- [ ] Contamination-resistant benchmarks integrated with temporal cutoffs preventing training data overlap
- [ ] Single-command evaluation across code generation and language understanding domains
- [ ] Statistical rigor with Pass@K metrics, template sensitivity analysis, and significance testing
- [ ] Model-agnostic architecture supporting HuggingFace, Ollama, and API-based models
- [ ] MCP-Bench evaluation framework testing 4 levels: Discovery, Selection, Recovery, Chaining
- [ ] Documented research identifying methodology gaps in existing tool-use benchmarks
- [ ] At least 3 contamination-resistant benchmark suites operational

## 3. Scope Boundaries

### In Scope
- Integration of contamination-resistant benchmarks (LiveCodeBench, BigCodeBench, SWE-bench Live)
- Temporal filtering preventing training cutoff contamination
- Unified runner orchestrating multiple evaluation harnesses
- Statistical evaluation methodology (Pass@K, template sensitivity, bootstrap confidence)
- Model adapters for HuggingFace, Ollama, and API providers
- MCP-Bench design for organic tool discovery evaluation
- Research analysis identifying gaps in existing tool-use benchmarks
- Predefined evaluation suites (quick, standard, comprehensive, research)
- Cross-domain analysis tools combining code and language results

### Out of Scope
- Creating new benchmarks from scratch (leverage existing work)
- Training models or fine-tuning (evaluation only)
- Real-time production model serving
- Custom benchmark dataset curation
- Benchmark result hosting or leaderboards
- Multi-GPU distributed evaluation infrastructure
- Commercial model comparison services
- Automated prompt engineering or optimization

### Future Consideration
- Additional contamination-resistant benchmarks as they emerge
- Expanded language support beyond Python-focused tasks
- Multi-modal evaluation (code + documentation + UI)
- Automated benchmark freshness monitoring
- Integration with CI/CD for model quality gates
- Custom benchmark creation tools
- Public leaderboard infrastructure

## 4. Key Stakeholders

| Role | Name/Entity | Interest | Influence Level |
|------|-------------|----------|-----------------|
| Owner/Researcher | cordlesssteve | Tool-use evaluation research, contamination resistance | High |
| Primary Users | AI researchers evaluating models | Accurate, contamination-free benchmarks | High |
| Secondary Users | Model developers | Validation of improvements | Medium |
| Research Community | Tool-use benchmark researchers | Methodology improvements | Medium |

## 5. Constraints

### Time Constraints
- No fixed deadline (research-driven project)
- Benchmark freshness depends on upstream updates
- Research publication cycles for methodology contributions

### Budget Constraints
- Self-funded compute for model evaluation
- API costs for commercial model testing (OpenAI, Anthropic)
- Infrastructure costs for benchmark storage and processing
- Limited access to high-end GPUs for local model evaluation

### Technology Constraints
- **Required:**
  - Python for evaluation harness integration
  - Git submodules for upstream benchmark integration
  - Docker for isolated evaluation environments
  - Statistical analysis libraries (scipy, numpy, pandas)
- **Prohibited:**
  - Solutions requiring expensive commercial licenses
  - Platforms with vendor lock-in preventing benchmark updates
  - Evaluation approaches that modify upstream benchmarks

### Resource Constraints
- Solo researcher/developer
- Limited compute resources for large model evaluation
- Dependence on upstream benchmark maintenance
- No dedicated infrastructure team

## 6. Assumptions

We are assuming:
1. Contamination-resistant benchmarks will continue to receive updates maintaining temporal separation
2. Upstream benchmark projects remain maintained and accessible
3. Model providers offer API access for evaluation purposes
4. Statistical methodology (Pass@K, template sensitivity) generalizes across domains
5. MCP tool-use evaluation insights apply to other AI assistant frameworks
6. Research community values Layer 0 (tool discovery) evaluation gap identification
7. Single-machine evaluation sufficient for research purposes (vs distributed infrastructure)
8. Benchmark licensing permits research and evaluation use
9. Python ecosystem remains standard for ML evaluation tooling

❓ **If any assumption proves false, this charter may need revision.**

## 7. Known Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|---------|---------------------|
| Benchmark contamination despite temporal filtering | Medium | High | Multi-benchmark validation, monitoring model training dates |
| Upstream benchmarks abandoned or unmaintained | Low | High | Fork critical benchmarks, maintain local copies |
| API costs for model evaluation prohibitive | Medium | Medium | Prioritize local models, batch evaluations, sample strategies |
| MCP-Bench design doesn't generalize to other frameworks | Medium | Medium | Collaborate with tool-use researchers, validate across assistants |
| Statistical methodology insufficient for dynamic benchmarks | Low | Medium | Consult with statistics experts, peer review |
| Compute resources inadequate for comprehensive evaluation | High | Medium | Focus on representative subsets, cloud burst capacity |
| Model providers restrict API access for benchmarking | Low | High | Local model fallback, academic research partnerships |

## 8. Background & Context

**What led to this project?**

The project originated from two convergent research observations:

1. **Benchmark Contamination Crisis**: Popular code benchmarks (HumanEval, MBPP) appeared in training datasets, making evaluation meaningless. The 2024 research landscape shifted toward contamination-resistant approaches with temporal guarantees (LiveCodeBench with monthly updates, BigCodeBench with enhanced quality, SWE-bench Live with repository-level tasks).

2. **Tool-Use Evaluation Gap**: While developing MetaMCP-RAG for Claude Code, we discovered that existing tool-use benchmarks (Recovery-Bench, τ-bench, ARTIST) only test execution quality (Layers 1-3: selection, parameter filling, recovery). No benchmark tests whether AI assistants recognize when to use tools versus answering directly - the "Layer 0" organic discovery decision.

This gap is critical: an assistant that executes tools perfectly but doesn't recognize when to trigger them will hallucinate answers instead of checking authoritative sources. The ai-benchmark-suite combines contamination-resistant evaluation with tool-use research to address both gaps systematically.

**Related Projects:**
- MetaMCP-RAG (inspired tool-use evaluation research)
- LiveCodeBench (contamination-resistant code benchmark)
- BigCodeBench (enhanced quality code evaluation)
- SWE-bench Live (repository-level tasks)

## 9. Dependencies

**External Dependencies:**
- Upstream contamination-resistant benchmarks (LiveCodeBench, BigCodeBench, SWE-bench Live)
- Model APIs (OpenAI, Anthropic) for commercial model evaluation
- Ollama for local model serving
- HuggingFace transformers library
- Statistical libraries (scipy, numpy, pandas)
- Git submodule infrastructure for benchmark integration

**Internal Dependencies:**
- None (independent project)

## 10. Success Metrics

**How will we measure success?**
- **Contamination Resistance**: Temporal separation verified between model training dates and benchmark content
- **Coverage**: ≥3 contamination-resistant benchmark suites operational
- **Statistical Rigor**: Pass@K metrics with confidence intervals, template sensitivity analysis
- **Model Support**: HuggingFace, Ollama, and ≥2 API providers functional
- **Research Impact**: MCP-Bench design framework documented with Layer 0 evaluation methodology
- **Usability**: Single-command evaluation across domains in <10 minutes for quick suite
- **Reproducibility**: Consistent results across evaluation runs (variance <5%)
- **Documentation**: Comprehensive research report on tool-use benchmark gaps published

---

## Charter Approval

**Approved By:** cordlesssteve (Project Owner)
**Approval Date:** 2025-01-16

---

## Document History

This is the original charter. All changes logged in `/docs/charter/CHARTER_CHANGELOG.md`.
