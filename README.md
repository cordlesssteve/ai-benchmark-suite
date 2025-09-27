# Unified AI Benchmarking Suite

A comprehensive, unified framework for evaluating AI models across multiple domains using established evaluation harnesses.

## 🏗️ Architecture

```
ai-benchmark-suite/
├── harnesses/                          # Git submodules (READ-ONLY)
│   ├── bigcode-evaluation-harness/     # Code generation evaluation
│   └── lm-evaluation-harness/          # General language model evaluation
├── src/
│   ├── unified_runner.py               # Main orchestration interface
│   ├── statistical_analysis.py        # Advanced statistical metrics
│   └── model_interfaces/               # Model adapters (Ollama, HF, API)
├── config/
│   ├── suite_definitions.yaml         # Benchmark suite configurations
│   ├── harness_mappings.yaml          # Task routing to harnesses
│   └── models.yaml                     # Model definitions
└── results/
    ├── code_generation/                # BigCode results
    ├── language_tasks/                 # LM-Eval results
    └── cross_analysis/                 # Unified analysis
```

## ✨ Key Features

- **Unified Interface**: Single command to run evaluations across multiple harnesses
- **Multi-Domain**: Code generation (BigCode) + Language understanding (LM-Eval)
- **Model Agnostic**: Supports local HF models, Ollama, and API models
- **Statistical Rigor**: Template sensitivity analysis, Pass@K metrics, significance testing
- **Predefined Suites**: Quick, Standard, Comprehensive, and Research-grade evaluations
- **Safe Submodules**: Read-only integration with upstream evaluation frameworks

## 🚀 Quick Start

### 1. Clone and Initialize

```bash
git clone <your-repo>
cd ai-benchmark-suite

# Initialize submodules
git submodule init
git submodule update

# Set up evaluation harnesses
python src/unified_runner.py --setup
```

### 2. Run Quick Evaluation

```bash
# Single task
python src/unified_runner.py --task humaneval --model codeparrot-small --limit 5

# Predefined suite
python src/unified_runner.py --suite quick --models phi3,qwen-coder
```

### 3. Available Suites

- **quick**: Fast validation (~5-10 min) - humaneval + hellaswag
- **standard**: Balanced evaluation (~30-60 min) - 5 core tasks
- **comprehensive**: Multi-domain assessment (~2-4 hours) - 10 tasks
- **research**: Statistical rigor (~6-12 hours) - High sample counts

## 🎯 Supported Evaluations

### Code Generation (BigCode Harness)
- **HumanEval**: Python code completion (164 problems)
- **MBPP**: Python programming (974 problems)
- **APPS**: Algorithmic programming
- **DS-1000**: Data science tasks

### Language Understanding (LM-Eval Harness)
- **HellaSwag**: Common sense reasoning
- **ARC**: Science questions (easy/challenge)
- **WinoGrande**: Pronoun resolution
- **MathQA**: Mathematical reasoning
- **GSM8K**: Grade school math

## 🤖 Supported Models

### Local Models
- **HuggingFace**: codeparrot, starcoder, any HF model
- **Ollama**: phi3, qwen-coder, codellama, mistral

### API Models
- **OpenAI**: GPT-4, GPT-4-Turbo
- **Anthropic**: Claude 3 Sonnet/Opus

## 📊 Advanced Features

### Statistical Analysis
- **Pass@K Metrics**: Proper statistical evaluation (K=1,5,10,100)
- **Template Sensitivity**: Measures prompt engineering dependence
- **Bootstrap Confidence**: Statistical significance testing
- **Multi-Template Evaluation**: 5 research-validated prompt formats

### Template Types
1. **Direct**: Raw completion
2. **Instruction**: Structured prompts
3. **Conversational**: Natural dialogue
4. **Few-Shot**: Example-based learning
5. **Chain-of-Thought**: Step-by-step reasoning

## ⚙️ Configuration

### Adding Models
Edit `config/models.yaml`:
```yaml
local_models:
  my_model:
    model_id: "path/to/model"
    type: "code"
    temperature: 0.2
```

### Creating Custom Suites
Edit `config/suite_definitions.yaml`:
```yaml
suites:
  my_suite:
    description: "Custom evaluation"
    tasks: [humaneval, hellaswag]
    settings:
      limit: 20
      n_samples: 5
```

## 🔒 Submodule Safety

**CRITICAL**: Harnesses are read-only submodules. See `SUBMODULE_SAFETY.md` for detailed guidelines.

- ✅ Update submodules: `git submodule update --remote`
- ❌ Never edit files in `harnesses/` directories
- ❌ Never commit or push from submodule directories

## 📈 Example Results

```bash
# Research-grade evaluation with statistical analysis
python src/unified_runner.py --suite research --models qwen-coder,phi3.5 --n_samples 50

# Results automatically saved to:
# - results/code_generation/qwen-coder_humaneval_timestamp.json
# - results/language_tasks/phi3.5_hellaswag_timestamp.json
# - results/cross_analysis/model_comparison_timestamp.json
```

## 🛠️ Development

### Project Structure
- All customizations go in `src/` - never modify submodules
- Configuration in `config/` using YAML
- Results organized by harness type and timestamp
- Statistical analysis separate from execution

### Adding New Harnesses
1. Add as git submodule in `harnesses/`
2. Update `config/harness_mappings.yaml`
3. Implement interface in `src/unified_runner.py`

## 🤝 Contributing

This project unifies established evaluation frameworks:
- [BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness)
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)

Focus contributions on:
- New model interfaces
- Statistical analysis improvements
- Suite definitions
- Results visualization

---

**Status**: ✅ Core framework implemented
**Next**: Model interface completion and result analysis tools