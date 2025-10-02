# Adaptive Ollama Deployment Guide

**Phase 2 Multi-Armed Bandit Adaptive Prompting System**

This guide covers production deployment of the adaptive prompting system that uses contextual multi-armed bandits to automatically optimize prompting strategies.

## 🎯 What This System Does

The adaptive system automatically:
- **Learns optimal prompting strategies** for different types of code completion tasks
- **Adapts in real-time** based on quality feedback and context
- **Maintains exploration** to discover better strategies over time
- **Provides comprehensive analytics** on adaptation performance

## 📋 Prerequisites

### Required
- **Python 3.8+** with pip
- **Ollama installed and running** (tested with models like `qwen2.5-coder:3b`, `phi3:latest`)
- **SQLite** (for bandit state persistence)

### Optional
- **PyYAML** for configuration files
- **Docker** for containerized deployment

### Install Dependencies

```bash
pip install requests numpy sqlite3 pyyaml
```

## 🚀 Quick Start

### 1. Basic CLI Usage

```bash
# Generate single completion with adaptive strategy selection
python adaptive_ollama_cli.py generate "def fibonacci(n):" --model qwen2.5-coder:3b

# Run benchmark with multiple problems
python adaptive_ollama_cli.py benchmark --model phi3:latest --problems 10

# Check model and bandit status
python adaptive_ollama_cli.py status --model deepseek-coder:6.7b
```

### 2. Python API Usage

```python
from src.model_interfaces.adaptive_ollama_interface import AdaptiveOllamaInterface

# Create adaptive interface
interface = AdaptiveOllamaInterface(
    model_name="qwen2.5-coder:3b",
    base_url="http://localhost:11434"
)

# Generate with automatic strategy selection
response = interface.generate_adaptive("def add_numbers(a, b):")

print(f"Strategy used: {response.selected_strategy}")
print(f"Quality score: {response.quality_score}")
print(f"Completion: {response.text}")
```

### 3. Drop-in Replacement for Existing Code

```python
from src.model_interfaces.adaptive_benchmark_adapter import create_ollama_interface

# Replace existing ollama interface creation with this line
interface = create_ollama_interface("qwen2.5-coder:3b")

# Use exactly the same as before - adapter handles adaptation automatically
completion = interface.generate("def factorial(n):")
```

## ⚙️ Configuration

### Configuration File

Create `config/adaptive_ollama.yaml`:

```yaml
# Ollama configuration
ollama:
  base_url: "http://localhost:11434"
  timeout: 120
  quality_threshold: 0.6

# Bandit learning configuration
bandit:
  exploration_alpha: 2.0              # Higher = more exploration
  target_exploration_rate: 0.3        # Target 30% exploration
  target_diversity: 0.6               # Target 60% strategy diversity

# Model-specific settings
models:
  "qwen2.5-coder:3b":
    exploration_alpha: 2.2           # More exploration for smaller model
    quality_threshold: 0.55

  "phi3:latest":
    exploration_alpha: 1.8           # Less exploration for larger model
    quality_threshold: 0.65
```

### Use Configuration

```bash
python adaptive_ollama_cli.py --config config/adaptive_ollama.yaml generate "code here"
```

## 🔧 Integration with Existing Systems

### HumanEval Integration

```python
from src.model_interfaces.adaptive_benchmark_adapter import create_adaptive_humaneval_adapter

# Create adapter optimized for HumanEval
adapter = create_adaptive_humaneval_adapter("qwen2.5-coder:3b")

# Use in existing HumanEval runner
with adapter as interface:
    results = interface.generate_batch(humaneval_prompts)

# Get adaptation analytics
print(interface.get_adaptation_stats())
```

### BigCode Integration

```python
from src.model_interfaces.adaptive_benchmark_adapter import create_adaptive_bigcode_adapter

adapter = create_adaptive_bigcode_adapter("deepseek-coder:6.7b")
results = adapter.generate_batch_with_details(bigcode_prompts)
```

### Custom Integration

Replace your existing Ollama interface creation:

```python
# OLD CODE:
# interface = OllamaInterface(model_name="qwen2.5-coder:3b")

# NEW CODE:
from src.model_interfaces.adaptive_benchmark_adapter import AdaptiveBenchmarkAdapter
interface = AdaptiveBenchmarkAdapter(model_name="qwen2.5-coder:3b")

# Everything else works the same!
```

## 📊 Monitoring and Analytics

### Real-time Status

```bash
# Check current learning status
python adaptive_ollama_cli.py status --model qwen2.5-coder:3b
```

Output includes:
- **Exploration rate**: How often the system tries new strategies
- **Strategy diversity**: Percentage of available strategies being used
- **Strategy performance**: Quality scores for each strategy
- **Total requests**: Number of completions generated

### Detailed Analytics

```python
interface = AdaptiveOllamaInterface(model_name="qwen2.5-coder:3b")

# Get comprehensive analytics
analytics = interface.get_adaptation_analytics()
print(f"Total requests: {analytics['total_requests']}")
print(f"Successful adaptations: {analytics['successful_adaptations']}")

# Get strategy performance breakdown
performance = interface.bandit_selector.get_strategy_performance()
for strategy, stats in performance.items():
    print(f"{strategy}: {stats['mean_reward']:.3f} quality, {stats['total_trials']} trials")

# Get exploration statistics
exploration = interface.bandit_selector.get_exploration_stats()
print(f"Exploration rate: {exploration['exploration_rate']:.2f}")
print(f"Strategy diversity: {exploration['recent_diversity']:.2f}")
```

## 🎛️ Tuning and Optimization

### Automatic Tuning

The system automatically tunes exploration parameters:

```python
interface = AdaptiveOllamaInterface(model_name="qwen2.5-coder:3b")

# Automatic tuning (called automatically every ~50 requests)
interface.tune_exploration()

# Manual tuning with custom targets
interface.bandit_selector.tune_exploration_parameter(
    target_exploration_rate=0.4,  # 40% exploration
    target_diversity=0.7           # 70% strategy diversity
)
```

### Manual Parameter Adjustment

```python
# Check current exploration parameter
current_alpha = interface.bandit_selector.alpha
print(f"Current alpha: {current_alpha}")

# Increase exploration (useful for new domains)
interface.bandit_selector.alpha = 2.5

# Decrease exploration (useful when optimal strategies are known)
interface.bandit_selector.alpha = 1.5
```

### Strategy Performance Analysis

```python
# Get insights on which contexts work best for each strategy
insights = interface.bandit_selector.get_context_insights()

for strategy, insight in insights.items():
    if 'feature_preferences' in insight:
        print(f"{strategy} performs best with:")
        for feature, preference in insight['feature_preferences'].items():
            if abs(preference) > 0.1:  # Significant preference
                print(f"  {feature}: {'high' if preference > 0 else 'low'}")
```

## 🛠️ Production Deployment

### Basic Production Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama** with required models:
   ```bash
   ollama pull qwen2.5-coder:3b
   ollama pull phi3:latest
   ollama serve
   ```

3. **Configure system**:
   ```bash
   cp config/adaptive_ollama.yaml.example config/adaptive_ollama.yaml
   # Edit configuration for your environment
   ```

4. **Test deployment**:
   ```bash
   python adaptive_ollama_cli.py status --model qwen2.5-coder:3b
   ```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Start Ollama in background and run adaptive system
CMD ["python", "adaptive_ollama_cli.py", "benchmark", "--model", "qwen2.5-coder:3b"]
```

### Batch Processing

```python
from src.model_interfaces.adaptive_benchmark_adapter import AdaptiveBenchmarkAdapter

# Efficient batch processing
adapter = AdaptiveBenchmarkAdapter("qwen2.5-coder:3b")

with adapter as interface:
    # Process large batches with automatic progress tracking
    results = interface.generate_batch(prompts, progress=True)

    # Save analytics
    interface.save_analytics("batch_results.json")
```

## 🔍 Troubleshooting

### Common Issues

**1. "Model not available" error**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull the model if needed
ollama pull qwen2.5-coder:3b
```

**2. "Connection refused" error**
```bash
# Start Ollama service
ollama serve

# Or use different URL
python adaptive_ollama_cli.py generate "code" --model qwen2.5-coder:3b --base-url http://your-ollama-server:11434
```

**3. Low quality scores**
- **Check model performance**: Some models work better than others
- **Adjust quality threshold**: Lower threshold in config for smaller models
- **Let system learn**: Quality improves over time as bandit learns

**4. Too much exploration**
```python
# Reduce exploration parameter
interface.bandit_selector.alpha = 1.5

# Or set target exploration rate
interface.bandit_selector.tune_exploration_parameter(target_exploration_rate=0.2)
```

**5. Too little exploration**
```python
# Increase exploration parameter
interface.bandit_selector.alpha = 2.5

# Or reset bandit to start fresh learning
interface.bandit_selector.reset_bandit()
```

### Error Recovery

The system includes automatic error recovery:

- **Connection errors**: Falls back to basic generation or cached responses
- **Bandit errors**: Uses deterministic strategy as fallback
- **Context analysis errors**: Continues with simplified feature extraction

```python
# Check error statistics
from src.model_interfaces.error_handling import AdaptiveErrorHandler

handler = AdaptiveErrorHandler()
stats = handler.get_error_statistics()
print(f"Error recovery rate: {stats['recovery_rate']:.1%}")
```

## 📈 Performance Expectations

### Quality Improvements

Based on validation testing:

- **Initial Performance**: Variable, as system explores strategies
- **After ~50 requests**: 1.0-2.0% quality improvement over baseline
- **Long-term**: Continued optimization as system learns optimal contexts

### Strategy Convergence

- **Exploration Phase** (requests 1-30): High diversity, testing all strategies
- **Learning Phase** (requests 30-100): Balanced exploration/exploitation
- **Optimization Phase** (requests 100+): Focus on best strategies with continued exploration

### Resource Usage

- **Memory**: ~50MB additional for bandit state and analytics
- **Storage**: ~1MB SQLite database per model for learning persistence
- **CPU**: <5% overhead for context analysis and strategy selection
- **Network**: Same as baseline (strategy selection is local)

## 🔬 Advanced Features

### Custom Strategy Configuration

```python
# Add custom strategy
custom_strategy = {
    "prompt": "You are a senior developer. Complete this code professionally:",
    "temperature": 0.15,
    "stop_tokens": ["\n\n", "```"]
}

interface.strategy_prompts[PromptingStrategy.ROLE_BASED] = custom_strategy
```

### Context Feature Engineering

```python
# Custom context analysis
from src.prompting.context_analyzer import PromptContextAnalyzer

analyzer = PromptContextAnalyzer()

# Add custom domain keywords
analyzer.domain_keywords[CodeDomain.WEB_DEVELOPMENT]['vue3'] = 2.0
analyzer.domain_keywords[CodeDomain.WEB_DEVELOPMENT]['typescript'] = 1.5
```

### Multi-Model Learning

```python
# Train separate bandits for different models
models = ["qwen2.5-coder:3b", "phi3:latest", "deepseek-coder:6.7b"]

interfaces = {}
for model in models:
    interfaces[model] = AdaptiveOllamaInterface(model_name=model)

# Each model learns independently and optimally
```

## 📚 API Reference

### AdaptiveOllamaInterface

**Main Methods**:
- `generate_adaptive(prompt: str) -> AdaptiveOllamaResponse`
- `get_adaptation_analytics() -> Dict[str, Any]`
- `tune_exploration(target_rate: float = 0.2)`

### AdaptiveBenchmarkAdapter

**Compatibility Methods**:
- `generate(prompt: str) -> str` - Drop-in replacement
- `generate_batch(prompts: List[str]) -> List[str]` - Batch processing
- `get_adaptation_stats() -> Dict[str, Any]` - Analytics

### CLI Commands

```bash
# Generation
python adaptive_ollama_cli.py generate PROMPT --model MODEL [--output FILE]

# Benchmarking
python adaptive_ollama_cli.py benchmark --model MODEL [--problems N] [--output FILE]

# Status and monitoring
python adaptive_ollama_cli.py status --model MODEL [--output FILE]
```

## 🎉 Success Metrics

Your deployment is successful when you see:

1. **Strategy Diversity**: >60% of available strategies being used
2. **Exploration Rate**: 20-40% depending on use case
3. **Quality Improvement**: 1-2% improvement after learning period
4. **Error Recovery**: >90% error recovery rate
5. **System Health**: No critical errors in recent operations

```bash
# Quick health check
python adaptive_ollama_cli.py status --model qwen2.5-coder:3b | grep -E "(Exploration|Diversity|Success)"
```

The system is production-ready and will automatically optimize performance over time!