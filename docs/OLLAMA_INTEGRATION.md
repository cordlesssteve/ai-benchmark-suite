# Ollama Integration Guide

Your AI Benchmark Suite now supports local Ollama models! Here's how to use your phi3, phi3.5, Qwen, and other Ollama models for code generation benchmarking.

## ✅ **Available Ollama Models**

Your configured models:
- **phi3:latest** - Microsoft Phi-3 (2.2GB) - Small but capable model
- **phi3.5:latest** - Microsoft Phi-3.5 (2.2GB) - Latest small model  
- **qwen2.5-coder:3b** - Qwen 2.5 Coder 3B (1.9GB) - Specialized coding model
- **codellama:13b-instruct** - CodeLlama 13B Instruct (7.4GB) - Large coding model
- **mistral:7b-instruct** - Mistral 7B Instruct (4.4GB) - General purpose model

## 🚀 **Quick Start**

### 1. Start Ollama Server
```bash
ollama serve
```

### 2. Test Simple Benchmarks
```bash
# Test with simple coding problems
python3 run_ollama_benchmark.py --model phi3:latest --problems 3

# Test coding-specific model
python3 run_ollama_benchmark.py --model qwen2.5-coder:3b --problems 5
```

### 3. Check Results
```bash
ls results/ollama_*_simple_test_*.json
```

## 📊 **Understanding Results**

### **Model Behavior Observed:**
- **Instruction-tuned models** (phi3, qwen2.5-coder) generate explanatory text
- **Models prefer completeness** over raw code completion
- **Different prompting strategies** needed for different model types

### **Sample Output Patterns:**
```json
{
  "model": "phi3:latest",
  "problems_tested": 3,
  "results": [
    {
      "problem_id": "test_1",
      "prompt": "def add_numbers(a, b): return",
      "generated": "```python\\ndef add_numbers(a, b):\\n    return a + b\\n```",
      "expected_pattern": "a + b",
      "passed": true
    }
  ],
  "summary": {
    "passed": 2,
    "total": 3, 
    "score": 0.67
  }
}
```

## 🔧 **Model Configuration**

All models are configured in `config/models.yaml`:

```yaml
ollama_models:
  phi3:
    type: "ollama"
    model_id: "phi3:latest"
    temperature: 0.2
    max_tokens: 512
    
  qwen-coder:
    type: "ollama" 
    model_id: "qwen2.5-coder:3b"
    temperature: 0.2
    max_tokens: 512
```

## 📈 **Benchmarking Approaches**

### **1. Simple Function Completion**
```bash
python3 run_ollama_benchmark.py --model phi3:latest
```
- Tests basic coding patterns
- Evaluates simple completions
- Good for quick model evaluation

### **2. Complex HumanEval Integration**
```bash
python3 src/ollama_integration.py --model qwen2.5-coder:3b --task humaneval --limit 5
```
- Full HumanEval problem solving
- More challenging evaluation
- Research-grade benchmarking

### **3. Model Comparison Suite**
```bash
# Test all small models
for model in phi3:latest phi3.5:latest qwen2.5-coder:3b; do
    python3 run_ollama_benchmark.py --model $model --problems 3
done
```

## 🎯 **Best Practices**

### **1. Model Selection by Use Case**
- **Quick tests**: phi3:latest, phi3.5:latest  
- **Code-focused**: qwen2.5-coder:3b
- **Large problems**: codellama:13b-instruct
- **General purpose**: mistral:7b-instruct

### **2. Temperature Settings**
- **0.1-0.2**: Deterministic, consistent code
- **0.5-0.7**: More creative solutions
- **0.8+**: Experimental, varied outputs

### **3. Prompt Engineering**
Different models respond better to different prompt styles:

```python
# Direct completion (base models)
prompt = "def function_name(args):\\n    "

# Instruction format (instruct models)  
prompt = "Complete this Python function:\\n\\ndef function_name(args):"

# Conversational (chat models)
prompt = "Please complete this Python function for me:\\n..."
```

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **"Ollama server not responding"**
   ```bash
   ollama serve  # Start Ollama server
   ```

2. **"Model not found"**
   ```bash
   ollama list   # Check available models
   ollama pull phi3:latest  # Download model if needed
   ```

3. **"Models generate explanations instead of code"**
   - This is expected behavior for instruction-tuned models
   - Use different evaluation criteria
   - Focus on solution quality over format

4. **"Slow generation"**
   ```bash
   # Check system resources
   htop
   # Consider smaller models for testing
   python3 run_ollama_benchmark.py --model phi3:latest
   ```

## 📊 **Performance Expectations**

### **Generation Speed (approximate):**
- **phi3:latest**: ~2-5 seconds per completion
- **phi3.5:latest**: ~2-5 seconds per completion  
- **qwen2.5-coder:3b**: ~3-8 seconds per completion
- **codellama:13b-instruct**: ~10-30 seconds per completion

### **Model Sizes:**
- **Small models** (phi3, qwen-3b): 2-4 GB RAM
- **Medium models** (mistral-7b): 6-8 GB RAM
- **Large models** (codellama-13b): 12-16 GB RAM

## 🏆 **Example Evaluation Session**

```bash
# Start Ollama
ollama serve &

# Test different models
python3 run_ollama_benchmark.py --model phi3:latest --problems 3
python3 run_ollama_benchmark.py --model qwen2.5-coder:3b --problems 3
python3 run_ollama_benchmark.py --model phi3.5:latest --problems 3

# Compare results
python3 src/results_organizer.py --organize
python3 src/results_organizer.py --compare phi3 qwen-coder phi3.5
```

## 🔮 **Advanced Usage**

### **Custom Problem Sets**
Edit `run_ollama_benchmark.py` to add your own coding problems:

```python
problems = [
    {
        "id": "custom_1",
        "prompt": "def your_function(args):\n    \"\"\"Your description\"\"\"\n    return",
        "expected_pattern": "expected_code_pattern"
    }
]
```

### **Integration with Results Organization**
All Ollama results automatically integrate with the results organization system:

```bash
# Organize results
python3 src/results_organizer.py --organize

# Create leaderboards  
python3 src/results_organizer.py --leaderboard simple_test

# Find results
ls results/organized/by_model/phi3/
```

## 📋 **Model Recommendations**

Based on testing:

### **For Quick Development Testing:**
- ✅ **phi3:latest** - Fast, lightweight, decent quality
- ✅ **phi3.5:latest** - Latest improvements, still fast

### **For Code-Focused Tasks:**  
- ✅ **qwen2.5-coder:3b** - Specialized for coding, good balance
- 🔄 **codellama:13b-instruct** - High quality but slower

### **For General Purpose:**
- ✅ **mistral:7b-instruct** - Good general capabilities

---

## 🎉 **Success!**

**Your Ollama models are now integrated with the AI Benchmark Suite!**

Key achievements:
- ✅ All 5 Ollama models configured  
- ✅ Simple benchmark runner working
- ✅ Results integration with main system
- ✅ Model comparison capabilities
- ✅ Organized result storage

**Next steps:**
1. Run benchmarks on models you're interested in
2. Compare results across different models  
3. Tune prompting strategies for better performance
4. Use results to guide model selection for your projects

The integration provides both simple testing capabilities and full research-grade evaluation options!