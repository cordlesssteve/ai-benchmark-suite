# Results Organization & Analysis Guide

The AI Benchmark Suite now includes a comprehensive results organization system that makes it easy to compare models, track progress over time, and generate analysis reports.

## 📁 **Organized Directory Structure**

Results are automatically organized into a clean, searchable structure:

```
results/
├── organized/
│   ├── by_date/           # Results organized by date
│   │   └── 2025-09-09/
│   │       ├── metrics_*.json
│   │       ├── generations_*.json
│   │       └── suite_*.json
│   ├── by_model/          # Results organized by model
│   │   └── codeparrot_codeparrot-small/
│   │       ├── humaneval_2025-09-09_metrics_*.json
│   │       └── mbpp_2025-09-09_metrics_*.json
│   ├── by_benchmark/      # Results organized by benchmark
│   │   ├── humaneval/
│   │   │   └── codeparrot_codeparrot-small_2025-09-09_metrics_*.json
│   │   └── mbpp/
│   │       └── codeparrot_codeparrot-small_2025-09-09_metrics_*.json
│   └── suites/           # Suite results
│       └── quick_2025-09-09_suite_*.json
├── comparisons/          # Generated comparison reports
│   ├── model_vs_model/   # Model comparison reports
│   ├── benchmark_vs_benchmark/
│   ├── time_series/      # Progress tracking over time
│   └── leaderboard_*.json # Benchmark leaderboards
└── archive/              # Original unorganized files
    └── (moved here after organization)
```

## 🚀 **Quick Usage**

### Organize Existing Results
```bash
# Organize all existing result files
python3 src/results_organizer.py --organize
```

### Generate Leaderboards
```bash
# Create HumanEval leaderboard
python3 src/results_organizer.py --leaderboard humaneval

# Create MBPP leaderboard  
python3 src/results_organizer.py --leaderboard mbpp
```

### Compare Models
```bash
# Compare two models across all benchmarks
python3 src/results_organizer.py --compare codeparrot-small starcoder

# Compare models on specific benchmark
python3 src/results_organizer.py --compare codeparrot-small starcoder --benchmark humaneval
```

## 📊 **Analysis Features**

### **1. Automatic Leaderboards**
- Ranks all models by benchmark performance
- Includes detailed configuration and scores
- Tracks primary metrics (Pass@1) and secondary metrics
- Sortable JSON format for further analysis

**Sample Leaderboard Output:**
```json
{
  "benchmark": "humaneval",
  "generated_at": "2025-09-09T10:05:00.720520",
  "rankings": [
    {
      "model": "starcoder",
      "primary_score": 0.33,
      "all_scores": {"humaneval": {"pass@1": 0.33}},
      "config": {...}
    },
    {
      "model": "codeparrot-small", 
      "primary_score": 0.0,
      "all_scores": {"humaneval": {"pass@1": 0.0}},
      "config": {...}
    }
  ]
}
```

### **2. Model Comparisons**
- Side-by-side performance analysis
- Identifies strengths and weaknesses
- Configuration difference highlighting
- Statistical significance tracking

### **3. Time Series Analysis**
- Track model improvements over time
- Compare different training runs
- Monitor benchmark score evolution
- Regression detection

## 📋 **File Organization Logic**

### **Metrics Files** (`metrics_*.json`)
- **by_model/**: `{model_name}/{benchmark}_{date}_{original_filename}`
- **by_benchmark/**: `{benchmark}/{model_name}_{date}_{original_filename}`
- **by_date/**: `{date}/{original_filename}`

### **Generation Files** (`generations_*.json`)
- **by_date/**: `{date}/{original_filename}`
- Linked to corresponding metrics files by timestamp

### **Suite Files** (`suite_*.json`)
- **suites/**: `{suite_name}_{date}_{original_filename}`
- **by_date/**: `{date}/{original_filename}`

## 🔍 **Finding Results**

### **Find All Results for a Model**
```bash
ls results/organized/by_model/codeparrot_codeparrot-small/
```

### **Find All Results for a Benchmark**
```bash
ls results/organized/by_benchmark/humaneval/
```

### **Find Results from Specific Date**
```bash
ls results/organized/by_date/2025-09-09/
```

### **Find Recent Comparisons**
```bash
ls -lt results/comparisons/leaderboard_*.json
```

## ⚡ **Integration with Benchmark Runner**

The results organization happens automatically when you run benchmarks:

```bash
# This will automatically organize results
python3 run_benchmark.py --benchmark humaneval --model starcoder --limit 10

# Results are immediately available in organized structure
ls results/organized/by_model/starcoder/
```

## 📈 **Advanced Analysis Examples**

### **1. Track Model Performance Over Time**
```bash
# Find all results for a model
find results/organized/by_model/starcoder/ -name "*.json" | sort

# Compare configurations between runs
python3 src/results_organizer.py --compare starcoder starcoder --benchmark humaneval
```

### **2. Benchmark Comparison Analysis**
```bash
# Create leaderboards for multiple benchmarks
python3 src/results_organizer.py --leaderboard humaneval
python3 src/results_organizer.py --leaderboard mbpp

# Compare which models perform better on which benchmarks
```

### **3. Configuration Impact Analysis**
Look at the detailed config in leaderboard files to understand:
- Temperature settings impact
- Sample count vs. accuracy tradeoffs  
- Precision (fp16 vs fp32) effects
- Batch size performance implications

## 🔧 **Customization**

### **Add Custom Organization Rules**
Edit `src/results_organizer.py` to add:
- Custom file naming conventions
- Additional organization dimensions
- New comparison metrics
- Custom analysis functions

### **Export to Other Formats**
The JSON results can be easily converted:
```python
import json
import csv

# Convert leaderboard to CSV
with open('results/comparisons/leaderboard_humaneval_*.json') as f:
    data = json.load(f)
    
# Export rankings to CSV for spreadsheet analysis
```

## 📝 **File Naming Convention**

After organization, files follow clear naming patterns:

- **Original**: `metrics_1757429359.json` 
- **Organized**: `codeparrot_codeparrot-small_2025-09-09_metrics_1757429359.json`

This makes it easy to:
- ✅ Identify model and date at a glance
- ✅ Sort chronologically
- ✅ Group by model or benchmark
- ✅ Avoid filename conflicts

## 🎯 **Best Practices**

1. **Run Organization Regularly**
   ```bash
   python3 src/results_organizer.py --organize
   ```

2. **Generate Leaderboards After New Results**
   ```bash
   python3 src/results_organizer.py --leaderboard humaneval
   ```

3. **Use Descriptive Model Names**
   - Good: `starcoder-7b-instruct`
   - Poor: `model1`

4. **Archive Old Results**
   - Original files are moved to `archive/` directory
   - Organized structure keeps only clean, searchable copies

5. **Regular Comparison Reports**
   ```bash
   # Weekly model comparison
   python3 src/results_organizer.py --compare starcoder codeparrot gpt-4
   ```

---

## 🚀 **Quick Start Checklist**

- [ ] Run `python3 src/results_organizer.py --organize` to organize existing results
- [ ] Check `results/organized/` structure 
- [ ] Generate first leaderboard: `python3 src/results_organizer.py --leaderboard humaneval`
- [ ] Run new benchmark to see automatic organization
- [ ] Create model comparison report
- [ ] Set up regular organization in your workflow

**The organized results system transforms your benchmark data from a messy pile of timestamped files into a clean, searchable, comparable dataset that grows more valuable over time!**