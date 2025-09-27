#!/usr/bin/env python3
"""
AI Benchmark Suite - Main Benchmark Runner

Orchestrates benchmark execution across multiple models and frameworks.
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from results_organizer import ResultsOrganizer

class BenchmarkRunner:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.results_dir = Path("results")
        self.logs_dir = Path("logs")
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load configurations
        self.models = self.load_yaml(self.config_dir / "models.yaml")
        self.benchmarks = self.load_yaml(self.config_dir / "benchmarks.yaml")
        
        # BigCode harness path
        self.bigcode_dir = Path("bigcode-evaluation-harness")
        
        # Results organizer
        self.results_organizer = ResultsOrganizer()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.logs_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_yaml(self, file_path: Path) -> Dict:
        """Load YAML configuration file"""
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {file_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML file {file_path}: {e}")
            sys.exit(1)
            
    def get_model_config(self, model_name: str) -> Dict:
        """Get model configuration by name"""
        # Check API models
        if model_name in self.models.get('api_models', {}):
            return self.models['api_models'][model_name]
        
        # Check local models
        if model_name in self.models.get('local_models', {}):
            return self.models['local_models'][model_name]
            
        raise ValueError(f"Model '{model_name}' not found in configuration")
        
    def get_benchmark_config(self, benchmark_name: str) -> Dict:
        """Get benchmark configuration by name"""
        if benchmark_name not in self.benchmarks.get('benchmarks', {}):
            raise ValueError(f"Benchmark '{benchmark_name}' not found in configuration")
        return self.benchmarks['benchmarks'][benchmark_name]
        
    def build_bigcode_command(self, model_config: Dict, benchmark_config: Dict, 
                            limit: Optional[int] = None, n_samples: Optional[int] = None) -> List[str]:
        """Build BigCode evaluation harness command"""
        
        # Get python executable from virtual environment
        venv_python = Path.cwd() / self.bigcode_dir / "venv" / "bin" / "python"
        
        # Generate absolute paths for result files
        timestamp = int(time.time())
        results_dir = Path.cwd() / "results"
        metrics_path = results_dir / f"metrics_{timestamp}.json"
        generations_path = results_dir / f"generations_{timestamp}.json"
        
        # Base command
        cmd = [
            str(venv_python), "main.py",
            "--model", model_config['model_id'],
            "--tasks", benchmark_config['task'],
            "--temperature", str(model_config.get('temperature', 0.2)),
            "--save_generations",
            "--metric_output_path", str(metrics_path),
            "--save_generations_path", str(generations_path)
        ]
        
        # Add model-specific options
        if model_config.get('precision'):
            cmd.extend(["--precision", model_config['precision']])
            
        if model_config.get('batch_size'):
            cmd.extend(["--batch_size", str(model_config['batch_size'])])
            
        if model_config.get('trust_remote_code'):
            cmd.append("--trust_remote_code")
            
        # Add benchmark-specific options
        if benchmark_config.get('allow_code_execution', False):
            cmd.append("--allow_code_execution")
            
        # Override with run-specific parameters
        if limit:
            cmd.extend(["--limit", str(limit)])
            
        if n_samples:
            cmd.extend(["--n_samples", str(n_samples)])
        elif benchmark_config.get('n_samples'):
            cmd.extend(["--n_samples", str(benchmark_config['n_samples'])])
            
        return cmd
        
    def run_single_benchmark(self, model_name: str, benchmark_name: str,
                           limit: Optional[int] = None, n_samples: Optional[int] = None) -> Dict:
        """Run a single benchmark on a single model"""
        
        start_time = time.time()
        self.logger.info(f"Starting benchmark: {model_name} on {benchmark_name}")
        
        try:
            model_config = self.get_model_config(model_name)
            benchmark_config = self.get_benchmark_config(benchmark_name)
            
            # Set environment variables for API models
            if model_config.get('api_key_env'):
                api_key = os.getenv(model_config['api_key_env'])
                if not api_key:
                    raise ValueError(f"API key not found in environment: {model_config['api_key_env']}")
            
            # Build command
            cmd = self.build_bigcode_command(model_config, benchmark_config, limit, n_samples)
            
            self.logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run benchmark
            result = subprocess.run(
                cmd,
                cwd=self.bigcode_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Process results
            benchmark_result = {
                'model': model_name,
                'benchmark': benchmark_name,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'duration_seconds': duration,
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd)
            }
            
            if result.returncode == 0:
                self.logger.info(f"✅ Completed: {model_name} on {benchmark_name} ({duration:.1f}s)")
            else:
                self.logger.error(f"❌ Failed: {model_name} on {benchmark_name}")
                self.logger.error(f"Error: {result.stderr}")
                
            return benchmark_result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.error(f"❌ Exception in {model_name} on {benchmark_name}: {str(e)}")
            
            return {
                'model': model_name,
                'benchmark': benchmark_name,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'end_time': datetime.fromtimestamp(end_time).isoformat(),
                'duration_seconds': duration,
                'success': False,
                'error': str(e),
                'command': 'N/A'
            }
            
    def run_benchmark_suite(self, suite_name: str, models: List[str]) -> Dict:
        """Run a complete benchmark suite on multiple models"""
        
        if suite_name not in self.benchmarks.get('suites', {}):
            raise ValueError(f"Suite '{suite_name}' not found in configuration")
            
        suite_config = self.benchmarks['suites'][suite_name]
        
        self.logger.info(f"🚀 Starting benchmark suite: {suite_name}")
        self.logger.info(f"Models: {', '.join(models)}")
        self.logger.info(f"Description: {suite_config['description']}")
        
        start_time = time.time()
        results = {
            'suite': suite_name,
            'models': models,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'benchmarks': []
        }
        
        # Run each benchmark
        for benchmark_spec in suite_config['benchmarks']:
            benchmark_name = benchmark_spec['name']
            limit = benchmark_spec.get('limit')
            n_samples = benchmark_spec.get('n_samples')
            
            benchmark_results = []
            
            # Run on each model
            for model_name in models:
                result = self.run_single_benchmark(model_name, benchmark_name, limit, n_samples)
                benchmark_results.append(result)
                
            results['benchmarks'].append({
                'name': benchmark_name,
                'results': benchmark_results
            })
            
        end_time = time.time()
        results['end_time'] = datetime.fromtimestamp(end_time).isoformat()
        results['duration_seconds'] = end_time - start_time
        
        # Save results
        results_file = self.results_dir / f"suite_{suite_name}_{int(start_time)}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"🎉 Suite completed: {suite_name} ({(end_time - start_time)/60:.1f} minutes)")
        self.logger.info(f"Results saved to: {results_file}")
        
        return results
        
    def list_available_items(self):
        """List available models, benchmarks, and suites"""
        print("\n📊 AI Benchmark Suite - Available Items\n")
        
        # Models
        print("🤖 Available Models:")
        api_models = list(self.models.get('api_models', {}).keys())
        local_models = list(self.models.get('local_models', {}).keys())
        
        if api_models:
            print(f"  API Models: {', '.join(api_models)}")
        if local_models:
            print(f"  Local Models: {', '.join(local_models)}")
            
        # Model groups
        groups = self.models.get('groups', {})
        if groups:
            print(f"  Model Groups: {', '.join(groups.keys())}")
            
        # Benchmarks
        benchmarks = list(self.benchmarks.get('benchmarks', {}).keys())
        if benchmarks:
            print(f"\n🎯 Available Benchmarks: {', '.join(benchmarks)}")
            
        # Suites
        suites = list(self.benchmarks.get('suites', {}).keys())
        if suites:
            print(f"\n📋 Available Suites: {', '.join(suites)}")
            for suite_name, suite_config in self.benchmarks['suites'].items():
                print(f"  {suite_name}: {suite_config['description']}")

def main():
    parser = argparse.ArgumentParser(description="AI Benchmark Suite - Automated Code Generation Evaluation")
    
    # Primary actions
    parser.add_argument('--list', action='store_true', help='List available models, benchmarks, and suites')
    parser.add_argument('--suite', type=str, help='Run a benchmark suite')
    parser.add_argument('--benchmark', type=str, help='Run a single benchmark')
    
    # Model selection
    parser.add_argument('--model', type=str, help='Single model to run')
    parser.add_argument('--models', type=str, nargs='+', help='Multiple models to run')
    parser.add_argument('--group', type=str, help='Model group to run')
    
    # Benchmark options
    parser.add_argument('--limit', type=int, help='Limit number of samples')
    parser.add_argument('--n-samples', type=int, help='Number of generations per sample')
    
    # Configuration
    parser.add_argument('--config-dir', type=str, default='config', help='Configuration directory')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = BenchmarkRunner(args.config_dir)
    
    # Handle list command
    if args.list:
        runner.list_available_items()
        return
        
    # Determine models to run
    models = []
    if args.model:
        models = [args.model]
    elif args.models:
        models = args.models
    elif args.group:
        if args.group not in runner.models.get('groups', {}):
            print(f"❌ Model group '{args.group}' not found")
            sys.exit(1)
        models = runner.models['groups'][args.group]
    else:
        print("❌ Must specify --model, --models, or --group")
        sys.exit(1)
        
    # Run suite or single benchmark
    if args.suite:
        runner.run_benchmark_suite(args.suite, models)
    elif args.benchmark:
        for model in models:
            runner.run_single_benchmark(model, args.benchmark, args.limit, args.n_samples)
    else:
        print("❌ Must specify --suite or --benchmark")
        sys.exit(1)

if __name__ == "__main__":
    main()