#!/usr/bin/env python3
"""
Real Ollama Experimental Adapter
Uses proper BigCode evaluation with sandboxed execution
"""

import requests
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add project path
sys.path.append(str(Path(__file__).parent))

from real_experimental_framework import RealExperimentalFramework, ModelType

logger = logging.getLogger(__name__)

class OllamaModelInterface:
    """Real interface to Ollama models"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Verify model is available
        self._verify_model()
    
    def _verify_model(self):
        """Verify model is available in Ollama"""
        try:
            models_url = f"{self.base_url}/api/tags"
            response = requests.get(models_url, timeout=10)
            response.raise_for_status()
            
            available_models = [model['name'] for model in response.json()['models']]
            if self.model_name not in available_models:
                raise ValueError(f"Model {self.model_name} not found. Available: {available_models}")
                
            logger.info(f"✅ Model {self.model_name} verified and ready")
            
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}: {e}")
    
    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        """Generate code completion from model"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
                # Remove ALL stop sequences - let the model complete naturally
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            if not generated_text:
                logger.warning(f"Empty response from {self.model_name}")
                return ""
            
            return generated_text
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise

def run_real_ollama_experiment(model_name: str, num_problems: int = 3, 
                              samples_per_template: int = 5) -> Dict[str, Any]:
    """Run REAL experimental evaluation on Ollama model"""
    
    logger.info(f"🧪 Starting REAL experimental evaluation: {model_name}")
    logger.info(f"📝 Testing {num_problems} problems with {samples_per_template} samples per template")
    
    # Create model interface
    model_interface = OllamaModelInterface(model_name)
    
    # Create real experimental framework
    framework = RealExperimentalFramework()
    
    # Run experiment with real execution
    result = framework.run_experiment(
        model_interface=model_interface,
        num_problems=num_problems,
        samples_per_template=samples_per_template
    )
    
    # Print REAL results
    print(f"\n🎯 REAL Results for {model_name}:")
    print(f"   Model Type: {result.model_type}")
    print(f"   Problems Tested: {result.problems_tested}")
    
    print(f"\n📊 REAL Pass@K Metrics:")
    for k, score in result.pass_at_k.items():
        print(f"   Pass@{k}: {score:.3f} ({score*100:.1f}%)")
    
    print(f"\n🎯 Template Analysis:")
    print(f"   Sensitivity: {result.statistical_metrics['sensitivity']:.3f}")
    print(f"   Mean Score: {result.statistical_metrics['mean_score']:.3f}")
    print(f"   Std Dev: {result.statistical_metrics['std_dev']:.3f}")
    
    print(f"\n📋 Template Scores (REAL execution results):")
    for template, score in result.statistical_metrics['template_scores'].items():
        print(f"   {template}: {score:.3f} ({score*100:.1f}%)")
    
    # Show some actual generations
    print(f"\n🔍 Sample Generations:")
    for template_name, template_data in result.template_results.items():
        generations = template_data['generations']
        if generations:
            sample = generations[0]  # First generation
            print(f"\n  Template: {template_name}")
            print(f"  Task: {sample['task_id']}")
            print(f"  Result: {'✅ PASS' if sample['passed'] else '❌ FAIL'}")
            print(f"  Generated: {sample['generated_text'][:100]}...")
            print(f"  Extracted Code: {sample['extracted_code'][:100]}...")
            if sample['error_message']:
                print(f"  Error: {sample['error_message']}")
    
    return result

def compare_real_ollama_models(model_names: List[str], num_problems: int = 3,
                              samples_per_template: int = 5) -> Dict[str, Any]:
    """Compare multiple Ollama models with REAL evaluation"""
    print(f"🔬 REAL COMPARISON: {len(model_names)} models on {num_problems} problems")
    print("Using REAL BigCode evaluation with sandboxed execution")
    
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        try:
            result = run_real_ollama_experiment(
                model_name, 
                num_problems, 
                samples_per_template
            )
            results[model_name] = result
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
    
    # REAL comparison summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("📊 REAL COMPARISON SUMMARY")
        print("(Based on actual code execution, not string matching)")
        print(f"{'='*60}")
        
        print(f"\n🏆 REAL Pass@1 Rankings:")
        pass1_scores = [(name, result.pass_at_k[1]) for name, result in results.items()]
        pass1_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, score) in enumerate(pass1_scores, 1):
            print(f"   {i}. {name}: {score:.3f} ({score*100:.1f}%)")
        
        print(f"\n🎯 Template Sensitivity (lower = more consistent):")
        for name, result in results.items():
            sensitivity = result.statistical_metrics['sensitivity']
            print(f"   {name}: {sensitivity:.3f}")
        
        # Show actual execution results
        print(f"\n⚡ Execution Success Rates by Template:")
        all_templates = set()
        for result in results.values():
            all_templates.update(result.statistical_metrics['template_scores'].keys())
        
        for template in sorted(all_templates):
            print(f"\n  {template}:")
            for name, result in results.items():
                score = result.statistical_metrics['template_scores'].get(template, 0.0)
                print(f"    {name}: {score:.3f} ({score*100:.1f}%)")
    
    return results

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="REAL experimental evaluation of Ollama models")
    parser.add_argument('--model', help='Single model to evaluate')
    parser.add_argument('--compare', nargs='+', help='Multiple models to compare')
    parser.add_argument('--problems', type=int, default=3, help='Number of problems to test')
    parser.add_argument('--samples', type=int, default=5, help='Samples per template')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🔥 REAL Ollama Experimental Evaluation")
    print("Using BigCode evaluation harness with sandboxed execution")
    print("This will show ACTUAL code generation performance\\n")
    
    if args.model:
        run_real_ollama_experiment(args.model, args.problems, args.samples)
    elif args.compare:
        compare_real_ollama_models(args.compare, args.problems, args.samples)
    else:
        print("Please specify --model or --compare")

if __name__ == "__main__":
    main()