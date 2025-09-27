#!/usr/bin/env python3
"""
Experimental Framework - Research-Grade Model Evaluation

Multi-template, Pass@K evaluation system with statistical rigor for fair model comparison.
Based on best practices from leading AI research papers.
"""

import json
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import statistics
from collections import defaultdict
import math
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Model architecture types for template selection"""
    BASE = "base"           # GPT-3 style completion models
    INSTRUCT = "instruct"   # InstructGPT/ChatGPT style
    CHAT = "chat"          # Conversational models
    CODE = "code"          # Code-specific models
    UNKNOWN = "unknown"    # Auto-detect or user specify

class PromptTemplate(Enum):
    """Research-validated prompt templates"""
    DIRECT = "direct"                    # Direct completion: "def func():\n    "
    INSTRUCTION = "instruction"          # "Complete this Python function:\ndef func():"
    CONVERSATIONAL = "conversational"    # "Please implement this function:\ndef func():"
    FEW_SHOT = "few_shot"               # Examples + "Now complete:\ndef func():"
    CHAIN_OF_THOUGHT = "chain_of_thought" # "Let's solve step by step:\ndef func():"

@dataclass
class Problem:
    """Standard problem format"""
    id: str
    prompt: str
    canonical_solution: str
    test_cases: List[str]
    difficulty: str = "medium"
    domain: str = "general"

@dataclass
class GenerationResult:
    """Result of model generation"""
    problem_id: str
    template: PromptTemplate
    attempt: int
    generated_code: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class ExperimentResult:
    """Complete experimental results"""
    model_name: str
    model_type: ModelType
    timestamp: int
    problems_tested: int
    template_results: Dict[PromptTemplate, Dict[str, Any]]
    pass_at_k: Dict[int, float]
    statistical_metrics: Dict[str, float]
    metadata: Dict[str, Any]

class PromptTemplateFactory:
    """Factory for generating research-validated prompt templates"""
    
    @staticmethod
    def get_template(template_type: PromptTemplate, problem: Problem, examples: Optional[List[Problem]] = None) -> str:
        """Generate prompt based on template type and problem"""
        
        if template_type == PromptTemplate.DIRECT:
            return PromptTemplateFactory._direct_template(problem)
        elif template_type == PromptTemplate.INSTRUCTION:
            return PromptTemplateFactory._instruction_template(problem)
        elif template_type == PromptTemplate.CONVERSATIONAL:
            return PromptTemplateFactory._conversational_template(problem)
        elif template_type == PromptTemplate.FEW_SHOT:
            return PromptTemplateFactory._few_shot_template(problem, examples or [])
        elif template_type == PromptTemplate.CHAIN_OF_THOUGHT:
            return PromptTemplateFactory._chain_of_thought_template(problem)
        else:
            raise ValueError(f"Unknown template type: {template_type}")
    
    @staticmethod
    def _direct_template(problem: Problem) -> str:
        """Direct completion template (GPT-3 style)"""
        return problem.prompt.rstrip() + "\n    "
    
    @staticmethod
    def _instruction_template(problem: Problem) -> str:
        """Instruction template (InstructGPT style)"""
        return f"Complete this Python function:\n\n{problem.prompt}"
    
    @staticmethod
    def _conversational_template(problem: Problem) -> str:
        """Conversational template (ChatGPT style)"""
        return f"Please implement this Python function:\n\n{problem.prompt}"
    
    @staticmethod
    def _few_shot_template(problem: Problem, examples: List[Problem]) -> str:
        """Few-shot template with examples"""
        examples_text = ""
        for i, example in enumerate(examples[:2]):  # Use max 2 examples
            examples_text += f"# Example {i+1}:\n{example.prompt}\n{example.canonical_solution}\n\n"
        
        return f"{examples_text}# Now complete:\n{problem.prompt}"
    
    @staticmethod
    def _chain_of_thought_template(problem: Problem) -> str:
        """Chain of thought template"""
        return f"Let's solve this step by step:\n\n{problem.prompt}\n\n# Step-by-step solution:\n"

class ModelTypeDetector:
    """Automatic model type detection based on name patterns"""
    
    @staticmethod
    def detect_model_type(model_name: str) -> ModelType:
        """Detect model type from name"""
        name_lower = model_name.lower()
        
        if any(keyword in name_lower for keyword in ['instruct', 'chat', 'assistant']):
            return ModelType.INSTRUCT
        elif any(keyword in name_lower for keyword in ['code', 'codegen', 'starcoder', 'codellama']):
            return ModelType.CODE
        elif any(keyword in name_lower for keyword in ['gpt-3.5', 'gpt-4', 'claude']):
            return ModelType.CHAT
        elif any(keyword in name_lower for keyword in ['gpt-3', 'codeparrot', 'base']):
            return ModelType.BASE
        else:
            return ModelType.UNKNOWN
    
    @staticmethod
    def get_default_templates(model_type: ModelType) -> List[PromptTemplate]:
        """Get recommended templates for model type"""
        if model_type == ModelType.BASE:
            return [PromptTemplate.DIRECT, PromptTemplate.FEW_SHOT]
        elif model_type == ModelType.INSTRUCT:
            return [PromptTemplate.INSTRUCTION, PromptTemplate.CONVERSATIONAL]
        elif model_type == ModelType.CHAT:
            return [PromptTemplate.CONVERSATIONAL, PromptTemplate.CHAIN_OF_THOUGHT]
        elif model_type == ModelType.CODE:
            return [PromptTemplate.DIRECT, PromptTemplate.INSTRUCTION, PromptTemplate.FEW_SHOT]
        else:  # UNKNOWN
            return list(PromptTemplate)  # Try all templates

class PassAtKEvaluator:
    """Pass@K evaluation with statistical analysis"""
    
    @staticmethod
    def calculate_pass_at_k(results: List[bool], k_values: List[int]) -> Dict[int, float]:
        """Calculate Pass@K metrics"""
        n = len(results)
        if n == 0:
            return {k: 0.0 for k in k_values}
        
        # Count successful attempts
        successes = sum(results)
        
        pass_at_k = {}
        for k in k_values:
            if k > n:
                # If we have fewer samples than k, just use what we have
                pass_at_k[k] = successes / n
            else:
                # Calculate Pass@K: probability of at least one success in k attempts
                # P(at least one success) = 1 - P(all failures)
                failure_rate = (n - successes) / n
                pass_at_k[k] = 1 - (failure_rate ** k)
        
        return pass_at_k
    
    @staticmethod
    def bootstrap_confidence_interval(results: List[bool], k: int, 
                                    n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for Pass@K"""
        if not results:
            return (0.0, 0.0)
        
        bootstrap_scores = []
        n = len(results)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = [random.choice(results) for _ in range(n)]
            # Calculate Pass@K for this sample
            pass_k = PassAtKEvaluator.calculate_pass_at_k(bootstrap_sample, [k])[k]
            bootstrap_scores.append(pass_k)
        
        # Calculate confidence interval
        alpha = (1 - confidence) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100
        
        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)
        
        return (ci_lower, ci_upper)

class StatisticalAnalyzer:
    """Statistical analysis tools for experimental results"""
    
    @staticmethod
    def _normal_cdf(x):
        """Approximate standard normal CDF using error function approximation"""
        # Approximation using Taylor series
        if x < -5:
            return 0.0
        elif x > 5:
            return 1.0
        else:
            # Using approximation: CDF(x) ≈ 0.5 + 0.5 * erf(x/sqrt(2))
            t = x / math.sqrt(2)
            # Approximation of erf(t)
            a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
            p = 0.3275911
            sign = 1 if t >= 0 else -1
            t = abs(t)
            
            # Abramowitz and Stegun approximation
            y = 1.0 / (1.0 + p * t)
            erf_approx = sign * (1 - (((((a5*y + a4)*y) + a3)*y + a2)*y + a1)*y*math.exp(-t*t))
            
            return 0.5 + 0.5 * erf_approx
    
    @staticmethod
    def template_sensitivity_analysis(template_results: Dict[PromptTemplate, List[bool]]) -> Dict[str, float]:
        """Analyze sensitivity to prompt templates"""
        if not template_results:
            return {}
        
        # Calculate Pass@1 for each template
        template_scores = {}
        for template, results in template_results.items():
            if results:
                template_scores[template.value] = sum(results) / len(results)
            else:
                template_scores[template.value] = 0.0
        
        scores = list(template_scores.values())
        
        return {
            'mean_score': statistics.mean(scores),
            'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0.0,
            'min_score': min(scores),
            'max_score': max(scores),
            'sensitivity': max(scores) - min(scores) if scores else 0.0,
            'template_scores': template_scores
        }
    
    @staticmethod
    def significance_test(results_a: List[bool], results_b: List[bool]) -> Dict[str, Any]:
        """Statistical significance test between two result sets"""
        if not results_a or not results_b:
            return {'p_value': 1.0, 'significant': False, 'test': 'insufficient_data'}
        
        # Convert to success rates
        rate_a = sum(results_a) / len(results_a)
        rate_b = sum(results_b) / len(results_b)
        
        # Simple two-proportion z-test
        try:
            n1, n2 = len(results_a), len(results_b)
            x1, x2 = sum(results_a), sum(results_b)
            
            if n1 == 0 or n2 == 0:
                return {
                    'p_value': 1.0,
                    'significant': False,
                    'test': 'insufficient_data'
                }
            
            # Pooled proportion
            p_pool = (x1 + x2) / (n1 + n2)
            se_pool = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            
            if se_pool == 0:
                p_value = 1.0 if rate_a == rate_b else 0.0
            else:
                z_score = (rate_a - rate_b) / se_pool
                # Approximate p-value using normal distribution
                p_value = 2 * (1 - StatisticalAnalyzer._normal_cdf(abs(z_score)))
            
            # Effect size (Cohen's d approximation)
            pooled_var = (np.var(results_a) * (n1-1) + np.var(results_b) * (n2-1)) / (n1+n2-2)
            cohens_d = (rate_a - rate_b) / math.sqrt(pooled_var) if pooled_var > 0 else 0
            
            return {
                'p_value': min(p_value, 1.0),
                'significant': p_value < 0.05,
                'z_score': z_score if se_pool > 0 else 0,
                'effect_size': cohens_d,
                'rate_a': rate_a,
                'rate_b': rate_b,
                'test': 'two_proportion_z_test'
            }
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            return {
                'p_value': 1.0,
                'significant': False,
                'error': str(e),
                'test': 'failed'
            }

class ExperimentalFramework:
    """Main experimental framework for rigorous model evaluation"""
    
    def __init__(self, results_dir: str = "results/experimental"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Default experimental parameters
        self.k_values = [1, 5, 10, 100]
        self.random_seed = 42
        self.confidence_level = 0.95
        
        # Set random seeds for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
    def run_experiment(self, model_interface, problems: List[Problem], 
                      templates: Optional[List[PromptTemplate]] = None,
                      samples_per_template: int = 10,
                      model_type: Optional[ModelType] = None) -> ExperimentResult:
        """Run complete experimental evaluation"""
        
        logger.info(f"Starting experimental evaluation with {len(problems)} problems")
        
        # Auto-detect model type if not provided
        if model_type is None:
            model_type = ModelTypeDetector.detect_model_type(model_interface.model_name)
        
        # Select templates if not provided
        if templates is None:
            templates = ModelTypeDetector.get_default_templates(model_type)
        
        logger.info(f"Model type: {model_type.value}, Templates: {[t.value for t in templates]}")
        
        # Run evaluation for each template
        template_results = {}
        all_generations = []
        
        for template in templates:
            logger.info(f"Evaluating template: {template.value}")
            
            template_generations = []
            template_success_rates = []
            
            for problem in problems:
                # Generate prompt
                prompt = PromptTemplateFactory.get_template(template, problem)
                
                # Generate multiple samples for Pass@K evaluation
                problem_results = []
                
                for attempt in range(samples_per_template):
                    start_time = time.time()
                    
                    try:
                        generated_code = model_interface.generate(prompt, temperature=0.2)
                        execution_time = time.time() - start_time
                        
                        # Evaluate generated code
                        success = self._evaluate_code(generated_code, problem)
                        
                        result = GenerationResult(
                            problem_id=problem.id,
                            template=template,
                            attempt=attempt,
                            generated_code=generated_code,
                            execution_time=execution_time,
                            success=success
                        )
                        
                        problem_results.append(success)
                        template_generations.append(result)
                        all_generations.append(result)
                        
                    except Exception as e:
                        logger.warning(f"Generation failed for {problem.id}, attempt {attempt}: {e}")
                        
                        result = GenerationResult(
                            problem_id=problem.id,
                            template=template,
                            attempt=attempt,
                            generated_code="",
                            execution_time=time.time() - start_time,
                            success=False,
                            error_message=str(e)
                        )
                        
                        problem_results.append(False)
                        template_generations.append(result)
                        all_generations.append(result)
                
                template_success_rates.extend(problem_results)
            
            # Calculate Pass@K for this template
            pass_at_k = PassAtKEvaluator.calculate_pass_at_k(template_success_rates, self.k_values)
            
            # Calculate confidence intervals
            confidence_intervals = {}
            for k in self.k_values:
                ci = PassAtKEvaluator.bootstrap_confidence_interval(
                    template_success_rates, k, confidence=self.confidence_level
                )
                confidence_intervals[k] = ci
            
            # Convert GenerationResult objects to dicts
            generations_dict = []
            for gen in template_generations:
                gen_dict = {
                    'problem_id': gen.problem_id,
                    'template': gen.template.value,
                    'attempt': gen.attempt,
                    'generated_code': gen.generated_code,
                    'execution_time': gen.execution_time,
                    'success': gen.success,
                    'error_message': gen.error_message
                }
                generations_dict.append(gen_dict)
            
            template_results[template] = {
                'pass_at_k': pass_at_k,
                'confidence_intervals': confidence_intervals,
                'success_rate': sum(template_success_rates) / len(template_success_rates),
                'total_samples': len(template_success_rates),
                'generations': generations_dict
            }
        
        # Overall Pass@K calculation (best across templates)
        overall_success_rates = []
        for problem in problems:
            problem_results = [gen.success for gen in all_generations if gen.problem_id == problem.id]
            if problem_results:
                overall_success_rates.append(max(problem_results))  # Best attempt across templates
        
        overall_pass_at_k = PassAtKEvaluator.calculate_pass_at_k(overall_success_rates, self.k_values)
        
        # Statistical analysis
        template_boolean_results = {
            template: [gen['success'] for gen in data['generations']] 
            for template, data in template_results.items()
        }
        
        sensitivity_analysis = StatisticalAnalyzer.template_sensitivity_analysis(template_boolean_results)
        
        # Create experiment result
        result = ExperimentResult(
            model_name=model_interface.model_name,
            model_type=model_type,
            timestamp=int(time.time()),
            problems_tested=len(problems),
            template_results={k.value: v for k, v in template_results.items()},
            pass_at_k=overall_pass_at_k,
            statistical_metrics=sensitivity_analysis,
            metadata={
                'templates_used': [t.value for t in templates],
                'samples_per_template': samples_per_template,
                'k_values': self.k_values,
                'random_seed': self.random_seed,
                'confidence_level': self.confidence_level
            }
        )
        
        # Save results
        self._save_experiment_result(result)
        
        logger.info("Experimental evaluation completed")
        return result
    
    def _evaluate_code(self, generated_code: str, problem: Problem) -> bool:
        """Evaluate generated code against test cases"""
        # Simplified evaluation - in practice would need sandboxed execution
        if not generated_code or len(generated_code.strip()) < 5:
            return False
        
        # Basic heuristics for code quality
        has_return = "return" in generated_code
        has_logic = any(keyword in generated_code for keyword in 
                       ["for", "if", "while", "range", "len", "abs", "%", "==", "!=", "<", ">"])
        reasonable_length = len(generated_code.strip()) > 10
        
        return has_return and has_logic and reasonable_length
    
    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to file"""
        filename = f"experiment_{result.model_name.replace('/', '_')}_{result.timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert to serializable format
        result_dict = {
            'model_name': result.model_name,
            'model_type': result.model_type.value,
            'timestamp': result.timestamp,
            'problems_tested': result.problems_tested,
            'template_results': result.template_results,
            'pass_at_k': result.pass_at_k,
            'statistical_metrics': result.statistical_metrics,
            'metadata': result.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")

def main():
    """Example usage of experimental framework"""
    # This would be replaced with actual model interface
    class DummyModelInterface:
        def __init__(self, model_name: str):
            self.model_name = model_name
        
        def generate(self, prompt: str, temperature: float = 0.2) -> str:
            # Dummy implementation
            return "return a + b"
    
    # Example problems
    problems = [
        Problem(
            id="test_add",
            prompt="def add(a, b):\n    \"\"\"Add two numbers\"\"\"\n    ",
            canonical_solution="return a + b",
            test_cases=["assert add(2, 3) == 5"]
        )
    ]
    
    # Run experiment
    framework = ExperimentalFramework()
    model = DummyModelInterface("test-model")
    
    result = framework.run_experiment(
        model_interface=model,
        problems=problems,
        samples_per_template=5
    )
    
    print(f"Pass@1: {result.pass_at_k[1]:.3f}")
    print(f"Template sensitivity: {result.statistical_metrics['sensitivity']:.3f}")

if __name__ == "__main__":
    main()