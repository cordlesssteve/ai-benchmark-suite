#!/usr/bin/env python3
"""
Real Experimental Framework for Code Generation Benchmarking
Uses actual BigCode evaluation harness with sandboxed execution
"""

import sys
import os
import time
import json
import re
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Add BigCode path
bigcode_path = Path(__file__).parent.parent / "bigcode-evaluation-harness"
sys.path.insert(0, str(bigcode_path))

from datasets import load_dataset
from bigcode_eval.tasks.custom_metrics.execute import check_correctness

logger = logging.getLogger(__name__)

@dataclass
class RealGenerationResult:
    """Result from a single code generation attempt"""
    task_id: str
    template: str
    attempt: int
    prompt: str
    generated_text: str
    extracted_code: str
    execution_time: float
    passed: bool
    execution_result: str
    error_message: Optional[str] = None

@dataclass 
class RealExperimentResult:
    """Complete experimental evaluation result"""
    model_name: str
    model_type: str
    timestamp: int
    problems_tested: int
    template_results: Dict[str, Dict]
    pass_at_k: Dict[int, float]
    statistical_metrics: Dict[str, Any]
    metadata: Dict[str, Any]

class PromptTemplate(Enum):
    """Research-validated prompt templates"""
    DIRECT = "direct"
    INSTRUCTION = "instruction" 
    CONVERSATIONAL = "conversational"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"

class ModelType(Enum):
    """Model type classification for template selection"""
    CODE = "code"
    CHAT = "chat"
    UNKNOWN = "unknown"

class CodeExtractor:
    """Extract executable code from model responses"""
    
    @staticmethod
    def extract_python_code(text: str) -> str:
        """Extract Python code from mixed text/code responses"""
        
        # First try to find complete code blocks
        code_block_patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```', 
            r'```python(.*?)```',
            r'```(.*?)```'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                code = matches[0].strip()
                if CodeExtractor._is_executable_code(code):
                    return code
        
        # Handle incomplete code blocks (common with truncated responses)
        incomplete_patterns = [
            r'```python\n(.*)',  # Starts with ```python but doesn't close
            r'```\n(.*)',        # Starts with ``` but doesn't close
        ]
        
        for pattern in incomplete_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                code = matches[0].strip()
                # Clean up incomplete blocks
                if code.endswith('```'):
                    code = code[:-3].strip()
                if CodeExtractor._is_executable_code(code):
                    return code
        
        # Try to extract function definitions
        function_patterns = [
            r'(def\s+\w+.*?(?=\ndef|\nclass|\Z))',  # Function until next def/class
            r'(def\s+.*?return\s+[^;\n]+)',        # Function with return statement
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            if matches:
                code = matches[0].strip()
                if CodeExtractor._is_executable_code(code):
                    return code
        
        # Advanced extraction: look for indented code blocks after explanatory text
        lines = text.split('\n')
        code_lines = []
        found_def = False
        current_indent = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Skip markdown or explanatory lines
            if any(marker in line for marker in ['**', '##', '1.', '2.', '3.', '4.', '-']):
                continue
                
            # Found function definition
            if stripped.startswith('def '):
                found_def = True
                code_lines = [line]
                current_indent = len(line) - len(line.lstrip())
                continue
            
            # If we found a def, collect indented lines
            if found_def:
                if line.strip() == '':
                    code_lines.append(line)  # Keep empty lines
                elif line.startswith(' ' * (current_indent + 4)):  # Properly indented
                    code_lines.append(line)
                elif line.startswith(' ' * current_indent) and stripped:  # Same level as def
                    code_lines.append(line)
                elif not line.startswith(' ') and stripped:  # Top level, stop
                    break
        
        if code_lines and found_def:
            code = '\n'.join(code_lines)
            if CodeExtractor._is_executable_code(code):
                return code
        
        # Return original text if no code found (will likely fail execution)
        return text.strip()
    
    @staticmethod
    def _is_executable_code(code: str) -> bool:
        """Check if code looks executable"""
        code = code.strip()
        
        # Must have some basic code structure
        if not code:
            return False
            
        # Must have function definition for HumanEval
        if 'def ' not in code:
            return False
            
        # Must have return statement
        if 'return' not in code:
            return False
            
        # Should not be mostly explanatory text
        code_indicators = ['def ', '    ', 'return', 'if ', 'for ', 'while ', '=']
        text_indicators = ['to solve', 'the function', 'we need', 'here is', 'this will']
        
        code_score = sum(1 for indicator in code_indicators if indicator in code)
        text_score = sum(1 for indicator in text_indicators if indicator in code.lower())
        
        return code_score > text_score

class PromptTemplateFactory:
    """Generate different prompt templates for experiments"""
    
    @staticmethod
    def get_template(template_type: PromptTemplate, problem_data: Dict) -> str:
        """Generate prompt based on template type"""
        
        base_prompt = problem_data['prompt']
        
        if template_type == PromptTemplate.DIRECT:
            return base_prompt
            
        elif template_type == PromptTemplate.INSTRUCTION:
            return f"Please complete the following Python function. Only return the code, no explanation:\n\n{base_prompt}"
            
        elif template_type == PromptTemplate.CONVERSATIONAL:
            return f"I need help completing this Python function:\n\n{base_prompt}\n\nCould you complete it for me?"
            
        elif template_type == PromptTemplate.FEW_SHOT:
            # Add a simple example
            example = """Example:
def add_numbers(a, b):
    return a + b

Now complete this function:

"""
            return example + base_prompt
            
        elif template_type == PromptTemplate.CHAIN_OF_THOUGHT:
            return f"Let me think step by step about this function:\n\n{base_prompt}\n\nFirst, I need to understand what this function should do, then implement it:"
            
        else:
            return base_prompt

class RealExperimentalFramework:
    """Real experimental framework with proper evaluation"""
    
    def __init__(self, results_dir: str = "results/real_experimental"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load real HumanEval dataset
        logger.info("Loading HumanEval dataset...")
        self.dataset = load_dataset("openai_humaneval")['test']
        
        # Experimental parameters
        self.k_values = [1, 5, 10, 100]
        self.timeout = 3.0  # Match BigCode timeout
        self.random_seed = 42
        
        random.seed(self.random_seed)
        
    def run_experiment(self, model_interface, num_problems: int = 3, 
                      samples_per_template: int = 10,
                      model_type: Optional[ModelType] = None) -> RealExperimentResult:
        """Run complete experimental evaluation with real execution"""
        
        logger.info(f"Starting REAL experimental evaluation with {num_problems} problems")
        
        # Select subset of problems
        problems = list(self.dataset.select(range(num_problems)))
        
        # Auto-detect model type
        if model_type is None:
            model_type = self._detect_model_type(model_interface.model_name)
        
        # Select templates
        templates = self._get_templates_for_model_type(model_type)
        logger.info(f"Model type: {model_type.value}, Templates: {[t.value for t in templates]}")
        
        # Run evaluation for each template
        template_results = {}
        all_generations = []
        
        for template in templates:
            logger.info(f"Evaluating template: {template.value}")
            
            template_generations = []
            template_passes = []
            
            for i, problem in enumerate(problems):
                logger.info(f"Problem {i+1}/{len(problems)}: {problem['task_id']}")
                
                # Generate prompt
                prompt = PromptTemplateFactory.get_template(template, problem)
                
                # Generate multiple samples for Pass@K
                problem_passes = []
                
                for attempt in range(samples_per_template):
                    start_time = time.time()
                    
                    try:
                        # Generate code
                        generated_text = model_interface.generate(prompt)
                        
                        # Extract actual code
                        extracted_code = CodeExtractor.extract_python_code(generated_text)
                        
                        execution_time = time.time() - start_time
                        
                        # Execute with real test cases
                        passed, execution_result, error_msg = self._execute_code(
                            extracted_code, problem
                        )
                        
                        result = RealGenerationResult(
                            task_id=problem['task_id'],
                            template=template.value,
                            attempt=attempt,
                            prompt=prompt,
                            generated_text=generated_text,
                            extracted_code=extracted_code,
                            execution_time=execution_time,
                            passed=passed,
                            execution_result=execution_result,
                            error_message=error_msg
                        )
                        
                        template_generations.append(result)
                        problem_passes.append(passed)
                        
                        logger.debug(f"  Attempt {attempt}: {'PASS' if passed else 'FAIL'}")
                        
                    except Exception as e:
                        logger.error(f"Generation error: {e}")
                        result = RealGenerationResult(
                            task_id=problem['task_id'],
                            template=template.value,
                            attempt=attempt,
                            prompt=prompt,
                            generated_text="",
                            extracted_code="",
                            execution_time=0.0,
                            passed=False,
                            execution_result="generation_error",
                            error_message=str(e)
                        )
                        template_generations.append(result)
                        problem_passes.append(False)
                
                template_passes.extend(problem_passes)
            
            # Calculate Pass@K for this template
            pass_at_k = self._calculate_pass_at_k(template_passes, self.k_values)
            
            all_generations.extend(template_generations)
            
            template_results[template.value] = {
                'pass_at_k': pass_at_k,
                'success_rate': sum(template_passes) / len(template_passes) if template_passes else 0.0,
                'total_samples': len(template_passes),
                'generations': [self._generation_to_dict(gen) for gen in template_generations]
            }
            
            logger.info(f"Template {template.value} Pass@1: {pass_at_k[1]:.3f}")
        
        # Overall statistics
        overall_passes = []
        for problem in problems:
            problem_results = [gen.passed for gen in all_generations if gen.task_id == problem['task_id']]
            if problem_results:
                overall_passes.append(max(problem_results))  # Best across templates
        
        overall_pass_at_k = self._calculate_pass_at_k(overall_passes, self.k_values)
        
        # Statistical analysis
        statistical_metrics = self._analyze_template_performance(template_results)
        
        # Create result
        result = RealExperimentResult(
            model_name=model_interface.model_name,
            model_type=model_type.value,
            timestamp=int(time.time()),
            problems_tested=len(problems),
            template_results=template_results,
            pass_at_k=overall_pass_at_k,
            statistical_metrics=statistical_metrics,
            metadata={
                'templates_used': [t.value for t in templates],
                'samples_per_template': samples_per_template,
                'k_values': self.k_values,
                'timeout': self.timeout,
                'random_seed': self.random_seed
            }
        )
        
        # Save results
        self._save_result(result)
        
        logger.info("REAL experimental evaluation completed")
        return result
    
    def _execute_code(self, code: str, problem: Dict) -> tuple[bool, str, Optional[str]]:
        """Execute code with real test cases using BigCode's execution engine"""
        
        try:
            # Construct complete program with test
            full_program = code + "\n" + problem['test'] + f"\ncheck({problem['entry_point']})"
            
            # Use BigCode's execution engine
            execution_result = check_correctness(
                check_program=full_program,
                timeout=self.timeout,
                task_id=problem['task_id'],
                completion_id=0
            )
            
            return execution_result['passed'], execution_result['result'], None
            
        except Exception as e:
            return False, "execution_error", str(e)
    
    def _calculate_pass_at_k(self, results: List[bool], k_values: List[int]) -> Dict[int, float]:
        """Calculate Pass@K metrics"""
        if not results:
            return {k: 0.0 for k in k_values}
            
        n = len(results)
        successes = sum(results)
        
        pass_at_k = {}
        for k in k_values:
            if k >= n:
                pass_at_k[k] = successes / n
            else:
                # Pass@K: probability of at least one success in k attempts
                if successes == 0:
                    pass_at_k[k] = 0.0
                elif successes == n:
                    pass_at_k[k] = 1.0
                else:
                    failure_rate = (n - successes) / n
                    pass_at_k[k] = 1 - (failure_rate ** k)
        
        return pass_at_k
    
    def _detect_model_type(self, model_name: str) -> ModelType:
        """Detect model type for template selection"""
        model_name_lower = model_name.lower()
        
        if any(keyword in model_name_lower for keyword in ['code', 'coder', 'codellama']):
            return ModelType.CODE
        elif any(keyword in model_name_lower for keyword in ['chat', 'instruct', 'assistant']):
            return ModelType.CHAT
        else:
            return ModelType.UNKNOWN
    
    def _get_templates_for_model_type(self, model_type: ModelType) -> List[PromptTemplate]:
        """Get optimal templates for model type"""
        if model_type == ModelType.CODE:
            return [PromptTemplate.DIRECT, PromptTemplate.INSTRUCTION, PromptTemplate.FEW_SHOT]
        elif model_type == ModelType.CHAT:
            return [PromptTemplate.CONVERSATIONAL, PromptTemplate.INSTRUCTION, PromptTemplate.CHAIN_OF_THOUGHT]
        else:
            return list(PromptTemplate)  # Try all templates
    
    def _analyze_template_performance(self, template_results: Dict) -> Dict[str, Any]:
        """Statistical analysis of template performance"""
        template_scores = {}
        for template, data in template_results.items():
            template_scores[template] = data['success_rate']
        
        scores = list(template_scores.values())
        if not scores:
            return {
                'template_scores': template_scores,
                'mean_score': 0.0,
                'std_dev': 0.0,
                'sensitivity': 0.0
            }
        
        mean_score = sum(scores) / len(scores)
        variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
        std_dev = variance ** 0.5
        sensitivity = std_dev / mean_score if mean_score > 0 else 0.0
        
        return {
            'template_scores': template_scores,
            'mean_score': mean_score,
            'std_dev': std_dev,
            'sensitivity': sensitivity
        }
    
    def _generation_to_dict(self, gen: RealGenerationResult) -> Dict:
        """Convert GenerationResult to dictionary for JSON serialization"""
        return {
            'task_id': gen.task_id,
            'template': gen.template,
            'attempt': gen.attempt,
            'prompt': gen.prompt[:200] + "..." if len(gen.prompt) > 200 else gen.prompt,
            'generated_text': gen.generated_text,
            'extracted_code': gen.extracted_code,
            'execution_time': gen.execution_time,
            'passed': gen.passed,
            'execution_result': gen.execution_result,
            'error_message': gen.error_message
        }
    
    def _save_result(self, result: RealExperimentResult):
        """Save experiment result to file"""
        filename = f"real_experiment_{result.model_name.replace('/', '_')}_{result.timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert to dictionary for JSON serialization
        result_dict = {
            'model_name': result.model_name,
            'model_type': result.model_type,
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

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test the framework
    print("✅ Real Experimental Framework loaded successfully")
    print("Ready for integration with model interfaces")