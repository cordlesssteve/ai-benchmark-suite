#!/usr/bin/env python3
"""
Contextual Feature Extraction for Adaptive Prompting

Based on Phase 2 roadmap and academic research (ProCC Framework, 2024-2025).
Extracts contextual features from prompts to enable intelligent strategy selection
via multi-armed bandit algorithms.

Version: 1.0 (Phase 2 Implementation)
"""

import re
import ast
import keyword
import math
import logging
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CompletionType(Enum):
    """Types of code completion tasks"""
    FUNCTION_BODY = "function_body"
    FUNCTION_DEFINITION = "function_definition"
    CLASS_BODY = "class_body"
    CLASS_DEFINITION = "class_definition"
    EXPRESSION = "expression"
    STATEMENT = "statement"
    DOCSTRING = "docstring"
    COMMENT = "comment"
    IMPORT = "import"
    UNKNOWN = "unknown"

class CodeDomain(Enum):
    """Code domains for specialized completion"""
    GENERAL = "general"
    WEB_DEVELOPMENT = "web_development"
    DATA_SCIENCE = "data_science"
    SYSTEM_PROGRAMMING = "system_programming"
    MACHINE_LEARNING = "machine_learning"
    ALGORITHMS = "algorithms"
    DATABASE = "database"
    TESTING = "testing"

@dataclass
class ContextFeatures:
    """Structured context features for bandit algorithms"""
    # Core prompt characteristics
    prompt_complexity: float        # 0.0-1.0, normalized complexity score
    context_length: float          # 0.0-1.0, normalized prompt length
    completion_type: float         # 0.0-1.0, encoded completion type
    code_domain: float            # 0.0-1.0, domain specialization score

    # Syntactic features
    indentation_level: float      # 0.0-1.0, normalized indentation depth
    nesting_level: float         # 0.0-1.0, normalized code nesting
    has_function_def: float      # 0.0 or 1.0, binary feature
    has_class_def: float         # 0.0 or 1.0, binary feature

    # Semantic features
    keyword_density: float       # 0.0-1.0, Python keyword density
    variable_complexity: float   # 0.0-1.0, variable naming complexity

    # Model-specific features
    model_preference: float      # 0.0-1.0, model-specific bias

    def to_vector(self) -> List[float]:
        """Convert to feature vector for bandit algorithms"""
        return [
            self.prompt_complexity,
            self.context_length,
            self.completion_type,
            self.code_domain,
            self.indentation_level,
            self.nesting_level,
            self.has_function_def,
            self.has_class_def,
            self.keyword_density,
            self.variable_complexity,
            self.model_preference
        ]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easier analysis"""
        return {
            'prompt_complexity': self.prompt_complexity,
            'context_length': self.context_length,
            'completion_type': self.completion_type,
            'code_domain': self.code_domain,
            'indentation_level': self.indentation_level,
            'nesting_level': self.nesting_level,
            'has_function_def': self.has_function_def,
            'has_class_def': self.has_class_def,
            'keyword_density': self.keyword_density,
            'variable_complexity': self.variable_complexity,
            'model_preference': self.model_preference
        }

class PromptContextAnalyzer:
    """Extract contextual features for adaptive strategy selection"""

    def __init__(self):
        # Calibrated domain-specific keywords with weights
        self.domain_keywords = {
            CodeDomain.GENERAL: {
                # General programming keywords
                'python': 1.0, 'function': 1.0, 'variable': 1.0, 'class': 1.0, 'method': 1.0,
                'code': 1.0, 'program': 1.0, 'script': 1.0, 'programming': 1.0, 'syntax': 1.0,
                'loop': 1.0, 'condition': 1.0, 'algorithm': 1.0, 'logic': 1.0, 'simple': 1.0
            },
            CodeDomain.WEB_DEVELOPMENT: {
                # High-weight web indicators
                'flask': 2.0, 'django': 2.0, 'fastapi': 2.0, 'express': 2.0,
                'react': 2.0, 'vue': 2.0, 'angular': 2.0,
                # Medium-weight web indicators
                'html': 1.5, 'css': 1.5, 'javascript': 1.5, 'dom': 1.5,
                'fetch': 1.5, 'axios': 1.5, 'component': 1.5, 'hooks': 1.5,
                # Low-weight web indicators
                'browser': 1.0, 'window': 1.0, 'document': 1.0, 'jquery': 1.0,
                'http': 1.0, 'url': 1.0, 'web': 1.0, 'application': 1.0
            },
            CodeDomain.DATA_SCIENCE: {
                # High-weight data science indicators
                'pandas': 2.0, 'numpy': 2.0, 'matplotlib': 2.0, 'sklearn': 2.0, 'scipy': 2.0,
                # Medium-weight data science indicators
                'dataframe': 1.5, 'array': 1.5, 'plot': 1.5, 'chart': 1.5, 'seaborn': 1.5, 'plotly': 1.5,
                # Low-weight data science indicators
                'csv': 1.0, 'json': 1.0, 'data': 1.0, 'dataset': 1.0, 'analysis': 1.0, 'statistics': 1.0
            },
            CodeDomain.MACHINE_LEARNING: {
                # High-weight ML indicators
                'tensorflow': 2.0, 'pytorch': 2.0, 'keras': 2.0, 'sklearn': 2.0,
                # Medium-weight ML indicators
                'model': 1.5, 'train': 1.5, 'predict': 1.5, 'neural': 1.5, 'network': 1.5, 'deep': 1.5,
                'complex': 1.5,
                # Low-weight ML indicators
                'accuracy': 1.0, 'loss': 1.0, 'learning': 1.0, 'classification': 1.0, 'regression': 1.0
            },
            CodeDomain.SYSTEM_PROGRAMMING: {
                # High-weight system indicators
                'subprocess': 2.0, 'threading': 2.0, 'multiprocessing': 2.0, 'socket': 2.0,
                # Medium-weight system indicators
                'os': 1.5, 'sys': 1.5, 'process': 1.5, 'thread': 1.5, 'memory': 1.5,
                # Low-weight system indicators
                'file': 1.0, 'path': 1.0, 'directory': 1.0, 'cpu': 1.0, 'system': 1.0
            },
            CodeDomain.ALGORITHMS: {
                # High-weight algorithm indicators
                'algorithm': 2.0, 'complexity': 2.0, 'recursive': 2.0, 'dynamic': 2.0,
                # Medium-weight algorithm indicators
                'sort': 1.5, 'search': 1.5, 'binary': 1.5, 'tree': 1.5, 'graph': 1.5,
                # Low-weight algorithm indicators
                'hash': 1.0, 'queue': 1.0, 'stack': 1.0, 'heap': 1.0, 'optimization': 1.0
            },
            CodeDomain.DATABASE: {
                # High-weight database indicators
                'sql': 2.0, 'sqlite': 2.0, 'postgres': 2.0, 'mysql': 2.0, 'mongodb': 2.0,
                # Medium-weight database indicators
                'database': 1.5, 'query': 1.5, 'table': 1.5, 'schema': 1.5,
                # Low-weight database indicators
                'select': 1.0, 'insert': 1.0, 'update': 1.0, 'delete': 1.0, 'join': 1.0, 'index': 1.0
            },
            CodeDomain.TESTING: {
                # High-weight testing indicators
                'pytest': 2.0, 'unittest': 2.0, 'mock': 2.0, 'fixture': 2.0,
                # Medium-weight testing indicators
                'test': 1.5, 'assert': 1.5, 'coverage': 1.5,
                # Low-weight testing indicators
                'setup': 1.0, 'teardown': 1.0, 'tdd': 1.0, 'bdd': 1.0, 'spec': 1.0
            }
        }

        # Model-specific preferences (learned from research/experience)
        self.model_preferences = {
            'phi3.5:latest': 0.8,       # High preference for structured prompts
            'mistral:7b-instruct': 0.6,  # Medium preference
            'qwen2.5-coder:3b': 0.9,    # Very high for code-specific tasks
            'codellama:13b-instruct': 0.7,  # Good for code but slower
            'tinyllama:1.1b': 0.3,      # Lower capability, simpler prompts better
        }

        # Python keywords for density calculation
        self.python_keywords = set(keyword.kwlist)

        # Common programming patterns
        self.code_patterns = {
            'function_def': r'def\s+\w+\s*\([^)]*\)\s*:',
            'class_def': r'class\s+\w+\s*(?:\([^)]*\))?\s*:',
            'method_call': r'\w+\.\w+\s*\(',
            'list_comp': r'\[.*for.*in.*\]',
            'dict_comp': r'\{.*for.*in.*\}',
            'lambda': r'lambda\s+.*:',
            'decorator': r'@\w+',
            'import': r'(?:from\s+\w+\s+)?import\s+\w+',
            'return': r'return\s+.+',
            'yield': r'yield\s+.+',
            'except': r'except\s+\w+',
            'with': r'with\s+.+:',
        }

    def extract_features(self, prompt: str, model_name: str) -> ContextFeatures:
        """Extract comprehensive contextual features from prompt"""
        if not prompt or not prompt.strip():
            return self._create_empty_features(model_name)

        try:
            return ContextFeatures(
                prompt_complexity=self.calculate_complexity(prompt),
                context_length=self.normalize_length(prompt),
                completion_type=self.encode_completion_type(prompt),
                code_domain=self.calculate_domain_score(prompt),
                indentation_level=self.calculate_indentation_level(prompt),
                nesting_level=self.calculate_nesting_level(prompt),
                has_function_def=1.0 if self.has_function_definition(prompt) else 0.0,
                has_class_def=1.0 if self.has_class_definition(prompt) else 0.0,
                keyword_density=self.calculate_keyword_density(prompt),
                variable_complexity=self.calculate_variable_complexity(prompt),
                model_preference=self.get_model_preference(model_name)
            )
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}, using fallback")
            return self._create_fallback_features(prompt, model_name)

    def calculate_complexity(self, prompt: str) -> float:
        """Calculate prompt complexity score 0.0-1.0"""
        factors = {
            'length': min(len(prompt) / 1000.0, 1.0),  # Normalize by 1000 chars
            'lines': min(len(prompt.split('\n')) / 50.0, 1.0),  # Normalize by 50 lines
            'nesting': self.calculate_nesting_level(prompt),
            'keywords': min(len(self.extract_python_keywords(prompt)) / 10.0, 1.0),
            'patterns': min(len(self.extract_code_patterns(prompt)) / 8.0, 1.0)
        }

        # Weighted average of complexity factors
        weights = {'length': 0.2, 'lines': 0.2, 'nesting': 0.3, 'keywords': 0.15, 'patterns': 0.15}
        complexity = sum(factors[k] * weights[k] for k in factors)

        return min(complexity, 1.0)

    def normalize_length(self, prompt: str) -> float:
        """Normalize prompt length to 0.0-1.0 scale"""
        # Use sigmoid-like normalization for better distribution
        length = len(prompt)
        normalized = 1.0 / (1.0 + math.exp(-(length - 500) / 200))
        return min(normalized, 1.0)

    def encode_completion_type(self, prompt: str) -> float:
        """Detect and encode completion type as 0.0-1.0"""
        completion_type = self.detect_completion_type(prompt)

        # Encode completion types with different complexity scores
        type_scores = {
            CompletionType.EXPRESSION: 0.1,
            CompletionType.STATEMENT: 0.2,
            CompletionType.FUNCTION_BODY: 0.5,
            CompletionType.FUNCTION_DEFINITION: 0.6,
            CompletionType.CLASS_BODY: 0.7,
            CompletionType.CLASS_DEFINITION: 0.8,
            CompletionType.DOCSTRING: 0.3,
            CompletionType.COMMENT: 0.2,
            CompletionType.IMPORT: 0.4,
            CompletionType.UNKNOWN: 0.0
        }

        return type_scores.get(completion_type, 0.0)

    def detect_completion_type(self, prompt: str) -> CompletionType:
        """Detect the type of completion needed"""
        prompt_lower = prompt.lower()

        # Check for specific patterns
        if re.search(r'def\s+\w+\s*\([^)]*\)\s*:\s*$', prompt):
            return CompletionType.FUNCTION_BODY
        elif 'def ' in prompt and prompt.strip().endswith(':'):
            return CompletionType.FUNCTION_DEFINITION
        elif re.search(r'class\s+\w+.*:\s*$', prompt):
            return CompletionType.CLASS_BODY
        elif 'class ' in prompt and ':' in prompt:
            return CompletionType.CLASS_DEFINITION
        elif '"""' in prompt or "'''" in prompt:
            return CompletionType.DOCSTRING
        elif prompt.strip().startswith('#'):
            return CompletionType.COMMENT
        elif 'import' in prompt_lower:
            return CompletionType.IMPORT
        elif '=' in prompt or 'return' in prompt_lower:
            return CompletionType.STATEMENT
        else:
            return CompletionType.EXPRESSION

    def calculate_domain_score(self, prompt: str) -> float:
        """Calculate domain specialization score 0.0-1.0 using weighted keywords"""
        prompt_lower = prompt.lower()
        domain_scores = {}

        # Expected score ranges for different domain types
        # Format: (min_for_0.0, max_for_1.0) to map scores to 0.0-1.0 range
        domain_ranges = {
            CodeDomain.GENERAL: (0.5, 3.5),      # 0.5-3.5: single to multiple general keywords
            CodeDomain.WEB_DEVELOPMENT: (1.5, 6.0),  # 1.5-6.0: single framework to multiple tech
            CodeDomain.DATA_SCIENCE: (1.5, 7.0),     # 1.5-7.0: single tool to full stack
            CodeDomain.MACHINE_LEARNING: (1.5, 7.0), # 1.5-7.0: similar to data science
            CodeDomain.SYSTEM_PROGRAMMING: (1.5, 6.0),
            CodeDomain.ALGORITHMS: (1.5, 6.0),
            CodeDomain.DATABASE: (1.5, 7.0),     # 1.5-7.0: sql+database+query+joins can hit high
            CodeDomain.TESTING: (1.5, 6.0)
        }

        for domain, keyword_weights in self.domain_keywords.items():
            weighted_score = 0.0

            # Count weighted matches only (don't normalize by total possible)
            for keyword, weight in keyword_weights.items():
                if keyword in prompt_lower:
                    weighted_score += weight

            # Store raw weighted score
            if weighted_score > 0:
                domain_scores[domain] = weighted_score

        if not domain_scores:
            return 0.0  # No domain keywords detected

        # Return the highest domain score, normalized using domain-specific ranges
        max_score = max(domain_scores.values())
        max_domain = max(domain_scores, key=domain_scores.get)

        # Get expected range for the dominant domain
        min_expected, max_expected = domain_ranges.get(max_domain, (1.0, 4.0))

        # Normalize: 0.0 at min_expected, 1.0 at max_expected
        normalized = (max_score - min_expected) / (max_expected - min_expected)
        return max(0.0, min(normalized, 1.0))

    def calculate_indentation_level(self, prompt: str) -> float:
        """Calculate normalized indentation depth 0.0-1.0"""
        lines = prompt.split('\n')
        indentations = []

        for line in lines:
            if line.strip():  # Skip empty lines
                indent = len(line) - len(line.lstrip())
                indentations.append(indent)

        if not indentations:
            return 0.0

        max_indent = max(indentations)
        # Normalize by 16 spaces (4 levels of 4-space indentation)
        return min(max_indent / 16.0, 1.0)

    def calculate_nesting_level(self, prompt: str) -> float:
        """Calculate code nesting level 0.0-1.0"""
        nesting_keywords = ['if', 'for', 'while', 'with', 'try', 'def', 'class']
        lines = prompt.split('\n')

        max_nesting = 0
        current_nesting = 0

        for line in lines:
            stripped = line.strip().lower()

            # Count nesting increases
            for keyword in nesting_keywords:
                if stripped.startswith(keyword + ' ') or stripped.startswith(keyword + ':'):
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)

            # Count nesting decreases (simplified)
            if stripped.startswith(('else:', 'elif ', 'except:', 'finally:')):
                current_nesting = max(0, current_nesting - 1)

        # Normalize by 5 levels
        return min(max_nesting / 5.0, 1.0)

    def has_function_definition(self, prompt: str) -> bool:
        """Check if prompt contains function definition"""
        return bool(re.search(r'def\s+\w+\s*\([^)]*\)\s*:', prompt))

    def has_class_definition(self, prompt: str) -> bool:
        """Check if prompt contains class definition"""
        return bool(re.search(r'class\s+\w+\s*(?:\([^)]*\))?\s*:', prompt))

    def calculate_keyword_density(self, prompt: str) -> float:
        """Calculate Python keyword density 0.0-1.0"""
        words = re.findall(r'\b\w+\b', prompt.lower())
        if not words:
            return 0.0

        keyword_count = sum(1 for word in words if word in self.python_keywords)
        density = keyword_count / len(words)

        # Normalize - typical code has 5-15% keywords
        return min(density * 10.0, 1.0)

    def calculate_variable_complexity(self, prompt: str) -> float:
        """Calculate variable naming complexity 0.0-1.0"""
        # Find variable-like patterns
        variables = re.findall(r'\b[a-z_][a-z0-9_]*\b', prompt)

        if not variables:
            return 0.0

        # Calculate complexity based on naming patterns
        complexity_factors = {
            'length': sum(len(var) for var in variables) / len(variables) / 20.0,  # Avg length
            'underscore_usage': sum(1 for var in variables if '_' in var) / len(variables),
            'camel_case': sum(1 for var in variables if any(c.isupper() for c in var)) / len(variables),
            'descriptive': sum(1 for var in variables if len(var) > 3) / len(variables)
        }

        # Weighted combination
        weights = {'length': 0.3, 'underscore_usage': 0.2, 'camel_case': 0.2, 'descriptive': 0.3}
        complexity = sum(complexity_factors[k] * weights[k] for k in complexity_factors)

        return min(complexity, 1.0)

    def get_model_preference(self, model_name: str) -> float:
        """Get model-specific preference score 0.0-1.0"""
        # Direct match
        if model_name in self.model_preferences:
            return self.model_preferences[model_name]

        # Pattern matching
        for model_pattern, preference in self.model_preferences.items():
            if model_pattern.split(':')[0] in model_name:
                return preference

        # Default for unknown models
        return 0.5

    def extract_python_keywords(self, prompt: str) -> Set[str]:
        """Extract Python keywords from prompt"""
        words = re.findall(r'\b\w+\b', prompt.lower())
        return {word for word in words if word in self.python_keywords}

    def extract_code_patterns(self, prompt: str) -> Set[str]:
        """Extract code patterns from prompt"""
        patterns_found = set()

        for pattern_name, pattern_regex in self.code_patterns.items():
            if re.search(pattern_regex, prompt, re.MULTILINE):
                patterns_found.add(pattern_name)

        return patterns_found

    def _create_empty_features(self, model_name: str) -> ContextFeatures:
        """Create features for empty prompt"""
        return ContextFeatures(
            prompt_complexity=0.0,
            context_length=0.0,
            completion_type=0.0,
            code_domain=0.0,
            indentation_level=0.0,
            nesting_level=0.0,
            has_function_def=0.0,
            has_class_def=0.0,
            keyword_density=0.0,
            variable_complexity=0.0,
            model_preference=self.get_model_preference(model_name)
        )

    def _create_fallback_features(self, prompt: str, model_name: str) -> ContextFeatures:
        """Create fallback features when extraction fails"""
        return ContextFeatures(
            prompt_complexity=min(len(prompt) / 500.0, 1.0),  # Simple length-based
            context_length=min(len(prompt) / 1000.0, 1.0),
            completion_type=0.5,  # Unknown
            code_domain=0.0,  # General
            indentation_level=0.5,  # Assume medium
            nesting_level=0.5,  # Assume medium
            has_function_def=1.0 if 'def ' in prompt else 0.0,
            has_class_def=1.0 if 'class ' in prompt else 0.0,
            keyword_density=0.5,  # Assume medium
            variable_complexity=0.5,  # Assume medium
            model_preference=self.get_model_preference(model_name)
        )

    def analyze_prompt_characteristics(self, prompt: str) -> Dict[str, any]:
        """Comprehensive prompt analysis for debugging and insights"""
        features = self.extract_features(prompt, "unknown")

        return {
            'raw_features': features.to_dict(),
            'completion_type': self.detect_completion_type(prompt).value,
            'detected_domains': [
                domain.value for domain, keywords in self.domain_keywords.items()
                if any(keyword in prompt.lower() for keyword in keywords)
            ],
            'code_patterns': list(self.extract_code_patterns(prompt)),
            'python_keywords': list(self.extract_python_keywords(prompt)),
            'statistics': {
                'length': len(prompt),
                'lines': len(prompt.split('\n')),
                'words': len(prompt.split()),
                'unique_words': len(set(prompt.lower().split()))
            }
        }

def create_context_analyzer() -> PromptContextAnalyzer:
    """Factory function to create context analyzer"""
    return PromptContextAnalyzer()

# Test utility functions
def test_feature_extraction():
    """Test feature extraction with various prompts"""
    analyzer = PromptContextAnalyzer()

    test_prompts = [
        "def fibonacci(n):",
        "class DataProcessor:\n    def __init__(self):",
        "import pandas as pd\ndf = pd.read_csv('data.csv')",
        "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
        ""
    ]

    for prompt in test_prompts:
        features = analyzer.extract_features(prompt, "phi3.5:latest")
        print(f"Prompt: {repr(prompt[:50])}")
        print(f"Features: {features.to_dict()}")
        print()

if __name__ == "__main__":
    test_feature_extraction()