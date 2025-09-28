#!/usr/bin/env python3
"""
Language Detection System (Sprint 2.2)

Detects programming languages from generated code to enable multi-language
evaluation with appropriate execution environments and test runners.
"""

import re
from typing import Dict, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass


class ProgrammingLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    TYPESCRIPT = "typescript"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    SCALA = "scala"
    UNKNOWN = "unknown"


@dataclass
class LanguageDetectionResult:
    """Result of language detection"""
    language: ProgrammingLanguage
    confidence: float  # 0.0 to 1.0
    detected_features: List[str]
    file_extension: str
    execution_command: str
    compile_command: Optional[str] = None


class LanguagePatterns:
    """Language-specific patterns for detection"""

    # Language patterns with confidence scores
    PATTERNS = {
        ProgrammingLanguage.PYTHON: {
            'keywords': [
                r'\bdef\s+\w+\s*\(',
                r'\bclass\s+\w+\s*[:(\s]',
                r'\bimport\s+\w+',
                r'\bfrom\s+\w+\s+import',
                r'\bif\s+__name__\s*==\s*[\'"]__main__[\'"]',
                r'\bprint\s*\(',
                r'\brange\s*\(',
                r'\blen\s*\(',
            ],
            'syntax': [
                r':\s*$',  # Colon at end of line (if/for/def)
                r'^\s*#',  # Comments starting with #
                r'\bself\.',  # self references
                r'""".*?"""',  # Triple quote strings
                r"'''.*?'''",  # Triple quote strings
            ],
            'extension': '.py',
            'execute': 'python3',
        },

        ProgrammingLanguage.JAVASCRIPT: {
            'keywords': [
                r'\bfunction\s+\w+\s*\(',
                r'\bconst\s+\w+\s*=',
                r'\blet\s+\w+\s*=',
                r'\bvar\s+\w+\s*=',
                r'\bconsole\.log\s*\(',
                r'\breturn\s+',
                r'\brequire\s*\(',
                r'\bmodule\.exports\s*=',
            ],
            'syntax': [
                r';\s*$',  # Semicolons at end of line
                r'//.*$',  # Single line comments
                r'/\*.*?\*/',  # Multi-line comments
                r'{\s*$',  # Opening braces
                r'}\s*$',  # Closing braces
            ],
            'extension': '.js',
            'execute': 'node',
        },

        ProgrammingLanguage.JAVA: {
            'keywords': [
                r'\bpublic\s+class\s+\w+',
                r'\bpublic\s+static\s+void\s+main',
                r'\bpublic\s+\w+\s+\w+\s*\(',
                r'\bprivate\s+\w+\s+\w+',
                r'\bSystem\.out\.print',
                r'\bString\s+\w+\s*=',
                r'\bint\s+\w+\s*=',
                r'\bnew\s+\w+\s*\(',
            ],
            'syntax': [
                r';\s*$',  # Semicolons
                r'//.*$',  # Comments
                r'{\s*$',  # Braces
                r'}\s*$',
                r'\bpackage\s+[\w.]+;',  # Package declarations
                r'\bimport\s+[\w.]+;',  # Import statements
            ],
            'extension': '.java',
            'compile': 'javac',
            'execute': 'java',
        },

        ProgrammingLanguage.CPP: {
            'keywords': [
                r'#include\s*<[\w./]+>',
                r'#include\s*"[\w./]+"',
                r'\bint\s+main\s*\(',
                r'\bstd::\w+',
                r'\bcout\s*<<',
                r'\bcin\s*>>',
                r'\bnamespace\s+\w+',
                r'\bclass\s+\w+\s*{',
                r'\bpublic:\s*$',
                r'\bprivate:\s*$',
            ],
            'syntax': [
                r';\s*$',  # Semicolons
                r'//.*$',  # Comments
                r'/\*.*?\*/',  # Multi-line comments
                r'{\s*$',  # Braces
                r'}\s*$',
                r'::\w+',  # Scope resolution
            ],
            'extension': '.cpp',
            'compile': 'g++',
            'execute': './program',
        },

        ProgrammingLanguage.GO: {
            'keywords': [
                r'\bpackage\s+main',
                r'\bfunc\s+main\s*\(\s*\)',
                r'\bfunc\s+\w+\s*\(',
                r'\bimport\s*\(',
                r'\bfmt\.Print',
                r'\bvar\s+\w+\s+\w+',
                r'\b:=\s*',
                r'\bgo\s+\w+\s*\(',
            ],
            'syntax': [
                r'{\s*$',  # Braces
                r'}\s*$',
                r'//.*$',  # Comments
                r'/\*.*?\*/',  # Multi-line comments
            ],
            'extension': '.go',
            'compile': 'go build',
            'execute': 'go run',
        },

        ProgrammingLanguage.RUST: {
            'keywords': [
                r'\bfn\s+main\s*\(\s*\)',
                r'\bfn\s+\w+\s*\(',
                r'\blet\s+\w+\s*=',
                r'\blet\s+mut\s+\w+',
                r'\bprintln!\s*\(',
                r'\bmatch\s+\w+\s*{',
                r'\bpub\s+fn\s+\w+',
                r'\buse\s+\w+',
            ],
            'syntax': [
                r';\s*$',  # Semicolons
                r'//.*$',  # Comments
                r'/\*.*?\*/',  # Multi-line comments
                r'{\s*$',  # Braces
                r'}\s*$',
                r'!\s*\(',  # Macro calls
            ],
            'extension': '.rs',
            'compile': 'rustc',
            'execute': './program',
        },

        ProgrammingLanguage.TYPESCRIPT: {
            'keywords': [
                r'\bfunction\s+\w+\s*\(',
                r'\bconst\s+\w+:\s*\w+\s*=',
                r'\blet\s+\w+:\s*\w+\s*=',
                r'\binterface\s+\w+\s*{',
                r'\btype\s+\w+\s*=',
                r'\bconsole\.log\s*\(',
                r'\bexport\s+\w+',
                r'\bimport\s+.*\bfrom\s+',
            ],
            'syntax': [
                r';\s*$',  # Semicolons
                r'//.*$',  # Comments
                r'/\*.*?\*/',  # Multi-line comments
                r':\s*\w+\s*[=;]',  # Type annotations
                r'{\s*$',  # Braces
                r'}\s*$',
            ],
            'extension': '.ts',
            'compile': 'tsc',
            'execute': 'node',
        },
    }


class LanguageDetector:
    """Detects programming language from generated code"""

    def __init__(self):
        self.patterns = LanguagePatterns.PATTERNS

    def detect_language(self, code: str) -> LanguageDetectionResult:
        """
        Detect the programming language of the given code.

        Args:
            code: The source code to analyze

        Returns:
            LanguageDetectionResult with detected language and metadata
        """
        if not code or not code.strip():
            return self._create_result(ProgrammingLanguage.UNKNOWN, 0.0, [])

        # Score each language
        language_scores = {}
        language_features = {}

        for language, patterns in self.patterns.items():
            score, features = self._calculate_language_score(code, patterns)
            language_scores[language] = score
            language_features[language] = features

        # Find best match
        best_language = max(language_scores.keys(), key=lambda k: language_scores[k])
        best_score = language_scores[best_language]
        best_features = language_features[best_language]

        # Apply confidence threshold
        if best_score < 0.2:  # Too low confidence
            best_language = ProgrammingLanguage.UNKNOWN
            best_score = 0.0
            best_features = []

        return self._create_result(best_language, best_score, best_features)

    def _calculate_language_score(self, code: str, patterns: Dict) -> Tuple[float, List[str]]:
        """Calculate confidence score for a specific language"""
        total_patterns = len(patterns.get('keywords', [])) + len(patterns.get('syntax', []))
        if total_patterns == 0:
            return 0.0, []

        matched_patterns = []
        score = 0.0

        # Check keyword patterns (higher weight)
        for pattern in patterns.get('keywords', []):
            if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                score += 1.0
                matched_patterns.append(f"keyword: {pattern}")

        # Check syntax patterns (lower weight)
        for pattern in patterns.get('syntax', []):
            if re.search(pattern, code, re.MULTILINE):
                score += 0.5
                matched_patterns.append(f"syntax: {pattern}")

        # Normalize score with boost for keyword matches
        keyword_boost = 1.5 if len([f for f in matched_patterns if f.startswith('keyword:')]) > 2 else 1.0
        normalized_score = min((score * keyword_boost) / total_patterns, 1.0)

        return normalized_score, matched_patterns

    def _create_result(self, language: ProgrammingLanguage, confidence: float,
                      features: List[str]) -> LanguageDetectionResult:
        """Create a LanguageDetectionResult"""
        if language == ProgrammingLanguage.UNKNOWN:
            return LanguageDetectionResult(
                language=language,
                confidence=confidence,
                detected_features=features,
                file_extension='.txt',
                execution_command='cat',
                compile_command=None
            )

        pattern_info = self.patterns[language]

        return LanguageDetectionResult(
            language=language,
            confidence=confidence,
            detected_features=features,
            file_extension=pattern_info['extension'],
            execution_command=pattern_info['execute'],
            compile_command=pattern_info.get('compile')
        )

    def detect_from_prompt(self, prompt: str) -> Optional[ProgrammingLanguage]:
        """
        Detect intended language from the prompt/task description.

        This is useful when the generated code doesn't have enough features
        for reliable detection.
        """
        prompt_lower = prompt.lower()

        # Check for explicit language mentions
        language_hints = {
            ProgrammingLanguage.PYTHON: ['python', 'py', 'def ', 'import '],
            ProgrammingLanguage.JAVASCRIPT: ['javascript', 'js', 'node', 'function '],
            ProgrammingLanguage.JAVA: ['java', 'class ', 'public static void main'],
            ProgrammingLanguage.CPP: ['c++', 'cpp', '#include', 'std::'],
            ProgrammingLanguage.GO: ['go', 'golang', 'func main', 'package main'],
            ProgrammingLanguage.RUST: ['rust', 'fn main', 'cargo'],
            ProgrammingLanguage.TYPESCRIPT: ['typescript', 'ts', 'interface '],
        }

        for language, hints in language_hints.items():
            if any(hint in prompt_lower for hint in hints):
                return language

        return None


# Utility functions for multi-language support
def get_docker_image_for_language(language: ProgrammingLanguage) -> str:
    """Get appropriate Docker image for language execution"""
    docker_images = {
        ProgrammingLanguage.PYTHON: "python:3.11-slim",
        ProgrammingLanguage.JAVASCRIPT: "node:18-slim",
        ProgrammingLanguage.JAVA: "openjdk:17-slim",
        ProgrammingLanguage.CPP: "gcc:12-slim",
        ProgrammingLanguage.GO: "golang:1.21-slim",
        ProgrammingLanguage.RUST: "rust:1.70-slim",
        ProgrammingLanguage.TYPESCRIPT: "node:18-slim",
        ProgrammingLanguage.PHP: "php:8.2-cli-slim",
        ProgrammingLanguage.RUBY: "ruby:3.2-slim",
        ProgrammingLanguage.SWIFT: "swift:5.8-slim",
        ProgrammingLanguage.SCALA: "openjdk:17-slim",
    }

    return docker_images.get(language, "ubuntu:22.04")


def get_bigcode_task_name(language: ProgrammingLanguage) -> str:
    """Get BigCode task name for the language"""
    task_mappings = {
        ProgrammingLanguage.PYTHON: "humaneval",
        ProgrammingLanguage.JAVASCRIPT: "multiple-js",
        ProgrammingLanguage.JAVA: "multiple-java",
        ProgrammingLanguage.CPP: "multiple-cpp",
        ProgrammingLanguage.GO: "multiple-go",
        ProgrammingLanguage.RUST: "multiple-rs",
        ProgrammingLanguage.TYPESCRIPT: "multiple-ts",
        ProgrammingLanguage.PHP: "multiple-php",
        ProgrammingLanguage.RUBY: "multiple-rb",
        ProgrammingLanguage.SWIFT: "multiple-swift",
        ProgrammingLanguage.SCALA: "multiple-scala",
    }

    return task_mappings.get(language, "humaneval")


# Testing and examples
if __name__ == "__main__":
    detector = LanguageDetector()

    # Test examples
    test_codes = [
        # Python
        """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
        """,

        # JavaScript
        """
function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n-1) + fibonacci(n-2);
}

console.log(fibonacci(10));
        """,

        # Java
        """
public class Fibonacci {
    public static int fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n-1) + fibonacci(n-2);
    }

    public static void main(String[] args) {
        System.out.println(fibonacci(10));
    }
}
        """,

        # C++
        """
#include <iostream>
using namespace std;

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n-1) + fibonacci(n-2);
}

int main() {
    cout << fibonacci(10) << endl;
    return 0;
}
        """,
    ]

    print("🔍 Language Detection Test Results:")
    print("=" * 50)

    for i, code in enumerate(test_codes):
        result = detector.detect_language(code)
        print(f"\nTest {i+1}:")
        print(f"  Language: {result.language.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Extension: {result.file_extension}")
        print(f"  Execute: {result.execution_command}")
        if result.compile_command:
            print(f"  Compile: {result.compile_command}")
        print(f"  Features: {len(result.detected_features)} detected")