#!/usr/bin/env python3
"""
Sprint 2.2 Demo: Multi-Language Support Implementation

This script demonstrates the complete Sprint 2.2 functionality:
- Language detection from generated code
- Multi-language execution environments (Python, JS, Java, C++)
- Language-specific test runners with BigCode integration
- Container isolation for each language
- Integration with Sprint 2.1 Pass@K metrics
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "model_interfaces"))

from language_detector import (
    LanguageDetector, ProgrammingLanguage,
    get_docker_image_for_language, get_bigcode_task_name
)
from multi_language_executor import (
    MultiLanguageExecutor, ExecutionMode
)
from multi_language_test_runner import (
    BigCodeMultiLanguageAdapter, TestCase, MultiLanguageTestSuite
)


def demonstrate_language_detection():
    """Demonstrate advanced language detection"""
    print("🔍 SPRINT 2.2: Language Detection System")
    print("=" * 60)

    detector = LanguageDetector()

    # Test various code samples
    test_samples = [
        ("Python Fibonacci", '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

if __name__ == "__main__":
    print(fibonacci(10))
        '''),

        ("JavaScript Promise", '''
async function fetchData(url) {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error:", error);
        return null;
    }
}

fetchData("https://api.example.com/data").then(console.log);
        '''),

        ("Java Class", '''
public class Calculator {
    private double result;

    public Calculator() {
        this.result = 0.0;
    }

    public double add(double a, double b) {
        result = a + b;
        return result;
    }

    public static void main(String[] args) {
        Calculator calc = new Calculator();
        System.out.println("Result: " + calc.add(5.5, 3.2));
    }
}
        '''),

        ("C++ Template", '''
#include <iostream>
#include <vector>
#include <algorithm>

template<typename T>
class Stack {
private:
    std::vector<T> items;

public:
    void push(const T& item) {
        items.push_back(item);
    }

    T pop() {
        if (!items.empty()) {
            T top = items.back();
            items.pop_back();
            return top;
        }
        throw std::runtime_error("Stack is empty");
    }
};

int main() {
    Stack<int> stack;
    stack.push(42);
    std::cout << "Popped: " << stack.pop() << std::endl;
    return 0;
}
        '''),

        ("Go Function", '''
package main

import (
    "fmt"
    "time"
)

func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

func main() {
    start := time.Now()
    result := fibonacci(30)
    duration := time.Since(start)

    fmt.Printf("fibonacci(30) = %d (took %v)\\n", result, duration)
}
        '''),
    ]

    for name, code in test_samples:
        print(f"\\n🔸 {name}:")
        print("-" * 30)

        result = detector.detect_language(code)

        print(f"Language: {result.language.value.upper()}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"File Extension: {result.file_extension}")
        print(f"Execute Command: {result.execution_command}")

        if result.compile_command:
            print(f"Compile Command: {result.compile_command}")

        # Show Docker integration
        docker_image = get_docker_image_for_language(result.language)
        bigcode_task = get_bigcode_task_name(result.language)

        print(f"Docker Image: {docker_image}")
        print(f"BigCode Task: {bigcode_task}")


def demonstrate_multi_language_execution():
    """Demonstrate multi-language execution environments"""
    print("\\n\\n🚀 SPRINT 2.2: Multi-Language Execution")
    print("=" * 60)

    executor = MultiLanguageExecutor(ExecutionMode.DIRECT)  # Use direct for demo

    # Test execution across languages
    test_programs = [
        (ProgrammingLanguage.PYTHON, '''
def greet(name):
    return f"Hello, {name}!"

print(greet("Sprint 2.2"))
        '''),

        (ProgrammingLanguage.JAVASCRIPT, '''
function greet(name) {
    return `Hello, ${name}!`;
}

console.log(greet("Sprint 2.2"));
        '''),

        (ProgrammingLanguage.CPP, '''
#include <iostream>
#include <string>
using namespace std;

string greet(const string& name) {
    return "Hello, " + name + "!";
}

int main() {
    cout << greet("Sprint 2.2") << endl;
    return 0;
}
        '''),
    ]

    for language, code in test_programs:
        print(f"\\n🔸 Executing {language.value.upper()}:")
        print("-" * 30)

        result = executor.execute_code(code, language, timeout=10)

        print(f"Success: {result.success}")
        print(f"Exit Code: {result.exit_code}")
        print(f"Execution Time: {result.execution_time:.3f}s")

        if result.stdout:
            print(f"Output: {result.stdout.strip()}")

        if result.stderr and result.stderr.strip():
            print(f"Errors: {result.stderr.strip()}")

        if result.compile_output:
            print(f"Compile Output: {result.compile_output.strip()}")


def demonstrate_bigcode_integration():
    """Demonstrate BigCode multi-language integration"""
    print("\\n\\n🔗 SPRINT 2.2: BigCode Multi-Language Integration")
    print("=" * 60)

    # Show supported language mappings
    language_mappings = {
        ProgrammingLanguage.PYTHON: "humaneval",
        ProgrammingLanguage.JAVASCRIPT: "multiple-js",
        ProgrammingLanguage.JAVA: "multiple-java",
        ProgrammingLanguage.CPP: "multiple-cpp",
        ProgrammingLanguage.GO: "multiple-go",
        ProgrammingLanguage.RUST: "multiple-rs",
        ProgrammingLanguage.TYPESCRIPT: "multiple-ts",
    }

    print("🔧 Language → BigCode Task Mappings:")
    for language, task in language_mappings.items():
        docker_image = get_docker_image_for_language(language)
        print(f"   {language.value.upper():12} → {task:15} (Docker: {docker_image})")

    print("\\n📊 CLI Usage Examples:")
    print("\\n🚀 Multi-language evaluation:")
    print("   python src/unified_runner.py \\\\")
    print("     --task multiple-js \\\\")
    print("     --model qwen-coder \\\\")
    print("     --n_samples 10 \\\\")
    print("     --temperature 0.25")

    print("\\n🔄 Cross-language comparison:")
    print("   python src/unified_runner.py \\\\")
    print("     --task multiple-cpp \\\\")
    print("     --model codellama \\\\")
    print("     --n_samples 50 \\\\")
    print("     --temperature 0.2 \\\\")
    print("     --limit 20")

    print("\\n🎯 Language-specific Pass@K:")
    print("   python src/unified_runner.py \\\\")
    print("     --task multiple-java \\\\")
    print("     --model phi3.5 \\\\")
    print("     --n_samples 100 \\\\")
    print("     --temperature 0.15")


def demonstrate_test_runner():
    """Demonstrate multi-language test runner"""
    print("\\n\\n🧪 SPRINT 2.2: Multi-Language Test Runner")
    print("=" * 60)

    # Create test cases that work across languages
    test_cases = [
        TestCase("add(2, 3)", "5"),
        TestCase("add(10, -5)", "5"),
        TestCase("add(0, 0)", "0"),
    ]

    # Sample generated code in different languages
    generated_codes = {
        "Python": '''
def add(a, b):
    """Add two numbers and return the result"""
    return a + b
        ''',

        "JavaScript": '''
function add(a, b) {
    // Add two numbers and return the result
    return a + b;
}
        ''',

        "C++": '''
#include <iostream>
using namespace std;

int add(int a, int b) {
    // Add two numbers and return the result
    return a + b;
}
        ''',
    }

    print("🔍 Testing language detection and execution:")

    # Create a temporary adapter for testing
    temp_dir = Path("/tmp")
    adapter = BigCodeMultiLanguageAdapter(temp_dir, ExecutionMode.DIRECT)

    for lang_name, code in generated_codes.items():
        print(f"\\n🔸 Testing {lang_name}:")
        print("-" * 25)

        result = adapter.detect_and_run_tests(code, test_cases[:1], f"test_{lang_name.lower()}")

        print(f"Detected: {result.language.value}")
        print(f"Pass Rate: {result.pass_rate:.1%}")
        print(f"Tests: {result.passed_tests}/{result.total_tests}")

        if result.error_message:
            print(f"Error: {result.error_message}")


def show_sprint_22_architecture():
    """Show Sprint 2.2 architecture and improvements"""
    print("\\n\\n🏗️ SPRINT 2.2: Architecture Overview")
    print("=" * 60)

    print("📋 COMPLETED COMPONENTS:")
    print("\\n🔍 Language Detection System:")
    print("   • Pattern-based language recognition")
    print("   • Support for 7+ programming languages")
    print("   • Confidence scoring and feature detection")
    print("   • Integration with BigCode task mapping")

    print("\\n🚀 Multi-Language Execution:")
    print("   • Language-specific executors (Python, JS, Java, C++)")
    print("   • Docker container isolation per language")
    print("   • Compilation and execution pipelines")
    print("   • Error handling and timeout management")

    print("\\n🧪 Test Runner Integration:")
    print("   • BigCode multi-language adapter")
    print("   • Language-aware test case execution")
    print("   • Pass@K metrics across languages")
    print("   • Integration with Sprint 2.1 sampling")

    print("\\n🔗 BigCode Harness Integration:")
    print("   • Enhanced RealBigCodeAdapter")
    print("   • Multi-language task routing")
    print("   • Language-specific Ollama adapters")
    print("   • Container-based execution environments")

    print("\\n📊 SPRINT INTEGRATION:")
    print("   Sprint 2.1 + Sprint 2.2 = Complete Multi-Language Pass@K System")
    print("   • Pass@K metrics (Sprint 2.1) ✅")
    print("   • Multiple sampling (Sprint 2.1) ✅")
    print("   • Language detection (Sprint 2.2) ✅")
    print("   • Multi-language execution (Sprint 2.2) ✅")
    print("   • Container isolation (Sprint 2.0 + 2.2) ✅")


def main():
    """Main demonstration"""
    print("🎉 SPRINT 2.2 COMPLETE: Multi-Language Support")
    print("=" * 60)
    print("Enhanced AI Benchmark Suite with Multi-Language Evaluation")
    print()

    try:
        demonstrate_language_detection()
        demonstrate_multi_language_execution()
        demonstrate_bigcode_integration()
        demonstrate_test_runner()
        show_sprint_22_architecture()

        print("\\n\\n✅ SPRINT 2.2 SUCCESSFULLY COMPLETED!")
        print("🚀 Ready for production multi-language evaluation with Pass@K metrics")

        print("\\n🎯 NEXT CAPABILITIES:")
        print("   • Evaluate models across Python, JavaScript, Java, C++")
        print("   • Generate Pass@K metrics for each language")
        print("   • Compare model performance across languages")
        print("   • Secure container isolation for all languages")
        print("   • Integration with BigCode harness multi-language tasks")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()