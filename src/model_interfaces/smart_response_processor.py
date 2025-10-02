#!/usr/bin/env python3
"""
Smart Response Processor with Quality-Based Evaluation

Addresses critical Phase 1 issues:
1. Smart response cleaning that preserves code content while removing conversational wrappers
2. Dual success flag system (HTTP success + content quality success)
3. Quality scoring framework for content evaluation

Version: 1.0 (Phase 1 Implementation)
"""

import re
import ast
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ContentQuality(Enum):
    """Content quality levels"""
    EXCELLENT = "excellent"      # Clean, executable code
    GOOD = "good"               # Code with minor conversational elements
    FAIR = "fair"               # Mixed code and conversation, but salvageable
    POOR = "poor"               # Mostly conversational, little code
    EMPTY = "empty"             # No meaningful content

@dataclass
class ProcessedResponse:
    """Enhanced response with quality metrics and dual success flags"""
    text: str                           # Cleaned response text
    raw_text: str                      # Original response text
    http_success: bool                 # HTTP request succeeded
    content_quality_success: bool      # Content meets quality threshold
    overall_success: bool              # Combined success indicator
    quality_level: ContentQuality      # Detailed quality assessment
    quality_score: float               # Numerical quality score (0.0-1.0)
    is_conversational: bool           # Contains conversational elements
    is_executable: bool               # Appears to be executable code
    has_syntax_errors: bool           # Contains syntax errors
    cleaning_applied: bool            # Whether cleaning was applied
    metadata: Dict[str, Any]          # Additional processing metadata

class SmartResponseProcessor:
    """Smart response processor with quality-based evaluation"""

    def __init__(self, quality_threshold: float = 0.3):
        """
        Initialize processor with configurable quality threshold

        Args:
            quality_threshold: Minimum quality score for content_quality_success (0.0-1.0)
        """
        self.quality_threshold = quality_threshold

        # Smart conversational patterns - more precise matching
        self.conversational_starters = [
            r"^(here's|here is|certainly|sure|i'll|let me|you can|i can)",
            r"^(to complete|the function|this function|this code)",
            r"^(explanation|note that|as you can see|let's|we can)",
            r"^(i would|i will|we need to|you should|you need to)"
        ]

        # Patterns that indicate conversational but might contain code
        self.mixed_content_patterns = [
            r"^(# explanation|# this|# here|# note)",
            r"(here's the|here is the).*(code|function|implementation)",
            r"(to solve this|to implement this|to complete this)"
        ]

        # Code indicators - patterns that suggest executable code
        self.code_indicators = [
            r"^\s*(def |class |import |from |if |for |while |try |with )",
            r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=",  # Variable assignment
            r"^\s*return\s+",                     # Return statement
            r"^\s*print\s*\(",                   # Print function
            r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\(",  # Function call
            r"^\s*#[^#]",                        # Single hash comment (not explanation)
        ]

        # Markdown code block patterns
        self.markdown_patterns = [
            r"^```\w*$",      # Code block start/end
            r"^```$",         # Code block start/end
        ]

    def is_likely_code(self, line: str) -> bool:
        """Check if a line is likely to be code"""
        line_strip = line.strip()
        if not line_strip:
            return False

        # Check for code indicators
        for pattern in self.code_indicators:
            if re.match(pattern, line_strip, re.IGNORECASE):
                return True

        # Check if it looks like Python syntax
        try:
            # Try to parse as Python AST (for single expressions)
            compile(line_strip, '<string>', 'eval')
            return True
        except:
            try:
                # Try to parse as Python statement
                compile(line_strip, '<string>', 'exec')
                return True
            except:
                pass

        return False

    def is_conversational_line(self, line: str) -> bool:
        """Check if a line is conversational"""
        line_lower = line.strip().lower()
        if not line_lower:
            return False

        # Check conversational starters
        for pattern in self.conversational_starters:
            if re.match(pattern, line_lower):
                return True

        return False

    def is_mixed_content_line(self, line: str) -> bool:
        """Check if line is mixed conversational/code content"""
        line_lower = line.strip().lower()
        if not line_lower:
            return False

        for pattern in self.mixed_content_patterns:
            if re.search(pattern, line_lower):
                return True

        return False

    def smart_clean_response(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """
        Smart cleaning that preserves code while removing conversational wrappers

        Returns:
            Tuple of (cleaned_text, cleaning_metadata)
        """
        if not response or not response.strip():
            return "", {"lines_removed": 0, "patterns_matched": [], "cleaning_applied": False}

        lines = response.split('\n')
        cleaned_lines = []
        metadata = {
            "lines_removed": 0,
            "patterns_matched": [],
            "cleaning_applied": False,
            "original_line_count": len(lines),
            "code_lines_preserved": 0,
            "conversational_lines_removed": 0
        }

        in_code_block = False
        skip_next_empty = False

        for i, line in enumerate(lines):
            line_strip = line.strip()

            # Handle markdown code blocks
            if re.match(r"^```", line_strip):
                if not in_code_block:
                    in_code_block = True
                    metadata["patterns_matched"].append("markdown_start")
                    metadata["cleaning_applied"] = True
                    continue  # Skip opening ```
                else:
                    in_code_block = False
                    metadata["patterns_matched"].append("markdown_end")
                    skip_next_empty = True
                    continue  # Skip closing ```

            # If we're in a code block, preserve everything
            if in_code_block:
                cleaned_lines.append(line)
                metadata["code_lines_preserved"] += 1
                continue

            # Skip empty lines immediately after closing code blocks
            if skip_next_empty and not line_strip:
                skip_next_empty = False
                metadata["lines_removed"] += 1
                continue
            skip_next_empty = False

            # Skip empty lines at the beginning
            if not cleaned_lines and not line_strip:
                metadata["lines_removed"] += 1
                continue

            # Check if line is likely code - preserve it
            if self.is_likely_code(line):
                cleaned_lines.append(line)
                metadata["code_lines_preserved"] += 1
                continue

            # Check if line is purely conversational - remove it
            if self.is_conversational_line(line):
                metadata["lines_removed"] += 1
                metadata["conversational_lines_removed"] += 1
                metadata["patterns_matched"].append("conversational")
                metadata["cleaning_applied"] = True
                continue

            # Mixed content - keep but mark as suspicious
            if self.is_mixed_content_line(line):
                cleaned_lines.append(line)
                metadata["patterns_matched"].append("mixed_content")
                continue

            # Default: preserve the line
            cleaned_lines.append(line)

        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
            metadata["lines_removed"] += 1

        cleaned_text = '\n'.join(cleaned_lines)
        metadata["final_line_count"] = len(cleaned_lines)

        return cleaned_text, metadata

    def calculate_quality_score(self, text: str, metadata: Dict[str, Any]) -> Tuple[float, ContentQuality]:
        """
        Calculate quality score and level for processed text

        Returns:
            Tuple of (quality_score, quality_level)
        """
        if not text or not text.strip():
            return 0.0, ContentQuality.EMPTY

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return 0.0, ContentQuality.EMPTY

        total_lines = len(lines)
        code_lines = sum(1 for line in lines if self.is_likely_code(line))
        conversational_lines = sum(1 for line in lines if self.is_conversational_line(line))

        # Base score from code content ratio
        code_ratio = code_lines / total_lines if total_lines > 0 else 0.0

        # Penalty for conversational content
        conversational_penalty = (conversational_lines / total_lines) * 0.5 if total_lines > 0 else 0.0

        # Bonus for having actual code
        has_code_bonus = 0.2 if code_lines > 0 else 0.0

        # Length bonus (reasonable length suggests completeness)
        length_bonus = min(0.1, len(text) / 1000) if len(text) > 20 else 0.0

        # Calculate final score
        quality_score = max(0.0, min(1.0, code_ratio + has_code_bonus + length_bonus - conversational_penalty))

        # Determine quality level
        if quality_score >= 0.8:
            quality_level = ContentQuality.EXCELLENT
        elif quality_score >= 0.6:
            quality_level = ContentQuality.GOOD
        elif quality_score >= 0.4:
            quality_level = ContentQuality.FAIR
        elif quality_score >= 0.1:
            quality_level = ContentQuality.POOR
        else:
            quality_level = ContentQuality.EMPTY

        return quality_score, quality_level

    def check_syntax_validity(self, text: str) -> bool:
        """Check if text contains valid Python syntax"""
        if not text or not text.strip():
            return False

        try:
            # Try to compile as Python code
            compile(text, '<string>', 'exec')
            return True
        except SyntaxError:
            # Check if it's a partial completion that could be valid
            try:
                # Try wrapping in a function to see if it's a valid code fragment
                wrapped = f"def temp_func():\n    {text.replace(chr(10), chr(10) + '    ')}"
                compile(wrapped, '<string>', 'exec')
                return True
            except SyntaxError:
                return False
        except Exception:
            # Other errors (like NameError) don't indicate syntax problems
            return True

    def process_response(self, raw_response: str, http_success: bool = True) -> ProcessedResponse:
        """
        Process response with smart cleaning and quality evaluation

        Args:
            raw_response: Original response text
            http_success: Whether the HTTP request succeeded

        Returns:
            ProcessedResponse with all quality metrics
        """
        # Step 1: Smart cleaning
        cleaned_text, cleaning_metadata = self.smart_clean_response(raw_response)

        # Step 2: Quality scoring
        quality_score, quality_level = self.calculate_quality_score(cleaned_text, cleaning_metadata)

        # Step 3: Additional checks
        is_conversational = any(self.is_conversational_line(line) for line in cleaned_text.split('\n'))
        is_executable = self.check_syntax_validity(cleaned_text)
        has_syntax_errors = not is_executable and bool(cleaned_text.strip())

        # Step 4: Determine content quality success
        content_quality_success = quality_score >= self.quality_threshold and bool(cleaned_text.strip())

        # Step 5: Overall success (both HTTP and content quality)
        overall_success = http_success and content_quality_success

        # Step 6: Compile metadata
        metadata = {
            "cleaning": cleaning_metadata,
            "quality_score": quality_score,
            "quality_level": quality_level.value,
            "line_analysis": {
                "total_lines": len([l for l in cleaned_text.split('\n') if l.strip()]),
                "code_lines": sum(1 for line in cleaned_text.split('\n') if self.is_likely_code(line)),
                "conversational_lines": sum(1 for line in cleaned_text.split('\n') if self.is_conversational_line(line))
            },
            "thresholds": {
                "quality_threshold": self.quality_threshold,
                "meets_threshold": quality_score >= self.quality_threshold
            }
        }

        return ProcessedResponse(
            text=cleaned_text,
            raw_text=raw_response,
            http_success=http_success,
            content_quality_success=content_quality_success,
            overall_success=overall_success,
            quality_level=quality_level,
            quality_score=quality_score,
            is_conversational=is_conversational,
            is_executable=is_executable,
            has_syntax_errors=has_syntax_errors,
            cleaning_applied=cleaning_metadata["cleaning_applied"],
            metadata=metadata
        )

    def set_quality_threshold(self, threshold: float):
        """Update quality threshold for content success evaluation"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Quality threshold must be between 0.0 and 1.0")
        self.quality_threshold = threshold

    def get_quality_report(self, processed_response: ProcessedResponse) -> Dict[str, Any]:
        """Generate detailed quality report for a processed response"""
        return {
            "summary": {
                "http_success": processed_response.http_success,
                "content_quality_success": processed_response.content_quality_success,
                "overall_success": processed_response.overall_success,
                "quality_level": processed_response.quality_level.value,
                "quality_score": f"{processed_response.quality_score:.3f}"
            },
            "content_analysis": {
                "is_conversational": processed_response.is_conversational,
                "is_executable": processed_response.is_executable,
                "has_syntax_errors": processed_response.has_syntax_errors,
                "cleaning_applied": processed_response.cleaning_applied,
                "text_length": len(processed_response.text),
                "raw_length": len(processed_response.raw_text)
            },
            "detailed_metadata": processed_response.metadata
        }

def create_default_processor() -> SmartResponseProcessor:
    """Create processor with default settings for production use"""
    return SmartResponseProcessor(quality_threshold=0.3)

def create_strict_processor() -> SmartResponseProcessor:
    """Create processor with strict quality requirements"""
    return SmartResponseProcessor(quality_threshold=0.6)

def create_lenient_processor() -> SmartResponseProcessor:
    """Create processor with lenient quality requirements for research"""
    return SmartResponseProcessor(quality_threshold=0.1)