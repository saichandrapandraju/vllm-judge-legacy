import re
import json
from typing import Any, Dict, Optional, List

class OutputParser:
    """Parses outputs from LLMs into structured format."""
    
    def parse_single_evaluation(
        self, 
        raw_output: str, 
        template_id: Optional[str] = None,
        parser_rules: Optional[Dict[str, Any]] = None,
        provide_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Parse the output from a single evaluation.
        
        Args:
            raw_output: The raw output from the judge LLM
            template_id: ID of the template used (optional)
            parser_rules: Custom parser rules (optional)
            provide_reasoning: Whether reasoning was requested
            
        Returns:
            A dictionary containing the parsed judgment and reasoning
        """
        # Extract reasoning if it was requested
        reasoning = None
        judgment_text = raw_output
        
        if provide_reasoning:
            # Look for a reasoning section
            reasoning_match = re.search(r"Reasoning:(.+?)$", raw_output, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                # Remove the reasoning section from the judgment text
                judgment_text = raw_output[:reasoning_match.start()].strip()
            else:
                # Try to split by newlines and assume reasoning comes after
                lines = raw_output.split("\n")
                if len(lines) > 1:
                    judgment_text = lines[0].strip()
                    reasoning = "\n".join(lines[1:]).strip()
        
        # Try to parse the judgment
        judgment = None
        
        # First, check if the output is JSON
        if judgment_text.strip().startswith("{") and judgment_text.strip().endswith("}"):
            try:
                judgment = json.loads(judgment_text.strip())
                return {
                    "judgment": judgment,
                    "reasoning": reasoning
                }
            except json.JSONDecodeError:
                # Not valid JSON, continue with other parsing methods
                pass
        
        # Apply parser rules if provided
        if parser_rules:
            judgment = self._apply_parser_rules(judgment_text, parser_rules)
        
        # If no result from parser rules, try some common patterns
        if judgment is None:
            # Try binary classification patterns
            binary_judgment = self._parse_binary_classification(judgment_text)
            if binary_judgment is not None:
                judgment = binary_judgment
        
        # If no binary judgment, try numeric rating
        if judgment is None:
            numeric_judgment = self._parse_numeric_rating(judgment_text)
            if numeric_judgment is not None:
                judgment = numeric_judgment
        
        # If all else fails, use the raw text
        if judgment is None:
            judgment = judgment_text.strip()
        
        return {
            "judgment": judgment,
            "reasoning": reasoning
        }
    
    def parse_pairwise_comparison(
        self, 
        raw_output: str, 
        template_id: Optional[str] = None,
        parser_rules: Optional[Dict[str, Any]] = None,
        provide_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Parse the output from a pairwise comparison.
        
        Args:
            raw_output: The raw output from the judge LLM
            template_id: ID of the template used (optional)
            parser_rules: Custom parser rules (optional)
            provide_reasoning: Whether reasoning was requested
            
        Returns:
            A dictionary containing the parsed preference and reasoning
        """
        # Extract reasoning if it was requested
        reasoning = None
        judgment_text = raw_output
        
        if provide_reasoning:
            # Look for a reasoning section
            reasoning_match = re.search(r"Reasoning:(.+?)$", raw_output, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                # Remove the reasoning section from the judgment text
                judgment_text = raw_output[:reasoning_match.start()].strip()
            else:
                # Try to split by newlines and assume reasoning comes after
                lines = raw_output.split("\n")
                if len(lines) > 1:
                    judgment_text = lines[0].strip()
                    reasoning = "\n".join(lines[1:]).strip()
        
        # Try to determine preference (A, B, or EQUAL)
        preference = None
        
        # First, check if the output is JSON
        if judgment_text.strip().startswith("{") and judgment_text.strip().endswith("}"):
            try:
                preference_data = json.loads(judgment_text.strip())
                if "preference" in preference_data:
                    preference = preference_data["preference"]
                elif "preferred_text" in preference_data:
                    preference = preference_data["preferred_text"]
                elif "better" in preference_data:
                    preference = preference_data["better"]
                else:
                    # Just use the whole object
                    preference = preference_data
                
                return {
                    "judgment": preference,
                    "reasoning": reasoning
                }
            except json.JSONDecodeError:
                # Not valid JSON, continue with other parsing methods
                pass
        
        # Apply parser rules if provided
        if parser_rules:
            preference = self._apply_parser_rules(judgment_text, parser_rules)
        
        # If no result from parser rules, try some common patterns
        if preference is None:
            # Look for A or B mentions
            a_pattern = re.compile(r"\b(?:a|text a|option a)\b", re.IGNORECASE)
            b_pattern = re.compile(r"\b(?:b|text b|option b)\b", re.IGNORECASE)
            equal_pattern = re.compile(r"\b(?:equal|same|tie|equivalent)\b", re.IGNORECASE)
            
            a_matches = a_pattern.findall(judgment_text.lower())
            b_matches = b_pattern.findall(judgment_text.lower())
            equal_matches = equal_pattern.findall(judgment_text.lower())
            
            # Simple heuristic: choose the option with more mentions
            if len(equal_matches) > 0 and (len(equal_matches) >= len(a_matches) and len(equal_matches) >= len(b_matches)):
                preference = "EQUAL"
            elif len(a_matches) > len(b_matches):
                preference = "A"
            elif len(b_matches) > len(a_matches):
                preference = "B"
            elif "a" in judgment_text.lower():
                preference = "A"
            elif "b" in judgment_text.lower():
                preference = "B"
            else:
                preference = "EQUAL"  # Default if no clear preference
        
        return {
            "judgment": preference,
            "reasoning": reasoning
        }
    
    def _apply_parser_rules(self, text: str, rules: Dict[str, Any]) -> Any:
        """Apply parser rules to extract structured data from text."""
        rule_type = rules.get("type", "text")
        
        if rule_type == "binary":
            # Binary classification (e.g., yes/no, safe/unsafe)
            positive_patterns = rules.get("positive_patterns", ["yes", "true", "positive"])
            negative_patterns = rules.get("negative_patterns", ["no", "false", "negative"])
            
            text_lower = text.lower()
            
            for pattern in positive_patterns:
                if pattern.lower() in text_lower:
                    return True
            
            for pattern in negative_patterns:
                if pattern.lower() in text_lower:
                    return False
            
            # No match found
            return None
        
        elif rule_type == "numeric":
            # Numeric rating (e.g., 1-5 scale)
            pattern = rules.get("pattern", r"\b([1-5])\b")
            match = re.search(pattern, text)
            
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return None
            
            return None
        
        elif rule_type == "preference":
            # Preference (e.g., A vs B)
            pattern = rules.get("pattern", r"(?:(?:Text|Option|Response)\s*)?([AB])")
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                return match.group(1).upper()
            
            # Check for EQUAL or similar
            equal_pattern = re.compile(r"\b(?:equal|same|tie|equivalent)\b", re.IGNORECASE)
            if equal_pattern.search(text):
                return "EQUAL"
            
            return None
        
        elif rule_type == "json":
            # Try to parse as JSON
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
        
        elif rule_type == "regex":
            # Use a custom regex pattern
            pattern = rules.get("pattern", "")
            match = re.search(pattern, text)
            
            if match:
                # Return the capture group or the entire match
                if match.groups():
                    return match.group(1)
                else:
                    return match.group(0)
            
            return None
        
        else:
            # Default: return the raw text
            return text.strip()
    
    def _parse_binary_classification(self, text: str) -> Optional[bool]:
        """Parse binary classification output (yes/no, true/false, etc.)."""
        text_lower = text.lower().strip()
        
        # Common positive terms
        positive_terms = ["yes", "true", "positive", "correct", "acceptable", "safe", "allow", "allowed", "approve", "approved"]
        # Common negative terms
        negative_terms = ["no", "false", "negative", "incorrect", "unacceptable", "unsafe", "deny", "denied", "reject", "rejected"]
        
        for term in positive_terms:
            if term in text_lower or term == text_lower:
                return True
        
        for term in negative_terms:
            if term in text_lower or term == text_lower:
                return False
        
        return None
    
    def _parse_numeric_rating(self, text: str) -> Optional[int]:
        """Parse numeric rating output (e.g., 1-5 scale)."""
        # Look for digits
        matches = re.findall(r"\b(\d+)\b", text)
        
        if matches:
            try:
                # Return the first number found
                return int(matches[0])
            except ValueError:
                return None
        
        return None
