# src/nlp_to_logic.py

import string
import re


class SimpleMapper:
    """
    Maps natural language concepts to propositional variables.
    """
    
    def __init__(self):
        self.map = {}
        self.reverse_map = {}
        self.current_var = 0
        self.variables = list(string.ascii_uppercase)

    def reset(self):
        """Reset the mapper for a new problem."""
        self.map = {}
        self.reverse_map = {}
        self.current_var = 0

    def get_var(self, word):
        """Get or create a variable for a word."""
        word = word.lower().strip()
        
        # Skip empty words
        if not word:
            return None
            
        if word not in self.map:
            if self.current_var >= len(self.variables):
                return None  # Ran out of variables
            self.map[word] = self.variables[self.current_var]
            self.reverse_map[self.variables[self.current_var]] = word
            self.current_var += 1
        
        return self.map[word]

    def extract_key_words(self, text):
        """Extract meaningful words from text."""
        # Remove common words
        stop_words = {'all', 'are', 'is', 'if', 'then', 'a', 'an', 'the', 'be', 
                      'to', 'of', 'and', 'that', 'have', 'it', 'for', 'not',
                      'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this',
                      'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
                      'she', 'or', 'an', 'will', 'my', 'one', 'would', 'there',
                      'their', 'what', 'so', 'up', 'out', 'about', 'who', 'get',
                      'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time',
                      'no', 'just', 'him', 'know', 'take', 'people', 'into',
                      'year', 'your', 'good', 'some', 'could', 'them', 'see',
                      'other', 'than', 'now', 'look', 'only', 'come', 'its',
                      'over', 'think', 'also', 'back', 'after', 'use', 'two',
                      'how', 'our', 'work', 'first', 'well', 'way', 'even',
                      'new', 'want', 'because', 'any', 'these', 'give', 'day',
                      'most', 'us', 'true', 'false', 'every'}
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [w for w in words if w not in stop_words and len(w) > 1]

    def convert(self, sentence):
        """
        Convert an English sentence to propositional logic.
        
        Supported patterns:
        - All X are Y -> X -> Y
        - If X then Y -> X -> Y
        - X implies Y -> X -> Y
        - X is Y -> X (as a fact)
        - Is X true? -> X (as query)
        - Single word/concept -> that variable
        """
        s = sentence.strip()
        s_lower = s.lower()

        # Pattern: All X are Y
        match = re.match(r'all\s+(\w+)\s+are\s+(\w+)', s_lower)
        if match:
            x = self.get_var(match.group(1))
            y = self.get_var(match.group(2))
            if x and y:
                return f"{x} -> {y}"

        # Pattern: If X then Y
        match = re.match(r'if\s+(.+?)\s+then\s+(.+)', s_lower)
        if match:
            x_words = self.extract_key_words(match.group(1))
            y_words = self.extract_key_words(match.group(2))
            if x_words and y_words:
                x = self.get_var(x_words[0])
                y = self.get_var(y_words[0])
                if x and y:
                    return f"{x} -> {y}"

        # Pattern: X implies Y
        match = re.match(r'(\w+)\s+implies\s+(\w+)', s_lower)
        if match:
            x = self.get_var(match.group(1))
            y = self.get_var(match.group(2))
            if x and y:
                return f"{x} -> {y}"

        # Pattern: Is X true? / Is X Y?
        match = re.match(r'is\s+(\w+)\s+(\w+)', s_lower)
        if match:
            # Could be "Is Socrates mortal?" -> query about mortal(Socrates)
            subject = match.group(1)
            predicate = match.group(2)
            
            if predicate in ['true', 'valid', 'correct']:
                x = self.get_var(subject)
                if x:
                    return x
            else:
                y = self.get_var(predicate)
                if y:
                    return y

        # Pattern: X is Y / X is a Y
        match = re.match(r'(\w+)\s+is\s+(?:a\s+)?(\w+)', s_lower)
        if match:
            x = self.get_var(match.group(1))
            y = self.get_var(match.group(2))
            if x and y:
                # This is typically stating a fact that x belongs to category y
                # In propositional logic, we represent this as asserting x
                return x

        # Pattern: Single variable (A, B, C...)
        if len(s) == 1 and s.isupper():
            return s

        # Pattern: Simple fact - extract first meaningful word
        words = self.extract_key_words(s)
        if words:
            x = self.get_var(words[0])
            if x:
                return x

        return None

    def get_mapping_explanation(self):
        """Return human-readable mapping."""
        return {v: k for k, v in self.reverse_map.items()}


# Rule-based parser for direct logical input
def rule_based_parser(sentence):
    """
    Handles direct logical patterns safely.
    Returns logical form or None.
    """
    s = sentence.strip()

    # Already in logical form: A -> B
    if re.match(r'^[A-Z]\s*->\s*[A-Z]$', s):
        return s.replace(" ", "")

    # Single variable: A
    if re.match(r'^[A-Z]$', s):
        return s

    # Negation: ~A or NOT A
    match = re.match(r'^[~!]([A-Z])$', s)
    if match:
        return f"~{match.group(1)}"

    match = re.match(r'^NOT\s+([A-Z])$', s, re.IGNORECASE)
    if match:
        return f"~{match.group(1)}"

    # All X are Y (single letters)
    match = re.match(r'All\s+([A-Z])\s+are\s+([A-Z])', s, re.IGNORECASE)
    if match:
        return f"{match.group(1).upper()} -> {match.group(2).upper()}"

    # If X then Y (single letters)
    match = re.match(r'If\s+([A-Z])\s+then\s+([A-Z])', s, re.IGNORECASE)
    if match:
        return f"{match.group(1).upper()} -> {match.group(2).upper()}"

    # X implies Y (single letters)
    match = re.match(r'([A-Z])\s+implies\s+([A-Z])', s, re.IGNORECASE)
    if match:
        return f"{match.group(1).upper()} -> {match.group(2).upper()}"

    # Is X true?
    match = re.match(r'Is\s+([A-Z])\s+true', s, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def clean_logic_output(text):
    """
    Extract only valid logical tokens from text.
    """
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ->~ ")
    cleaned = "".join(c for c in text if c in allowed)
    return cleaned.strip()
