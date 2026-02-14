import re

class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        """
        Normalizes text:
        - Removes excessive whitespace
        - Strips leading/trailing whitespace
        - Preserves newlines but removes multiple empty lines
        """
        if not text:
            return ""
        
        # Replace multiple spaces with single space (excluding newlines)
        # We want to preserve paragraph structure (double newlines)
        
        # 1. Strip basic whitespace
        text = text.strip()
        
        # 2. Replace weird whitespace characters (tabs, non-breaking spaces) with space
        text = re.sub(r'[ \t\r\f\v]+', ' ', text)
        
        # 3. Handle newlines:
        # We want to keep real paragraphs. 
        # Strategy: Replace 3+ newlines with 2 newlines.
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
