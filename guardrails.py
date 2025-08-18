from typing import Tuple, Optional

class ContentGuardRails:
    def __init__(self):
        # Define categories of content to filter
        self.harmful_content_patterns = [
            # Hate speech and discrimination
            r'hate\s*speech|racist|discrimination|bigot|prejudice',
            # Violence
            r'violence|violent|kill|murder|harm',
            # Inappropriate content
            r'explicit|pornographic|nsfw|obscene',
            # Harassment
            r'harass|bully|threat|abuse',
        ]
        
        # Add Bangla patterns for the same categories
        self.bangla_harmful_patterns = [
            # Hate speech and discrimination
            r'বিদ্বেষ|ঘৃণা|বৈষম্য|জাতিবাদী',
            # Violence
            r'হিংসা|হত্যা|আঘাত',
            # Harassment
            r'হয়রানি|ভয়ভীতি|হুমকি',
        ]
        
        self.safety_prompt_prefix = """
        You are a helpful AI assistant that provides accurate and ethical information. Please ensure your response:
        1. Is factual and based on the provided context
        2. Avoids harmful, discriminatory, or inappropriate content
        3. Respects privacy and personal information
        4. Maintains professional and respectful language
        
        Context for answering:
        """
        
    def check_content_safety(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if the content contains potentially harmful material.
        Returns (is_safe, reason_if_unsafe)
        """
        import re
        
        # Convert to lowercase for better pattern matching
        text_lower = text.lower()
        
        # Check English patterns
        for pattern in self.harmful_content_patterns:
            if re.search(pattern, text_lower):
                return False, f"Content contains potentially harmful material"
        
        # Check Bangla patterns
        for pattern in self.bangla_harmful_patterns:
            if re.search(pattern, text):
                return False, f"Content contains potentially harmful material"
                
        return True, None
    
    def sanitize_prompt(self, prompt: str) -> str:
        """Add safety guidelines to the prompt."""
        return self.safety_prompt_prefix + prompt
    
    def sanitize_response(self, response: str) -> Tuple[str, bool]:
        """
        Check and potentially modify the response for safety.
        Returns (sanitized_response, was_modified)
        """
        is_safe, reason = self.check_content_safety(response)
        if not is_safe:
            return ("I apologize, but I cannot provide that information as it may contain inappropriate or harmful content.", True)
        return (response, False)
