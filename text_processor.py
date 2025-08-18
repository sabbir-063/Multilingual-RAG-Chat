import re
from typing import Literal

def remove_extra_whitespace(text: str) -> str:
    """Remove extra whitespace, newlines, and tabs."""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove spaces at the beginning and end
    return text.strip()

def clean_english_text(text: str) -> str:
    """Clean English text by removing special characters while preserving meaning."""
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
    return remove_extra_whitespace(text)

def clean_bangla_text(text: str) -> str:
    """Clean Bangla text while preserving Bangla characters and meaningful punctuation."""
    # Preserve Bangla unicode range (0980-09FF), basic punctuation, and numbers
    text = re.sub(r'[^\u0980-\u09FF\s.,!?0-9-]', '', text)
    return remove_extra_whitespace(text)

def detect_language(text: str) -> Literal['bangla', 'english', 'mixed']:
    """Detect if the text is primarily in Bangla or English."""
    # Count Bangla characters (Unicode range for Bangla: 0980-09FF)
    bangla_pattern = re.compile(r'[\u0980-\u09FF]')
    english_pattern = re.compile(r'[a-zA-Z]')
    
    bangla_chars = len(re.findall(bangla_pattern, text))
    english_chars = len(re.findall(english_pattern, text))
    
    # If more than 60% of characters are Bangla, consider it Bangla text
    total_chars = bangla_chars + english_chars
    if total_chars == 0:
        return 'english'  # Default to English for non-text content
    
    bangla_ratio = bangla_chars / total_chars
    english_ratio = english_chars / total_chars
    
    if bangla_ratio > 0.6:
        return 'bangla'
    elif english_ratio > 0.6:
        return 'english'
    else:
        return 'mixed'

def clean_text(text: str) -> str:
    """Clean text based on detected language."""
    lang = detect_language(text)
    if lang == 'bangla':
        return clean_bangla_text(text)
    elif lang == 'english':
        return clean_english_text(text)
    else:
        # For mixed text, apply both cleaners but be more conservative
        text = re.sub(r'[^\u0980-\u09FFa-zA-Z0-9\s.,!?-]', '', text)
        return remove_extra_whitespace(text)
