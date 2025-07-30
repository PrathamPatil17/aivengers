import tiktoken
from typing import List, Dict

encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def recursive_split(text: str, chunk_size: int = 500, overlap: int = 150) -> List[Dict]:
    """
    Improved chunking with better semantic preservation and overlap
    Reduced chunk size but increased overlap for better context retention
    """
    splits = []
    current_pos = 0
    text_length = len(text)
    chunk_index = 0
    
    # Split text into sentences first to maintain semantic boundaries
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_text = ""
    current_start = 0
    
    for sentence in sentences:
        test_text = current_text + " " + sentence if current_text else sentence
        test_tokens = count_tokens(test_text)
        
        if test_tokens > chunk_size and current_text:
            # Save current chunk
            splits.append({
                "chunk_id": f"chunk_{chunk_index:04d}",
                "text": current_text.strip(),
                "token_count": count_tokens(current_text),
                "char_range": (current_start, current_start + len(current_text))
            })
            
            # Start new chunk with overlap
            overlap_tokens = encoding.encode(current_text)[-overlap:]
            overlap_text = encoding.decode(overlap_tokens) if overlap_tokens else ""
            current_text = overlap_text + " " + sentence if overlap_text else sentence
            current_start = current_start + len(current_text) - len(overlap_text) - len(sentence) - 1
            chunk_index += 1
        else:
            current_text = test_text
    
    # Add final chunk if there's remaining text
    if current_text.strip():
        splits.append({
            "chunk_id": f"chunk_{chunk_index:04d}",
            "text": current_text.strip(),
            "token_count": count_tokens(current_text),
            "char_range": (current_start, current_start + len(current_text))
        })
    
    return splits