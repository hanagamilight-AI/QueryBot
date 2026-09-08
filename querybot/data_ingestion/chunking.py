"""
Document chunking strategies for optimal retrieval
"""
from typing import List
from loguru import logger

from config.settings import settings


def chunk_document(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    strategy: str = "recursive"
) -> List[str]:
    """
    Split document into chunks for embedding and retrieval
    
    Args:
        text: Document text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks
        strategy: Chunking strategy ("recursive", "sentence", "paragraph")
        
    Returns:
        List of text chunks
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    if not text or len(text.strip()) == 0:
        return []
    
    if strategy == "recursive":
        return _recursive_chunk(text, chunk_size, chunk_overlap)
    elif strategy == "sentence":
        return _sentence_chunk(text, chunk_size, chunk_overlap)
    elif strategy == "paragraph":
        return _paragraph_chunk(text, chunk_size, chunk_overlap)
    else:
        logger.warning(f"Unknown chunking strategy: {strategy}, using recursive")
        return _recursive_chunk(text, chunk_size, chunk_overlap)


def _recursive_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Recursive character-based chunking with separators
    Tries to split on natural boundaries (paragraphs, sentences, words)
    """
    separators = [
        "\n\n",      # Paragraph breaks
        "\n",        # Line breaks
        ". ",        # Sentence ends
        "! ",        # Exclamation
        "? ",        # Questions
        " ",         # Word boundaries
        ""           # Character level (last resort)
    ]
    
    chunks = []
    
    def _split(text: str, separator_index: int = 0) -> List[str]:
        if len(text) <= chunk_size:
            return [text]
        
        if separator_index >= len(separators):
            # Fall back to character-level splitting
            return _fixed_split(text, chunk_size)
        
        separator = separators[separator_index]
        
        if separator and separator in text:
            parts = text.split(separator)
            result = []
            
            current_chunk = ""
            for part in parts:
                if len(current_chunk) + len(separator) + len(part) <= chunk_size:
                    if current_chunk:
                        current_chunk += separator + part
                    else:
                        current_chunk = part
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    
                    if len(part) > chunk_size:
                        # Recursively split large parts
                        result.extend(_split(part, separator_index + 1))
                    else:
                        current_chunk = part
            
            if current_chunk:
                result.append(current_chunk)
            
            return result
        else:
            # Try next separator
            return _split(text, separator_index + 1)
    
    chunks = _split(text)
    
    # Add overlap between chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, chunk_overlap)
    
    return chunks


def _sentence_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Sentence-based chunking
    Tries to keep complete sentences together
    """
    import re
    
    # Simple sentence tokenization
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            if len(sentence) > chunk_size:
                # Split long sentences
                chunks.extend(_fixed_split(sentence, chunk_size))
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Add overlap
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, chunk_overlap)
    
    return chunks


def _paragraph_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Paragraph-based chunking
    Keeps paragraphs together, splits large ones
    """
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            if len(paragraph) > chunk_size:
                # Split large paragraphs
                chunks.extend(_recursive_chunk(paragraph, chunk_size, 0))
            else:
                current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Add overlap
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, chunk_overlap)
    
    return chunks


def _fixed_split(text: str, chunk_size: int) -> List[str]:
    """
    Fixed-size character splitting (last resort)
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def _add_overlap(chunks: List[str], overlap_size: int) -> List[str]:
    """
    Add overlap between consecutive chunks
    """
    if len(chunks) <= 1:
        return chunks
    
    result = []
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            result.append(chunk)
        else:
            # Get overlap from previous chunk
            prev_chunk = result[-1]
            overlap_start = max(0, len(prev_chunk) - overlap_size)
            overlap_text = prev_chunk[overlap_start:]
            
            # Prepend overlap to current chunk
            if overlap_text and not chunk.startswith(overlap_text):
                new_chunk = overlap_text + " " + chunk
                result[-1] = prev_chunk  # Keep previous as is
                result.append(new_chunk)
            else:
                result.append(chunk)
    
    return result


# Test function
if __name__ == "__main__":
    sample_text = """
    This is a sample political document. It contains multiple paragraphs.
    
    The first paragraph discusses election results. Party A won 45% of the vote.
    Party B secured 38%. Party C got the remaining 17%.
    
    The second paragraph covers voter sentiment. Surveys show high engagement.
    Youth turnout increased by 15% compared to previous elections.
    
    This is a very long sentence that might need to be split because it exceeds the normal chunk size limit and contains too much information to be processed effectively in a single embedding vector without losing semantic coherence or exceeding token limits.
    """
    
    chunks = chunk_document(sample_text, chunk_size=100, chunk_overlap=20)
    
    print(f"Generated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} ({len(chunk)} chars):")
        print(chunk)
        print("-" * 50)
