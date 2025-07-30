from document_pipeline.vectorstore import query_similar_chunks
from document_pipeline.chunk_schema import DocumentChunk
import re
from typing import List, Dict, Any

def retrieve_relevant_chunks(query: str, top_k: int = 8):  # Increased default back to 8
    """
    Comprehensive retrieval system for detailed answers
    """
    print(f"🔍 Searching for: {query}")
    
    # Get embedding for the query
    from document_pipeline.embedding_cache import embed_with_cache
    
    temp_chunk = DocumentChunk(
        chunk_id="query", 
        text=query, 
        token_count=len(query.split()), 
        char_range=(0, len(query)), 
        embedding=[]
    )
    
    embedded_query = embed_with_cache(temp_chunk)
    query_vector = embedded_query.embedding

    print("🎯 Querying vector database...")
    # Increased search scope for comprehensive answers
    matches = query_similar_chunks(query_vector, top_k=25)  # Increased from 20 to 25
    
    print(f"📊 Found {len(matches)} matches")

    if not matches:
        print("❌ No matches found")
        return []

    # Use lower thresholds for comprehensive coverage
    query_lower = query.lower()
    if any(term in query_lower for term in ['what is', 'define', 'definition']):
        min_score = 0.10  # Slightly lower for definitions
    elif any(term in query_lower for term in ['how', 'procedure', 'process']):
        min_score = 0.08  # Lower for procedures
    else:
        min_score = 0.06  # Very low threshold for comprehensive coverage
    
    relevant_matches = [match for match in matches if match.score >= min_score]
    
    if not relevant_matches:
        print("⚠️ No matches above threshold, using top matches")
        relevant_matches = matches[:top_k * 2]
    else:
        print(f"✅ {len(relevant_matches)} matches above {min_score} threshold")
    
    # Enhanced relevance scoring
    def calculate_enhanced_relevance_score(match):
        text = match.metadata.get("text", "").lower()
        query_lower = query.lower()
        base_score = match.score
        
        # Extract key terms from query
        query_words = [word.strip("?.,!()") for word in query_lower.split() if len(word) > 2]
        
        # Word matching boost
        word_matches = sum(1 for word in query_words if word in text)
        word_boost = word_matches * 0.02
        
        # Domain-specific term boosting
        domain_terms = {
            'grace': 0.05, 'period': 0.04, 'waiting': 0.05, 'premium': 0.04,
            'coverage': 0.04, 'benefit': 0.04, 'claim': 0.04, 'exclusion': 0.04,
            'age': 0.03, 'limit': 0.03, 'hospital': 0.03, 'policy': 0.03,
            'renewal': 0.04, 'maternity': 0.04, 'sum': 0.03, 'insured': 0.03
        }
        
        domain_boost = sum(boost for term, boost in domain_terms.items() 
                          if term in query_lower and term in text)
        
        # Phrase matching boost
        if len(query_words) >= 2:
            for i in range(len(query_words) - 1):
                phrase = f"{query_words[i]} {query_words[i+1]}"
                if phrase in text:
                    domain_boost += 0.03
        
        return base_score + word_boost + domain_boost
    
    # Sort by enhanced relevance score
    relevant_matches.sort(key=calculate_enhanced_relevance_score, reverse=True)
    
    # Return top results
    final_chunks = relevant_matches[:top_k]
    print(f"📤 Returning {len(final_chunks)} most relevant chunks")
    
    # Debug: Show what we're returning
    if final_chunks:
        first_text = final_chunks[0].metadata.get("text", "")[:150]
        top_score = calculate_enhanced_relevance_score(final_chunks[0])
        print(f"🔍 Top result (score: {top_score:.3f}): {first_text}...")
    
    return [
        {
            "id": match.id,
            "score": calculate_enhanced_relevance_score(match),
            "text": match.metadata.get("text", "No text available")
        }
        for match in final_chunks
    ]
