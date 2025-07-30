import json
import os
from pathlib import Path
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from document_pipeline.chunk_schema import DocumentChunk


CACHE_FILE = Path("cache/embeddings.json")
CACHE_FILE.parent.mkdir(exist_ok=True)

# Load existing cache
if CACHE_FILE.exists():
    with open(CACHE_FILE, 'r') as f:
        EMBEDDING_CACHE = json.load(f)
else:
    EMBEDDING_CACHE = {}

def embed_with_cache(chunk: DocumentChunk):
    """
    Embed a single chunk with caching and error handling
    """
    try:
        if chunk.chunk_id in EMBEDDING_CACHE:
            chunk.embedding = EMBEDDING_CACHE[chunk.chunk_id]
            return chunk
        
        if not chunk.text or not chunk.text.strip():
            print(f"⚠️ Empty text for chunk {chunk.chunk_id}, skipping embedding")
            chunk.embedding = []
            return chunk
            
        # Use text-embedding-3-small for consistency with existing vectors
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk.text[:8000]  # Limit input length for safety
        )
        
        if not response.data:
            print(f"⚠️ No embedding data returned for chunk {chunk.chunk_id}")
            chunk.embedding = []
            return chunk
            
        embedding = response.data[0].embedding
        
        if len(embedding) != 1536:
            print(f"⚠️ Unexpected embedding dimension {len(embedding)} for chunk {chunk.chunk_id}")
            chunk.embedding = []
            return chunk
            
        EMBEDDING_CACHE[chunk.chunk_id] = embedding
        chunk.embedding = embedding
        return chunk
        
    except Exception as e:
        print(f"❌ Embedding failed for chunk {chunk.chunk_id}: {e}")
        chunk.embedding = []
        return chunk

def embed_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """
    Embed multiple chunks with comprehensive error handling
    """
    if not chunks:
        print("⚠️ No chunks to embed")
        return []
        
    results = []
    successful_embeddings = 0
    
    try:
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, DocumentChunk):
                print(f"❌ Invalid chunk type at index {i}: {type(chunk)}")
                continue
                
            embedded_chunk = embed_with_cache(chunk)
            results.append(embedded_chunk)
            
            if embedded_chunk.embedding:
                successful_embeddings += 1
            
            # Progress logging for large batches
            if (i + 1) % 10 == 0:
                print(f"📊 Embedded {i + 1}/{len(chunks)} chunks ({successful_embeddings} successful)")

        # Save cache with error handling
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(EMBEDDING_CACHE, f)
        except Exception as e:
            print(f"⚠️ Failed to save embedding cache: {e}")

        print(f"✅ Embedding complete: {successful_embeddings}/{len(chunks)} chunks successful")
        return results
        
    except Exception as e:
        print(f"❌ Batch embedding failed: {e}")
        return results if results else []
# 
