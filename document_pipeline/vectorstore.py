# document_pipeline/vectorstore.py

import os
from dotenv import load_dotenv
try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    # Fallback for older versions
    import pinecone
    Pinecone = pinecone.Pinecone
    ServerlessSpec = pinecone.ServerlessSpec
from document_pipeline.chunk_schema import DocumentChunk
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "hackathon-doc-index")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
existing_indexes = [index.name for index in pc.list_indexes()]
if PINECONE_INDEX not in existing_indexes:
    print(f"📦 Creating Pinecone index: {PINECONE_INDEX}")
    try:
        from pinecone import ServerlessSpec
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✅ Successfully created index: {PINECONE_INDEX}")
        # Wait a moment for index to be ready
        import time
        time.sleep(10)
    except Exception as e:
        print(f"❌ Failed to create index: {e}")
        raise

# Connect to index
try:
    index = pc.Index(PINECONE_INDEX)
    print(f"✅ Connected to Pinecone index: {PINECONE_INDEX}")
except Exception as e:
    print(f"❌ Failed to connect to index: {e}")
    raise

def upsert_chunks(chunks: list[DocumentChunk]):
    """
    Upsert chunks to Pinecone with comprehensive error handling
    """
    if not chunks:
        print("⚠️ No chunks to upsert")
        return False
        
    try:
        batch_size = 100
        vectors = []
        
        for chunk in chunks:
            if not chunk.embedding:
                print(f"⚠️ Skipping chunk {chunk.chunk_id} - no embedding")
                continue
                
            if len(chunk.embedding) != 1536:
                print(f"⚠️ Skipping chunk {chunk.chunk_id} - invalid embedding dimension: {len(chunk.embedding)}")
                continue
                
            vectors.append({
                "id": chunk.chunk_id,
                "values": chunk.embedding,
                "metadata": {
                    "text": chunk.text[:1000],  # Limit metadata size
                    "token_count": chunk.token_count,
                    "char_start": chunk.char_range[0],
                    "char_end": chunk.char_range[1],
                }
            })

        if not vectors:
            print("❌ No valid vectors to upsert")
            return False

        def upsert_batch(batch):
            try:
                index.upsert(vectors=batch)
                return True
            except Exception as e:
                print(f"❌ Batch upsert failed: {e}")
                return False

        batches = [vectors[i:i + batch_size] for i in range(0, len(vectors), batch_size)]
        successful_batches = 0
        
        with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers for stability
            results = list(executor.map(upsert_batch, batches))
            successful_batches = sum(results)

        print(f"✅ Upserted {len(vectors)} chunks to Pinecone index '{PINECONE_INDEX}' ({successful_batches}/{len(batches)} batches successful)")
        return successful_batches == len(batches)
        
    except Exception as e:
        print(f"❌ Upsert operation failed: {e}")
        return False

def query_similar_chunks(query_embedding: list[float], top_k: int = 10):
    """
    Enhanced Pinecone query with comprehensive error handling
    """
    if not query_embedding:
        print("❌ No query embedding provided")
        return []
        
    if len(query_embedding) != 1536:
        print(f"❌ Invalid query embedding dimension: {len(query_embedding)}")
        return []
    
    try:
        response = index.query(
            vector=query_embedding,
            top_k=min(top_k, 100),  # Allow larger retrieval set for better reranking
            include_metadata=True,
            include_values=False,  # Don't need embedding values, saves bandwidth
            namespace=""  # Use default namespace
        )
        
        if not response or not response.matches:
            print("⚠️ No matches found in Pinecone")
            return []
        
        # Filter out very low similarity matches early
        filtered_matches = [
            match for match in response.matches 
            if match.score > 0.05  # Lower threshold for more results
        ]
        
        print(f"📊 Pinecone returned {len(response.matches)} matches, {len(filtered_matches)} above threshold")
        return filtered_matches
        
    except Exception as e:
        print(f"❌ Pinecone query failed: {e}")
        # Return empty list to allow the system to continue gracefully
        return []
