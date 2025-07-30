# document_pipeline/pipeline_runner.py

from .parser import extract_text_from_pdf
from .cleaner import remove_common_headers_footers, normalize_whitespace
from .chunker import recursive_split
from .embedding_cache import embed_chunks  # Uses embed_with_cache internally
from .chunk_schema import DocumentChunk
from .vectorstore import upsert_chunks
from .retriever import retrieve_relevant_chunks  # Optional: can be used to test retrieval
from typing import List, Optional
from datetime import datetime
import uuid


def run_pipeline(
    pdf_path: str,
    doc_id: Optional[str] = None,
    pipeline_version: str = "v1.0"
) -> List[DocumentChunk]:
    """
    Orchestrates the full document processing pipeline with comprehensive error handling:
    PDF → Clean Text → Chunks → Embeddings → Upsert to Pinecone → DocumentChunk
    """
    try:
        print(f"🔍 Loading: {pdf_path}")
        if not pdf_path or not isinstance(pdf_path, str):
            raise ValueError("Invalid PDF path provided")
        
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            raise ValueError("No text extracted from PDF")

        print("🧼 Cleaning text...")
        cleaned_pages = remove_common_headers_footers(pages)
        full_text = "\n".join(normalize_whitespace(p) for p in cleaned_pages)
        
        if not full_text.strip():
            raise ValueError("No meaningful text found after cleaning")

        print("✂️ Chunking text...")
        raw_chunks = recursive_split(full_text)
        
        if not raw_chunks:
            raise ValueError("No chunks created from text")

        # Create DocumentChunk objects before embedding
        now = datetime.utcnow()
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        document_chunks = []
        for i, chunk_data in enumerate(raw_chunks):
            try:
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_data["text"],
                    token_count=chunk_data["token_count"],
                    char_range=chunk_data["char_range"],
                    embedding=[],  # will be filled by embed_chunks
                    doc_id=doc_id,
                    pipeline_version=pipeline_version,
                    page_num=None,
                    section_title=None,
                    created_at=now
                )
                document_chunks.append(chunk)
            except Exception as e:
                print(f"⚠️ Warning: Failed to create chunk {i}: {e}")
                continue

        if not document_chunks:
            raise ValueError("No valid chunks created")

        print(f"🤖 Embedding {len(document_chunks)} chunks with cache...")
        embedded_chunks = embed_chunks(document_chunks)
        
        if not embedded_chunks:
            raise ValueError("No chunks were successfully embedded")

        print("📦 Uploading chunks to Pinecone index...")
        upsert_result = upsert_chunks(embedded_chunks)

        print(f"✅ Pipeline completed. {len(embedded_chunks)} chunks processed and uploaded.")
        return embedded_chunks
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise Exception(f"Document processing pipeline failed: {str(e)}")
