from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, ValidationError
from typing import List, Optional
import httpx
import tempfile
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from document_pipeline.pipeline_runner import run_pipeline
from document_pipeline.retriever import retrieve_relevant_chunks
from document_pipeline.chunk_schema import DocumentChunk
from document_pipeline.embedding_cache import embed_with_cache
from database.service import get_database_service
from openai import OpenAI

load_dotenv()

# Initialize database service
db_service = get_database_service()

app = FastAPI(
    title="HackRX 6.0 - LLM-Powered Document Query System with PostgreSQL",
    description="Intelligent query-retrieval system for insurance, legal, HR, and compliance documents with PostgreSQL metadata storage",
    version="1.0.0",
    swagger_ui_parameters={
        "persistAuthorization": True
    }
)

# Add CORS middleware for web interface compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler for JSON validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle JSON validation errors with better error messages"""
    error_details = []
    
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        error_details.append(f"{field}: {message}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": error_details,
            "message": "Please check your JSON format and required fields",
            "required_format": {
                "documents": "string (URL to document)",
                "questions": ["array of strings (at least one question required)"]
            }
        }
    )

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple in-memory cache for similar questions (token optimization)
question_cache = {}

# Request/Response Models
class QueryRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]
    
    class Config:
        # Allow extra fields to be ignored instead of causing validation errors
        extra = "ignore"
    
    # Add validation for questions
    @validator('questions')
    def validate_questions(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one question is required')
        if len(v) > 50:  # Reasonable limit
            raise ValueError('Too many questions (max 50)')
        return v
    
    # Add validation for documents URL
    @validator('documents')
    def validate_documents(cls, v):
        if not v or not v.strip():
            raise ValueError('Document URL is required')
        return v.strip()

class QueryResponse(BaseModel):
    answers: List[str]

# Authentication
EXPECTED_TOKEN = "880b4911f53f0dc33bb443bfc2c5831f87db7bc9d8bf084d6f42acb6918b02f7"
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the Bearer token"""
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return credentials.credentials

async def download_document(url: str) -> str:
    """Download document from URL or handle local file path"""
    
    # Handle local file paths (for testing)
    if url.startswith('file://'):
        local_path = url.replace('file://', '')
        if os.path.exists(local_path):
            return local_path
        else:
            raise HTTPException(status_code=400, detail=f"Local file not found: {local_path}")
    
    # Handle relative paths for local testing
    if not url.startswith('http'):
        if os.path.exists(url):
            return url
        else:
            raise HTTPException(status_code=400, detail=f"Local file not found: {url}")
    
    # Handle HTTP/HTTPS URLs
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(response.content)
                return tmp_file.name
                
        except httpx.RequestError as e:
            raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=400, detail=f"HTTP error downloading document: {e.response.status_code}")

def generate_answer_with_context(question: str, relevant_chunks: List[dict]) -> str:
    """Generate comprehensive answer using OpenAI with balanced token usage"""
    
    # Check cache first to save tokens
    question_key = question.lower().strip()
    if question_key in question_cache:
        print("💾 Using cached answer")
        return question_cache[question_key]
    
    # Enhanced context preparation for better answers
    context_parts = []
    total_chars = 0
    max_context_chars = 3000  # Increased for better context coverage
    
    # Sort chunks by score and take top 6 for better coverage
    sorted_chunks = sorted(relevant_chunks, key=lambda x: x.get('score', 0), reverse=True)[:6]
    
    for i, chunk in enumerate(sorted_chunks):
        chunk_text = chunk['text'].strip()
        
        # Skip very short chunks
        if len(chunk_text) < 30:
            continue
            
        if total_chars + len(chunk_text) > max_context_chars:
            # More generous truncation for better context
            remaining_chars = max_context_chars - total_chars
            if remaining_chars > 200:
                chunk_text = chunk_text[:remaining_chars-10] + "..."
                context_parts.append(chunk_text)
            break
        else:
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
    
    context = "\n".join(context_parts)
    
    # Enhanced prompt for comprehensive answers
    prompt = f"""Based on the following policy document context, provide a comprehensive and detailed answer to the question. Include specific details, numbers, conditions, and timeframes when available.

Context from Policy Document:
{context}

Question: {question}

Instructions:
- Provide a complete, detailed answer based on the context
- Include specific numbers, percentages, timeframes, and conditions
- If multiple related pieces of information exist, combine them coherently  
- Be thorough but concise
- Only state "information not available" if absolutely no relevant information exists

Answer:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,  # Increased for more detailed answers
            temperature=0.2  # Slightly higher for more natural responses
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Cache the answer to save future tokens
        question_cache[question_key] = answer
        
        # Keep cache size manageable (max 50 entries)
        if len(question_cache) > 50:
            oldest_key = next(iter(question_cache))
            del question_cache[oldest_key]
        
        return answer
        
    except Exception as e:
        # Fallback: return context-based answer without GPT
        print(f"Warning: GPT generation failed ({str(e)}), using fallback")
        if relevant_chunks:
            fallback_answer = f"Based on the document: {relevant_chunks[0]['text'][:300]}..."
        else:
            fallback_answer = "Information not available in the provided document."
        
        # Cache fallback too
        question_cache[question_key] = fallback_answer
        return fallback_answer

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_document_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Main endpoint to process document queries with PostgreSQL logging
    
    1. Download document from provided URL
    2. Process document through pipeline (parse, chunk, embed, store)
    3. Log document processing to PostgreSQL
    4. For each question, retrieve relevant chunks and generate answers
    5. Log query session to PostgreSQL
    6. Return structured JSON response
    """
    
    temp_file_path = None
    processing_start_time = time.time()
    document_id = None
    
    try:
        # Validate input parameters
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=422, detail="At least one question is required")
        
        if not request.documents or not request.documents.strip():
            raise HTTPException(status_code=422, detail="Document URL is required")
        
        # Step 1: Download document
        print(f"📥 Downloading document from: {request.documents}")
        temp_file_path = await download_document(request.documents)
        
        # Get file size for logging
        file_size = os.path.getsize(temp_file_path) if temp_file_path else 0
        doc_hash = f"hackrx-doc-{hash(request.documents)}"
        
        # Step 2: Check if document already processed (Performance Optimization)
        print("🔍 Checking if document already processed...")
        existing_doc = db_service.get_document_by_url(request.documents)
        
        if existing_doc and existing_doc.processing_status == "completed":
            print(f"⚡ Document already processed ({existing_doc.chunks_created} chunks). Skipping processing.")
            chunks_count = existing_doc.chunks_created
            processing_time = 0  # No processing needed
            document_id = existing_doc.id
        else:
            # Process document through pipeline
            print("🔄 Processing document through pipeline...")
            processing_start = time.time()
            chunks = run_pipeline(temp_file_path, doc_id=doc_hash)
            processing_time = time.time() - processing_start
            chunks_count = len(chunks)
            print(f"✅ Document processed: {chunks_count} chunks created in {processing_time:.2f}s")
            
            # Log document processing to PostgreSQL (with fallback)
            try:
                document_id = db_service.log_document_processing(
                    document_url=request.documents,
                    file_size=file_size,
                    chunks_created=chunks_count,
                    processing_time=processing_time,
                    status="completed"
                )
                print("✅ Document processing logged to PostgreSQL")
            except Exception as e:
                print(f"⚠️ Failed to log document processing: {e}")
                document_id = None  # Continue without database logging
        
        # Step 4: Process each question (Optimized for Performance)
        query_start_time = time.time()
        answers = []
        
        for i, question in enumerate(request.questions):
            question_start = time.time()
            print(f"⚡ Processing question {i+1}/{len(request.questions)}: {question}")
            
            try:
                # Retrieve more chunks for comprehensive answers (increased from 5 to 8)
                relevant_chunks = retrieve_relevant_chunks(question, top_k=8)
                
                # Simplified debugging to reduce console output
                print(f"📊 Retrieved {len(relevant_chunks)} chunks")
                if relevant_chunks:
                    print(f"🔍 Top score: {relevant_chunks[0]['score']:.3f}")
                
                if not relevant_chunks:
                    answers.append("Information not available in the provided document.")
                    continue
                
                # Generate answer using GPT with context
                answer = generate_answer_with_context(question, relevant_chunks)
                answers.append(answer)
                
                question_time = time.time() - question_start
                print(f"⚡ Q{i+1} done in {question_time:.1f}s")  # Simplified logging
                
            except Exception as e:
                print(f"❌ Error Q{i+1}: {str(e)}")
                answers.append(f"Error: {str(e)}")
        
        # Step 5: Log query session to PostgreSQL (with fallback)
        query_time = time.time() - query_start_time
        try:
            db_service.log_query_session(
                document_id=document_id or 0,
                questions=request.questions,
                answers=answers,
                response_time=query_time,
                user_session=f"hackrx-session-{int(time.time())}"
            )
            print("✅ Query session logged to PostgreSQL")
        except Exception as e:
            print(f"⚠️ Failed to log query session: {e}")
            # Continue without failing the entire request
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        # Log failed processing
        if 'processing_start_time' in locals():
            processing_time = time.time() - processing_start_time
            db_service.log_document_processing(
                document_url=request.documents,
                file_size=file_size if 'file_size' in locals() else 0,
                chunks_created=0,
                processing_time=processing_time,
                status="failed",
                error_message=str(e)
            )
        
        print(f"❌ Pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
        
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print("🗑️ Cleaned up temporary file")
            except Exception as e:
                print(f"Warning: Failed to clean up temp file: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "HackRX 6.0 LLM-Powered Document Query System with PostgreSQL",
        "status": "running",
        "version": "1.0.0",
        "optimizations": "Token-optimized with GPT-3.5-turbo and intelligent caching",
        "technologies": {
            "backend": "FastAPI",
            "llm": "OpenAI GPT-3.5-turbo",  # Updated to reflect actual model
            "vector_database": "Pinecone", 
            "relational_database": "PostgreSQL",
            "document_processing": "PyMuPDF"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check with all system components"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {},
        "database_stats": {},
        "optimizations": {
            "model": "gpt-3.5-turbo",
            "caching": "enabled",
            "token_optimization": "active"
        }
    }
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    health_status["components"]["openai"] = "configured" if openai_key else "missing"
    
    # Check Pinecone
    pinecone_key = os.getenv("PINECONE_API_KEY")
    health_status["components"]["pinecone"] = "configured" if pinecone_key else "missing"
    
    # Test Pinecone connection
    try:
        from document_pipeline.vectorstore import index
        # Try a small test query
        test_vector = [0.0] * 1536
        result = index.query(vector=test_vector, top_k=1, include_metadata=False)
        health_status["components"]["pinecone_connection"] = "working"
    except Exception as e:
        health_status["components"]["pinecone_connection"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check PostgreSQL
    try:
        health_status["components"]["postgresql"] = "configured" if db_service.postgres_enabled else "unavailable"
        if db_service.postgres_enabled:
            stats = db_service.get_system_stats()
            health_status["database_stats"] = stats
    except Exception as e:
        health_status["components"]["postgresql"] = f"error: {str(e)}"
    
    # Check embedding cache
    try:
        from document_pipeline.embedding_cache import CACHE_FILE
        cache_exists = CACHE_FILE.exists()
        cache_size = CACHE_FILE.stat().st_size if cache_exists else 0
        health_status["components"]["embedding_cache"] = {
            "status": "available" if cache_exists else "empty",
            "size_bytes": cache_size
        }
    except Exception as e:
        health_status["components"]["embedding_cache"] = f"error: {str(e)}"
    
    # Check question cache
    health_status["components"]["question_cache"] = {
        "status": "active",
        "entries": len(question_cache)
    }
    
    return health_status

@app.get("/admin/documents")
async def get_document_history(
    limit: int = 10,
    token: str = Depends(verify_token)
):
    """Get recent document processing history from PostgreSQL"""
    return {
        "document_history": db_service.get_document_history(limit=limit),
        "postgresql_enabled": db_service.postgres_enabled
    }

@app.get("/admin/stats")
async def get_system_statistics(
    token: str = Depends(verify_token)
):
    """Get comprehensive system statistics from PostgreSQL"""
    return db_service.get_system_stats()

@app.post("/admin/setup-db") 
async def setup_database(
    token: str = Depends(verify_token)
):
    """Initialize PostgreSQL database tables"""
    success = db_service.setup_database()
    
    if success:
        return {
            "message": "PostgreSQL database tables created successfully",
            "status": "success"
        }
    else:
        raise HTTPException(
            status_code=500, 
            detail="Failed to setup PostgreSQL database. Check configuration and connection."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
