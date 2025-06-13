from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import sqlite3
import uuid
from datetime import datetime
import asyncio
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()

app = FastAPI(title="PDF Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories and database
Path("uploads").mkdir(exist_ok=True)
Path("faiss_index").mkdir(exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect("pdf_qa.db")
    conn.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT,
            file_path TEXT,
            upload_date TEXT,
            status TEXT DEFAULT 'pending'
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Global vector store
vector_store = None

# Initialize components
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # <-- Use correct model name for embeddings
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # This is correct for chat/completions
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

async def process_pdf(doc_id: str, file_path: str):
    """Background task to process PDF"""
    global vector_store
    
    try:
        # Update status to processing
        conn = sqlite3.connect("pdf_qa.db")
        conn.execute("UPDATE documents SET status = 'processing' WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()
        
        # Load and split PDF
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        texts = text_splitter.split_documents(documents)
        
        # Add metadata
        for text in texts:
            text.metadata['doc_id'] = doc_id
            text.metadata['source'] = file_path
        
        # Create or update vector store
        if vector_store is None:
            vector_store = FAISS.from_documents(texts, embeddings)
            vector_store.save_local("faiss_index")
        else:
            # Load existing and add new documents
            temp_store = FAISS.from_documents(texts, embeddings)
            vector_store.merge_from(temp_store)
            vector_store.save_local("faiss_index")
        
        # Update status to completed
        conn = sqlite3.connect("pdf_qa.db")
        conn.execute("UPDATE documents SET status = 'completed' WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()
        
        print(f"Successfully processed document {doc_id}")
        
    except Exception as e:
        # Update status to failed
        conn = sqlite3.connect("pdf_qa.db")
        conn.execute("UPDATE documents SET status = 'failed' WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()
        print(f"Error processing document {doc_id}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load existing vector store on startup."""
    global vector_store
    if os.path.exists("faiss_index/index.faiss"):
        try:
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            print("Loaded existing FAISS index")
        except Exception as e:
            print(f"Could not load existing index: {e}")
    yield
    # (Optional) Add any shutdown/cleanup logic here

app = FastAPI(title="PDF Q&A API", lifespan=lifespan)

@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, pdf: UploadFile = File(...)):
    """Upload PDF and process in background"""
    
    if not pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique ID and filename
    doc_id = str(uuid.uuid4())
    filename = f"{doc_id}_{pdf.filename}"
    file_path = f"uploads/{filename}"
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            content = await pdf.read()
            buffer.write(content)
        
        # Save to database
        conn = sqlite3.connect("pdf_qa.db")
        conn.execute(
            "INSERT INTO documents (id, filename, file_path, upload_date, status) VALUES (?, ?, ?, ?, ?)",
            (doc_id, pdf.filename, file_path, datetime.now().isoformat(), "pending")
        )
        conn.commit()
        conn.close()
        
        # Process in background
        background_tasks.add_task(process_pdf, doc_id, file_path)
        
        return {
            "message": "PDF uploaded successfully",
            "document_id": doc_id,
            "filename": pdf.filename,
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/chat")
async def chat(message: str = Query(..., description="Your question")):
    """Chat with uploaded documents"""
    
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if vector_store is None:
        raise HTTPException(status_code=404, detail="No documents uploaded yet. Please upload PDFs first.")
    
    try:
        # Search for relevant documents
        docs = vector_store.similarity_search(message, k=5)
        
        if not docs:
            return {"message": "No relevant information found in uploaded documents."}
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response
        prompt = f"""Based on the following context from uploaded PDF documents, answer the user's question.
        If the context doesn't contain enough information, say so clearly.
        
        Context:
        {context}
        
        Question: {message}
        
        Answer:"""
        
        response = llm.invoke(prompt)
        
        return {
            "message": response.content,
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get('source', 'Unknown')
                } for doc in docs[:3]
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    conn = sqlite3.connect("pdf_qa.db")
    cursor = conn.execute("SELECT id, filename, upload_date, status FROM documents ORDER BY upload_date DESC")
    documents = [
        {
            "id": row[0],
            "filename": row[1],
            "upload_date": row[2],
            "status": row[3]
        }
        for row in cursor.fetchall()
    ]
    conn.close()
    return {"documents": documents}

@app.get("/")
async def root():
    return {"message": "PDF Q&A API - Upload PDFs and ask questions!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)