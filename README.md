# Palm Mind AI Backend Assessment

An intelligent backend service built with **FastAPI**, **Qdrant**, **Redis**, **SQLite**, and **Groq LLM** supporting document ingestion, multi-turn RAG question answering, and automated interview scheduling.

---

## **Tech Stack**
| Component | Choice | Reason |
|---|---|---|
| **Framework** | FastAPI | High performance, automatic OpenAPI docs, strict typing |
| **Vector DB** | Qdrant (Docker) | Local, fast, developer-friendly vector storage |
| **Relational DB** | SQLite via SQLAlchemy 2.0 | Zero-setup, transactional persistence for metadata & bookings |
| **Chat Memory** | Redis (Docker) | Ephemeral conversation history & temporary booking state with TTL |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local 384-dimensional embeddings (free, no API key required) |
| **LLM Client** | Groq (`qwen/qwen3.8-27b`) via OpenAI SDK | High-speed inference behind an swappable `LLMClient` adapter |

---

## **Project Structure**
```
app/
  main.py            # FastAPI app initialization, lifespan, router inclusion
  config.py          # Environment settings using pydantic-settings
  db.py              # SQLAlchemy 2.0 models (DocumentRecord, BookingRecord) & DB session
  schemas.py         # Pydantic validation models and enums
  routers/
    ingestion.py     # POST /ingest (PDF/TXT upload, chunking, embeddings, Qdrant upsert)
    chat.py          # POST /chat (Multi-turn RAG & interview booking)
  services/
    extraction.py    # PDF and TXT text extraction
    chunking.py      # Fixed-size & sentence-based chunkers with factory pattern
    embeddings.py    # Lazy-loaded sentence transformer embedder
    vector_store.py  # Qdrant client wrapper (collection init, upsert, vector search)
    memory.py        # Redis session history and booking state management
    llm.py           # Swappable LLM client adapter
docker-compose.yml   # Qdrant and Redis services
requirements.txt
.env.example
README.md
```

---

## **Setup & Installation Instructions**

### **Prerequisites**
- Python 3.10+
- Docker & Docker Compose (for Qdrant & Redis)
- Groq API Key (free tier from [console.groq.com](https://console.groq.com))

### **1. Clone & Set Up Virtual Environment**
```powershell
git clone <repository-url>
cd palm-mind-ai-backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### **2. Configure Environment Variables**
Copy `.env.example` to `.env` and fill in your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/1
DATABASE_URL=sqlite:///./app.db
```

### **3. Start Infrastructure (Qdrant & Redis)**
```powershell
docker compose up -d
```

### **4. Run the FastAPI Server**
```powershell
uvicorn app.main:app --reload
```
The interactive API documentation will be available at: **http://localhost:8000/docs**

---

## **API Documentation**

### **1. Health Check**
- **Endpoint**: `GET /health`
- **Description**: Verifies API status.

### **2. Document Ingestion API**
- **Endpoint**: `POST /ingest`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: PDF or TXT file (`UploadFile`)
  - `chunking_strategy`: `fixed` or `sentence` (`Form`)
- **Response**:
  ```json
  {
    "document_id": 1,
    "chunk_count": 12,
    "strategy": "fixed"
  }
  ```

### **3. Conversational RAG & Booking API**
- **Endpoint**: `POST /chat`
- **Content-Type**: `application/json`
- **Request**:
  ```json
  {
    "session_id": "optional-uuid-session-id",
    "message": "What is the primary topic of the document?"
  }
  ```
- **Response**:
  ```json
  {
    "session_id": "123e4567-e89b-12d3-a456-426614174000",
    "intent": "question",
    "answer": "The primary topic is...",
    "sources": ["document.pdf"],
    "booking": null
  }
  ```

---

## **Architecture Flow Diagram**

```
[ User Request ] 
       │
       ▼
  POST /chat ──> Load Redis Chat History & Booking State
       │
       ▼
LLM Router / Analyzer (Intent Classification & Query Rewriting)
       ├──> If Intent = "question":
       │         ├──> Embed query (sentence-transformers)
       │         ├──> Vector Search (Qdrant)
       │         └──> Construct Grounded Prompt + LLM Generation -> Answer
       │
       └──> If Intent = "booking":
                 ├──> Merge & Validate Fields (Name, Email, Date, Time)
                 ├──> If incomplete: Ask clarifying question
                 └──> If complete: Save to SQLite (BookingRecord) & Clear Redis State
```

---

## **Testing Scenarios**

1. **Ingestion Test**: Upload a `.pdf` file with `fixed` strategy and a `.txt` file with `sentence` strategy. Confirm chunk counts differ and records are saved in SQLite (`app.db`) and Qdrant.
2. **RAG Question Flow**: Ask a question answerable by the uploaded document. Confirm the bot replies with accurate information and lists source filenames.
3. **Multi-Turn Query Rewriting**: Ask a follow-up question (e.g., *"What about its pricing?"*). Confirm the LLM correctly rewrites it into a standalone query using chat history.
4. **Interview Booking Flow**: 
   - Say *"I want to book an interview"*.
   - Provide name, email, date (`YYYY-MM-DD`), and time (`HH:MM`) across turns.
   - Test invalid email/past date validation rejection.
   - Confirm successful booking creates exactly one row in the SQLite `bookings` table.
5. **Session Persistence**: Restart the server, reuse the same `session_id`, and verify chat history persists in Redis.

---

## **Interview Q&A Guide**

- **Why Qdrant?** Simple, lightweight, containerized, high-performance vector database with no cloud dependency or API keys required.
- **Why two chunking strategies?** `fixed` size chunking splits text uniformly by character count with overlap, whereas `sentence` chunking respects semantic sentence boundaries, preventing mid-sentence cuts.
- **Why manual RAG instead of RetrievalQAChain?** Manual RAG (`embed -> search -> prompt -> LLM`) gives complete control over prompt formatting, source attribution, error handling, and debugging without heavy abstraction layers.
- **How does multi-turn work?** We store rolling conversation history in Redis and use an LLM router to rewrite follow-up user messages into self-contained standalone queries before vector search.
- **How is booking state managed?** Temporary slot collection is tracked in Redis hashes with TTL. Once all four mandatory fields (`name`, `email`, `date`, `time`) pass validation, the record is durably committed to SQLite.
