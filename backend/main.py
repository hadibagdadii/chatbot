from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging

from .database import load_csv_to_df, persist_to_sqlite
from .retrieval import create_or_load_faiss, retrieve_semantic
from .chatbot import stream_response, is_technical_query
from .config import DB_PATH, CSV_PATH, VSTORE_DIR
from .models import ChatRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Comet Support Chatbot (Local Streaming)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vectorstore placeholder
app.state.vectorstore = None

@app.on_event("startup")
async def startup_event():
    """Initialize database and vectorstore with error handling"""
    try:
        logger.info("🚀 Starting Comet Support Chatbot...")
        
        # Check CSV exists
        if not CSV_PATH.exists():
            raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
        logger.info(f"✓ Found CSV at {CSV_PATH}")
        
        # Load and persist data
        logger.info("Loading CSV data...")
        df = load_csv_to_df(CSV_PATH)
        logger.info(f"✓ Loaded {len(df)} records")
        
        logger.info("Persisting to SQLite...")
        persist_to_sqlite(df, DB_PATH)
        logger.info(f"✓ Database saved to {DB_PATH}")
        
        # Create vectorstore
        logger.info("Building/loading FAISS vectorstore...")
        app.state.vectorstore = create_or_load_faiss(df, VSTORE_DIR)
        logger.info(f"✓ Vectorstore ready at {VSTORE_DIR}")
        
        logger.info("✅ Startup complete - ready to serve requests!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    index_path = frontend_dir / "index.html"
    
    if not index_path.exists():
        raise HTTPException(
            status_code=500, 
            detail=f"Frontend not found at {index_path}"
        )
    
    return index_path.read_text(encoding="utf-8")


@app.get("/health")
async def health_check():
    """Health check endpoint for debugging"""
    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    return {
        "status": "ok",
        "vectorstore_loaded": app.state.vectorstore is not None,
        "frontend_dir": str(frontend_dir),
        "frontend_exists": frontend_dir.exists(),
        "csv_path": str(CSV_PATH),
        "csv_exists": CSV_PATH.exists(),
        "db_path": str(DB_PATH),
        "db_exists": DB_PATH.exists()
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """Handle chat requests with streaming response"""
    if app.state.vectorstore is None:
        raise HTTPException(
            status_code=503, 
            detail="Vectorstore not initialized"
        )
    
    try:
        query = req.query
        logger.info(f"Received query: {query}")
        
        # Check if this is a technical query that needs semantic search
        if is_technical_query(query):
            logger.info("Technical query detected - running semantic search")
            vs = app.state.vectorstore
            agg = retrieve_semantic(query, vs)
            return StreamingResponse(
                stream_response(query, agg), 
                media_type="text/plain"
            )
        else:
            logger.info("Casual query detected - responding conversationally")
            # For casual queries, don't run expensive semantic search
            return StreamingResponse(
                stream_response(query, agg=None), 
                media_type="text/plain"
            )
            
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files LAST (after all routes)
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if FRONTEND_DIR.exists():
    app.mount(
        "/static", 
        StaticFiles(directory=FRONTEND_DIR), 
        name="static"
    )
    logger.info(f"Static files mounted from {FRONTEND_DIR}")
else:
    logger.warning(f"Frontend directory not found at {FRONTEND_DIR}")