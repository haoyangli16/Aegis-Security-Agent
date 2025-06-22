from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import routers
try:
    from .analyze_stream import router as analyze_router
    from .upload_video import router as upload_router
    logger.info("✅ Successfully imported all routers")
except ImportError as e:
    logger.error(f"❌ Failed to import routers: {e}")
    raise

app = FastAPI(
    title="viclab Real-time Video Analysis API",
    description="API for real-time video analysis using SmolVLM",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analyze_router, prefix="/api", tags=["analysis"])
app.include_router(upload_router, prefix="/api", tags=["upload"])

@app.get("/")
async def root():
    return {
        "message": "viclab Real-time Video Analysis API",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 viclab API is starting up...")
    
    # Create necessary directories
    os.makedirs("temp", exist_ok=True)
    logger.info("📁 Directories created")
    
    logger.info("✅ viclab API startup completed successfully")

# Shutdown event  
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 viclab API is shutting down...")
    logger.info("✅ viclab API shutdown completed")
