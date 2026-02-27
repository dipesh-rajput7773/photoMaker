import sys
import os
from fastapi import FastAPI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CPU before anything loads
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ORT_DISABLE_GPU"] = "1"

app = FastAPI(
    title="Photomaker AI Backend",
    description="Backend for Photogov.net clone with AI Photo Processing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/web", StaticFiles(directory="frontend", html=True), name="web")

# Health check FIRST — so Render detects the port
@app.get("/")
async def root():
    return {"status": "online", "message": "Welcome to Photogov.net Clone API"}

# LAZY import — load photos router AFTER server starts
@app.on_event("startup")
async def startup():
    logger.info("Server started, loading routes...")
    from app.api.v1 import photos
    app.include_router(photos.router, prefix="/api/v1/photos", tags=["Photos"])
    logger.info("Routes loaded successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
