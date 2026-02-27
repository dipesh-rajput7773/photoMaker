import sys
import os
from fastapi import FastAPI

# Add the parent directory (backend/) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.v1 import photos
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Photomaker AI Backend",
    description="Backend for Photogov.net clone with AI Photo Processing",
    version="1.0.0"
)

# CORS Configuration (React Frontend ke liye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Production mein specific domains rakhenge
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files (Public access to photos)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/web", StaticFiles(directory="frontend", html=True), name="web")

# Include Routers
app.include_router(photos.router, prefix="/api/v1/photos", tags=["Photos"])

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Welcome to Photogov.net Clone API",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

