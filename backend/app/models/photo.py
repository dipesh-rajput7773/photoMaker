from sqlalchemy import Column, String, Integer, DateTime, JSON, DECIMAL
from sqlalchemy.sql import func
from app.db.database import Base
import uuid

class Photo(Base):
    __tablename__ = "photos"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(100), index=True) # 
    
    # Payload se aane wali dynamic specs yahan store karenge
    target_width_mm = Column(DECIMAL(6, 2))
    target_height_mm = Column(DECIMAL(6, 2))
    target_bg_color = Column(String(7), default="#FFFFFF")
    
    # File Paths
    original_path = Column(String(500))
    processed_path = Column(String(500))
    
    # Status: 'uploaded', 'processing', 'ready', 'failed'
    status = Column(String(20), default="uploaded")
    
    # Compliance results (MediaPipe landmarks and checks)
    compliance_result = Column(JSON) 
    compliance_score = Column(Integer)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True))

    def __repr__(self):
        return f"<Photo {self.id} Status: {self.status}>"