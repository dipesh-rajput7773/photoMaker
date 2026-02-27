from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.ext.asyncio import AsyncSession
from PIL import Image
import io
import os
import uuid
import shutil
from datetime import datetime

from app.db.database import get_db
from app.models.photo import Photo
from app.image.background import remove_background, apply_background_color
from app.image.face_detect import detect_face_bbox, get_face_landmarks
from app.image.smart_crop import smart_crop_passport
from app.image.overlays import draw_overlays

router = APIRouter()

UPLOAD_DIR = "uploads/original"
PROCESSED_DIR = "uploads/processed"

@router.post("/upload")
async def upload_photo(
    file: UploadFile = File(...),
    target_width_mm: float = Form(50.8), # 600px at 300 DPI
    target_height_mm: float = Form(50.8), # 600px at 300 DPI
    target_bg_color: str = Form("#ffffff"),
    draw_debug_overlays: bool = Form(True),
    transparent_png: bool = Form(False),
    session_id: str = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Main Photo Upload & Core Process.
    1. Saves original photo.
    2. Runs AI Background Removal.
    3. Detects Face for initial validation.
    4. Saves record to DB.
    """
    
    # 1. Validation: File Type Check
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate Unique ID
    photo_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    original_filename = f"{photo_id}{file_extension}"
    original_path = os.path.join(UPLOAD_DIR, original_filename)

    # 2. Save Original File
    try:
        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    # 3. Core Processing Logic
    try:
        # Load Image for processing
        with Image.open(original_path) as img:
            img = img.convert("RGB")
            
            # A. Detect Face & Landmarks
            face_bbox = detect_face_bbox(img)
            face_landmarks = get_face_landmarks(img)
            
            if not face_bbox or not face_landmarks:
                # Passport photo ke liye face hona zaruri hai
                status = "failed"
                compliance_result = {"error": "No face or landmarks detected"}
            else:
                # 1. SMART CROP (ON ORIGINAL IMAGE)
                # Following PhotoGov order: Detect Landmarks -> Smart Crop
                cropped_img = smart_crop_passport(
                    img, 
                    face_landmarks, 
                    target_width_mm, 
                    target_height_mm
                )
                
                # 2. BACKGROUND PROCESSING (ON CROPPED IMAGE)
                # Following PhotoGov order: Smart Crop -> Remove BG -> Apply White BG
                img = remove_background(cropped_img)
                
                # Determine output format and extension
                save_format = "JPEG"
                save_ext = "_processed.jpg"
                
                if transparent_png:
                    save_format = "PNG"
                    save_ext = "_processed.png"
                    # We keep the RGBA image from remove_background
                    final_img = img
                else:
                    final_img = apply_background_color(img, target_bg_color)
                    final_img = final_img.convert("RGB")
                
                # Save Processed Image
                processed_path = os.path.join(PROCESSED_DIR, f"{photo_id}{save_ext}")
                
                # 3. APPLY OVERLAYS (FOR VERIFICATION)
                # We re-detect landmarks on the final square image for precise overlay placement
                w_f, h_f = final_img.size
                p_h, p_v = int(w_f * 0.2), int(h_f * 0.2)
                
                # For overlay padding, use white for JPEG, transparent for PNG
                pad_color = (255, 255, 255) if not transparent_png else (0, 0, 0, 0)
                padded_temp = Image.new(final_img.mode, (w_f + 2*p_h, h_f + 2*p_v), pad_color)
                padded_temp.paste(final_img, (p_h, p_v))
                
                final_landmarks = get_face_landmarks(padded_temp)
                if final_landmarks:
                    # Shift landmarks back to cropped image coordinate space
                    for lm in final_landmarks:
                        lm['x'] -= p_h
                        lm['y'] -= p_v
                        
                    if draw_debug_overlays:
                        final_img = draw_overlays(
                            final_img, 
                            final_landmarks, 
                            target_width_mm, 
                            target_height_mm
                        )
                
                if save_format == "JPEG":
                    final_img.save(processed_path, format="JPEG", quality=95, subsampling=0)
                else:
                    # Save as PNG with same quality level
                    final_img.save(processed_path, format="PNG")
                
                status = "ready"
                compliance_result = {"face_found": True, "bbox": face_bbox}

        # 4. Save to Database
        new_photo = Photo(
            id=photo_id,
            session_id=session_id,
            target_width_mm=target_width_mm,
            target_height_mm=target_height_mm,
            target_bg_color=target_bg_color,
            original_path=original_path,
            processed_path=processed_path if status == "ready" else None,
            status=status,
            compliance_result=compliance_result,
            compliance_score=100 if status == "ready" else 0,
            created_at=datetime.utcnow()
        )
        
        db.add(new_photo)
        await db.commit()
        await db.refresh(new_photo)

        return {
            "photo_id": new_photo.id,
            "status": new_photo.status,
            "compliance": new_photo.compliance_result,
            "processed_url": f"/uploads/processed/{os.path.basename(new_photo.processed_path)}" if new_photo.processed_path else None
        }

    except Exception as e:
        await db.rollback()
        # Clean up if DB fails?
        raise HTTPException(status_code=500, detail=f"Processing Pipeline Error: {e}")

@router.get("/{photo_id}")
async def get_photo_status(photo_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get the current status of a photo job.
    """
    result = await db.execute(f"SELECT * FROM photos WHERE id = '{photo_id}'")
    photo = result.first()
    
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
        
    return {
        "id": photo.id,
        "status": photo.status,
        "compliance": photo.compliance_result,
        "processed_url": f"/uploads/processed/{os.path.basename(photo.processed_path)}" if photo.processed_path else None
    }
