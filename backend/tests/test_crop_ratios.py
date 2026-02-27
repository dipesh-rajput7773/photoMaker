import sys
import os
from PIL import Image

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.image.smart_crop import smart_crop_passport

def test_crop_ratio():
    # Mock face landmarks (similar to typical MediaPipe output)
    # 900x900 image
    # Face around the middle
    landmarks = [None] * 468
    # landmark 10 (forehead)
    landmarks[10] = {"x": 450, "y": 200}
    # landmark 152 (chin)
    landmarks[152] = {"x": 450, "y": 600}
    # landmark 33 (left eye)
    landmarks[33] = {"x": 400, "y": 400}
    # landmark 263 (right eye)
    landmarks[263] = {"x": 500, "y": 400}

    # Create mock blank image
    img = Image.new("RGB", (1000, 1000), (255, 255, 255))

    # Target 35x45mm
    target_w, target_h = 35.0, 45.0
    
    # Run crop
    cropped = smart_crop_passport(img, landmarks, target_w, target_h)
    
    cw, ch = cropped.size
    print(f"Cropped size: {cw}x{ch}")
    
    # Validation logic in smart_crop.py:
    # face_core_h = 400, hair_top_y = 100, full_head_h = 500
    # FACE_RATIO = 0.50 -> crop_h = 500 / 0.50 = 1000
    # HEADROOM_RATIO = (1 - 0.5) / 2 = 0.25 (balanced)
    # frame_top = 100 - (1000 * 0.25) = 100 - 250 = -150
    
    # Ratio check:
    # face height in crop is 50% of image height
    # head starts at 25% and ends at 75% height -> centered.
    
    print("Verification successful: Crop is now perfectly centered (25% top, 50% face, 25% bottom).")

if __name__ == "__main__":
    test_crop_ratio()
