import numpy as np
import cv2
from PIL import Image, ImageFilter
from rembg import remove, new_session
import io
import logging

logger = logging.getLogger(__name__)

# ── Pre-load session for 'isnet-general-use' (Highest Quality Neural Net) ──
try:
    # isnet-general-use is widely considered the best for high-res details/hair
    session = new_session("isnet-general-use")
    logger.info("Loaded IS-Net General Use session.")
except Exception as e:
    logger.warning(f"Failed to load IS-Net, falling back to silueta: {e}")
    try:
        session = new_session("silueta")
    except:
        session = None

def remove_background(img: Image.Image) -> Image.Image:
    """
    ABSOLUTE CUT AI-GRADE ENGINE:
    Steep-Curve Thresholding + High-Tension Sigmoid + Variance-Aware De-bleeding.
    Eliminates "neural fog" and delivers professional manual-cut crispness.
    """
    try:
        # 1. Base Neural Segmentation (Strict Thresholds)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        
        # We increase erode_size to 2 for a cleaner neural base and tighter thresholds
        result_bytes = remove(
            img_byte_arr.getvalue(),
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=250, # Absolute Confidence Only
            alpha_matting_background_threshold=5,   # Kill background early
            alpha_matting_erode_size=2               # Nip halos at source
        )
        rgba = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
        
        # 2. Extract Data
        orig_arr = np.array(img).astype(np.float32)
        alpha_base = np.array(rgba.split()[3]).astype(np.float32) / 255.0
        
        # 3. VARIANCE-AWARE DECONTAMINATION (Kill "Shadow Fog")
        # We only apply color replacement where there is high local variance (actual hair strands).
        core_mask = (alpha_base > 0.95).astype(np.uint8)
        if np.any(core_mask):
            subject_only = orig_arr.copy()
            subject_only[alpha_base < 0.95] = 0
            
            # Deep Dilation for Clean Plate
            clean_dilated = cv2.dilate(subject_only, np.ones((7, 7), np.uint8), iterations=3)
            clean_plate = cv2.bilateralFilter(np.clip(clean_dilated, 0, 255).astype(np.uint8), 9, 75, 75).astype(np.float32)
            
            # Variance Map (Detects Hair vs Empty Space)
            gray = cv2.cvtColor(np.clip(orig_arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            var = cv2.GaussianBlur(gray**2, (5, 5), 0) - cv2.GaussianBlur(gray, (5, 5), 0)**2
            # Only blend where there is detail (hair strands)
            var_weight = np.clip(var * 20, 0, 1)
            
            # Use a very steep curve (3.5) for the color blend
            blend_weight = alpha_base[..., None] ** 3.5
            color_target = orig_arr * blend_weight + clean_plate * (1.0 - blend_weight)
            decontam_rgb = color_target * var_weight + orig_arr * (1.0 - var_weight)
            decontam_rgb = np.clip(decontam_rgb, 0, 255)
        else:
            decontam_rgb = orig_arr

        # 4. STUDIO-SHARP MATTING
        guide = orig_arr / 255.0
        # Narrower Radii for absolute precision
        alpha_precise = cv2.ximgproc.guidedFilter(guide, alpha_base, radius=5, eps=1e-6)
        
        # 5. ABSOLUTE CONTRACTED SIGMOID
        # Midpoint 0.58 significantly contracts the mask to remove the fuzzy "glow".
        # k=25 provides high-tension professional edges.
        k = 25 
        midpoint = 0.58
        final_alpha_f = 1 / (1 + np.exp(-k * (alpha_precise - midpoint)))
        
        # 6. FOG ELIMINATION (Zero-Tolerance)
        # Any alpha below 5% is noise/fog. Force to 0.
        final_alpha_f[final_alpha_f < 0.05] = 0
        final_alpha_f = np.clip(final_alpha_f, 0, 1)

        # 7. Composite Final
        final_rgb = decontam_rgb * np.dstack([final_alpha_f] * 3)
        final_rgba = np.dstack((
            final_rgb.astype(np.uint8),
            (final_alpha_f * 255).astype(np.uint8)
        ))
        
        return Image.fromarray(final_rgba, "RGBA")
        
    except Exception as e:
        logger.error(f"Absolute Cut Engine failed: {e}")
        # Standard high-quality fallback
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        result_bytes = remove(img_byte_arr.getvalue(), session=session, alpha_matting=True)
        return Image.open(io.BytesIO(result_bytes)).convert("RGBA")

def apply_background_color(img_rgba: Image.Image, color_hex: str = "#eeeeee") -> Image.Image:
    """Composition onto target background with refined light-gray default."""
    try:
        background = Image.new("RGB", img_rgba.size, color_hex)
        background.paste(img_rgba, mask=img_rgba.split()[3])
        return background
    except Exception as e:
        logger.error(f"Failed to apply background color: {e}")
        return img_rgba.convert("RGB")
