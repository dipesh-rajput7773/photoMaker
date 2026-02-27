import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Landmark 10 tracks forehead, not hair. This baseline buffer protects common styles.
DEFAULT_HAIR_BUFFER = 0.32


def mm_to_px(mm, dpi=300):
    """Utility to convert millimeters to pixels for printing."""
    return int((mm * dpi) / 25.4)


def estimate_hair_top_y(
    img: Image.Image,
    eye_center_x: float,
    top_head_y: float,
    chin_y: float,
    face_core_h: float,
) -> float:
    """
    Estimate visible top-of-hair with robust fallback.

    1) Start with a landmark estimate.
    2) Refine with a pixel scan in a face-centered vertical band when possible.
    """
    fallback_hair_top = float(top_head_y - (face_core_h * DEFAULT_HAIR_BUFFER))

    try:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        h, w = rgb.shape[:2]
        if h < 20 or w < 20:
            return fallback_hair_top

        border_samples = np.concatenate(
            (
                rgb[:8, :, :].reshape(-1, 3),
                rgb[:, :4, :].reshape(-1, 3),
                rgb[:, -4:, :].reshape(-1, 3),
            ),
            axis=0,
        )
        bg_color = np.median(border_samples, axis=0)
        color_distance = np.linalg.norm(rgb - bg_color, axis=2)

        band_half_w = max(30, int(face_core_h * 0.45))
        x1 = max(0, int(eye_center_x - band_half_w))
        x2 = min(w, int(eye_center_x + band_half_w))
        if x2 - x1 < 12:
            return fallback_hair_top

        max_scan_y = min(h, max(1, int(chin_y)))
        if max_scan_y < 5:
            return fallback_hair_top

        dynamic_floor = float(np.percentile(color_distance[:8, :], 97))
        fg_threshold = max(18.0, dynamic_floor + 6.0)
        min_fg_pixels = max(6, int((x2 - x1) * 0.06))

        streak = 0
        first_subject_y = None
        for y in range(max_scan_y):
            fg_count = int(np.count_nonzero(color_distance[y, x1:x2] > fg_threshold))
            if fg_count >= min_fg_pixels:
                streak += 1
                if streak >= 3:
                    first_subject_y = y - 2
                    break
            else:
                streak = 0

        if first_subject_y is None:
            return fallback_hair_top

        safety = max(2, int(face_core_h * 0.02))
        candidate = float(first_subject_y - safety)

        # Guard against extreme lifts from busy backgrounds.
        max_lift = float(face_core_h * 0.45)
        min_candidate = fallback_hair_top - max_lift
        candidate = max(candidate, min_candidate)

        # Higher forehead/hairline in image => smaller y.
        return min(fallback_hair_top, candidate)
    except Exception as e:
        logger.debug(f"Hair top pixel estimate fallback used: {e}")
        return fallback_hair_top


def smart_crop_passport(
    img: Image.Image, face_landmarks: list, target_w_mm: float, target_h_mm: float
) -> Image.Image:
    """
    Crop image for passport framing.
    """
    try:
        w, h = img.size

        # 1. Identify key landmarks (MediaPipe indices)
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        top_head = face_landmarks[10]
        chin = face_landmarks[152]

        # 2. Calculate center point
        eye_center_x = (left_eye["x"] + right_eye["x"]) / 2

        # 3. Calculate geometry
        face_core_h = abs(chin["y"] - top_head["y"])
        hair_top_y = estimate_hair_top_y(
            img=img,
            eye_center_x=eye_center_x,
            top_head_y=top_head["y"],
            chin_y=chin["y"],
            face_core_h=face_core_h,
        )
        full_head_h = abs(chin["y"] - hair_top_y)

        # 4. Framing ratios
        face_ratio = 0.50
        headroom_ratio = (1.0 - face_ratio) / 2.0
        crop_h = full_head_h / face_ratio
        crop_w = (target_w_mm / target_h_mm) * crop_h

        # 5. Vertical + horizontal placement
        frame_top = hair_top_y - (crop_h * headroom_ratio)
        frame_bottom = frame_top + crop_h
        frame_left = eye_center_x - (crop_w / 2)
        frame_right = eye_center_x + (crop_w / 2)

        # 6. White-pad canvas and crop
        pad = int(max(w, h) * 0.6)
        canvas = Image.new(img.mode, (w + 2 * pad, h + 2 * pad), (255, 255, 255))
        canvas.paste(img, (pad, pad))

        box = (
            int(frame_left + pad),
            int(frame_top + pad),
            int(frame_right + pad),
            int(frame_bottom + pad),
        )
        cropped = canvas.crop(box)

        # 7. Resize to exact output dimensions
        final_w = mm_to_px(target_w_mm)
        final_h = mm_to_px(target_h_mm)
        return cropped.resize((final_w, final_h), Image.Resampling.LANCZOS)

    except Exception as e:
        logger.error(f"Smart crop failed: {e}")
        return img.resize((mm_to_px(target_w_mm), mm_to_px(target_h_mm)))
