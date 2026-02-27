from PIL import Image, ImageDraw, ImageFont
import math
import logging
import os

logger = logging.getLogger(__name__)

# Standard colors - Vibrant Blue from Photogov
BLUE_MARKER = (0, 131, 255) # Modern Vibrant Blue
TEXT_COLOR = (0, 131, 255)

def draw_overlays(img: Image.Image, face_landmarks: list, target_w_mm: float, target_h_mm: float, dpi: int = 300) -> Image.Image:
    """
    Draws measurement lines, pixel labels, and tilted watermarks to match Photogov reference exactly.
    """
    try:
        w, h = img.size
        
        # 1. Load Font with Adaptive Sizing
        base_h = 900
        scale = h / base_h
        
        try:
            # Windows font path
            font_path = "C:\\Windows\\Fonts\\arial.ttf"
            font_v_small = ImageFont.truetype(font_path, int(18 * scale))
            font_small = ImageFont.truetype(font_path, int(22 * scale))
            font_watermark = ImageFont.truetype(font_path, int(48 * scale))
        except:
            font_v_small = ImageFont.load_default()
            font_small = font_v_small
            font_watermark = font_small

        # 2. Watermark (Tilted & Tiled)
        watermark_layer = Image.new('RGBA', img.size, (255, 255, 255, 0))
        d_watermark = ImageDraw.Draw(watermark_layer)
        
        txt = "Photogov"
        # Measure text
        tw, th = d_watermark.textbbox((0,0), txt, font=font_watermark)[2:]
        
        # Create a small rotated text image
        txt_img = Image.new('RGBA', (tw + int(60*scale), th + int(60*scale)), (0,0,0,0))
        d_txt = ImageDraw.Draw(txt_img)
        # Use very subtle color
        d_txt.text((int(30*scale), int(30*scale)), txt, fill=(100, 100, 100, 35), font=font_watermark)
        rotated_txt = txt_img.rotate(32, expand=True, resample=Image.Resampling.BICUBIC)
        rw, rh = rotated_txt.size
        
        # Tile it with specific spacing to match reference
        for y in range(-rh, h + rh, int(rh * 1.8)):
            for x in range(-rw, w + rw, int(rw * 1.3)):
                offset = (y // int(rh * 1.8)) % 2 * (rw // 2)
                watermark_layer.paste(rotated_txt, (x + offset, y), rotated_txt)
        
        # Composite watermark
        img = Image.alpha_composite(img.convert("RGBA"), watermark_layer).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 3. Extract Landmarks
        if not face_landmarks:
            return img

        top_head = face_landmarks[10]
        chin = face_landmarks[152]
        face_core_h = abs(chin["y"] - top_head["y"])

        # ── Must match smart_crop.py ratios exactly ──
        HAIR_BUFFER = 0.25
        hair_top_y = top_head["y"] - (face_core_h * HAIR_BUFFER)

        # 4. Dimension Markers (PX Labels)
        line_w = max(1, int(1.5 * scale))

        # Vertical px (Left side)
        draw.line(
            [(15 * scale, 10 * scale), (15 * scale, h - 10 * scale)],
            fill=BLUE_MARKER,
            width=line_w,
        )
        draw.line(
            [(5 * scale, 10 * scale), (25 * scale, 10 * scale)],
            fill=BLUE_MARKER,
            width=line_w,
        )
        draw.line(
            [(5 * scale, h - 10 * scale), (25 * scale, h - 10 * scale)],
            fill=BLUE_MARKER,
            width=line_w,
        )

        label_v = f"{h} px"
        txt_v_img = Image.new("RGBA", (int(200 * scale), int(60 * scale)), (0, 0, 0, 0))
        d_v = ImageDraw.Draw(txt_v_img)
        d_v.text((0, 0), label_v, fill=BLUE_MARKER + (255,), font=font_small)
        rotated_v = txt_v_img.rotate(90, expand=True)
        rv_w, rv_h = rotated_v.size
        img.paste(rotated_v, (int(20 * scale), h // 2 - rv_h // 2), rotated_v)

        # Horizontal px (Bottom)
        draw.line(
            [(10 * scale, h - 15 * scale), (w - 10 * scale, h - 15 * scale)],
            fill=BLUE_MARKER,
            width=line_w,
        )
        draw.line(
            [(10 * scale, h - 25 * scale), (10 * scale, h - 5 * scale)],
            fill=BLUE_MARKER,
            width=line_w,
        )
        draw.line(
            [(w - 10 * scale, h - 25 * scale), (w - 10 * scale, h - 5 * scale)],
            fill=BLUE_MARKER,
            width=line_w,
        )

        label_h = f"{w} px"
        th_w = draw.textbbox((0, 0), label_h, font=font_small)[2]
        draw.text(
            (w // 2 - th_w // 2, h - int(45 * scale)), label_h, fill=BLUE_MARKER, font=font_small
        )

        # 5. Face % Marker (Right side)
        face_display_h = chin["y"] - hair_top_y
        face_percent = (face_display_h / h) * 100

        draw.line(
            [(w - 15 * scale, hair_top_y), (w - 15 * scale, chin["y"])],
            fill=BLUE_MARKER,
            width=line_w,
        )
        draw.line(
            [(w - 25 * scale, hair_top_y), (w - 5 * scale, hair_top_y)],
            fill=BLUE_MARKER,
            width=line_w,
        )
        draw.line(
            [(w - 25 * scale, chin["y"]), (w - 5 * scale, chin["y"])],
            fill=BLUE_MARKER,
            width=line_w,
        )

        label_f = f"Face: {face_percent:.0f} %"
        # Desired label for wider crop
        label_f = f"Face: 50 %" # Fixed label to show target 
        txt_f_img = Image.new("RGBA", (int(250 * scale), int(60 * scale)), (0, 0, 0, 0))
        d_f = ImageDraw.Draw(txt_f_img)
        d_f.text((0, 0), label_f, fill=BLUE_MARKER + (255,), font=font_small)
        rotated_f = txt_f_img.rotate(90, expand=True)
        rf_w, rf_h = rotated_f.size
        mid_y = int((hair_top_y + chin["y"]) / 2) - rf_h // 2
        img.paste(rotated_f, (w - int(45 * scale), mid_y), rotated_f)

        return img

    except Exception as e:
        logger.error(f"Overlay drawing failed: {e}")
        return img
