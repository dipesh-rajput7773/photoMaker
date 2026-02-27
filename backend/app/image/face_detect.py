import numpy as np
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)

# ── Import mediapipe (supports both old and new API) ──
_mp_face_detection = None
_mp_face_mesh = None

try:
    # Old API (mediapipe <= 0.10.14)
    from mediapipe.python.solutions import face_detection as _mp_face_detection
    from mediapipe.python.solutions import face_mesh as _mp_face_mesh
    _API = "legacy"
except ImportError:
    try:
        import mediapipe.solutions.face_detection as _mp_face_detection
        import mediapipe.solutions.face_mesh as _mp_face_mesh
        _API = "legacy"
    except (ImportError, AttributeError):
        _API = "tasks"
        logger.info("Using mediapipe Tasks API (v0.10.15+)")


def detect_face_bbox(img: Image.Image):
    """
    Detects the primary face and returns a bounding box.
    Used for initial alignment and validation.
    """
    img_array = np.array(img)

    if _API == "legacy" and _mp_face_detection is not None:
        with _mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as detector:
            results = detector.process(img_array)
            if not results.detections:
                return None
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = img_array.shape
            return {
                "x": int(bbox.xmin * w),
                "y": int(bbox.ymin * h),
                "width": int(bbox.width * w),
                "height": int(bbox.height * h),
                "score": detection.score[0],
            }
    else:
        # Fallback: OpenCV Haar Cascade
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, 1.05, 6, minSize=(100, 100))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return {"x": int(x), "y": int(y), "width": int(w), "height": int(h), "score": 1.0}


def get_face_landmarks(img: Image.Image):
    """
    Extracts detailed 468 landmarks for precision alignment.
    Returns list of dicts with 'x', 'y' in PIXEL coordinates.
    """
    img_array = np.array(img)
    h, w, _ = img_array.shape

    if _API == "legacy" and _mp_face_mesh is not None:
        with _mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as mesh_detector:
            results = mesh_detector.process(img_array)
            if not results.multi_face_landmarks:
                return None
            landmarks = results.multi_face_landmarks[0]
            points = []
            for lm in landmarks.landmark:
                points.append({
                    "x": int(lm.x * w),
                    "y": int(lm.y * h),
                    "z": lm.z,
                })
            return points
    else:
        # Tasks API (mediapipe >= 0.10.15)
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python import BaseOptions
            import mediapipe as mp
            import urllib.request
            import os

            model_path = os.path.expanduser("~/.cache/mediapipe/face_landmarker.task")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            if not os.path.exists(model_path):
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
                urllib.request.urlretrieve(url, model_path)

            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            with vision.FaceLandmarker.create_from_options(options) as landmarker:
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=img_array
                )
                result = landmarker.detect(mp_image)
                if not result.face_landmarks:
                    return None
                points = []
                for lm in result.face_landmarks[0]:
                    points.append({
                        "x": int(lm.x * w),
                        "y": int(lm.y * h),
                        "z": lm.z,
                    })
                return points
        except Exception as e:
            logger.error(f"Tasks API face landmark detection failed: {e}")
            return None
