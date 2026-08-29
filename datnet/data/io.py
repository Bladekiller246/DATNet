"""Image IO with a cv2 fast path and a PIL fallback."""
import os

import numpy as np

try:
    import cv2
    cv2.setNumThreads(0)  # DataLoader workers must not each spawn a thread pool
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def imread(path):
    """Read an image as HWC uint8 RGB."""
    if _HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if _HAS_PIL:
        return np.array(Image.open(path).convert("RGB"))
    raise ImportError("need opencv-python or Pillow to read images")


def imwrite(path, img):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255.0).round().astype(np.uint8)
    if _HAS_CV2:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    elif _HAS_PIL:
        Image.fromarray(img).save(path)
    else:
        raise ImportError("need opencv-python or Pillow to write images")


def list_images(root):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"no such directory: {root}")
    files = [os.path.join(root, f) for f in sorted(os.listdir(root))
             if f.lower().endswith(IMG_EXTS)]
    if not files:
        raise FileNotFoundError(f"no images found in {root}")
    return files
