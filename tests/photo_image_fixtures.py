"""Decodable, distinguishable raw photos for multimodal request tests."""

import cv2
import numpy as np


def encoded_photo(extension: str, color: tuple[int, int, int]) -> bytes:
    photo = np.full((96, 64, 3), color, dtype=np.uint8)
    # Asymmetric markers make accidental flips and rotations observable.
    photo[8:16, 4:12] = (0, 0, 0)
    ok, encoded = cv2.imencode(extension, photo)
    if not ok:
        raise RuntimeError("Could not encode test photo.")
    return encoded.tobytes()
