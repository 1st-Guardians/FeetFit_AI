from __future__ import annotations

import cv2
import numpy as np


PLANTAR_COMPOSITE_LABEL = "BILATERAL RAW PLANTAR PHOTOS (LEFT PANEL / RIGHT PANEL)"


def combine_raw_plantar_photos(left_image_bytes: bytes, right_image_bytes: bytes) -> bytes:
    """Keep decoded photo pixels intact; add equal panels, padding and labels."""
    photos = []
    for side, image_bytes in (("Left", left_image_bytes), ("Right", right_image_bytes)):
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError(f"{side} plantar photo is empty or invalid.")
        photo = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if photo is None:
            raise ValueError(f"{side} plantar photo could not be decoded.")
        photos.append(photo)

    padding, gap, header = 24, 32, 64
    panel_width = max(160, *(photo.shape[1] for photo in photos))
    panel_height = max(photo.shape[0] for photo in photos)
    canvas = np.full(
        (header + panel_height + 2 * padding, 2 * panel_width + gap + 2 * padding, 3),
        255,
        dtype=np.uint8,
    )
    for index, (label, photo) in enumerate(zip(("LEFT", "RIGHT"), photos, strict=True)):
        panel_x = padding + index * (panel_width + gap)
        x = panel_x + (panel_width - photo.shape[1]) // 2
        y = header + padding + (panel_height - photo.shape[0]) // 2
        canvas[y:y + photo.shape[0], x:x + photo.shape[1]] = photo
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
        cv2.putText(
            canvas, label, (panel_x + (panel_width - text_width) // 2, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA,
        )

    encoded, png = cv2.imencode(".png", canvas)
    if not encoded:
        raise ValueError("Combined plantar photo could not be encoded.")
    return png.tobytes()
