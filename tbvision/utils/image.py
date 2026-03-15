"""Utilities for decoding user-provided images."""

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile


def content_type_is_image(content_type: str) -> bool:
    return content_type.startswith("image/")


async def decode_upload_image(upload: UploadFile) -> np.ndarray:
    contents = await upload.read()
    
    array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise HTTPException(
            status_code=400, detail="Unable to decode the uploaded image"
        )

    return image
