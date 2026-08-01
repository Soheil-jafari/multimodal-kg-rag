"""Region OCR via RapidOCR (onnxruntime, CPU) — pure-pip, no system binary.

Implements :class:`~platform_core.ingestion.base.RegionOCR`. Crops the region
from the in-memory page image and OCRs it. FIGURE regions return "" (kept as
crops for CLIP, not OCR'd). The engine is loaded once and reused. Behind the
interface, so swapping in EasyOCR/Tesseract is a one-line config change.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from platform_core.ingestion.base import RegionOCR
from platform_core.types import BBox, RegionType


def crop_bbox(page_image: Any, bbox: BBox):
    """Return an integer-clamped PIL crop for ``bbox``, or ``None`` if degenerate."""
    width, height = page_image.size
    x0 = max(0, int(math.floor(bbox.x0)))
    y0 = max(0, int(math.floor(bbox.y0)))
    x1 = min(width, int(math.ceil(bbox.x1)))
    y1 = min(height, int(math.ceil(bbox.y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return page_image.crop((x0, y0, x1, y1))


class RapidOCRRegionOCR(RegionOCR):
    """OCR a single region crop with RapidOCR (loaded once, CPU)."""

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

    def ocr_region(self, page_image: Any, bbox: BBox, region_type: RegionType) -> str:
        if region_type == RegionType.FIGURE:
            return ""
        crop = crop_bbox(page_image, bbox)
        if crop is None:
            return ""
        result = self._engine(np.asarray(crop))
        # RapidOCR (onnxruntime) returns (result, elapse); result = [[box, text, score], ...] or None.
        lines = result[0] if isinstance(result, tuple) else result
        if not lines:
            return ""
        texts = [str(item[1]) for item in lines if len(item) >= 2]
        return "\n".join(texts).strip()
