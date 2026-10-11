"""The integration core: run all three models and MERGE their detections.

This is the whole point of the project. No single model knows every object:
    yolov8  -> {mobile, fan}
    rtdetr  -> {clock, watch}
    dfine   -> {laptop, mobile}

We run all three on one image, concatenate every detection, then do a
CROSS-MODEL NMS so the duplicated `mobile` class (predicted by BOTH yolov8 and
dfine) collapses to a single box instead of showing two overlapping boxes.
"""
from __future__ import annotations

import time
from typing import List, Dict, Any

import numpy as np

from detectors import BaseDetector, _nms


def _cross_model_nms(dets: List[Dict[str, Any]], iou_thr: float = 0.55) -> List[Dict[str, Any]]:
    """De-duplicate overlapping boxes of the SAME label across models.

    NMS is applied per-label so a `mobile` and a `laptop` that overlap are both
    kept, but two `mobile` boxes (one from yolov8, one from dfine) merge to the
    higher-confidence one.
    """
    if not dets:
        return []
    kept: List[Dict[str, Any]] = []
    labels = {d["label"] for d in dets}
    for label in labels:
        group = [d for d in dets if d["label"] == label]
        boxes = np.array([d["box"] for d in group], dtype=np.float32)
        scores = np.array([d["confidence"] for d in group], dtype=np.float32)
        keep_idx = _nms(boxes, scores, iou_thr)
        kept.extend(group[i] for i in keep_idx)
    kept.sort(key=lambda d: d["confidence"], reverse=True)
    return kept


class Ensemble:
    def __init__(self, detectors: List[BaseDetector]):
        self.detectors = detectors

    @property
    def available_models(self) -> List[str]:
        return [d.name for d in self.detectors if d.available]

    def detect(self, image: np.ndarray, iou_thr: float = 0.55) -> Dict[str, Any]:
        """Run every available model, merge, return detections + timing."""
        all_dets: List[Dict[str, Any]] = []
        per_model = {}
        for det in self.detectors:
            if not det.available:
                per_model[det.name] = {"available": False, "ms": None, "count": 0}
                continue
            t0 = time.perf_counter()
            dets = det.predict(image)
            ms = (time.perf_counter() - t0) * 1000.0
            per_model[det.name] = {
                "available": True,
                "arch": det.arch,
                "ms": round(ms, 1),
                "count": len(dets),
            }
            all_dets.extend(dets)

        merged = _cross_model_nms(all_dets, iou_thr)
        return {
            "detections": merged,
            "raw_count": len(all_dets),
            "merged_count": len(merged),
            "per_model": per_model,
        }

    def detect_single(self, image: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Run just one model (for the architecture-comparison story)."""
        for det in self.detectors:
            if det.name == model_name:
                if not det.available:
                    return {"detections": [], "per_model": {model_name: {"available": False}}}
                t0 = time.perf_counter()
                dets = det.predict(image)
                ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "detections": dets,
                    "merged_count": len(dets),
                    "per_model": {model_name: {"available": True, "arch": det.arch,
                                               "ms": round(ms, 1), "count": len(dets)}},
                }
        raise ValueError(f"unknown model: {model_name}")
