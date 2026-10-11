"""Per-architecture ONNX detector adapters.

Each architecture produces a DIFFERENT output format, so each gets its own
post-processing. Every adapter returns the SAME common shape:

    [
        {"label": "mobile", "confidence": 0.91, "box": [x1, y1, x2, y2], "source_model": "yolov8"},
        ...
    ]

Box coordinates are in ORIGINAL image pixels (not the model input size).

All three load as ONNX via onnxruntime so the backend needs no PyTorch /
Ultralytics / D-FINE repo at serving time.

NOTE ON OUTPUT FORMATS: exact ONNX output tensor layouts depend on how you
export each model. After you run training/export_to_onnx.py, print each
session's output shapes (see the __main__ block at the bottom) and, if needed,
adjust the small `_decode_*` sections. The assumptions used here match the
default Ultralytics and official D-FINE ONNX exports.
"""
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:  # keeps the module importable even before deps are installed
    ort = None


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _load_session(onnx_path: str):
    if ort is None:
        raise RuntimeError("onnxruntime not installed. pip install -r requirements.txt")
    # CPU is enough for small models; this is what runs free on HF Spaces.
    providers = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def _letterbox(img: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
    """Resize + pad to a square keeping aspect ratio (YOLO-style).

    Returns padded image, scale ratio r, and (dw, dh) padding.
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    dw, dh = (new_shape - nw) // 2, (new_shape - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


def _to_input(img_chw_rgb: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> NCHW float32 normalized 0..1."""
    x = img_chw_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # 1,3,H,W
    return np.ascontiguousarray(x)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> List[int]:
    """Plain NumPy NMS. boxes in xyxy. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


# --------------------------------------------------------------------------- #
# Base adapter
# --------------------------------------------------------------------------- #
class BaseDetector:
    """One ONNX model specialized on a subset of GLOBAL_LABELS.

    `class_names` is this model's own class order (index -> global label name).
    """

    arch = "base"

    def __init__(self, name: str, onnx_path: str, class_names: List[str],
                 imgsz: int = 640, conf_thr: float = 0.25, iou_thr: float = 0.45):
        self.name = name
        self.onnx_path = onnx_path
        self.class_names = class_names
        self.imgsz = imgsz
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.available = bool(onnx_path and os.path.exists(onnx_path))
        self._session = None
        if self.available:
            try:
                self._session = _load_session(onnx_path)
            except Exception as e:  # don't kill the whole backend over one model
                print(f"[{self.name}] failed to load {onnx_path}: {e}")
                self.available = False

    # Each subclass implements predict().
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _input_name(self) -> str:
        return self._session.get_inputs()[0].name


# --------------------------------------------------------------------------- #
# YOLOv8 (one-stage, needs NMS)
# --------------------------------------------------------------------------- #
class YoloV8Detector(BaseDetector):
    arch = "yolov8"

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        H0, W0 = image.shape[:2]
        padded, r, dw, dh = _letterbox(image, self.imgsz)
        out = self._session.run(None, {self._input_name(): _to_input(padded)})[0]
        # Ultralytics YOLOv8 export: (1, 4+nc, 8400). We want (N, 4+nc).
        # Orient using the KNOWN channel count so it works regardless of which
        # axis holds the predictions.
        pred = out[0]
        expected_c = 4 + len(self.class_names)
        if pred.shape[0] == expected_c and pred.shape[1] != expected_c:
            pred = pred.transpose(1, 0)          # (C, N) -> (N, C)
        elif pred.shape[1] != expected_c and pred.shape[0] < pred.shape[1]:
            pred = pred.transpose(1, 0)          # fallback heuristic
        boxes_xywh = pred[:, :4]
        scores_all = pred[:, 4:]
        cls_ids = scores_all.argmax(1)
        confs = scores_all.max(1)
        m = confs >= self.conf_thr
        boxes_xywh, cls_ids, confs = boxes_xywh[m], cls_ids[m], confs[m]
        if len(boxes_xywh) == 0:
            return []
        # xywh (center) -> xyxy in padded space, then undo letterbox.
        cx, cy, w, h = boxes_xywh.T
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / r
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W0)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H0)
        keep = _nms(xyxy, confs, self.iou_thr)
        return self._format(xyxy[keep], confs[keep], cls_ids[keep])

    def _format(self, boxes, confs, cls_ids):
        dets = []
        for b, c, k in zip(boxes, confs, cls_ids):
            if int(k) >= len(self.class_names):
                continue
            dets.append({
                "label": self.class_names[int(k)],
                "confidence": float(c),
                "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "source_model": self.name,
            })
        return dets


# --------------------------------------------------------------------------- #
# RT-DETR (DETR-style, NMS-free)
# --------------------------------------------------------------------------- #
class RTDetrDetector(BaseDetector):
    arch = "rtdetr"

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        H0, W0 = image.shape[:2]
        # Ultralytics RT-DETR uses a plain square resize (no letterbox).
        resized = np.array(Image.fromarray(image).resize((self.imgsz, self.imgsz), Image.BILINEAR))
        out = self._session.run(None, {self._input_name(): _to_input(resized)})[0]
        # Ultralytics RT-DETR export: (1, 300, 4+nc), boxes = cxcywh normalized 0..1.
        pred = out[0]
        boxes_xywh = pred[:, :4]
        scores_all = pred[:, 4:]
        cls_ids = scores_all.argmax(1)
        confs = scores_all.max(1)
        m = confs >= self.conf_thr
        boxes_xywh, cls_ids, confs = boxes_xywh[m], cls_ids[m], confs[m]
        if len(boxes_xywh) == 0:
            return []
        cx, cy, w, h = boxes_xywh.T
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        # normalized -> original pixels
        xyxy[:, [0, 2]] *= W0
        xyxy[:, [1, 3]] *= H0
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W0)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H0)
        dets = []
        for b, c, k in zip(xyxy, confs, cls_ids):
            if int(k) >= len(self.class_names):
                continue
            dets.append({
                "label": self.class_names[int(k)],
                "confidence": float(c),
                "box": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "source_model": self.name,
            })
        return dets


# --------------------------------------------------------------------------- #
# D-FINE (DETR-style, NMS-free, separate output tensors)
# --------------------------------------------------------------------------- #
class DFineDetector(BaseDetector):
    arch = "dfine"

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        H0, W0 = image.shape[:2]
        resized = np.array(Image.fromarray(image).resize((self.imgsz, self.imgsz), Image.BILINEAR))
        inputs = {self._input_name(): _to_input(resized)}
        # The official D-FINE ONNX export takes a second input, orig_target_sizes.
        in_names = [i.name for i in self._session.get_inputs()]
        if len(in_names) > 1:
            inputs[in_names[1]] = np.array([[W0, H0]], dtype=np.int64)
        outs = self._session.run(None, inputs)
        # Official D-FINE export returns (labels, boxes, scores).
        labels, boxes, scores = outs[0][0], outs[1][0], outs[2][0]
        dets = []
        for lab, b, sc in zip(labels, boxes, scores):
            if sc < self.conf_thr:
                continue
            k = int(lab)
            if k >= len(self.class_names):
                continue
            x1, y1, x2, y2 = b
            # If the export already mapped to orig size these are pixels; otherwise
            # they are in imgsz space and we rescale.
            if x2 <= self.imgsz + 1 and y2 <= self.imgsz + 1 and len(in_names) == 1:
                x1, x2 = x1 / self.imgsz * W0, x2 / self.imgsz * W0
                y1, y2 = y1 / self.imgsz * H0, y2 / self.imgsz * H0
            dets.append({
                "label": self.class_names[k],
                "confidence": float(sc),
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "source_model": self.name,
            })
        return dets


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_detector(arch: str, **kw) -> BaseDetector:
    arch = arch.lower()
    if arch == "yolov8":
        return YoloV8Detector(**kw)
    if arch == "rtdetr":
        return RTDetrDetector(**kw)
    if arch == "dfine":
        return DFineDetector(**kw)
    raise ValueError(f"unknown arch: {arch}")


if __name__ == "__main__":
    # Quick way to inspect your actual ONNX output shapes after export.
    import sys
    if len(sys.argv) > 1 and ort is not None:
        s = _load_session(sys.argv[1])
        print("inputs:", [(i.name, i.shape) for i in s.get_inputs()])
        print("outputs:", [(o.name, o.shape) for o in s.get_outputs()])
