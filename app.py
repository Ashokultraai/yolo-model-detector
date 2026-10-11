"""FastAPI backend: loads 3 ONNX detectors and serves a merged (ensemble) result.

Endpoints
---------
GET  /health                -> liveness + which models loaded
GET  /models                -> model metadata (arch, classes, available)
POST /detect                -> ensemble of all 3 models on an uploaded image
POST /detect?mode=single&model=yolov8 -> run one model only

Run locally:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
On Hugging Face Spaces the Dockerfile runs it on port 7860.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from detectors import build_detector
from ensemble import Ensemble
from labels import GLOBAL_LABELS, LABEL_COLORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backend")

# --------------------------------------------------------------------------- #
# Model configuration: each architecture + its own class order + weight path.
# The class_names list MUST match the order the model was TRAINED with.
# Weights live in ../weights (local) or /app/weights (Docker/HF). WEIGHTS_DIR
# can override.
# --------------------------------------------------------------------------- #
WEIGHTS_DIR = os.environ.get(
    "WEIGHTS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights"),
)

MODEL_CONFIG = [
    {
        "name": "yolov8",
        "arch": "yolov8",
        "file": "yolov8_mobile_fan.onnx",
        "class_names": ["mobile", "fan"],
    },
    # RT-DETR and D-FINE are trained the same way (see training/) — drop their
    # .onnx into weights/ and un-comment these to grow the ensemble to 3 models.
    # {
    #     "name": "rtdetr",
    #     "arch": "rtdetr",
    #     "file": "rtdetr_clock_watch.onnx",
    #     "class_names": ["clock", "watch"],
    # },
    # {
    #     "name": "dfine",
    #     "arch": "dfine",
    #     "file": "dfine_laptop_mobile.onnx",
    #     "class_names": ["laptop", "mobile"],
    # },
]

CONF_THR = float(os.environ.get("CONF_THR", "0.25"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))

detectors = []
for cfg in MODEL_CONFIG:
    path = os.path.join(WEIGHTS_DIR, cfg["file"])
    det = build_detector(
        cfg["arch"], name=cfg["name"], onnx_path=path,
        class_names=cfg["class_names"], imgsz=IMG_SIZE, conf_thr=CONF_THR,
    )
    status = "LOADED" if det.available else "MISSING (train & export first)"
    log.info("model %-7s [%s] classes=%s -> %s", det.name, det.arch, cfg["class_names"], status)
    detectors.append(det)

ensemble = Ensemble(detectors)

app = FastAPI(title="Tri-Model Detector", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _read_image(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": ensemble.available_models,
            "global_labels": GLOBAL_LABELS}


@app.get("/models")
def models():
    out = []
    for det, cfg in zip(detectors, MODEL_CONFIG):
        out.append({
            "name": det.name,
            "arch": det.arch,
            "classes": cfg["class_names"],
            "available": det.available,
            "weight_file": cfg["file"],
        })
    return {"models": out, "global_labels": GLOBAL_LABELS, "colors": LABEL_COLORS}


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    mode: str = Query("ensemble", enum=["ensemble", "single"]),
    model: Optional[str] = Query(None),
):
    try:
        img = _read_image(await image.read())
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad image: {e}"})

    H, W = img.shape[:2]
    if mode == "single":
        if not model:
            return JSONResponse(status_code=400, content={"error": "model= required for mode=single"})
        try:
            result = ensemble.detect_single(img, model)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
    else:
        result = ensemble.detect(img)

    result["image_size"] = {"width": W, "height": H}
    labels_found = sorted({d["label"] for d in result["detections"]})
    log.info("detect mode=%s file=%s found=%s per_model=%s",
             mode, image.filename, labels_found, result.get("per_model"))
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
