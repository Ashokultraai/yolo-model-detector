# Tri-Model Detector — Render deploy

Minimal, self-contained repo to host the FastAPI + onnxruntime backend on
Render's **free** tier. Contains only the backend code and the trained weight
(not the datasets).

```
render.yaml          # Render Blueprint (free web service)
backend/             # FastAPI app (app.py, detectors.py, ensemble.py, labels.py)
weights/             # yolov8_mobile_fan.onnx
```

## Deploy

1. Push this folder to a **new GitHub repo**.
2. On Render: **New + → Blueprint**, pick the repo. Render reads `render.yaml`
   and creates a free web service.
   (Or **New + → Web Service**: Root Directory `backend`,
    Build `pip install -r requirements.txt`,
    Start `uvicorn app:app --host 0.0.0.0 --port $PORT`.)
3. Wait for the build + deploy (~3–5 min). Your URL:
   `https://<service-name>.onrender.com`

## Test

- `GET /health`  → `{"status":"ok","loaded_models":["yolov8"],...}`
- `GET /docs`    → interactive API
- `POST /detect` (form field `image`) → detections JSON

Free tier sleeps after ~15 min idle; the first request then takes ~30–60 s to
wake, after which it's fast.
