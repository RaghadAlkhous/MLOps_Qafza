from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

START_TIME = time.time()
LOCK = threading.Lock()
ROUTER = APIRouter(tags=["monitoring"])

EXPECTED_LATE_RATE = float(os.getenv("EXPECTED_LATE_RATE", "0.0811"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))
MIN_SAMPLES_FOR_DRIFT = int(os.getenv("MIN_SAMPLES_FOR_DRIFT", "100"))
LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl"))

RECENT_LABELS: deque[int] = deque(maxlen=1000)

STATS = {
    "requests_total": 0,
    "errors_total": 0,
    "latency_sum_ms": 0.0,
    "latency_count": 0,
    "predictions_total": 0,
    "late_predictions": 0,
    "on_time_predictions": 0,
    "probability_sum": 0.0,
    "probability_count": 0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_log_dir() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def record_request(path: str, status_code: int, duration_ms: float) -> None:
    with LOCK:
        STATS["requests_total"] += 1
        STATS["latency_sum_ms"] += duration_ms
        STATS["latency_count"] += 1
        if status_code >= 500:
            STATS["errors_total"] += 1


def record_prediction_stats(label: int, probability: float) -> None:
    with LOCK:
        STATS["predictions_total"] += 1
        if int(label) == 1:
            STATS["late_predictions"] += 1
            RECENT_LABELS.append(1)
        else:
            STATS["on_time_predictions"] += 1
            RECENT_LABELS.append(0)

        STATS["probability_sum"] += float(probability)
        STATS["probability_count"] += 1


def write_prediction_log(payload: dict, latency_ms: float) -> None:
    ensure_log_dir()
    row = {
        "timestamp": utc_now(),
        "latency_ms": round(latency_ms, 3),
        "payload": payload,
    }
    with LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")


def process_prediction_payload(payload: dict, latency_ms: float) -> None:
    if not isinstance(payload, dict) or payload.get("status") != "success":
        return

    if "prediction" in payload and "probability" in payload:
        record_prediction_stats(int(payload["prediction"]), float(payload["probability"]))
    elif "predictions" in payload and isinstance(payload["predictions"], list):
        for item in payload["predictions"]:
            if isinstance(item, dict) and "prediction" in item and "probability" in item:
                record_prediction_stats(int(item["prediction"]), float(item["probability"]))

    write_prediction_log(payload, latency_ms)


def average_latency_ms() -> float:
    with LOCK:
        if STATS["latency_count"] == 0:
            return 0.0
        return STATS["latency_sum_ms"] / STATS["latency_count"]


def error_rate() -> float:
    with LOCK:
        if STATS["requests_total"] == 0:
            return 0.0
        return STATS["errors_total"] / STATS["requests_total"]


def positive_rate() -> float:
    with LOCK:
        if STATS["predictions_total"] == 0:
            return 0.0
        return STATS["late_predictions"] / STATS["predictions_total"]


def drift_status() -> dict:
    with LOCK:
        samples = len(RECENT_LABELS)
        recent_positive = sum(RECENT_LABELS) / samples if samples else 0.0
        delta = recent_positive - EXPECTED_LATE_RATE
        alert = samples >= MIN_SAMPLES_FOR_DRIFT and abs(delta) > DRIFT_THRESHOLD
        return {
            "expected_late_rate": EXPECTED_LATE_RATE,
            "recent_late_rate": recent_positive,
            "delta": delta,
            "samples": samples,
            "alert": alert,
        }


class MonitoringMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            record_request(request.url.path, 500, duration_ms)
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        status_code = response.status_code

        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        body = b"".join(chunks)

        record_request(request.url.path, status_code, duration_ms)

        if request.url.path == "/predict" and status_code == 200:
            try:
                payload = json.loads(body.decode("utf-8"))
                process_prediction_payload(payload, duration_ms)
            except Exception:
                pass

        headers = dict(response.headers)
        headers.pop("content-length", None)

        return Response(
            content=body,
            status_code=status_code,
            headers=headers,
            media_type=response.media_type,
        )


@ROUTER.get("/metrics")
def metrics() -> Response:
    with LOCK:
        lines = [
            "# HELP http_requests_total Total HTTP requests.",
            "# TYPE http_requests_total counter",
            f"http_requests_total {STATS['requests_total']}",
            "# HELP http_errors_total Total HTTP 5xx errors.",
            "# TYPE http_errors_total counter",
            f"http_errors_total {STATS['errors_total']}",
            "# HELP http_latency_ms_sum Sum of request latency in milliseconds.",
            "# TYPE http_latency_ms_sum counter",
            f"http_latency_ms_sum {STATS['latency_sum_ms']}",
            "# HELP http_latency_ms_count Count of measured requests.",
            "# TYPE http_latency_ms_count counter",
            f"http_latency_ms_count {STATS['latency_count']}",
            "# HELP ml_predictions_total Total predictions processed.",
            "# TYPE ml_predictions_total counter",
            f"ml_predictions_total {STATS['predictions_total']}",
            "# HELP ml_predictions_late_total Total late predictions.",
            "# TYPE ml_predictions_late_total counter",
            f"ml_predictions_late_total {STATS['late_predictions']}",
            "# HELP ml_predictions_on_time_total Total on-time predictions.",
            "# TYPE ml_predictions_on_time_total counter",
            f"ml_predictions_on_time_total {STATS['on_time_predictions']}",
            "# HELP ml_prediction_probability_sum Sum of late probabilities.",
            "# TYPE ml_prediction_probability_sum counter",
            f"ml_prediction_probability_sum {STATS['probability_sum']}",
            "# HELP ml_prediction_probability_count Count of probabilities.",
            "# TYPE ml_prediction_probability_count counter",
            f"ml_prediction_probability_count {STATS['probability_count']}",
        ]

    drift = drift_status()
    lines.extend(
        [
            "# HELP ml_prediction_positive_rate Current late prediction rate.",
            "# TYPE ml_prediction_positive_rate gauge",
            f"ml_prediction_positive_rate {positive_rate()}",
            "# HELP ml_drift_alert Whether prediction drift alert is active.",
            "# TYPE ml_drift_alert gauge",
            f"ml_drift_alert {1 if drift['alert'] else 0}",
        ]
    )

    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@ROUTER.get("/monitoring/stats")
def stats() -> dict:
    with LOCK:
        snapshot = dict(STATS)

    snapshot.update(
        {
            "average_latency_ms": average_latency_ms(),
            "error_rate": error_rate(),
            "positive_rate": positive_rate(),
            "drift": drift_status(),
            "uptime_seconds": time.time() - START_TIME,
            "log_path": str(LOG_PATH),
        }
    )
    return snapshot


@ROUTER.get("/monitoring/drift")
def drift() -> dict:
    return drift_status()


def setup_monitoring(app: FastAPI) -> None:
    if getattr(app.state, "monitoring_initialized", False):
        return

    app.add_middleware(MonitoringMiddleware)
    app.include_router(ROUTER)
    app.state.monitoring_initialized = True
