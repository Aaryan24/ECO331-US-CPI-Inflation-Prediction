"""
Lightweight Demo Web Server for the ECO331 Multi-Model Inflation Predictor.

Usage:
    uv run python app.py
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs

from src.predictor_service import LiveInflationPredictor

WEB_DIR = Path(__file__).resolve().parent / "src" / "web"

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


def create_handler(predictor: LiveInflationPredictor):
    class PredictorRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return  # Keep demo console clean

        def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", raw)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = (WEB_DIR / "index.html").read_bytes()
                self._send_bytes(HTTPStatus.OK, MIME_TYPES[".html"], body)
                return

            if path == "/api/meta":
                self._send_json(HTTPStatus.OK, predictor.get_metadata())
                return

            if path.startswith("/static/"):
                rel = path.replace("/static/", "", 1)
                file_path = (WEB_DIR / rel).resolve()
                if WEB_DIR in file_path.parents and file_path.is_file():
                    ctype = MIME_TYPES.get(file_path.suffix, "application/octet-stream")
                    self._send_bytes(HTTPStatus.OK, ctype, file_path.read_bytes())
                    return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != "/api/predict":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            ctype = self.headers.get("Content-Type", "")

            try:
                if "application/json" in ctype:
                    data = json.loads(raw_body or "{}")
                    horizon = int(data.get("horizon", 3))
                    preset_id = str(data.get("preset_id", "latest_2026_09"))
                    overrides = {k: float(v) for k, v in (data.get("overrides") or {}).items()}
                else:
                    # Progressive enhancement fallback for standard HTML form POST
                    parsed = parse_qs(raw_body)
                    horizon = int(parsed.get("horizon", ["3"])[0])
                    preset_id = str(parsed.get("preset_id", ["latest_2026_09"])[0])
                    overrides = {
                        k: float(vals[0])
                        for k, vals in parsed.items()
                        if k not in ("horizon", "preset_id") and vals
                    }
                result = predictor.predict(horizon=horizon, preset_id=preset_id, overrides=overrides)
                self._send_json(HTTPStatus.OK, result)
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return PredictorRequestHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ECO331 Multi-Model Inflation Predictor Demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    print("Loading 12 trained models across horizons h = [1, 3, 6, 12] months...")
    predictor = LiveInflationPredictor()
    server = ThreadingHTTPServer((args.host, args.port), create_handler(predictor))
    print(f"\nLive Multi-Model Inflation Predictor is running at: http://{args.host}:{args.port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down predictor server.")
        server.server_close()


if __name__ == "__main__":
    main()
