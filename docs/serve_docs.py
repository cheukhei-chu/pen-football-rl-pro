#!/usr/bin/env python3
"""Serve the GitHub Pages docs directory locally over HTTP."""

from __future__ import annotations

import argparse
import http.server
import socketserver
import webbrowser
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the docs/ site locally.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on.")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the landing page in the default browser after starting.",
    )
    args = parser.parse_args()

    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        httpd.allow_reuse_address = True
        url = f"http://127.0.0.1:{args.port}/"
        print(f"Serving {DOCS_DIR} at {url}")
        if args.open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    import os

    os.chdir(DOCS_DIR)
    main()
