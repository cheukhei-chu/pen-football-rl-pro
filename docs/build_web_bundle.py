#!/usr/bin/env python3
"""Build reproducible pygbag APK bundles for the web deployment."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent


BUNDLES = {
    "twoplayer": {
        "source_root": DOCS_DIR / "twoplayer",
        "output": DOCS_DIR / "twoplayer" / "penfootballweb.apk",
        "extra_outputs": [],
        "entries": [
            ("index.html", "assets/index.html"),
            ("main.py", "assets/main.py"),
            ("pygbag.json", "assets/pygbag.json"),
            ("soccer_base64.txt", "assets/soccer_base64.txt"),
            ("assets", "assets/assets"),
        ],
    },
    "singleplayer": {
        "source_root": DOCS_DIR / "singleplayer",
        "output": DOCS_DIR / "singleplayer" / "build" / "web" / "singleplayer.apk",
        "extra_outputs": [
            DOCS_DIR / "singleplayer" / "build" / "web" / "penfootball_singleplayer.apk",
        ],
        "entries": [
            ("main.py", "assets/main.py"),
            ("pen_football_web.py", "assets/pen_football_web.py"),
            ("assets", "assets/assets"),
        ],
    },
}


def iter_bundle_files(source_root: Path, entries):
    for rel_path, archive_path in entries:
        source_path = source_root / rel_path
        if source_path.is_dir():
            for file_path in sorted(source_path.rglob("*")):
                if file_path.is_file():
                    nested_rel = file_path.relative_to(source_path).as_posix()
                    yield file_path, f"{archive_path}/{nested_rel}"
        else:
            yield source_path, archive_path


def build_bundle(name: str) -> None:
    config = BUNDLES[name]
    source_root: Path = config["source_root"]
    output: Path = config["output"]
    extra_outputs = config["extra_outputs"]
    entries = config["entries"]

    missing = [rel_path for rel_path, _ in entries if not (source_root / rel_path).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Missing source files for {name}: {missing_str}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_path, archive_path in iter_bundle_files(source_root, entries):
            archive.write(source_path, archive_path)

    for extra_output in extra_outputs:
        extra_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output, extra_output)

    print(f"Built {name}: {output}")
    for extra_output in extra_outputs:
        print(f"Synced copy: {extra_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build web APK bundles from tracked source files.")
    parser.add_argument(
        "bundle",
        choices=["twoplayer", "singleplayer", "all"],
        help="Which bundle to build.",
    )
    args = parser.parse_args()

    targets = BUNDLES.keys() if args.bundle == "all" else [args.bundle]
    for name in targets:
        build_bundle(name)


if __name__ == "__main__":
    main()
