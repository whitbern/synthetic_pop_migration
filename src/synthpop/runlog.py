"""Run log: one JSON line per run, so every number traces to a logged run.

Each entry records the date, purpose, parameters, random seed, input files
(with SHA-256 hashes) and output paths. The log holds parameters only, never
microdata, so it can be committed.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from synthpop.truth import file_sha256

__all__ = ["log_run"]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def log_run(
    log_path: str | Path,
    purpose: str,
    params: Mapping[str, object],
    seed: int | None,
    inputs: Sequence[str | Path] = (),
    outputs: Sequence[str | Path] = (),
    results: Mapping[str, object] | None = None,
) -> dict:
    """Append one entry to ``log_path`` (JSON lines) and return it."""
    entry = {
        "date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "purpose": purpose,
        "git_commit": _git_commit(),
        "seed": seed,
        "params": dict(params),
        "inputs": [
            {"path": str(p), "sha256": file_sha256(p) if Path(p).is_file() else None} for p in inputs
        ],
        "outputs": [str(p) for p in outputs],
        "results": dict(results or {}),
    }
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True, default=float) + "\n")
    return entry
