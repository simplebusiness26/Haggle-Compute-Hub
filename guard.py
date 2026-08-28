#!/usr/bin/env python3
"""Prevent overlapping accelerator launches while a Kaggle job is active."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hub


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: guard.py <input-command> <output-command>")
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    command = json.loads(source.read_text(encoding="utf-8"))
    queue = hub.load_json(hub.QUEUE_FILE, {"jobs": []})
    active = [
        j for j in queue.get("jobs", [])
        if j.get("status") == "running" and j.get("compute") in {"gpu", "tpu"}
    ]
    if command.get("action") == "schedule" and active:
        command = {
            "action": "status",
            "request_id": str(command.get("request_id") or "guarded-status")[:100],
        }
        print(f"Accelerator dispatch held: {len(active)} Kaggle job(s) still active.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
