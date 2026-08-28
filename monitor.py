#!/usr/bin/env python3
"""Monitor active Kaggle jobs and collect finished outputs.

This runs before each scheduled dispatch. It never launches work; it only checks
jobs already marked running by the Compute Hub.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import hub

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def state_from_output(text: str) -> str:
    t = text.lower()
    if re.search(r"\bcomplete(?:d)?\b", t):
        return "completed"
    if any(word in t for word in ("error", "failed", "failure")):
        return "failed"
    if "cancel" in t:
        return "cancelled"
    if any(word in t for word in ("running", "queued", "pending")):
        return "running"
    return "unknown"


def main() -> int:
    queue = hub.load_json(hub.QUEUE_FILE, {"jobs": []})
    history = hub.load_json(hub.HISTORY_FILE, {"events": []})
    resources = hub.load_json(hub.RESOURCES_FILE, {})
    hub.normalize_week(resources)

    active = [
        job for job in queue.get("jobs", [])
        if job.get("status") == "running" and job.get("compute") in {"gpu", "tpu"}
    ]
    if not active:
        print("No active Kaggle accelerator jobs.")
        return 0

    if not os.environ.get("KAGGLE_API_TOKEN"):
        print("KAGGLE_API_TOKEN unavailable; active job status not refreshed.")
        return 0

    changed = False
    for job in active:
        ref = str(job.get("kaggle_ref") or "").strip()
        if not ref:
            continue
        code, output = run(["kaggle", "kernels", "status", ref])
        if code != 0:
            job["last_error"] = output[-1000:]
            job["updated_at"] = hub.iso_now()
            hub.event(history, "status_check_failed", job_id=job["id"], ref=ref, error=output[-1000:])
            changed = True
            continue

        status = state_from_output(output)
        job["last_kaggle_status"] = output[-1000:]
        job["updated_at"] = hub.iso_now()
        hub.event(history, "status_check", job_id=job["id"], ref=ref, status=status)
        changed = True

        if status == "completed":
            out_dir = ARTIFACTS / job["id"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_code, out_text = run(["kaggle", "kernels", "output", ref, "-p", str(out_dir)], timeout=300)
            job["status"] = "completed"
            job["finished_at"] = hub.iso_now()
            job["artifact_folder"] = f"artifacts/{job['id']}"
            if out_code != 0:
                job["last_error"] = f"Job completed but output download failed: {out_text[-1000:]}"
                hub.event(history, "output_failed", job_id=job["id"], ref=ref, error=out_text[-1000:])
            else:
                job["last_error"] = None
                hub.event(history, "output_collected", job_id=job["id"], ref=ref, folder=job["artifact_folder"])
        elif status in {"failed", "cancelled"}:
            job["status"] = status
            job["finished_at"] = hub.iso_now()
            job["last_error"] = output[-1000:]

    if changed:
        hub.save_json(hub.QUEUE_FILE, queue)
        hub.save_json(hub.HISTORY_FILE, history)
        hub.save_json(hub.RESOURCES_FILE, resources)
    print(f"Checked {len(active)} active Kaggle job(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
