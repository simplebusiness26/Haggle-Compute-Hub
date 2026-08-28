#!/usr/bin/env python3
"""Kaggle Compute Hub scheduler and command processor.

The hub is intentionally conservative:
- queue JSON cannot execute arbitrary shell commands;
- only repository kernel folders may be dispatched;
- CPU jobs never consume Kaggle quota;
- GPU/TPU jobs are held if estimated budget is insufficient;
- consequential competition entry/submission is outside this scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state"
QUEUE_FILE = STATE / "queue.json"
HISTORY_FILE = STATE / "history.json"
RESOURCES_FILE = STATE / "resources.json"
CATALOG_FILE = STATE / "catalog.json"
RESULT_FILE = ROOT / "results" / "latest.md"

SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
SAFE_KERNEL = re.compile(r"^kernels/[A-Za-z0-9._/-]+$")
ALLOWED_COMPUTE = {"cpu", "gpu", "tpu"}
ALLOWED_PRIORITY = {"low", "normal", "high", "urgent"}
ALLOWED_ACCELERATORS = {
    "NvidiaTeslaT4",
    "NvidiaTeslaT4Highmem",
    "NvidiaTeslaA100",
    "NvidiaL4",
    "NvidiaL4X1",
    "NvidiaH100",
    "NvidiaRtxPro6000",
    "TpuV38",
    "Tpu1VmV38",
    "TpuV5E8",
    "TpuV6E8",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utcnow().isoformat()


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_result(title: str, body: list[str]) -> None:
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text("\n".join([f"# {title}", "", *body, ""]) + "\n", encoding="utf-8")


def event(history: dict[str, Any], kind: str, **data: Any) -> None:
    history.setdefault("events", []).insert(0, {"at": iso_now(), "kind": kind, **data})
    history["events"] = history["events"][:500]


def normalize_week(resources: dict[str, Any]) -> None:
    now = utcnow()
    monday = (now - timedelta(days=now.weekday())).date().isoformat()
    if resources.get("week_started_utc") != monday:
        resources["week_started_utc"] = monday
        resources["estimated_gpu_hours_used"] = 0.0
        resources["active_kaggle_jobs"] = []


def job_id(title: str, request_id: str) -> str:
    digest = hashlib.sha1(f"{title}|{request_id}|{iso_now()}".encode()).hexdigest()[:10]
    return f"job-{digest}"


def validate_job(raw: dict[str, Any], request_id: str, catalog: dict[str, Any]) -> dict[str, Any]:
    title = str(raw.get("title") or "").strip()
    if not title or len(title) > 160:
        raise ValueError("job.title must be 1-160 characters")

    family = str(raw.get("family") or "").strip()
    families = catalog.get("families", {})
    if family not in families:
        raise ValueError(f"unknown job family: {family}")

    family_cfg = families[family]
    compute = str(raw.get("compute") or family_cfg.get("default_compute") or "cpu").lower()
    if compute not in ALLOWED_COMPUTE:
        raise ValueError(f"compute must be one of {sorted(ALLOWED_COMPUTE)}")

    priority = str(raw.get("priority") or "normal").lower()
    if priority not in ALLOWED_PRIORITY:
        raise ValueError(f"priority must be one of {sorted(ALLOWED_PRIORITY)}")

    try:
        value = float(raw.get("value", family_cfg.get("default_value", 50)))
    except (TypeError, ValueError):
        raise ValueError("value must be numeric")
    value = max(0.0, min(100.0, value))

    try:
        estimated = float(raw.get("estimated_minutes", 10 if compute == "cpu" else 30))
    except (TypeError, ValueError):
        raise ValueError("estimated_minutes must be numeric")
    if estimated <= 0 or estimated > 720:
        raise ValueError("estimated_minutes must be >0 and <=720")

    kernel_path = str(raw.get("kernel_path") or "").strip()
    if compute in {"gpu", "tpu"}:
        if not kernel_path or not SAFE_KERNEL.fullmatch(kernel_path) or ".." in kernel_path:
            raise ValueError("GPU/TPU jobs require a safe kernels/... kernel_path")
        resolved = (ROOT / kernel_path).resolve()
        if ROOT not in resolved.parents:
            raise ValueError("kernel_path escapes repository")

    deadline = raw.get("deadline")
    if deadline is not None:
        deadline = str(deadline)
        try:
            datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("deadline must be ISO-8601") from exc

    accelerator = raw.get("accelerator")
    if accelerator is None:
        accelerator = "TpuV38" if compute == "tpu" else "NvidiaTeslaT4" if compute == "gpu" else None
    if accelerator is not None and accelerator not in ALLOWED_ACCELERATORS:
        raise ValueError("accelerator is not allow-listed")

    return {
        "id": job_id(title, request_id),
        "title": title,
        "family": family,
        "compute": compute,
        "priority": priority,
        "value": round(value, 1),
        "estimated_minutes": round(estimated, 1),
        "kernel_path": kernel_path or None,
        "accelerator": accelerator,
        "deadline": deadline,
        "source_project": str(raw.get("source_project") or "").strip() or None,
        "notes": str(raw.get("notes") or "").strip()[:500] or None,
        "status": "queued",
        "created_at": iso_now(),
        "updated_at": iso_now(),
        "request_id": request_id,
        "kaggle_ref": None,
        "last_error": None,
    }


def urgency(job: dict[str, Any]) -> float:
    value = job.get("deadline")
    if not value:
        return 1.0
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        hours = (dt.astimezone(timezone.utc) - utcnow()).total_seconds() / 3600
    except ValueError:
        return 1.0
    if hours <= 0:
        return 0.25
    if hours <= 24:
        return 2.0
    if hours <= 72:
        return 1.5
    if hours <= 168:
        return 1.2
    return 1.0


def scheduler_score(job: dict[str, Any], catalog: dict[str, Any]) -> float:
    priority_weight = float(catalog.get("priorities", {}).get(job.get("priority"), 1.0))
    value = float(job.get("value", 0))
    mins = max(5.0, float(job.get("estimated_minutes", 30)))
    efficiency = value / mins
    # Scale to a convenient human-readable range. A score is relative, not a probability.
    return round(efficiency * priority_weight * urgency(job) * 20.0, 2)


def remaining_gpu_hours(resources: dict[str, Any], include_reserve: bool = False) -> float:
    total = float(resources.get("configured_gpu_hours_per_week", 30.0))
    used = float(resources.get("estimated_gpu_hours_used", 0.0))
    reserve = 0.0 if include_reserve else total * float(resources.get("reserve_fraction", 0.2))
    return max(0.0, total - used - reserve)


def queue_rank(queue: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = [j for j in queue.get("jobs", []) if j.get("status") == "queued"]
    for job in jobs:
        job["scheduler_score"] = scheduler_score(job, catalog)
    return sorted(jobs, key=lambda j: (j["scheduler_score"], j.get("value", 0)), reverse=True)


def can_run(job: dict[str, Any], resources: dict[str, Any]) -> tuple[bool, str]:
    compute = job.get("compute")
    cfg = resources.get("resources", {}).get(compute, {})
    if not cfg.get("enabled", False):
        return False, f"{compute} is disabled"
    if compute == "cpu":
        return True, "CPU work does not consume Kaggle GPU quota"
    needed = float(job.get("estimated_minutes", 0)) / 60.0
    if compute == "gpu" and needed > remaining_gpu_hours(resources):
        # Urgent work can consume the configured reserve, but never exceed the total budget.
        if job.get("priority") == "urgent" and needed <= remaining_gpu_hours(resources, include_reserve=True):
            return True, "urgent job may use reserved GPU budget"
        return False, "insufficient unreserved weekly GPU budget"
    return True, "resource budget permits dispatch"


def render_status(queue: dict[str, Any], history: dict[str, Any], resources: dict[str, Any], catalog: dict[str, Any]) -> list[str]:
    ranked = queue_rank(queue, catalog)
    total = float(resources.get("configured_gpu_hours_per_week", 30.0))
    used = float(resources.get("estimated_gpu_hours_used", 0.0))
    lines = [
        f"Generated: **{iso_now()}**",
        "",
        f"GPU budget estimate: **{used:.2f} / {total:.2f} hours** used this week.",
        f"Unreserved GPU time remaining: **{remaining_gpu_hours(resources):.2f} hours**.",
        f"Queued jobs: **{len(ranked)}**.",
        "",
        "## Queue",
        "",
    ]
    if not ranked:
        lines.append("Queue is empty.")
    else:
        lines += ["| # | Score | Compute | Priority | Value | Est. min | Job |", "|---:|---:|---|---|---:|---:|---|"]
        for i, job in enumerate(ranked[:20], 1):
            lines.append(
                f"| {i} | {job['scheduler_score']:.2f} | {job['compute']} | {job['priority']} | {job['value']:.0f} | {job['estimated_minutes']:.0f} | {job['title']} |"
            )
    running = [j for j in queue.get("jobs", []) if j.get("status") in {"dispatching", "running"}]
    if running:
        lines += ["", "## Active", ""] + [f"- **{j['title']}** — {j['status']} ({j.get('kaggle_ref') or 'no ref yet'})" for j in running]
    return lines


def _run(cmd: list[str], timeout: int = 90) -> tuple[int, str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    return proc.returncode, proc.stdout.strip()


def render_kernel_metadata(folder: Path, accelerator: str | None) -> str | None:
    path = folder / "kernel-metadata.json"
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    owner = str(__import__("os").environ.get("KAGGLE_OWNER") or "").strip()
    if "__OWNER__" in raw:
        if not owner:
            raise RuntimeError("KAGGLE_OWNER is required for metadata containing __OWNER__")
        raw = raw.replace("__OWNER__", owner)
    data = json.loads(raw)
    if accelerator:
        data["accelerator"] = accelerator
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return str(data.get("id") or "") or None


def dispatch_kaggle(job: dict[str, Any]) -> tuple[bool, str, str | None]:
    import os
    if not os.environ.get("KAGGLE_API_TOKEN"):
        return False, "KAGGLE_API_TOKEN is not configured in this repository", None
    kernel_path = ROOT / str(job["kernel_path"])
    if not kernel_path.is_dir():
        return False, f"kernel folder does not exist: {job['kernel_path']}", None
    try:
        ref = render_kernel_metadata(kernel_path, job.get("accelerator"))
    except Exception as exc:
        return False, f"metadata error: {exc}", None
    code, output = _run(["kaggle", "kernels", "push", "-p", str(kernel_path)], timeout=180)
    if code != 0:
        return False, output[-2000:], ref
    return True, output[-2000:] or "Kaggle accepted kernel push", ref


def execute_cpu(job: dict[str, Any]) -> tuple[bool, str]:
    # CPU jobs are intentionally limited to built-in utility types. They may not carry commands.
    family = job.get("family")
    if family not in {"dataset", "cpu_utility"}:
        return False, "CPU execution is only implemented for dataset/cpu_utility jobs in v1"
    return True, "CPU utility job acknowledged; no GPU quota consumed"


def dispatch_best(queue: dict[str, Any], history: dict[str, Any], resources: dict[str, Any], catalog: dict[str, Any]) -> str:
    ranked = queue_rank(queue, catalog)
    if not ranked:
        return "No queued jobs."
    blocked: list[str] = []
    for candidate in ranked:
        ok, why = can_run(candidate, resources)
        if not ok:
            blocked.append(f"{candidate['id']}: {why}")
            continue
        candidate["status"] = "dispatching"
        candidate["updated_at"] = iso_now()
        event(history, "dispatch_attempt", job_id=candidate["id"], title=candidate["title"], compute=candidate["compute"], reason=why)
        if candidate["compute"] == "cpu":
            success, message = execute_cpu(candidate)
            ref = None
        else:
            success, message, ref = dispatch_kaggle(candidate)
        if success:
            candidate["status"] = "running" if candidate["compute"] != "cpu" else "completed"
            candidate["kaggle_ref"] = ref
            candidate["updated_at"] = iso_now()
            candidate["started_at"] = iso_now()
            candidate["last_error"] = None
            if candidate["compute"] == "gpu":
                resources["estimated_gpu_hours_used"] = round(
                    float(resources.get("estimated_gpu_hours_used", 0.0)) + float(candidate["estimated_minutes"]) / 60.0,
                    3,
                )
            event(history, "dispatched", job_id=candidate["id"], title=candidate["title"], status=candidate["status"], ref=ref, message=message[:1000])
            return f"Dispatched **{candidate['title']}** ({candidate['compute']}). {message}"
        candidate["status"] = "queued"
        candidate["last_error"] = message[:1000]
        candidate["updated_at"] = iso_now()
        event(history, "dispatch_failed", job_id=candidate["id"], title=candidate["title"], error=message[:1000])
        return f"Dispatch failed for **{candidate['title']}**: {message}"
    return "No job could run. " + "; ".join(blocked[:5])


def mark_job(queue: dict[str, Any], history: dict[str, Any], job_id_value: str, status: str, note: str | None = None) -> str:
    allowed = {"completed", "failed", "queued", "cancelled"}
    if status not in allowed:
        raise ValueError(f"status must be one of {sorted(allowed)}")
    for job in queue.get("jobs", []):
        if job.get("id") == job_id_value:
            job["status"] = status
            job["updated_at"] = iso_now()
            if status in {"completed", "failed", "cancelled"}:
                job["finished_at"] = iso_now()
            if note:
                job["last_note"] = note[:1000]
            event(history, "job_status", job_id=job_id_value, status=status, note=note)
            return f"{job_id_value} -> {status}"
    raise ValueError("job id not found")


def process(command: dict[str, Any]) -> int:
    queue = load_json(QUEUE_FILE, {"jobs": []})
    history = load_json(HISTORY_FILE, {"events": []})
    resources = load_json(RESOURCES_FILE, {})
    catalog = load_json(CATALOG_FILE, {})
    normalize_week(resources)

    action = str(command.get("action") or "status").strip().lower()
    request_id = str(command.get("request_id") or f"request-{int(utcnow().timestamp())}")
    if not SAFE_ID.fullmatch(request_id):
        raise ValueError("request_id contains unsupported characters")

    message = ""
    if action == "idle":
        message = "Compute Hub is installed and idle."
    elif action == "status":
        message = "Status refreshed."
    elif action == "enqueue":
        raw = command.get("job")
        if not isinstance(raw, dict):
            raise ValueError("enqueue requires a job object")
        job = validate_job(raw, request_id, catalog)
        queue.setdefault("jobs", []).append(job)
        event(history, "enqueued", job_id=job["id"], title=job["title"], compute=job["compute"], value=job["value"])
        message = f"Queued **{job['title']}** as `{job['id']}`. Scheduler score: **{scheduler_score(job, catalog):.2f}**."
    elif action == "schedule":
        message = dispatch_best(queue, history, resources, catalog)
    elif action == "cancel":
        message = mark_job(queue, history, str(command.get("job_id") or ""), "cancelled", str(command.get("note") or "") or None)
    elif action == "set_status":
        message = mark_job(
            queue,
            history,
            str(command.get("job_id") or ""),
            str(command.get("status") or ""),
            str(command.get("note") or "") or None,
        )
    elif action == "set_gpu_usage":
        hours = float(command.get("hours"))
        total = float(resources.get("configured_gpu_hours_per_week", 30.0))
        if hours < 0 or hours > total:
            raise ValueError("hours must be between 0 and configured weekly budget")
        resources["estimated_gpu_hours_used"] = round(hours, 3)
        event(history, "usage_override", hours=hours)
        message = f"GPU usage estimate set to **{hours:.2f}h**."
    else:
        raise ValueError(f"unsupported action: {action}")

    save_json(QUEUE_FILE, queue)
    save_json(HISTORY_FILE, history)
    save_json(RESOURCES_FILE, resources)
    write_result("Kaggle Compute Hub", [message, "", *render_status(queue, history, resources, catalog)])
    print(message)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="control/command.json")
    args = parser.parse_args()
    try:
        command = load_json(ROOT / args.command, {})
        if not isinstance(command, dict):
            raise ValueError("command JSON must be an object")
        return process(command)
    except Exception as exc:
        write_result("Kaggle Compute Hub — ERROR", [f"**{type(exc).__name__}:** {exc}"])
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
