#!/usr/bin/env python3
"""Write non-sensitive Kaggle connection health to state/system.json."""

from __future__ import annotations

import os
import subprocess

import hub


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60, check=False)
    return proc.returncode, proc.stdout.strip()


def main() -> int:
    token_present = bool(os.environ.get("KAGGLE_API_TOKEN"))
    owner = str(os.environ.get("KAGGLE_OWNER") or "").strip() or None
    version_code, version = run(["kaggle", "--version"])
    authenticated = False
    message = "Kaggle token not configured in this repository."
    if token_present:
        code, out = run(["kaggle", "kernels", "list", "-m", "--page-size", "1"])
        authenticated = code == 0
        message = "Kaggle authentication verified." if authenticated else "Kaggle token is present but authentication failed."
        if not authenticated:
            # Keep diagnostic small and strip any accidental token-shaped text defensively.
            message += " " + out[-300:].replace(os.environ.get("KAGGLE_API_TOKEN", ""), "***")
    payload = {
        "checked_at": hub.iso_now(),
        "kaggle_cli": version if version_code == 0 else "unavailable",
        "token_present": token_present,
        "authenticated": authenticated,
        "owner": owner,
        "message": message,
    }
    hub.save_json(hub.STATE / "system.json", payload)
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
