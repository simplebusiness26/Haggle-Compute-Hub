from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

result = {
    "python": platform.python_version(),
    "torch": None,
    "cuda_available": False,
    "cuda_device_count": 0,
    "cuda_devices": [],
}

try:
    import torch

    result["torch"] = torch.__version__
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda_device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available():
        result["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception as exc:
    result["torch_error"] = str(exc)

try:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    result["nvidia_smi"] = proc.stdout.strip().splitlines()
except Exception as exc:
    result["nvidia_smi_error"] = str(exc)

out = Path("/kaggle/working/compute-hub-smoke-test.json")
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
