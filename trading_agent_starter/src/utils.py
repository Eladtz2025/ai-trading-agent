import os
from pathlib import Path

def load_env():
    # Minimal loader (python-dotenv installed, but manual parse is fine)
    env_path = Path(__file__).resolve().parents[1] / "config" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k.strip(), v)
