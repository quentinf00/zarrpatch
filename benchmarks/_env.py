"""Machine-specific benchmark paths, read from the environment / ``.env``.

Dataset and scratch locations are never hard-coded. They come from environment
variables, which ``pixi`` sets on activation by sourcing the repository's
gitignored ``.env`` (see ``.env.example``). When a script runs outside pixi,
``.env`` is parsed here directly; variables already set in the environment win.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), os.path.expandvars(value.strip().strip("'\"")))


_load_dotenv()


def env_path(name: str, default: str | os.PathLike | None = None) -> Path:
    """Return ``$name`` as a Path, or ``default``; fail clearly if neither is set."""
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    if default is not None:
        return Path(default)
    raise SystemExit(f"{name} is not set — copy .env.example to .env and fill it in")


def scratch(subdir: str) -> Path:
    """A per-benchmark scratch directory under ``$ZARRPATCH_SCRATCH``."""
    return env_path("ZARRPATCH_SCRATCH") / subdir
