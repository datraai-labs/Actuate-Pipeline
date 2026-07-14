"""FastAPI service — a CONSUMER of the actuate library, never a dependency of it.

Master Spec §2.1 / AWS Architecture §4.2. Nothing in a layer package may import this;
the import-linter contract in `.importlinter` enforces it.

**Stub only this increment.** Health check + routes that declare themselves unimplemented.
The v1 service (top-level `service/api.py`) still runs the existing pipeline; it keeps job
state in memory and regex-scrapes its own log file to recover it, which is why AWS
Architecture §4.3 replaces that with a persistent `jobs` table. That replacement is a
later increment.
"""

from __future__ import annotations

from actuate.service.app import create_app

__all__ = ["create_app"]
