"""Customer-facing documentation shipped beside run and dataset artifacts."""
from __future__ import annotations

import shutil
from pathlib import Path

from actuate.schema.version import SCHEMA_VERSION, frozen_schema_path


def write_schema_reference(root: Path) -> Path:
    destination = Path(root) / f"canonical_v{SCHEMA_VERSION}.schema.json"
    shutil.copy2(frozen_schema_path(), destination)
    return destination


def write_dataset_readme(root: Path, *, format_name: str) -> None:
    root = Path(root)
    schema = write_schema_reference(root)
    (root / "README.md").write_text(
        f"""# Actuate dataset export

Format: `{format_name}`
Canonical contract: schema v{SCHEMA_VERSION}, included as `{schema.name}`.

This is a training-oriented export, not the untouched source archive. Camera frames may be
decoded, resized, and re-encoded by the reference dataset writer. The original capture remains
identified by `capture_id` / `source_content_hash` in the provenance record.

## Identifiers

- `capture_id`: SHA-256 identity of the source capture bytes.
- `episode_id`: the capture prefix plus an episode suffix such as `_ep00`.
- `schema_version`: the canonical JSON contract used by the exporter.

## Trust records

Read `actuate_provenance.json` (under `meta/` for LeRobot) for measurement provenance,
consent/PII state, code lineage, dropped frames, and zero-variance warnings. `not measured`
means the signal was absent; it is not equivalent to zero.
""",
        encoding="utf-8",
    )


def write_run_readme(root: Path) -> None:
    root = Path(root)
    schema = write_schema_reference(root)
    (root / "README.md").write_text(
        f"""# Actuate processing run

- `canonical.json`: validated canonical episode (schema: `{schema.name}`).
- `run_manifest.json`: source identity, frame coverage, lineage, artifact index, and omissions.
- `checkpoint.json`: stage status and the exact reason for every skipped stage.
- `review_route.json`: the L8 privacy/review/delivery queue and its reasons.
- `pipeline.rrd`: optional Rerun visualization; open it with the Rerun viewer.
- `artifacts/depth/*.npz`: dense metric depth plus the model's relative-confidence raster.
- `artifacts/perception/*`: structured perception sidecars not representable in the schema.
- `artifacts/privacy_report.json`: detector scope, counts, and the redaction recall limitation.

The staged source may be copied for reproducible processing. Training exports may re-encode
video at model resolution. Neither is described as a byte-preserving no-copy operation.

Internal `.actuate_cache` files are implementation caches and are not part of this run contract.
""",
        encoding="utf-8",
    )
