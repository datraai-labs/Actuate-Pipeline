"""Pipeline orchestration (library layer) -- the single entry point both the CLI and the
SDK drive. Lifting the stages here (out of cli.run_all) is what lets `sdk.process` and
`actuate run all` share one code path without inverting the import-linter contract.
"""

from __future__ import annotations

from actuate.pipeline.cache import stage_cached
from actuate.pipeline.run import PipelineResult, run_pipeline

__all__ = ["run_pipeline", "PipelineResult", "stage_cached"]
