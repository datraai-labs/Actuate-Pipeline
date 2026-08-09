"""Source resolvers -- turn a source spec into a local session directory (Phase 6 Parts A+F).

    hf://     HuggingFace Hub (LeRobot datasets, raw videos)
    s3://     S3 (uses configured AWS creds) -- written-only, no deployed bucket
    http://   direct URL / Google Drive / Dropbox
    openx://  Open-X-Embodiment via tensorflow-datasets (small slice)
    <path>    local file / session directory (handled by the SDK directly)

`resolve(source, work_root)` is the single entry point the SDK calls for remote schemes.
`push_to_hub(...)` is the write side (export a processed dataset to the Hub).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["push_to_hub", "resolve"]


def resolve(source: str, work_root: Path, **kwargs) -> Path:
    """Dispatch a remote source spec to its resolver, returning a staged session directory."""
    from actuate.sources import resolvers

    work_root = Path(work_root)
    if source.startswith("hf://"):
        return resolvers.resolve_hf(source, work_root, **kwargs)
    if source.startswith("s3://"):
        return resolvers.resolve_s3(source, work_root)
    if source.startswith(("http://", "https://")):
        return resolvers.resolve_http(source, work_root)
    if source.startswith("openx://"):
        return resolvers.resolve_openx(source, work_root, **kwargs)
    raise ValueError(
        f"unrecognised source scheme in {source!r}. Use hf:// s3:// http(s):// openx:// "
        "or a local path.")


def push_to_hub(local_dir: Path, repo_id: str, *, private: bool = True,
                token: str | None = None) -> str:
    """Upload a processed/exported dataset directory to the HuggingFace Hub (Part F write).

    Token from `token`, else HF_TOKEN, else the `huggingface-cli login` cache. Returns the
    repo URL. Never logs or stores the token.
    """
    import os

    from huggingface_hub import HfApi, create_repo

    token = token or os.environ.get("HF_TOKEN")
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"nothing to push: {local_dir} is not a directory")

    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    HfApi().upload_folder(folder_path=str(local_dir), repo_id=repo_id,
                          repo_type="dataset", token=token)
    return f"https://huggingface.co/datasets/{repo_id}"
