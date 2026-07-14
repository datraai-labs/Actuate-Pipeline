"""THE LOAD+TRAIN GATE — Master Spec §L7, non-negotiable.

    "load the exported LeRobot v3 dataset with LeRobot's own loader and run one real
     training step, including the normalization round-trip. A schema-valid-but-untrainable
     export must be caught by this gate — do not assert schema-correctness in place of a
     real load+train."

So this file does not check that our Parquet has the right columns. It hands the export to
LeRobot, hands LeRobot's batches to a real LeRobot policy, and runs a real optimizer step.
If the loss is not finite, or no gradient flows, the export is not training-ready — no
matter how correct it looks.

Runs on CPU. Slow, and valid: one step on one 95-second clip.

### What this proves, and what it does not

PROVES: the exporter's mechanics. Format, chunking, the delta-timestamp action-chunk path,
normalization, and that a policy can actually consume the result.

DOES NOT PROVE: that we have a training-ready dataset. The capture is NON-DELIVERABLE
(consent=pending), it is n=1, and its action is ego-contaminated (no L1 SLAM, so a wrist
delta on a head-mounted rig is hand motion + head motion). See STATUS.md.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from actuate.canonical import build_episode
from actuate.package import (
    ExportRefused,
    compute_norm_stats,
    denormalize_p01_p99,
    export_lerobot_v3,
    normalize_p01_p99,
)

pytest.importorskip("lerobot", reason="lerobot not installed")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.real_data

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "processed" / "session_001"
VIDEO = PROCESSED / "redacted_compressed.mp4"

CAPTURE_HASH = "4cff6acb16f7f390a35034eb7ddc088d76e13f1e814214fb8e49180fc1d6bb83"
TASK = "Sort and staple paperwork at the workbench."
TASK_PROV = (
    "operator_supplied; grounded in two independent VLM reads of this footage recorded in "
    "docs/PIPELINE_STATUS.md. NOT the v1 classifier, which returned 'unknown'."
)
HORIZON = 16

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not VIDEO.exists(), reason="real redacted capture not on disk"),
]


@pytest.fixture(scope="module")
def episode():
    return build_episode(
        PROCESSED, CAPTURE_HASH, task=TASK, task_provenance=TASK_PROV
    )


@pytest.fixture(scope="module")
def exported(episode, tmp_path_factory):
    root = tmp_path_factory.mktemp("lerobot") / "ds"
    res = export_lerobot_v3(
        episode, root, repo_id="actuate/gate", overwrite=True, video=VIDEO
    )
    yield res
    shutil.rmtree(root, ignore_errors=True)


# --- the exporter refuses what it cannot honestly export ----------------------------------


def test_export_refuses_an_episode_with_no_task():
    """LeRobot does `frame.pop("task")` and would KeyError three layers down. We refuse
    here, where a human can read why — and we do NOT substitute v1's template, which says
    'Perform unknown task using right hand with power grasp.'"""
    ep = build_episode(PROCESSED, CAPTURE_HASH)  # task defaults to None
    assert ep.task is None

    with pytest.raises(ExportRefused, match="has no `task`"):
        export_lerobot_v3(ep, Path("/tmp/never"), video=VIDEO)


def test_export_refuses_a_state_only_dataset(episode, tmp_path):
    """A VLA dataset without observation.images.* is not a VLA dataset — and LeRobot's own
    policies say so: 'You must provide at least one image or the environment state among
    the inputs.' Caught at export, not at train time."""
    with pytest.raises(ExportRefused, match="no video supplied"):
        export_lerobot_v3(episode, tmp_path / "x", video=None)


# --- the export is what LeRobot v3 says it is ---------------------------------------------


def test_lerobot_wrote_a_v3_dataset(exported):
    import json

    info = json.loads((exported.root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert "observation.images.head" in info["features"]
    assert "observation.state" in info["features"]
    assert "action" in info["features"]

    # chunked Parquet + per-camera chunked MP4 — L7's contract
    assert list(exported.root.glob("data/chunk-*/file-*.parquet"))
    assert list(exported.root.glob("videos/observation.images.head/chunk-*/file-*.mp4"))


# --- THE GATE -------------------------------------------------------------------------------


def _load(root: Path, horizon: int = HORIZON):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        "actuate/gate",
        root=root,
        delta_timestamps={"action": [i / 30 for i in range(horizon)]},
    )


def _make_policy(ds):
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    feats = dataset_to_policy_features(ds.meta.features)
    out_f = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    in_f = {k: v for k, v in feats.items() if k not in out_f}
    cfg = ACTConfig(
        input_features=in_f,
        output_features=out_f,
        chunk_size=HORIZON,
        n_action_steps=HORIZON,
        device="cpu",
    )
    return ACTPolicy(cfg, dataset_stats=ds.meta.stats)


def test_lerobots_own_loader_reads_the_export(exported):
    ds = _load(exported.root)
    assert ds.num_frames == exported.n_frames
    assert ds.num_episodes == 1

    sample = ds[0]
    assert tuple(sample["observation.images.head"].shape) == (3, 224, 224)
    assert tuple(sample["observation.state"].shape) == (8,)
    assert sample["task"] == TASK


def test_action_chunking_via_delta_timestamps(exported):
    """LeRobot v3's native action-chunk loading. Continuous, dense, time-aligned — what a
    flow-matching action head is trained to predict (Master Spec §2.1)."""
    ds = _load(exported.root)
    assert tuple(ds[0]["action"].shape) == (HORIZON, 8)


def test_ONE_REAL_TRAINING_STEP(exported):
    """THE GATE. Not a schema assertion — a real policy, a real loss, a real optimizer step."""
    ds = _load(exported.root)
    policy = _make_policy(ds)
    policy.train()

    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    loss, _ = policy.forward(batch)
    loss_before = loss.item()
    assert np.isfinite(loss_before), f"loss is not finite ({loss_before}) — untrainable"

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0).item()
    assert np.isfinite(grad_norm), "gradients are not finite — untrainable"
    assert grad_norm > 0, "NO GRADIENT FLOWED. The export is schema-valid and untrainable."

    opt.step()
    opt.zero_grad()

    loss_after, _ = policy.forward(batch)
    assert np.isfinite(loss_after.item())


# --- normalization round-trip (TRI LBM: normalization dominates) -----------------------------


def test_normalization_round_trips(episode):
    """1/99 percentile -> [-1,1] -> back. Must recover the original within float tolerance.

    TRI LBM's finding is that normalization dominates downstream performance, so a
    normalization that silently loses information is not a cosmetic bug — it is the bug.
    """
    from actuate.canonical import state_and_action_vectors

    state, action, valid = state_and_action_vectors(episode)
    s, a = state[valid], action[valid]
    stats = compute_norm_stats(s, a)

    norm = normalize_p01_p99(s, stats.state)
    assert norm.min() >= -1.0 - 1e-6 and norm.max() <= 1.0 + 1e-6, "not in [-1, 1]"

    back = denormalize_p01_p99(norm, stats.state)
    # Values outside the 1/99 percentiles are clipped by design; check the bulk.
    inside = (s >= np.asarray(stats.state.p01)) & (s <= np.asarray(stats.state.p99))
    assert np.allclose(back[inside], s[inside], atol=1e-4), (
        "normalization round-trip lost information inside the 1/99 range"
    )


def test_raw_percentiles_AND_mean_std_are_both_shipped(exported):
    """A customer on 2/98-per-timestep (TRI LBM) or z-score (EgoMimic) must be able to
    re-derive their own normalization without a full pass over the dataset."""
    import json

    stats = json.loads((exported.root / "meta" / "actuate_norm_stats.json").read_text())
    for field in ("state", "action"):
        for key in ("p01", "p99", "mean", "std"):
            assert stats[field][key], f"{field}.{key} missing from shipped norm stats"


def test_provenance_travels_with_the_dataset(exported):
    """The gaps must not live in someone's head. They ship with the data."""
    import json

    prov = json.loads((exported.root / "meta" / "actuate_provenance.json").read_text())
    assert prov["source_content_hash"] == CAPTURE_HASH
    assert prov["consent"] == "pending"
    assert prov["pii_status"] == "pending"

    notes = prov["derivation_notes"]
    assert "EGO-CONTAMINATED" in notes["action_semantics"], (
        "the exported dataset does not carry the warning that its action conflates hand "
        "motion with head motion"
    )
    assert "INFERRED" in notes["rig_type"]


# --- THE BROKEN VARIANT: schema-valid, untrainable, and the gate MUST catch it ---------------


def test_a_schema_valid_but_UNTRAINABLE_export_is_caught_by_the_gate(exported):
    """Required by Master Spec §L7: *"A schema-valid-but-untrainable export must be caught
    by this gate."*

    So construct one. We poison the actions with NaN — which is the realistic failure: a
    frame with no detected hand, zero-filled or NaN-filled instead of dropped. The Parquet
    is still perfectly well-formed, `meta/info.json` is still correct, and every
    schema-correctness assertion we could write would still pass.

    The training step is what catches it. That is the whole argument for having this gate
    rather than a schema test.
    """
    import pyarrow.parquet as pq
    import pyarrow as pa

    root = exported.root
    parquet = next(root.glob("data/chunk-*/file-*.parquet"))

    original = parquet.read_bytes()
    try:
        table = pq.read_table(parquet)
        cols = table.to_pydict()

        # Poison 10% of actions with NaN — exactly what a "just fill the gaps" change does.
        actions = [list(a) for a in cols["action"]]
        for i in range(0, len(actions), 10):
            actions[i] = [float("nan")] * len(actions[i])
        cols["action"] = actions

        pq.write_table(pa.table(cols, schema=table.schema), parquet)

        # The dataset still LOADS. Schema is intact. Nothing looks wrong.
        ds = _load(root)
        assert ds.num_frames == exported.n_frames, "the poisoned export still loads fine"

        # And it is untrainable.
        policy = _make_policy(ds)
        policy.train()
        loader = torch.utils.data.DataLoader(
            ds, batch_size=64, shuffle=False, num_workers=0
        )
        batch = next(iter(loader))

        loss, _ = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0).item()

        assert not (
            np.isfinite(loss.item()) and np.isfinite(grad_norm)
        ), (
            "the NaN-poisoned export produced a finite loss AND finite gradients — the "
            "gate did not catch an untrainable dataset, which means the gate is not "
            "actually gating anything."
        )
    finally:
        parquet.write_bytes(original)
