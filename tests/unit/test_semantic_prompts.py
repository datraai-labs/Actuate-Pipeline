from __future__ import annotations

import pytest

from actuate.language.semantic_prompts import (
    PROMPTS,
    WINDOW_OBSERVER_SYSTEM,
    prompt_hash,
    render_user_prompt,
)


def test_approved_prompt_set_is_complete():
    assert set(PROMPTS) == {
        "window_observer",
        "targeted_followup",
        "episode_synthesis",
        "corpus_relations",
    }


def test_window_observer_keeps_forbidden_authority_out():
    assert "Never certify or decide rights, consent" in WINDOW_OBSERVER_SYSTEM
    assert "Do not emit numerical confidence" in WINDOW_OBSERVER_SYSTEM


def test_render_requires_exact_runtime_values():
    values = {
        "window_id": "w1",
        "source_start_ms": 0,
        "source_end_ms": 8000,
        "frame_timestamp_map_json": "[]",
        "declared_metadata_json": "{}",
    }
    rendered = render_user_prompt("window_observer", **values)
    assert "window w1 from 0 ms to 8000 ms" in rendered
    assert "{{" not in rendered

    with pytest.raises(AssertionError):
        render_user_prompt("window_observer", window_id="w1")


def test_prompt_hash_is_stable_and_prompt_specific():
    assert prompt_hash("window_observer") == prompt_hash("window_observer")
    assert prompt_hash("window_observer") != prompt_hash("targeted_followup")
