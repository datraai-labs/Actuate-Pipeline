from actuate.config import ConsentStatus, PiiStatus, RigType
from actuate.feedback import route_episode
from actuate.schema import CanonicalEpisode
from actuate.schema.episode import EpisodeMeta


def _episode(**updates):
    base = CanonicalEpisode(
        episode_id="ep",
        capture_id="c" * 64,
        rig=RigType.HEAD_MOUNTED,
        task="move cup",
        consent=ConsentStatus.GRANTED,
        pii_status=PiiStatus.PASSED,
        episode_meta=EpisodeMeta(quality=4),
    )
    return base.model_copy(update=updates)


def test_privacy_failure_routes_to_hard_block():
    route = route_episode(_episode(consent=ConsentStatus.REVOKED))
    assert route.queue == "privacy_block" and route.may_export is False


def test_low_quality_is_reviewable_robustness_data_not_silently_dropped():
    route = route_episode(_episode(episode_meta=EpisodeMeta(quality=2)))
    assert route.queue == "robustness_review"
    assert route.may_export is True and route.needs_human_review is True


def test_clean_episode_becomes_delivery_candidate():
    route = route_episode(_episode())
    assert route.queue == "delivery_candidate" and route.reasons == ()
