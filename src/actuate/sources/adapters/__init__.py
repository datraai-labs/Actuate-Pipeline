"""Native-format adapters: map a pre-annotated dataset into the canonical schema."""

from actuate.sources.adapters.egocentric_rgbd import adapt as adapt_egocentric_rgbd

__all__ = ["adapt_egocentric_rgbd"]
