"""End-to-end Phase 1 pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hft_research.pipeline.phase1 import Phase1Result, run_phase1_pipeline

__all__ = ["Phase1Result", "run_phase1_pipeline"]


def __getattr__(name: str):
    if name in __all__:
        from hft_research.pipeline import phase1 as _phase1

        return getattr(_phase1, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

