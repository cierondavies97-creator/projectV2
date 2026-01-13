from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from engine.core.config_models import ClusterPlan
from engine.core.ids import RunContext


@dataclass(frozen=True)
class FeatureBuildContext:
    """
    Context passed into every feature-family builder.

    Fields:
      - run_ctx    : full RunContext for this job
      - cluster    : ClusterPlan (instruments + anchor/entry TFs) for this microbatch
      - trading_day: trading date for this microbatch
      - family_ids : which family_id(s) this call is for; usually a single-element list
    """

    run_ctx: RunContext
    cluster: ClusterPlan
    trading_day: date
    family_ids: Sequence[str]
