"""Minimal competition demo for budgeted HAB hypothesis exploration."""

from .agent import ExperimentAction, HypothesisAgent
from .data import generate_demo_data
from .experiment import (
    evaluate_action,
    evaluate_baselines,
    evaluate_negative_controls,
    evaluate_seasonal_baseline,
    random_search_reference,
)
from .aquaculture import (
    EVIDENCE_CONFIDENCE,
    MECHANISM_LABELS,
    PRODUCTION_PROFILES,
    project_aquaculture_risk,
)
from .evidence import SOUTH_AUSTRALIA_CASE
from .workflow import run_exploration
from .scenario import SCENARIO_PRESETS, project_synthetic_scenario
from .multiscale import DEFAULT_SCALES, detect_multiscale_anomalies
from .router import route_methods
from .transfer_entropy import estimate_te_cte_network, summarise_te_cte_by_lag
from .spatial_durbin import build_weight_matrix, estimate_spatial_durbin_impacts
from .real_replay import (
    SPECIES,
    build_sa_replay,
    load_sa_real_case,
    project_real_aquaculture_priority,
    real_data_router,
)
from .global_cases import (
    GLOBAL_EVIDENCE_CASES,
    build_norway_replay,
    global_evidence_frame,
    load_norway_real_case,
)

__all__ = [
    "ExperimentAction",
    "HypothesisAgent",
    "generate_demo_data",
    "evaluate_action",
    "evaluate_baselines",
    "evaluate_negative_controls",
    "evaluate_seasonal_baseline",
    "random_search_reference",
    "SCENARIO_PRESETS",
    "project_synthetic_scenario",
    "PRODUCTION_PROFILES",
    "MECHANISM_LABELS",
    "EVIDENCE_CONFIDENCE",
    "project_aquaculture_risk",
    "SOUTH_AUSTRALIA_CASE",
    "run_exploration",
    "DEFAULT_SCALES",
    "detect_multiscale_anomalies",
    "route_methods",
    "estimate_te_cte_network",
    "summarise_te_cte_by_lag",
    "build_weight_matrix",
    "estimate_spatial_durbin_impacts",
    "SPECIES",
    "build_sa_replay",
    "load_sa_real_case",
    "project_real_aquaculture_priority",
    "real_data_router",
    "GLOBAL_EVIDENCE_CASES",
    "build_norway_replay",
    "global_evidence_frame",
    "load_norway_real_case",
]
