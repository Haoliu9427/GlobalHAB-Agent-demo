"""Minimal competition demo for budgeted HAB hypothesis exploration."""

from .agent import ExperimentAction, HypothesisAgent
from .data import generate_demo_data
from .experiment import evaluate_action, evaluate_seasonal_baseline
from .scenario import SCENARIO_PRESETS, project_synthetic_scenario

__all__ = [
    "ExperimentAction",
    "HypothesisAgent",
    "generate_demo_data",
    "evaluate_action",
    "evaluate_seasonal_baseline",
    "SCENARIO_PRESETS",
    "project_synthetic_scenario",
]
