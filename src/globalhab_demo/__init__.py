"""Minimal competition demo for budgeted HAB hypothesis exploration."""

from .agent import ExperimentAction, HypothesisAgent
from .data import generate_demo_data
from .experiment import evaluate_action, evaluate_seasonal_baseline

__all__ = [
    "ExperimentAction",
    "HypothesisAgent",
    "generate_demo_data",
    "evaluate_action",
    "evaluate_seasonal_baseline",
]

