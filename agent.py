"""A small auditable agent for budgeted scientific hypothesis selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentAction:
    route: str
    lag_days: int
    model: str

    @property
    def action_id(self) -> str:
        return f"{self.route}__{self.lag_days}d__{self.model}"


class HypothesisAgent:
    """Select untested actions using feedback, diversity, and a fixed budget."""

    def __init__(self, actions: list[ExperimentAction], budget: int):
        if budget < 1 or budget > len(actions):
            raise ValueError("budget must be within the candidate action count")
        self.actions = actions
        self.budget = budget
        self.log: list[dict[str, object]] = []

    def _priority(self, action: ExperimentAction) -> float:
        # Operational horizons receive a small transparent prior; feedback and
        # diversity still determine the subsequent order.
        horizon_prior = {7: 0.08, 14: 0.07, 30: 0.06}.get(action.lag_days, 0.02)
        if not self.log:
            return (
                horizon_prior
                + (0.03 if action.route == "local" else 0.0)
                - (0.02 if action.model == "random_forest" else 0.0)
            )
        best = max(self.log, key=lambda row: float(row["utility"]))
        similarity = 0.0
        if action.route == best["route"]:
            similarity += 0.18
        similarity += 0.12 / (1.0 + abs(action.lag_days - int(best["lag_days"])))
        if action.model == best["model"]:
            similarity += 0.05
        route_count = sum(row["route"] == action.route for row in self.log)
        lag_count = sum(int(row["lag_days"]) == action.lag_days for row in self.log)
        model_count = sum(row["model"] == action.model for row in self.log)
        diversity = 0.20 / (1 + route_count) + 0.10 / (1 + lag_count) + 0.05 / (1 + model_count)
        return similarity + diversity + horizon_prior

    def next_action(self) -> ExperimentAction:
        if len(self.log) >= self.budget:
            raise StopIteration("experiment budget exhausted")
        tested = {str(row["action_id"]) for row in self.log}
        candidates = [action for action in self.actions if action.action_id not in tested]
        return max(candidates, key=lambda action: (self._priority(action), action.action_id))

    def observe(self, feedback: dict[str, object]) -> None:
        record = dict(feedback)
        record["step"] = len(self.log) + 1
        record["budget_remaining"] = self.budget - record["step"]
        record["action_id"] = (
            f"{record['route']}__{record['lag_days']}d__{record['model']}"
        )
        self.log.append(record)

    def best_result(self) -> dict[str, object]:
        if not self.log:
            raise RuntimeError("no experiment has been observed")
        return max(self.log, key=lambda row: float(row["utility"]))
