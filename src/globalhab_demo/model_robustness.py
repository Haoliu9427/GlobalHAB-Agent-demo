"""Independent model-complexity robustness check for the selected hypothesis.

The 24-candidate / 8-step Agent search remains unchanged.  This module asks a
separate question: does a small causal temporal convolution network support the
same selected route and lag on exactly the same outer blocked test rows?

The implementation uses NumPy only so that the public Streamlit deployment does
not acquire a heavyweight deep-learning runtime.  Hyperparameters are selected
inside the outer training era; the held-out region and forward test window are
never used for model selection or early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .agent import ExperimentAction
from .experiment import (
    _ece,
    _experiment_frame,
    _metrics,
    _split,
    evaluate_action,
    evaluate_baselines,
)


FEATURES = (
    "candidate_signal",
    "nitrate_mmol_m3",
    "phosphate_mmol_m3",
    "silicate_mmol_m3",
    "season_sin",
    "season_cos",
)
DEFAULT_SEEDS = (17, 42, 73, 101, 149)


@dataclass(frozen=True)
class TCNConfig:
    name: str
    channels_1: int
    channels_2: int
    kernel_size: int
    learning_rate: float
    l2: float
    max_epochs: int


TCN_CANDIDATES = (
    TCNConfig("TCN-119", 4, 3, 3, 0.010, 1e-3, 150),
    TCNConfig("TCN-195", 6, 4, 3, 0.006, 5e-4, 180),
)


def _parameter_count(config: TCNConfig, n_features: int = len(FEATURES)) -> int:
    first = config.channels_1 * (config.kernel_size * n_features + 1)
    second = config.channels_2 * (config.kernel_size * config.channels_1 + 1)
    head = config.channels_2 + 1
    return int(first + second + head)


def _scale_frame(
    work: pd.DataFrame,
    fit_mask: np.ndarray,
) -> np.ndarray:
    values = work.loc[:, FEATURES].to_numpy(dtype=float)
    fitted = values[fit_mask]
    median = np.nanmedian(fitted, axis=0)
    values = np.where(np.isnan(values), median, values)
    fitted = values[fit_mask]
    mean = fitted.mean(axis=0)
    scale = fitted.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (values - mean) / scale


def _causal_sequences(
    work: pd.DataFrame,
    scaled: np.ndarray,
    window: int,
) -> np.ndarray:
    """Return left-padded causal sequences aligned one-to-one with work rows."""
    sequences = np.zeros((len(work), window, scaled.shape[1]), dtype=np.float64)
    for _, indexes in work.groupby("region", sort=False).groups.items():
        ordered = work.loc[indexes].sort_values("date").index.to_numpy(dtype=int)
        for position, row_index in enumerate(ordered):
            start = max(0, position - window + 1)
            history = ordered[start: position + 1]
            sequences[row_index, -len(history):, :] = scaled[history]
    return sequences


def _conv_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    dilation: int,
) -> np.ndarray:
    batch, steps, _ = x.shape
    output = np.broadcast_to(bias, (batch, steps, len(bias))).copy()
    for kernel_index in range(weight.shape[1]):
        shift = dilation * kernel_index
        if shift >= steps:
            continue
        output[:, shift:, :] += np.einsum(
            "btc,hc->bth", x[:, : steps - shift, :], weight[:, kernel_index, :]
        )
    return output


def _conv_backward(
    grad_output: np.ndarray,
    x: np.ndarray,
    weight: np.ndarray,
    dilation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, steps, _ = x.shape
    grad_x = np.zeros_like(x)
    grad_weight = np.zeros_like(weight)
    for kernel_index in range(weight.shape[1]):
        shift = dilation * kernel_index
        if shift >= steps:
            continue
        grad = grad_output[:, shift:, :]
        source = x[:, : steps - shift, :]
        grad_weight[:, kernel_index, :] = np.einsum("bth,btc->hc", grad, source)
        grad_x[:, : steps - shift, :] += np.einsum(
            "bth,hc->btc", grad, weight[:, kernel_index, :]
        )
    return grad_x, grad_weight, grad_output.sum(axis=(0, 1))


def _initialise(config: TCNConfig, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    fan_1 = config.kernel_size * len(FEATURES)
    fan_2 = config.kernel_size * config.channels_1
    return {
        "w1": rng.normal(0, np.sqrt(2.0 / fan_1),
                         (config.channels_1, config.kernel_size, len(FEATURES))),
        "b1": np.zeros(config.channels_1),
        "w2": rng.normal(0, np.sqrt(2.0 / fan_2),
                         (config.channels_2, config.kernel_size, config.channels_1)),
        "b2": np.zeros(config.channels_2),
        "wo": rng.normal(0, np.sqrt(1.0 / config.channels_2), config.channels_2),
        "bo": np.zeros(1),
    }


def _forward(
    sequences: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    z1 = _conv_forward(sequences, parameters["w1"], parameters["b1"], dilation=1)
    a1 = np.maximum(z1, 0.0)
    z2 = _conv_forward(a1, parameters["w2"], parameters["b2"], dilation=2)
    a2 = np.maximum(z2, 0.0)
    representation = a2[:, -1, :]
    logits = representation @ parameters["wo"] + parameters["bo"][0]
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return probability, (z1, a1, z2, a2, representation)


def _loss_and_gradients(
    sequences: np.ndarray,
    labels: np.ndarray,
    parameters: dict[str, np.ndarray],
    l2: float,
    positive_weight: float,
) -> tuple[float, dict[str, np.ndarray]]:
    probability, cache = _forward(sequences, parameters)
    z1, a1, z2, a2, representation = cache
    sample_weight = np.where(labels > 0.5, positive_weight, 1.0)
    sample_weight = sample_weight / sample_weight.sum()
    eps = 1e-8
    loss = -np.sum(sample_weight * (
        labels * np.log(probability + eps)
        + (1.0 - labels) * np.log(1.0 - probability + eps)
    ))
    loss += 0.5 * l2 * sum(
        float(np.square(parameters[name]).sum()) for name in ("w1", "w2", "wo")
    )

    grad_logits = (probability - labels) * sample_weight
    gradients: dict[str, np.ndarray] = {
        "wo": representation.T @ grad_logits + l2 * parameters["wo"],
        "bo": np.array([grad_logits.sum()]),
    }
    grad_representation = grad_logits[:, None] * parameters["wo"][None, :]
    grad_a2 = np.zeros_like(a2)
    grad_a2[:, -1, :] = grad_representation
    grad_z2 = grad_a2 * (z2 > 0.0)
    grad_a1, grad_w2, grad_b2 = _conv_backward(
        grad_z2, a1, parameters["w2"], dilation=2
    )
    grad_z1 = grad_a1 * (z1 > 0.0)
    _, grad_w1, grad_b1 = _conv_backward(
        grad_z1, sequences, parameters["w1"], dilation=1
    )
    gradients.update({
        "w2": grad_w2 + l2 * parameters["w2"],
        "b2": grad_b2,
        "w1": grad_w1 + l2 * parameters["w1"],
        "b1": grad_b1,
    })
    return float(loss), gradients


def _fit_tcn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TCNConfig,
    seed: int,
    epochs: int,
) -> tuple[dict[str, np.ndarray], list[float]]:
    parameters = _initialise(config, seed)
    first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    positive_weight = float(np.clip((len(y_train) - y_train.sum()) /
                                    max(1.0, y_train.sum()), 1.0, 4.0))
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        loss, gradients = _loss_and_gradients(
            x_train, y_train, parameters, config.l2, positive_weight
        )
        losses.append(loss)
        for name in parameters:
            gradient = np.clip(gradients[name], -5.0, 5.0)
            first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradient
            second_moment[name] = 0.999 * second_moment[name] + 0.001 * np.square(gradient)
            corrected_first = first_moment[name] / (1.0 - 0.9 ** epoch)
            corrected_second = second_moment[name] / (1.0 - 0.999 ** epoch)
            parameters[name] -= config.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )
    return parameters, losses


def _outer_masks(
    work: pd.DataFrame,
    holdout_region: str,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    dates = np.sort(work["date"].unique())
    cut = pd.Timestamp(dates[int(len(dates) * (1.0 - test_fraction))])
    train = work["date"].lt(cut) & work["region"].ne(holdout_region)
    test = work["date"].ge(cut) & work["region"].eq(holdout_region)
    return train.to_numpy(), test.to_numpy(), cut


def _select_configuration(
    work: pd.DataFrame,
    outer_train_mask: np.ndarray,
    window: int,
) -> tuple[TCNConfig, int, pd.DataFrame, float]:
    train_dates = np.sort(work.loc[outer_train_mask, "date"].unique())
    inner_cut = pd.Timestamp(train_dates[int(len(train_dates) * 0.80)])
    inner_train = outer_train_mask & work["date"].lt(inner_cut).to_numpy()
    inner_validation = outer_train_mask & work["date"].ge(inner_cut).to_numpy()
    scaled = _scale_frame(work, inner_train)
    sequences = _causal_sequences(work, scaled, window)
    labels = work["hab_event"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    selection_start = perf_counter()
    checkpoints = (60, 90, 120, 150, 180)
    for config_index, config in enumerate(TCN_CANDIDATES):
        parameters = _initialise(config, seed=2026 + config_index)
        # Train once to each checkpoint by restarting from the same seed.  The
        # small network makes this explicit audit trail inexpensive.
        for epochs in checkpoints:
            if epochs > config.max_epochs:
                continue
            parameters, _ = _fit_tcn(
                sequences[inner_train], labels[inner_train], config,
                seed=2026 + config_index, epochs=epochs,
            )
            probability, _ = _forward(sequences[inner_validation], parameters)
            rows.append({
                "configuration": config.name,
                "parameter_count": _parameter_count(config),
                "epochs": epochs,
                "inner_validation_ap": float(average_precision_score(
                    labels[inner_validation], probability
                )),
                "inner_validation_brier": float(brier_score_loss(
                    labels[inner_validation], probability
                )),
                "inner_validation_ece": _ece(labels[inner_validation], probability),
                "inner_cut_date": inner_cut,
            })
    trace = pd.DataFrame(rows).sort_values(
        ["inner_validation_ap", "inner_validation_brier", "parameter_count"],
        ascending=[False, True, True], ignore_index=True,
    )
    selected = trace.iloc[0]
    config = next(item for item in TCN_CANDIDATES
                  if item.name == selected["configuration"])
    return config, int(selected["epochs"]), trace, perf_counter() - selection_start


def _aggregate(seed_results: pd.DataFrame) -> pd.DataFrame:
    grouped = seed_results.groupby("model", sort=False)
    rows = []
    for model, subset in grouped:
        rows.append({
            "model": model,
            "seeds": int(len(subset)),
            "ap_median": float(subset["ap"].median()),
            "ap_mean": float(subset["ap"].mean()),
            "ap_sd": float(subset["ap"].std(ddof=0)),
            "brier_mean": float(subset["brier"].mean()),
            "ece_mean": float(subset["ece"].mean()),
            "fit_seconds_mean": float(subset["fit_seconds"].mean()),
            "complexity": str(subset["complexity"].iloc[0]),
            "test_rows": int(subset["test_rows"].iloc[0]),
            "test_events": int(subset["test_events"].iloc[0]),
        })
    return pd.DataFrame(rows)


def run_model_complexity_check(
    frame: pd.DataFrame,
    route: str,
    lag_days: int,
    holdout_region: str,
    test_fraction: float,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    window: int = 14,
) -> dict[str, object]:
    """Compare Logistic, RF and a tiny TCN without changing Agent selection."""
    work = _experiment_frame(frame, route, lag_days)
    train_frame, test_frame, cut_date = _split(work, holdout_region, test_fraction)
    outer_train, outer_test, mask_cut = _outer_masks(work, holdout_region, test_fraction)
    if cut_date != mask_cut or len(test_frame) != int(outer_test.sum()):
        raise RuntimeError("robustness check is not aligned with the Agent outer split")

    config, selected_epochs, tuning_trace, selection_seconds = _select_configuration(
        work, outer_train, window
    )
    scaled = _scale_frame(work, outer_train)
    sequences = _causal_sequences(work, scaled, window)
    labels = work["hab_event"].to_numpy(dtype=float)
    strongest_baseline = float(evaluate_baselines(
        frame, holdout_region, test_fraction, seeds[0]
    )["pr_auc"].max())
    rows: list[dict[str, object]] = []

    for seed in seeds:
        for model_name, display_name, parameter_count in (
            ("logistic", "Logistic", 7),
            ("random_forest", "Random Forest", None),
        ):
            start = perf_counter()
            feedback, _ = evaluate_action(
                frame, ExperimentAction(route, lag_days, model_name),
                holdout_region, test_fraction, seed, strongest_baseline,
            )
            elapsed = perf_counter() - start
            rows.append({
                "model": display_name,
                "seed": seed,
                "ap": float(feedback["pr_auc"]),
                "brier": float(feedback["brier"]),
                "ece": float(feedback["ece"]),
                "fit_seconds": elapsed,
                "parameter_count": parameter_count,
                "complexity": (
                    "7个预测系数" if display_name == "Logistic" else "120棵树"
                ),
                "test_rows": int(feedback["test_rows"]),
                "test_events": int(feedback["test_events"]),
            })

        start = perf_counter()
        parameters, losses = _fit_tcn(
            sequences[outer_train], labels[outer_train], config,
            seed=seed, epochs=selected_epochs,
        )
        fit_seconds = perf_counter() - start
        inference_start = perf_counter()
        probability, _ = _forward(sequences[outer_test], parameters)
        inference_seconds = perf_counter() - inference_start
        metrics, _ = _metrics(
            labels[outer_test], probability, float(labels[outer_train].mean())
        )
        rows.append({
            "model": "轻量TCN",
            "seed": seed,
            "ap": metrics["pr_auc"],
            "brier": metrics["brier"],
            "ece": metrics["ece"],
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "parameter_count": _parameter_count(config),
            "complexity": f"{_parameter_count(config)}个可训练参数",
            "epochs": selected_epochs,
            "final_training_loss": losses[-1],
            "test_rows": int(outer_test.sum()),
            "test_events": int(labels[outer_test].sum()),
        })

    seed_results = pd.DataFrame(rows)
    summary = _aggregate(seed_results)
    tcn = seed_results[seed_results["model"].eq("轻量TCN")].set_index("seed")
    classical = seed_results[seed_results["model"].ne("轻量TCN")]
    best_classical = classical.groupby("seed")["ap"].max()
    wins = int((tcn["ap"] > best_classical).sum())
    tcn_summary = summary[summary["model"].eq("轻量TCN")].iloc[0]
    classical_summary = summary[summary["model"].ne("轻量TCN")]
    best_classical_ap = float(classical_summary["ap_median"].max())
    best_classical_brier = float(classical_summary["brier_mean"].min())
    best_classical_ece = float(classical_summary["ece_mean"].min())
    stable_improvement = bool(
        wins >= max(1, len(seeds) - 1)
        and float(tcn_summary["ap_median"]) >= best_classical_ap + 0.01
        and float(tcn_summary["brier_mean"]) <= best_classical_brier + 0.01
        and float(tcn_summary["ece_mean"]) <= best_classical_ece + 0.02
    )
    status = (
        "模型容量提高后获得稳定增益"
        if stable_improvement
        else "未观察到跨随机种子的稳定增益，保留为负结果"
    )
    card = {
        "analysis_role": "independent_model_complexity_robustness_check",
        "main_agent_search_unchanged": True,
        "main_agent_candidate_count": 24,
        "main_agent_budget": 8,
        "selected_hypothesis": {"route": route, "lag_days": int(lag_days)},
        "outer_validation": {
            "holdout_region": holdout_region,
            "cut_date": str(cut_date.date()),
            "test_rows": int(outer_test.sum()),
            "test_events": int(labels[outer_test].sum()),
        },
        "tcn": {
            "configuration": config.name,
            "parameter_count": _parameter_count(config),
            "window_days": window,
            "epochs_selected_inside_training_window": selected_epochs,
            "random_seeds": list(seeds),
            "selection_seconds": selection_seconds,
        },
        "tcn_wins_over_best_classical_seed": wins,
        "stable_improvement": stable_improvement,
        "result": status,
        "boundary": (
            "This synthetic-data check tests model-capacity sensitivity only. "
            "It does not enlarge the Agent search, validate an operational HAB forecast, "
            "or transfer synthetic performance to the South Australia or Norway cases."
        ),
    }
    return {
        "summary": summary,
        "seed_results": seed_results,
        "tuning_trace": tuning_trace,
        "card": card,
    }


def evaluate_current_lightweight_tcn(
    frame: pd.DataFrame,
    route: str,
    lag_days: int,
    holdout_region: str,
    test_fraction: float,
    seed: int = 42,
    window: int = 14,
    epochs: int = 90,
) -> dict[str, object]:
    """Fast current-setting TCN benchmark using a pre-registered tiny architecture.

    The architecture and epoch budget are fixed before seeing the current outer
    test labels.  This is intentionally lighter than the five-seed registered
    robustness audit and is meant only for the broad live benchmark table.
    """
    work = _experiment_frame(frame, route, lag_days)
    train_frame, test_frame, cut = _split(work, holdout_region, test_fraction)
    outer_train, outer_test, mask_cut = _outer_masks(work, holdout_region, test_fraction)
    if cut != mask_cut or len(test_frame) != int(outer_test.sum()):
        raise RuntimeError("current TCN rows are not aligned with the Agent split")
    labels = work["hab_event"].to_numpy(dtype=float)
    config = TCN_CANDIDATES[0]
    scaled = _scale_frame(work, outer_train)
    sequences = _causal_sequences(work, scaled, window)
    start = perf_counter()
    parameters, losses = _fit_tcn(
        sequences[outer_train], labels[outer_train], config,
        seed=seed, epochs=min(epochs, config.max_epochs),
    )
    probability, _ = _forward(sequences[outer_test], parameters)
    metrics, _ = _metrics(labels[outer_test], probability, float(labels[outer_train].mean()))
    return {
        "model": "Lightweight TCN",
        "pr_auc": float(metrics["pr_auc"]),
        "brier_skill": float(metrics["brier_skill"]),
        "ece": float(metrics["ece"]),
        "recall_at_top20": float(metrics["recall_at_top20"]),
        "precision_at_top20": float(metrics["precision_at_top20"]),
        "fit_seconds": float(perf_counter() - start),
        "complexity": f"{_parameter_count(config)}个可训练参数",
        "test_rows": int(outer_test.sum()),
        "test_events": int(labels[outer_test].sum()),
        "cut_date": cut.date().isoformat(),
        "epochs": int(min(epochs, config.max_epochs)),
        "final_training_loss": float(losses[-1]),
        "selection": "fixed registered TCN-119 architecture and 90-epoch budget; no outer-test tuning",
    }
