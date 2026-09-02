"""Dynamic science-structured model comparison for GlobalHAB-Agent.

The main 24-candidate Agent search is intentionally unchanged.  This module
implements a compact dual-branch temporal model for the currently selected lag:

* a local MHW-history branch;
* an upstream MHW-history branch;
* a learnable gate driven by the transport/residence/convergence proxy;
* a shared probabilistic HAB-risk head.

All epoch selection occurs inside the outer training era.  Logistic, Random
Forest and STS-Gated TCN are evaluated on exactly the same held-out region and
forward test rows for each random seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .agent import ExperimentAction
from .data import UPSTREAM
from .experiment import _ece, _experiment_frame, _metrics, _split, evaluate_action, evaluate_baselines


COMMON_FEATURES = (
    "nitrate_mmol_m3",
    "phosphate_mmol_m3",
    "silicate_mmol_m3",
    "season_sin",
    "season_cos",
)
DEFAULT_SEEDS = (17, 42, 73, 101, 149)


@dataclass(frozen=True)
class GatedTCNConfig:
    name: str = "STS-Gated-TCN-241"
    channels_1: int = 4
    channels_2: int = 3
    kernel_size: int = 3
    learning_rate: float = 0.008
    l2: float = 8e-4
    max_epochs: int = 35


CONFIG = GatedTCNConfig()


def _parameter_count(config: GatedTCNConfig = CONFIG, branch_features: int = 1 + len(COMMON_FEATURES)) -> int:
    # Two independent temporal branches + transport gate + probabilistic head.
    one_branch = (
        config.channels_1 * (config.kernel_size * branch_features + 1)
        + config.channels_2 * (config.kernel_size * config.channels_1 + 1)
    )
    gate = 2  # weight + bias for scalar transport gate
    science_skip = 5  # transmitted signal, nutrient context, interaction, season sin/cos
    head = config.channels_2 + science_skip + 1
    return int(2 * one_branch + gate + head)


def _raw_lagged_signal(frame: pd.DataFrame, route: str, lag_days: int) -> pd.Series:
    """Lag MHW intensity without applying the transport gate."""
    source = frame[["date", "region", "mhw_intensity_c"]].copy()
    source["date"] = source["date"] + pd.to_timedelta(lag_days, unit="D")
    if route == "downstream":
        parts = []
        for target, upstream in UPSTREAM.items():
            part = source[source["region"].eq(upstream)].copy()
            part["region"] = target
            parts.append(part)
        source = pd.concat(parts, ignore_index=True)
    elif route != "local":
        raise ValueError(f"unsupported route: {route}")
    lookup = frame[["date", "region"]].merge(
        source, on=["date", "region"], how="left"
    )
    return lookup["mhw_intensity_c"].rename(f"{route}_raw_signal")


def _dual_branch_frame(frame: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    work = _experiment_frame(frame, "downstream", lag_days).copy()
    # _experiment_frame resets rows after dropping unavailable lag history.
    raw = frame.copy()
    raw["local_raw_signal"] = _raw_lagged_signal(raw, "local", lag_days)
    raw["upstream_raw_signal"] = _raw_lagged_signal(raw, "downstream", lag_days)
    raw = raw[["date", "region", "local_raw_signal", "upstream_raw_signal"]]
    work = work.merge(raw, on=["date", "region"], how="left")
    proxy = work["circulation_residence_proxy"].to_numpy(dtype=float)
    work["transmitted_signal"] = work["upstream_raw_signal"].to_numpy(dtype=float) * (0.55 + 0.90 * proxy)
    work["local_gated_signal"] = work["local_raw_signal"].to_numpy(dtype=float) * (0.55 + 0.90 * proxy)
    nitrate_z = (work["nitrate_mmol_m3"] - 3.0) / 0.9
    phosphate_z = (work["phosphate_mmol_m3"] - 0.42) / 0.13
    silicate_z = (work["silicate_mmol_m3"] - 4.8) / 1.35
    work["nutrient_context"] = 0.50 * nitrate_z + 0.30 * phosphate_z + 0.20 * silicate_z
    work["sts_interaction"] = work["transmitted_signal"] * work["nutrient_context"]
    return work.dropna(subset=["local_raw_signal", "upstream_raw_signal"]).reset_index(drop=True)


def _outer_masks(work: pd.DataFrame, holdout_region: str, test_fraction: float):
    dates = np.sort(work["date"].unique())
    cut = pd.Timestamp(dates[int(len(dates) * (1.0 - test_fraction))])
    train = work["date"].lt(cut) & work["region"].ne(holdout_region)
    test = work["date"].ge(cut) & work["region"].eq(holdout_region)
    return train.to_numpy(), test.to_numpy(), cut


def _scale_matrix(values: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    fitted = values[fit_mask]
    median = np.nanmedian(fitted, axis=0)
    values = np.where(np.isnan(values), median, values)
    fitted = values[fit_mask]
    mean = fitted.mean(axis=0)
    scale = fitted.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (values - mean) / scale


def _branch_matrices(work: pd.DataFrame, fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common = work.loc[:, COMMON_FEATURES].to_numpy(dtype=float)
    local = np.column_stack([work["local_raw_signal"].to_numpy(dtype=float), common])
    upstream = np.column_stack([work["upstream_raw_signal"].to_numpy(dtype=float), common])
    local = _scale_matrix(local, fit_mask)
    upstream = _scale_matrix(upstream, fit_mask)
    # Gate keeps its interpretable 0-1 scale; center only for optimisation.
    gate = work["circulation_residence_proxy"].to_numpy(dtype=float)
    gate_mean = float(gate[fit_mask].mean())
    gate_std = float(gate[fit_mask].std())
    if gate_std < 1e-8:
        gate_std = 1.0
    gate_scaled = (gate - gate_mean) / gate_std

    nitrate_z = (work["nitrate_mmol_m3"].to_numpy(dtype=float) - 3.0) / 0.9
    phosphate_z = (work["phosphate_mmol_m3"].to_numpy(dtype=float) - 0.42) / 0.13
    silicate_z = (work["silicate_mmol_m3"].to_numpy(dtype=float) - 4.8) / 1.35
    nutrient_context = 0.50 * nitrate_z + 0.30 * phosphate_z + 0.20 * silicate_z
    transmitted = work["upstream_raw_signal"].to_numpy(dtype=float) * (0.55 + 0.90 * gate)
    science_skip = np.column_stack([
        transmitted,
        nutrient_context,
        transmitted * nutrient_context,
        work["season_sin"].to_numpy(dtype=float),
        work["season_cos"].to_numpy(dtype=float),
    ])
    science_skip = _scale_matrix(science_skip, fit_mask)
    return local, upstream, gate_scaled, science_skip


def _causal_sequences(work: pd.DataFrame, scaled: np.ndarray, window: int) -> np.ndarray:
    sequences = np.zeros((len(work), window, scaled.shape[1]), dtype=np.float64)
    for _, indexes in work.groupby("region", sort=False).groups.items():
        ordered = work.loc[indexes].sort_values("date").index.to_numpy(dtype=int)
        for position, row_index in enumerate(ordered):
            start = max(0, position - window + 1)
            history = ordered[start: position + 1]
            sequences[row_index, -len(history):, :] = scaled[history]
    return sequences


def _conv_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, dilation: int) -> np.ndarray:
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


def _conv_backward(grad_output: np.ndarray, x: np.ndarray, weight: np.ndarray, dilation: int):
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
        grad_x[:, : steps - shift, :] += np.einsum("bth,hc->btc", grad, weight[:, kernel_index, :])
    return grad_x, grad_weight, grad_output.sum(axis=(0, 1))


def _init_branch(rng: np.random.Generator, prefix: str, config: GatedTCNConfig, n_features: int):
    fan1 = config.kernel_size * n_features
    fan2 = config.kernel_size * config.channels_1
    return {
        f"{prefix}_w1": rng.normal(0, np.sqrt(2.0 / fan1), (config.channels_1, config.kernel_size, n_features)),
        f"{prefix}_b1": np.zeros(config.channels_1),
        f"{prefix}_w2": rng.normal(0, np.sqrt(2.0 / fan2), (config.channels_2, config.kernel_size, config.channels_1)),
        f"{prefix}_b2": np.zeros(config.channels_2),
    }


def _initialise(seed: int, n_features: int, config: GatedTCNConfig = CONFIG):
    rng = np.random.default_rng(seed)
    parameters = {}
    parameters.update(_init_branch(rng, "local", config, n_features))
    parameters.update(_init_branch(rng, "up", config, n_features))
    parameters.update({
        "gate_w": np.array([0.8]),
        "gate_b": np.array([0.0]),
        "wo": rng.normal(0, np.sqrt(1.0 / (config.channels_2 + 5)), config.channels_2 + 5),
        "bo": np.zeros(1),
    })
    return parameters


def _branch_forward(x: np.ndarray, p: dict[str, np.ndarray], prefix: str):
    z1 = _conv_forward(x, p[f"{prefix}_w1"], p[f"{prefix}_b1"], dilation=1)
    a1 = np.maximum(z1, 0.0)
    z2 = _conv_forward(a1, p[f"{prefix}_w2"], p[f"{prefix}_b2"], dilation=2)
    a2 = np.maximum(z2, 0.0)
    rep = a2[:, -1, :]
    return rep, (z1, a1, z2, a2)


def _forward(local_seq: np.ndarray, upstream_seq: np.ndarray, gate_input: np.ndarray, science_skip: np.ndarray, p: dict[str, np.ndarray]):
    local_rep, local_cache = _branch_forward(local_seq, p, "local")
    up_rep, up_cache = _branch_forward(upstream_seq, p, "up")
    gate_logit = gate_input * p["gate_w"][0] + p["gate_b"][0]
    gate = 1.0 / (1.0 + np.exp(-np.clip(gate_logit, -20.0, 20.0)))
    fused = (1.0 - gate[:, None]) * local_rep + gate[:, None] * up_rep
    head_input = np.column_stack([fused, science_skip])
    logits = head_input @ p["wo"] + p["bo"][0]
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return probability, (local_rep, up_rep, gate, fused, head_input, local_cache, up_cache)


def _backward_branch(grad_rep: np.ndarray, seq: np.ndarray, p: dict[str, np.ndarray], prefix: str, cache):
    z1, a1, z2, a2 = cache
    grad_a2 = np.zeros_like(a2)
    grad_a2[:, -1, :] = grad_rep
    grad_z2 = grad_a2 * (z2 > 0.0)
    grad_a1, grad_w2, grad_b2 = _conv_backward(grad_z2, a1, p[f"{prefix}_w2"], dilation=2)
    grad_z1 = grad_a1 * (z1 > 0.0)
    _, grad_w1, grad_b1 = _conv_backward(grad_z1, seq, p[f"{prefix}_w1"], dilation=1)
    return {
        f"{prefix}_w1": grad_w1,
        f"{prefix}_b1": grad_b1,
        f"{prefix}_w2": grad_w2,
        f"{prefix}_b2": grad_b2,
    }


def _loss_and_gradients(local_seq, upstream_seq, gate_input, science_skip, labels, p, config: GatedTCNConfig = CONFIG):
    probability, cache = _forward(local_seq, upstream_seq, gate_input, science_skip, p)
    local_rep, up_rep, gate, fused, head_input, local_cache, up_cache = cache
    positive_weight = float(np.clip((len(labels) - labels.sum()) / max(1.0, labels.sum()), 1.0, 4.0))
    sample_weight = np.where(labels > 0.5, positive_weight, 1.0)
    sample_weight = sample_weight / sample_weight.sum()
    eps = 1e-8
    loss = -np.sum(sample_weight * (
        labels * np.log(probability + eps) + (1.0 - labels) * np.log(1.0 - probability + eps)
    ))
    weight_names = [name for name in p if name.endswith("w1") or name.endswith("w2") or name in ("wo", "gate_w")]
    loss += 0.5 * config.l2 * sum(float(np.square(p[name]).sum()) for name in weight_names)

    grad_logits = (probability - labels) * sample_weight
    gradients = {
        "wo": head_input.T @ grad_logits + config.l2 * p["wo"],
        "bo": np.array([grad_logits.sum()]),
    }
    grad_fused = grad_logits[:, None] * p["wo"][: fused.shape[1]][None, :]
    grad_local = grad_fused * (1.0 - gate[:, None])
    grad_up = grad_fused * gate[:, None]
    grad_gate = np.sum(grad_fused * (up_rep - local_rep), axis=1)
    grad_gate_logit = grad_gate * gate * (1.0 - gate)
    gradients["gate_w"] = np.array([np.sum(grad_gate_logit * gate_input)]) + config.l2 * p["gate_w"]
    gradients["gate_b"] = np.array([grad_gate_logit.sum()])

    gradients.update(_backward_branch(grad_local, local_seq, p, "local", local_cache))
    gradients.update(_backward_branch(grad_up, upstream_seq, p, "up", up_cache))
    for name in list(gradients):
        if name.endswith("w1") or name.endswith("w2"):
            gradients[name] = gradients[name] + config.l2 * p[name]
    return float(loss), gradients


def _fit(local_seq, upstream_seq, gate_input, science_skip, labels, seed: int, epochs: int, config: GatedTCNConfig = CONFIG):
    p = _initialise(seed, local_seq.shape[2], config)
    first = {name: np.zeros_like(value) for name, value in p.items()}
    second = {name: np.zeros_like(value) for name, value in p.items()}
    losses = []
    for epoch in range(1, epochs + 1):
        loss, gradients = _loss_and_gradients(local_seq, upstream_seq, gate_input, science_skip, labels, p, config)
        losses.append(loss)
        for name in p:
            grad = np.clip(gradients[name], -5.0, 5.0)
            first[name] = 0.9 * first[name] + 0.1 * grad
            second[name] = 0.999 * second[name] + 0.001 * np.square(grad)
            first_hat = first[name] / (1.0 - 0.9 ** epoch)
            second_hat = second[name] / (1.0 - 0.999 ** epoch)
            p[name] -= config.learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
    return p, losses


def _prepare_sequences(work: pd.DataFrame, fit_mask: np.ndarray, window: int):
    local, upstream, gate, science_skip = _branch_matrices(work, fit_mask)
    return (
        _causal_sequences(work, local, window),
        _causal_sequences(work, upstream, window),
        gate,
        science_skip,
    )


def _select_epochs(work: pd.DataFrame, outer_train: np.ndarray, window: int, config: GatedTCNConfig = CONFIG):
    train_dates = np.sort(work.loc[outer_train, "date"].unique())
    inner_cut = pd.Timestamp(train_dates[int(len(train_dates) * 0.80)])
    inner_train = outer_train & work["date"].lt(inner_cut).to_numpy()
    inner_valid = outer_train & work["date"].ge(inner_cut).to_numpy()
    local_seq, upstream_seq, gate, science_skip = _prepare_sequences(work, inner_train, window)
    labels = work["hab_event"].to_numpy(dtype=float)
    rows = []
    for epochs in (20, 35):
        p, _ = _fit(
            local_seq[inner_train], upstream_seq[inner_train], gate[inner_train], science_skip[inner_train],
            labels[inner_train], seed=2026, epochs=epochs, config=config,
        )
        probability, _ = _forward(local_seq[inner_valid], upstream_seq[inner_valid], gate[inner_valid], science_skip[inner_valid], p)
        rows.append({
            "epochs": epochs,
            "inner_validation_ap": float(average_precision_score(labels[inner_valid], probability)),
            "inner_validation_brier": float(brier_score_loss(labels[inner_valid], probability)),
            "inner_validation_ece": _ece(labels[inner_valid], probability),
            "inner_cut_date": inner_cut.date().isoformat(),
        })
    trace = pd.DataFrame(rows).sort_values(
        ["inner_validation_ap", "inner_validation_brier", "epochs"],
        ascending=[False, True, True], ignore_index=True,
    )
    return int(trace.iloc[0]["epochs"]), trace


INTERACTION_FEATURES = [
    "transmitted_signal", "nutrient_context", "sts_interaction",
    "local_gated_signal", "season_sin", "season_cos",
]


def _select_interaction_c(work: pd.DataFrame, outer_train: np.ndarray) -> tuple[float, pd.DataFrame]:
    train_dates = np.sort(work.loc[outer_train, "date"].unique())
    inner_cut = pd.Timestamp(train_dates[int(len(train_dates) * 0.80)])
    inner_train = outer_train & work["date"].lt(inner_cut).to_numpy()
    inner_valid = outer_train & work["date"].ge(inner_cut).to_numpy()
    rows = []
    for c_value in (0.1, 0.3, 1.0, 3.0):
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=600, C=c_value)),
        ])
        model.fit(work.loc[inner_train, INTERACTION_FEATURES], work.loc[inner_train, "hab_event"])
        probability = model.predict_proba(work.loc[inner_valid, INTERACTION_FEATURES])[:, 1]
        rows.append({
            "C": c_value,
            "inner_validation_ap": float(average_precision_score(work.loc[inner_valid, "hab_event"], probability)),
            "inner_validation_brier": float(brier_score_loss(work.loc[inner_valid, "hab_event"], probability)),
            "inner_validation_ece": _ece(work.loc[inner_valid, "hab_event"].to_numpy(), probability),
            "inner_cut_date": inner_cut.date().isoformat(),
        })
    trace = pd.DataFrame(rows).sort_values(
        ["inner_validation_ap", "inner_validation_brier", "C"], ascending=[False, True, True], ignore_index=True
    )
    return float(trace.iloc[0]["C"]), trace


def _fit_interaction_model(work: pd.DataFrame, train_mask: np.ndarray, c_value: float):
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=600, C=c_value)),
    ])
    model.fit(work.loc[train_mask, INTERACTION_FEATURES], work.loc[train_mask, "hab_event"])
    return model


def _aggregate(seed_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, subset in seed_results.groupby("model", sort=False):
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


def run_dynamic_model_comparison(
    frame: pd.DataFrame,
    lag_days: int,
    holdout_region: str,
    test_fraction: float,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    window: int = 14,
) -> dict[str, object]:
    """Run a leakage-safe, current-configuration model comparison."""
    work = _dual_branch_frame(frame, lag_days)
    # Confirm exact outer rows match the standard downstream candidate split.
    standard_work = _experiment_frame(frame, "downstream", lag_days)
    standard_train, standard_test, standard_cut = _split(standard_work, holdout_region, test_fraction)
    outer_train, outer_test, cut = _outer_masks(work, holdout_region, test_fraction)
    if cut != standard_cut:
        raise RuntimeError("gated model cut date is not aligned with Agent split")
    gated_test_keys = set(map(tuple, work.loc[outer_test, ["date", "region"]].to_numpy()))
    standard_test_keys = set(map(tuple, standard_test[["date", "region"]].to_numpy()))
    if gated_test_keys != standard_test_keys:
        raise RuntimeError("gated model test rows are not identical to downstream Agent test rows")

    selected_epochs, tuning_trace = _select_epochs(work, outer_train, window)
    selected_c, interaction_tuning_trace = _select_interaction_c(work, outer_train)
    local_seq, upstream_seq, gate, science_skip = _prepare_sequences(work, outer_train, window)
    labels = work["hab_event"].to_numpy(dtype=float)
    strongest_baseline = float(evaluate_baselines(frame, holdout_region, test_fraction, seeds[0])["pr_auc"].max())
    rows = []

    for seed in seeds:
        for model_name, display_name, complexity in (
            ("logistic", "Logistic", "7个预测系数"),
            ("random_forest", "Random Forest", "120棵树"),
        ):
            start = perf_counter()
            feedback, _ = evaluate_action(
                frame, ExperimentAction("downstream", lag_days, model_name),
                holdout_region, test_fraction, seed, strongest_baseline,
            )
            rows.append({
                "model": display_name,
                "seed": seed,
                "ap": float(feedback["pr_auc"]),
                "brier": float(feedback["brier"]),
                "ece": float(feedback["ece"]),
                "fit_seconds": perf_counter() - start,
                "complexity": complexity,
                "test_rows": int(feedback["test_rows"]),
                "test_events": int(feedback["test_events"]),
            })

        start = perf_counter()
        interaction_model = _fit_interaction_model(work, outer_train, selected_c)
        interaction_probability = interaction_model.predict_proba(work.loc[outer_test, INTERACTION_FEATURES])[:, 1]
        interaction_metrics, _ = _metrics(
            labels[outer_test], interaction_probability, float(labels[outer_train].mean())
        )
        rows.append({
            "model": "STS-Interaction GLM",
            "seed": seed,
            "ap": interaction_metrics["pr_auc"],
            "brier": interaction_metrics["brier"],
            "ece": interaction_metrics["ece"],
            "fit_seconds": perf_counter() - start,
            "complexity": "6个科学结构特征 + 显式交互",
            "test_rows": int(outer_test.sum()),
            "test_events": int(labels[outer_test].sum()),
        })

        start = perf_counter()
        p, losses = _fit(
            local_seq[outer_train], upstream_seq[outer_train], gate[outer_train], science_skip[outer_train], labels[outer_train],
            seed=seed, epochs=selected_epochs,
        )
        fit_seconds = perf_counter() - start
        probability, cache = _forward(local_seq[outer_test], upstream_seq[outer_test], gate[outer_test], science_skip[outer_test], p)
        metrics, _ = _metrics(labels[outer_test], probability, float(labels[outer_train].mean()))
        gate_values = cache[2]
        rows.append({
            "model": "STS-Gated TCN",
            "seed": seed,
            "ap": metrics["pr_auc"],
            "brier": metrics["brier"],
            "ece": metrics["ece"],
            "fit_seconds": fit_seconds,
            "complexity": f"{_parameter_count()}个可训练参数",
            "test_rows": int(outer_test.sum()),
            "test_events": int(labels[outer_test].sum()),
            "epochs": selected_epochs,
            "final_training_loss": losses[-1],
            "gate_mean": float(gate_values.mean()),
            "gate_min": float(gate_values.min()),
            "gate_max": float(gate_values.max()),
        })

    seed_results = pd.DataFrame(rows)
    summary = _aggregate(seed_results)
    gated = summary[summary["model"].eq("STS-Gated TCN")].iloc[0]
    interaction = summary[summary["model"].eq("STS-Interaction GLM")].iloc[0]
    classical = summary[summary["model"].isin(["Logistic", "Random Forest"])]
    best_classical_ap = float(classical["ap_median"].max())
    improvement = float(gated["ap_median"] - best_classical_ap)
    interaction_improvement = float(interaction["ap_median"] - best_classical_ap)
    status = (
        "获得跨种子AP增益"
        if improvement > 0
        else "未获得稳定AP增益，保留为结构改进的负结果"
    )
    return {
        "summary": summary,
        "seed_results": seed_results,
        "tuning_trace": tuning_trace,
        "interaction_tuning_trace": interaction_tuning_trace,
        "card": {
            "model": CONFIG.name,
            "parameter_count": _parameter_count(),
            "window_days": window,
            "lag_days": int(lag_days),
            "epochs_selected_inside_training_window": selected_epochs,
            "random_seeds": list(seeds),
            "holdout_region": holdout_region,
            "test_fraction": float(test_fraction),
            "cut_date": cut.date().isoformat(),
            "test_rows": int(outer_test.sum()),
            "test_events": int(labels[outer_test].sum()),
            "ap_gain_vs_best_classical_median": improvement,
            "interaction_glm_C_selected_inside_training": selected_c,
            "interaction_glm_ap_gain_vs_best_classical_median": interaction_improvement,
            "result": status,
            "boundary": (
                "This current-run synthetic comparison tests whether a science-structured "
                "dual-branch temporal model adds value on the same blocked holdout. It does not "
                "change the Agent search or transfer synthetic performance to real cases."
            ),
        },
    }
