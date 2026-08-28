"""Run the complete GlobalHAB-Agent GOAI semifinal workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo import (  # noqa: E402
    build_sa_replay,
    load_sa_real_case,
    project_real_aquaculture_priority,
    real_data_router,
    run_exploration,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/demo.json")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / args.output
    data_dir = ROOT / "data"
    output.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    result = run_exploration(
        days=config["days"],
        seed=config["seed"],
        budget=config["budget"],
        holdout_region=config["holdout_region"],
        test_fraction=config["test_fraction"],
        routes=tuple(config["routes"]),
        lags=tuple(config["lags"]),
        models=tuple(config["models"]),
    )
    frame = result["frame"]
    baselines = result["baselines"]
    log = result["log"]
    predictions = result["predictions"]
    controls = result["controls"]
    card = result["card"]
    best = result["best"]
    anomaly_daily = result["anomaly_daily"]
    anomaly_events = result["anomaly_events"]
    router_trace = result["router_trace"]
    te_cte_network = result["te_cte_network"]
    te_cte_lag_summary = result["te_cte_lag_summary"]
    spatial_effects = result["spatial_effects"]
    spatial_weights = result["spatial_weights"]
    real_observations, real_provenance = load_sa_real_case(data_dir)
    real_replay = build_sa_replay(
        real_observations,
        real_observations["sample_date"].min(),
        real_observations["sample_date"].max(),
    )
    real_router = real_data_router(real_replay["observations"], has_daily_environment=False)
    real_aquaculture = project_real_aquaculture_priority(
        real_replay["sites"], "贝类（牡蛎/贻贝）", 0.80
    )

    data_path = data_dir / "demo_hab.csv"
    frame.to_csv(data_path, index=False)
    baselines.to_csv(output / "baseline_results.csv", index=False)
    log.to_csv(output / "agent_log.csv", index=False)
    predictions.to_csv(output / "risk_predictions.csv", index=False)
    controls.to_csv(output / "negative_controls.csv", index=False)
    anomaly_daily.to_csv(output / "multiscale_anomaly_daily.csv", index=False)
    anomaly_events.to_csv(output / "multiscale_event_catalog.csv", index=False)
    router_trace.to_csv(output / "adaptive_router_trace.csv", index=False)
    te_cte_network.to_csv(output / "te_cte_network.csv", index=False)
    te_cte_lag_summary.to_csv(output / "te_cte_lag_summary.csv", index=False)
    spatial_effects.to_csv(output / "spatial_durbin_effects.csv", index=False)
    spatial_weights.to_csv(output / "spatial_weight_matrix.csv", index=False)
    real_replay["timeline"].to_csv(output / "sa_real_replay_timeline.csv", index=False)
    real_replay["sites"].to_csv(output / "sa_real_site_summary.csv", index=False)
    real_replay["species"].to_csv(output / "sa_real_species_summary.csv", index=False)
    real_router.to_csv(output / "sa_real_router_trace.csv", index=False)
    real_aquaculture.to_csv(output / "sa_real_aquaculture_priority.csv", index=False)
    (output / "sa_real_replay_card.json").write_text(
        json.dumps(real_replay["card"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "method_diagnostics.json").write_text(
        json.dumps({
            "router": result["router_diagnostics"],
            "spatial_durbin": result["spatial_diagnostics"],
            "te_cte_peak_lag_days": int(
                te_cte_lag_summary.loc[te_cte_lag_summary["mean_cte_bits"].idxmax(), "lag_days"]
            ),
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (output / "discovery_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    manifest = {
        "version": "3.2.0-real-event-replay",
        "config_sha256": _sha256(config_path),
        "data_sha256": _sha256(data_path),
        "real_qpcr_sha256": _sha256(data_dir / "real_case" / "derived" / "sa_qpcr_observations.csv"),
        "real_data_license": real_provenance["qpcr"]["license"],
        "seed": config["seed"],
        "outputs": [
            "baseline_results.csv",
            "agent_log.csv",
            "risk_predictions.csv",
            "negative_controls.csv",
            "discovery_card.json",
            "multiscale_anomaly_daily.csv",
            "multiscale_event_catalog.csv",
            "adaptive_router_trace.csv",
            "te_cte_network.csv",
            "te_cte_lag_summary.csv",
            "spatial_durbin_effects.csv",
            "spatial_weight_matrix.csv",
            "method_diagnostics.json",
            "sa_real_replay_timeline.csv",
            "sa_real_site_summary.csv",
            "sa_real_species_summary.csv",
            "sa_real_router_trace.csv",
            "sa_real_aquaculture_priority.csv",
            "sa_real_replay_card.json",
        ],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = (
        "# GlobalHAB-Agent GOAI复赛试跑摘要\n\n"
        f"- 研究信号状态：{card['research_signal_status']}\n"
        f"- 实验预算：{config['budget']} / 候选总数：{len(result['catalog'])}\n"
        f"- 完全留出海区：{config['holdout_region']}\n"
        f"- 最佳候选：{best['action_id']}\n"
        f"- PR-AUC：{float(best['pr_auc']):.3f}\n"
        f"- Brier Skill：{float(best['brier_skill']):.3f}\n"
        f"- ECE：{float(best['ece']):.3f}\n"
        f"- Top20%召回：{float(best['recall_at_top20']):.1%}\n"
        f"- Top20%虚警率（FPR）：{float(best['false_positive_rate_at_top20']):.1%}\n"
        f"- 相同预算随机搜索恢复率："
        f"{float(result['random_reference']['hidden_signal_recovery_rate']):.1%}\n"
        f"- 多尺度异常事件数：{len(anomaly_events)}\n"
        f"- TE/CTE平均峰值滞后："
        f"{int(te_cte_lag_summary.loc[te_cte_lag_summary['mean_cte_bits'].idxmax(), 'lag_days'])}天\n"
        f"- Durbin空间自回归系数 rho：{float(result['spatial_diagnostics']['rho']):.2f}\n"
        f"- 已运行路由分支：{int(router_trace['selected'].sum())}/4\n"
        f"- 南澳真实qPCR样本：{real_replay['card']['observations']}条 / "
        f"{real_replay['card']['sampling_dates']}个日期 / {real_replay['card']['locations']}个地点\n"
        f"- 真实事件K. cristata峰值：{real_replay['card']['peak_k_cristata']['cells_l']:,.0f} cells L⁻¹\n"
        f"- 真实数据定位：事件回放与复核优先级，不作为监督训练性能\n\n"
        "> 性能来自匿名合成数据的软件正确性验证；不代表真实HAB或养殖损失预测性能。\n"
    )
    (output / "run_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
