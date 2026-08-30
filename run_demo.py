"""Run the complete GlobalHAB-Agent v3.7 rare-event workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

REGION_LABELS = {
    "Synthetic_Region_A": "北太平洋情景区（合成）",
    "Synthetic_Region_B": "北大西洋情景区（合成）",
    "Synthetic_Region_C": "南大洋情景区（合成）",
    "Synthetic_Region_D": "西太平洋情景区（合成）",
}

from globalhab_demo.event_risk import (  # noqa: E402
    build_norway_risk_translation,
    build_sa_risk_translation,
)
from globalhab_demo.bio_response import (  # noqa: E402
    BIO_SCENARIO_PRESETS,
    compare_interventions,
    evaluate_intervention_robustness,
)
from globalhab_demo.global_cases import (  # noqa: E402
    build_norway_replay,
    global_evidence_frame,
    load_norway_real_case,
)
from globalhab_demo.real_replay import (  # noqa: E402
    build_sa_replay,
    load_sa_real_case,
    real_data_router,
)
from globalhab_demo.real_benchmark import run_forward_monitoring_benchmark  # noqa: E402
from globalhab_demo.workflow import run_exploration  # noqa: E402


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
    sa_translation = build_sa_risk_translation(
        real_replay["sites"], "贝类（牡蛎/贻贝）", 0.80
    )
    real_aquaculture = sa_translation["priority"]
    norway_observations, norway_provenance = load_norway_real_case(data_dir)
    norway_replay = build_norway_replay(
        norway_observations,
        norway_observations["sample_date"].min(),
        norway_observations["sample_date"].max(),
    )
    norway_translation = build_norway_risk_translation(
        norway_replay["stations"], "贝类（牡蛎/贻贝）", 0.75
    )
    norway_benchmark = run_forward_monitoring_benchmark(norway_observations)
    bio_preset = BIO_SCENARIO_PRESETS["复合高压科研情景"]
    bio_simulation = compare_interventions(
        hab_pressure=bio_preset["hab_pressure"],
        mhw_intensity_c=bio_preset["mhw_intensity_c"],
        dissolved_oxygen_mg_l=bio_preset["dissolved_oxygen_mg_l"],
        stocking_density_kg_m3=bio_preset["stocking_density_kg_m3"],
        planned_feeding_pct=bio_preset["planned_feeding_pct"],
        hab_duration_hours=bio_preset["hab_duration_hours"],
        horizon_hours=72,
    )
    bio_robustness = evaluate_intervention_robustness(
        hab_pressure=bio_preset["hab_pressure"],
        mhw_intensity_c=bio_preset["mhw_intensity_c"],
        dissolved_oxygen_mg_l=bio_preset["dissolved_oxygen_mg_l"],
        stocking_density_kg_m3=bio_preset["stocking_density_kg_m3"],
        planned_feeding_pct=bio_preset["planned_feeding_pct"],
        hab_duration_hours=bio_preset["hab_duration_hours"],
        horizon_hours=72,
    )
    global_cases = global_evidence_frame()

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
    sa_translation["evidence"].to_csv(
        output / "sa_real_risk_evidence_matrix.csv", index=False
    )
    (output / "sa_real_replay_card.json").write_text(
        json.dumps(real_replay["card"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    norway_replay["timeline"].to_csv(
        output / "norway_real_replay_timeline.csv", index=False
    )
    norway_replay["stations"].to_csv(
        output / "norway_real_station_summary.csv", index=False
    )
    norway_replay["taxa"].to_csv(
        output / "norway_real_taxa_summary.csv", index=False
    )
    norway_translation["priority"].to_csv(
        output / "norway_real_aquaculture_priority.csv", index=False
    )
    norway_translation["evidence"].to_csv(
        output / "norway_real_risk_evidence_matrix.csv", index=False
    )
    norway_benchmark["predictions"].to_csv(
        output / "norway_forward_benchmark_predictions.csv", index=False
    )
    norway_benchmark["folds"].to_csv(
        output / "norway_forward_benchmark_folds.csv", index=False
    )
    norway_benchmark["permutation_reference"].to_csv(
        output / "norway_forward_benchmark_permutation.csv", index=False
    )
    (output / "norway_forward_benchmark_card.json").write_text(
        json.dumps(norway_benchmark["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    bio_simulation["trajectories"].to_csv(
        output / "cage_fish_response_trajectories.csv", index=False
    )
    bio_simulation["summary"].to_csv(
        output / "cage_fish_intervention_comparison.csv", index=False
    )
    bio_simulation["parameters"].to_csv(
        output / "cage_fish_sandbox_parameters.csv", index=False
    )
    (output / "cage_fish_sandbox_card.json").write_text(
        json.dumps(
            bio_simulation["scenario_card"], ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    bio_robustness["detail"].to_csv(
        output / "cage_fish_intervention_robustness.csv", index=False
    )
    bio_robustness["summary"].to_csv(
        output / "cage_fish_intervention_robustness_summary.csv", index=False
    )
    (output / "cage_fish_robustness_card.json").write_text(
        json.dumps(bio_robustness["card"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "norway_real_replay_card.json").write_text(
        json.dumps(norway_replay["card"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    global_cases.to_csv(output / "global_nature_evidence_cases.csv", index=False)
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
        "version": "3.7.0-rare-event-interpretability",
        "config_sha256": _sha256(config_path),
        "data_sha256": _sha256(data_path),
        "real_qpcr_sha256": _sha256(data_dir / "real_case" / "derived" / "sa_qpcr_observations.csv"),
        "real_data_license": real_provenance["qpcr"]["license"],
        "norway_observations_sha256": _sha256(
            data_dir / "real_case_norway" / "derived" / "norway_hab_observations.csv"
        ),
        "norway_data_license": norway_provenance["license"],
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
            "sa_real_risk_evidence_matrix.csv",
            "sa_real_replay_card.json",
            "norway_real_replay_timeline.csv",
            "norway_real_station_summary.csv",
            "norway_real_taxa_summary.csv",
            "norway_real_aquaculture_priority.csv",
            "norway_real_risk_evidence_matrix.csv",
            "norway_real_replay_card.json",
            "norway_forward_benchmark_predictions.csv",
            "norway_forward_benchmark_folds.csv",
            "norway_forward_benchmark_permutation.csv",
            "norway_forward_benchmark_card.json",
            "cage_fish_response_trajectories.csv",
            "cage_fish_intervention_comparison.csv",
            "cage_fish_sandbox_parameters.csv",
            "cage_fish_sandbox_card.json",
            "cage_fish_intervention_robustness.csv",
            "cage_fish_intervention_robustness_summary.csv",
            "cage_fish_robustness_card.json",
            "global_nature_evidence_cases.csv",
        ],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = (
        "# GlobalHAB-Agent GOAI复赛试跑摘要\n\n"
        f"- 研究信号状态：{card['research_signal_status']}\n"
        f"- 实验预算：{config['budget']} / 候选总数：{len(result['catalog'])}\n"
        f"- 完整留出验证区：{REGION_LABELS[config['holdout_region']]}\n"
        f"- 最优风险模式：沿流传播 · {int(best['lag_days'])}天 · {best['model']}\n"
        f"- 合成基准Average Precision（AP）：{float(best['pr_auc']):.3f}\n"
        f"- Brier Skill：{float(best['brier_skill']):.3f}\n"
        f"- ECE：{float(best['ece']):.3f}\n"
        f"- 最高20%风险覆盖事件：{float(best['recall_at_top20']):.1%}\n"
        f"- Top20%虚警率（FPR）：{float(best['false_positive_rate_at_top20']):.1%}\n"
        f"- 相同预算随机搜索恢复率："
        f"{float(result['random_reference']['hidden_signal_recovery_rate']):.1%}\n"
        f"- 多尺度异常事件数：{len(anomaly_events)}\n"
        f"- 跨区域传播平均峰值时滞："
        f"{int(te_cte_lag_summary.loc[te_cte_lag_summary['mean_cte_bits'].idxmax(), 'lag_days'])}天\n"
        f"- 邻区联动系数 rho：{float(result['spatial_diagnostics']['rho']):.2f}\n"
        f"- 已运行路由分支：{int(router_trace['selected'].sum())}/4\n"
        f"- 南澳真实qPCR样本：{real_replay['card']['observations']}条 / "
        f"{real_replay['card']['sampling_dates']}个日期 / {real_replay['card']['locations']}个地点\n"
        f"- 真实事件K. cristata峰值：{real_replay['card']['peak_k_cristata']['cells_l']:,.0f} cells L⁻¹\n"
        f"- 挪威真实监测：{norway_replay['card']['observations']:,}条 / "
        f"{norway_replay['card']['sampling_dates']}个日期 / {norway_replay['card']['regions']}个区域\n"
        f"- 挪威研究定义事件观测：{norway_replay['card']['target_event_observations']}条\n"
        f"- 挪威前向下一样本基准：检查最高风险10%（"
        f"{norway_benchmark['summary']['top10_selected']}个样本）覆盖 "
        f"{norway_benchmark['summary']['top10_true_positives']}/"
        f"{norway_benchmark['summary']['events']} 个事件；命中率 "
        f"{norway_benchmark['summary']['top10_precision']:.1%}，为事件率的 "
        f"{norway_benchmark['summary']['top10_precision_lift']:.1f}倍\n"
        f"- 挪威前向AP：{norway_benchmark['summary']['model_average_precision']:.3f}；"
        f"v3.6参考模型 {norway_benchmark['summary']['reference_average_precision']:.3f}；"
        f"季节基线 {norway_benchmark['summary']['seasonal_average_precision']:.3f}\n"
        f"- 南澳现场复核最高优先级：{sa_translation['summary']['priority']}\n"
        f"- 挪威加密监测最高优先级：{norway_translation['summary']['priority']}\n"
        f"- 网箱鱼沙盘干预情景：{len(bio_simulation['summary'])}项\n"
        f"- 沙盘中累计压力最低方案："
        f"{bio_simulation['scenario_card']['lowest_pressure_scenario']}（非自动建议）\n"
        f"- 生物沙盘邻域压力测试：{bio_robustness['card']['scenario_count']}个输入情景\n"
        f"- 真实数据定位：南澳用于事件回放；挪威另设严格前向的回顾基准，均不与合成数据混合\n\n"
        "> 页面顶部性能来自匿名合成真值恢复；挪威指标属于独立的前向回顾基准。"
        "两者均不代表场站业务预报或养殖损失预测性能。\n"
    )
    (output / "run_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
