import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from globalhab_demo.aquaculture import project_aquaculture_risk  # noqa: E402
from globalhab_demo.bio_response import (  # noqa: E402
    BIO_PRODUCTION_REGIONS,
    INTERVENTIONS,
    compare_interventions,
    evaluate_intervention_robustness,
    production_region_frame,
)
from globalhab_demo.event_risk import (  # noqa: E402
    build_norway_risk_translation,
    build_sa_risk_translation,
)
from globalhab_demo.global_cases import (  # noqa: E402
    build_norway_replay,
    global_evidence_frame,
    load_norway_real_case,
)
from globalhab_demo.real_replay import (  # noqa: E402
    build_sa_replay,
    load_sa_real_case,
    project_real_aquaculture_priority,
    real_data_router,
)
from globalhab_demo.real_benchmark import run_forward_monitoring_benchmark  # noqa: E402
from globalhab_demo.scenario import project_synthetic_scenario  # noqa: E402
from globalhab_demo.workflow import run_exploration  # noqa: E402


@pytest.fixture(scope="session")
def exploration_result():
    return run_exploration(720, 42, 8, "Synthetic_Region_D", 0.25)


def test_cli_runs_and_writes_auditable_outputs(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_demo.py"),
            "--config",
            "config/demo.json",
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )
    log = pd.read_csv(tmp_path / "agent_log.csv")
    card = json.loads((tmp_path / "discovery_card.json").read_text(encoding="utf-8"))
    assert len(log) == 8
    assert log["budget_remaining"].iloc[-1] == 0
    assert card["validation"]["random_split_used"] is False
    assert card["synthetic_ground_truth"]["recovered_by_agent"] is True
    assert (tmp_path / "negative_controls.csv").exists()
    assert (tmp_path / "run_manifest.json").exists()
    for name in [
        "multiscale_event_catalog.csv", "adaptive_router_trace.csv",
        "te_cte_network.csv", "te_cte_lag_summary.csv",
        "spatial_durbin_effects.csv", "spatial_weight_matrix.csv",
        "method_diagnostics.json",
        "sa_real_replay_timeline.csv", "sa_real_site_summary.csv",
        "sa_real_species_summary.csv", "sa_real_router_trace.csv",
        "sa_real_aquaculture_priority.csv", "sa_real_risk_evidence_matrix.csv",
        "sa_real_replay_card.json",
        "norway_real_replay_timeline.csv", "norway_real_station_summary.csv",
        "norway_real_taxa_summary.csv", "norway_real_aquaculture_priority.csv",
        "norway_real_risk_evidence_matrix.csv", "norway_real_replay_card.json",
        "norway_forward_benchmark_predictions.csv",
        "norway_forward_benchmark_folds.csv",
        "norway_forward_benchmark_permutation.csv",
        "norway_forward_benchmark_card.json",
        "cage_fish_response_trajectories.csv",
        "cage_fish_intervention_comparison.csv",
        "cage_fish_sandbox_parameters.csv", "cage_fish_sandbox_card.json",
        "cage_fish_intervention_robustness.csv",
        "cage_fish_intervention_robustness_summary.csv",
        "cage_fish_robustness_card.json",
        "global_nature_evidence_cases.csv",
        "global_production_regions.csv",
        "model_complexity_summary.csv", "model_complexity_seed_results.csv",
        "model_complexity_training_selection.csv", "model_complexity_card.json",
    ]:
        assert (tmp_path / name).exists(), name
    model_card = json.loads(
        (tmp_path / "model_complexity_card.json").read_text(encoding="utf-8")
    )
    assert model_card["main_agent_search_unchanged"] is True
    assert model_card["main_agent_candidate_count"] == 24
    assert model_card["main_agent_budget"] == 8
    assert model_card["outer_validation"]["test_rows"] == 177
    assert model_card["outer_validation"]["test_events"] == 26
    assert len(model_card["tcn"]["random_seeds"]) == 5


def test_exploration_has_baselines_controls_and_random_reference(exploration_result):
    result = exploration_result
    assert set(result["baselines"]["baseline"]) == {"季节气候态", "事件持续性"}
    assert set(result["controls"]["control_name"]) == {"反向路径", "区内时间置换"}
    assert 0 <= result["random_reference"]["hidden_signal_recovery_rate"] <= 1
    assert result["card"]["minimum_references"]["negative_controls_lower_than_candidate"]


def test_competition_equivalent_modules_are_auditable_and_recover_signal(exploration_result):
    result = exploration_result
    anomaly = result["anomaly_daily"]
    assert {"anomaly_score_7d", "anomaly_score_14d", "anomaly_score_30d",
            "anomaly_score_60d", "scale_agreement", "event_id"}.issubset(anomaly.columns)
    assert len(result["anomaly_events"]) > 0

    router = result["router_trace"]
    assert set(router["branch"]) == {
        "blocked_prediction", "multiscale_anomaly", "te_cte_network", "spatial_durbin"
    }
    assert router["selected"].all()

    lag_summary = result["te_cte_lag_summary"]
    peak_lag = int(lag_summary.loc[lag_summary["mean_cte_bits"].idxmax(), "lag_days"])
    assert peak_lag == 14
    at_14 = lag_summary[lag_summary["lag_days"].eq(14)].iloc[0]
    assert at_14["mean_cte_bits"] > at_14["mean_reverse_cte_bits"]

    effects = result["spatial_effects"]
    assert set(effects["effect_type"]) == {"direct", "indirect", "total"}
    spillover = effects[
        effects["variable"].eq("multiscale_anomaly_score_lag14")
        & effects["effect_type"].eq("indirect")
    ].iloc[0]
    assert spillover["effect_per_1sd"] > 0
    assert spillover["ci90_lower"] > 0


def test_scenario_and_aquaculture_outputs_are_bounded():
    scenario = project_synthetic_scenario(
        issue_date=pd.Timestamp("2026-08-28").date(),
        horizon_days=14,
        mhw_intensity_c=2.8,
        nitrate_mmol_m3=5.5,
        phosphate_mmol_m3=0.75,
        silicate_mmol_m3=7.0,
        transport_proxy=0.85,
    )
    assert len(scenario) == 12
    assert {
        "东地中海—爱琴海", "西印度洋—阿拉伯海",
        "秘鲁—智利洪堡流", "智利巴塔哥尼亚峡湾",
    }.issubset(set(scenario["候选海区"]))
    assert scenario["综合风险指数"].between(0, 100).all()
    assert scenario["预计时间"].eq("2026-09-11").all()

    aquaculture = project_aquaculture_risk(
        scenario,
        "贝类（牡蛎/贻贝）",
        "toxigenic",
        0.85,
        "A：物种/毒素/危害确认",
    )
    assert len(aquaculture) == len(scenario)
    assert aquaculture["养殖响应优先指数"].between(0, 100).all()
    assert (aquaculture["不确定性下限"] <= aquaculture["养殖响应优先指数"]).all()
    assert (aquaculture["不确定性上限"] >= aquaculture["养殖响应优先指数"]).all()

    regions = production_region_frame()
    assert len(regions) == len(BIO_PRODUCTION_REGIONS)
    assert regions["cage_sandbox"].sum() >= 7
    cage_status = regions.set_index("region")["cage_sandbox"].to_dict()
    assert cage_status["东地中海—爱琴海"]
    assert cage_status["智利巴塔哥尼亚峡湾"]
    assert not cage_status["秘鲁—智利洪堡流"]
    assert not cage_status["西印度洋—阿拉伯海"]


def test_real_sa_replay_uses_bundled_qpcr_and_defers_unsupported_models():
    observations, provenance = load_sa_real_case(ROOT / "data")
    assert len(observations) == 115
    assert observations["sample_date"].nunique() == 22
    assert observations["location"].nunique() == 22
    assert provenance["qpcr"]["license"] == "CC BY 4.0"

    replay = build_sa_replay(
        observations, observations["sample_date"].min(), observations["sample_date"].max()
    )
    peak = replay["card"]["peak_k_cristata"]
    assert peak["date"] == "2025-07-07"
    assert peak["location"] == "Stansbury (slick on water)"
    assert abs(peak["cells_l"] - 15015439.54) < 0.01

    router = real_data_router(observations, has_daily_environment=False)
    decisions = dict(zip(router["branch"], router["decision"]))
    assert decisions["spatiotemporal_qpcr_replay"] == "run"
    assert decisions["supervised_hab_classifier"] == "defer"
    assert decisions["te_cte_network"] == "defer"

    priority = project_real_aquaculture_priority(
        replay["sites"], "贝类（牡蛎/贻贝）", 0.80
    )
    assert priority["verification_priority_index"].between(0, 100).all()
    translation = build_sa_risk_translation(
        replay["sites"], "贝类（牡蛎/贻贝）", 0.80
    )
    assert translation["priority"]["verification_priority_index"].between(0, 100).all()
    assert set(translation["evidence"]["证据属性"]) == {
        "真实观测", "事件/文献证据", "情景假设", "参数设定", "待补数据"
    }
    assert "不是损失概率" in translation["summary"]["boundary"]


def test_norway_real_monitoring_replay_and_global_evidence_are_auditable():
    observations, provenance = load_norway_real_case(ROOT / "data")
    assert len(observations) == 5919
    assert observations["sample_date"].nunique() == 868
    assert observations["region"].nunique() == 35
    assert provenance["license"] == "CC BY 4.0"

    replay = build_norway_replay(
        observations, observations["sample_date"].min(), observations["sample_date"].max()
    )
    card = replay["card"]
    assert card["target_event_observations"] == 139
    assert card["peak_a_tamarense"] == {
        "date": "2012-07-02", "region": "Tromsø", "cells_l": 3600.0
    }
    assert card["peak_d_acuta"] == {
        "date": "2011-09-29", "region": "Risør", "cells_l": 9632.0
    }
    translation = build_norway_risk_translation(
        replay["stations"], "贝类（牡蛎/贻贝）", 0.75
    )
    assert translation["priority"]["verification_priority_index"].between(0, 100).all()
    assert translation["priority"].iloc[0]["region"] == "Risør"
    assert "不是损失概率" in translation["summary"]["boundary"]
    cases = global_evidence_frame()
    assert len(cases) == 4
    assert (cases["product_status"] == "完整观测回放").sum() == 2


def test_norway_forward_benchmark_is_temporal_and_beats_seasonal_reference():
    observations, _ = load_norway_real_case(ROOT / "data")
    benchmark = run_forward_monitoring_benchmark(observations)
    summary = benchmark["summary"]
    predictions = benchmark["predictions"]
    assert summary["valid_folds"] == 4
    assert summary["samples"] == len(predictions)
    assert summary["events"] >= 60
    assert summary["metric_name"].startswith("Average Precision")
    assert summary["model_average_precision"] > summary["reference_average_precision"]
    assert summary["model_average_precision"] > summary["seasonal_average_precision"]
    assert summary["relative_improvement_over_reference"] > 0.20
    assert summary["model_brier"] < summary["seasonal_brier"]
    assert summary["top10_recall"] >= 0.40
    assert summary["top10_precision"] > summary["event_rate"]
    assert summary["top10_precision_lift"] > 4.0
    assert summary["top10_selected"] == 364
    assert summary["top10_true_positives"] + summary["top10_false_positives"] == 364
    assert summary["permutation_p"] <= 0.01
    assert set(benchmark["folds"]["selected_model"]) == {
        "reference_logistic", "eco_weight5", "eco_weight10"
    }
    assert (predictions["sample_date"] < predictions["next_sample_date"]).all()
    assert predictions["gap_days"].between(1, 14).all()


def test_cage_fish_sandbox_is_bounded_and_interventions_are_honest():
    simulation = compare_interventions(
        hab_pressure=80.0,
        mhw_intensity_c=2.8,
        dissolved_oxygen_mg_l=4.5,
        stocking_density_kg_m3=25.0,
        planned_feeding_pct=100.0,
        hab_duration_hours=48,
        horizon_hours=72,
    )
    trajectories = simulation["trajectories"]
    summary = simulation["summary"].set_index("intervention")
    assert set(summary.index) == set(INTERVENTIONS)
    assert trajectories["relative_physiological_pressure"].between(0, 100).all()
    assert trajectories["compound_challenge"].between(0, 100).all()
    assert trajectories["feeding_opportunity_pct"].between(0, 100).all()
    assert trajectories["effective_do_mg_l"].between(0, 14).all()
    assert (
        summary["peak_pressure_lower"] <= summary["peak_pressure_index"]
    ).all()
    assert (
        summary["peak_pressure_upper"] >= summary["peak_pressure_index"]
    ).all()

    baseline = trajectories[trajectories["intervention"].eq("维持监测")]
    transfer = trajectories[
        trajectories["intervention"].eq("转移准备（未执行）")
    ]
    assert baseline["relative_physiological_pressure"].reset_index(drop=True).equals(
        transfer["relative_physiological_pressure"].reset_index(drop=True)
    )
    assert summary.loc["转移准备（未执行）", "response_readiness_hours"] < summary.loc[
        "维持监测", "response_readiness_hours"
    ]
    assert summary.loc["启动增氧", "peak_pressure_index"] < summary.loc[
        "维持监测", "peak_pressure_index"
    ]
    assert summary.loc["降低投喂40%", "mean_feeding_opportunity_pct"] < summary.loc[
        "维持监测", "mean_feeding_opportunity_pct"
    ]
    assert "mortality prediction" in simulation["scenario_card"]["excluded_claims"]

    robustness = evaluate_intervention_robustness(
        hab_pressure=80.0,
        mhw_intensity_c=2.8,
        dissolved_oxygen_mg_l=4.5,
        stocking_density_kg_m3=25.0,
        planned_feeding_pct=100.0,
        hab_duration_hours=48,
        horizon_hours=72,
    )
    assert robustness["card"]["scenario_count"] == 81
    assert set(robustness["summary"]["intervention"]) == set(INTERVENTIONS)
    assert robustness["summary"]["pareto_frequency"].between(0, 100).all()
    assert robustness["summary"]["lowest_pressure_frequency"].between(0, 100).all()
