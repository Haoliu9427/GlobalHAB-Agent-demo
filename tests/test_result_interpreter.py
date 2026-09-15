from pathlib import Path

from globalhab_demo.real_training.result_interpreter import (
    benchmark_summary,
    core_discovery_summary,
    mainland_summary,
    mainland_tasks,
    make_prompt,
    norway_summary,
    real_training_summary,
    real_training_tasks,
    south_australia_summary,
    uploaded_result_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def test_registered_summaries_build_without_network():
    assert "Average Precision" in core_discovery_summary(ROOT)
    assert "19" in benchmark_summary(ROOT)
    assert "0.102" in norway_summary(ROOT)
    assert "独立采样日数=22" in south_australia_summary(ROOT)


def test_real_training_and_mainland_summaries():
    tasks = real_training_tasks(ROOT)
    assert tasks
    first = next(iter(tasks.values()))
    text = real_training_summary(first, "test")
    assert "真实观测训练与验证" in text
    mainland = mainland_tasks(ROOT)
    assert mainland
    path = next(iter(mainland.values()))
    text2 = mainland_summary(path, "全部")
    assert "中国近海观测与跨年检验" in text2


def test_uploaded_csv_is_bounded_summary():
    raw = b"model,AP,Brier\nA,0.5,0.1\nB,0.4,0.2\n"
    text = uploaded_result_summary("metrics.csv", raw)
    assert "行数=2" in text
    assert "model,AP,Brier" in text


def test_prompt_has_evidence_boundaries():
    messages = make_prompt("## result\n- AP=0.5", "答辩讲解", "怎么讲")
    system = messages[0]["content"]
    assert "不得把关联写成因果证明" in system
    assert "合成机制验证" in system
    assert "30秒口头讲法" in system
    assert "答辩讲解" in messages[1]["content"]
