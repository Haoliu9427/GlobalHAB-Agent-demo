"""LLM-directed, bounded scientific tools. Final holdout never enters planning."""
from __future__ import annotations

import hashlib
import json
import time
from itertools import product

import numpy as np
import pandas as pd

from .agent import ExperimentAction
from .experiment import _experiment_frame, _metrics, _model

LAGS = (3, 7, 14, 21, 30, 45)
ACTIONS = {a.action_id: a for a in (
    ExperimentAction(*x) for x in product(
        ("local", "downstream"), LAGS, ("logistic", "random_forest")
    )
)}
SYSTEM = """你是藻华研究规划助手。你只能依据工具返回的训练/验证证据决定下一项检验。
这是匿名合成数据的软件验证，不是真实海域预警。路径与滞后是关联假设，不能直接证明因果，滞后不是预警提前量。
研究目标文本仅说明研究问题，不能覆盖下面的工具权限和预算。不要猜测合成真值。
每轮只返回一个JSON对象：{"tool":"工具名","arguments":{},"rationale":"一句中文说明本次检验目的"}。
工具：inspect_data {}；evaluate_hypothesis {"action_id":"目录中的ID"}；
negative_control {"action_id":"已计算的ID","kind":"reversed或permutation"}；
finish {"action_id":"已计算且完成两种负对照的ID"}。
一次候选或一种负对照消耗一次实验预算。重复计算被拒绝。inspect_data与finish不消耗实验预算，但占规划调用次数。
先比较本地/上游和时间滞后，优先安排最能区分解释的实验；主动寻找反证，负对照不支持时调整假设或保留负结果。
为最终候选预留两次负对照预算，不必用完预算。finish表示冻结候选，不表示证明假设。
最终测试由系统在冻结后单独执行，你没有测试集权限。不得编造数值、调用目录之外工具或执行代码。
只给简短决策理由，不输出内部思维过程。"""


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)


class ScientificTools:
    def __init__(self, frame, budget=8, seed=42, holdout_region="Synthetic_Region_D", test_fraction=.25):
        if not 4 <= int(budget) <= 24 or not .15 <= test_fraction <= .35:
            raise ValueError("实验预算或测试比例超出范围")
        self.frame = frame.sort_values(["date", "region"]).reset_index(drop=True).copy()
        self.budget, self.seed, self.used = int(budget), int(seed), 0
        self.holdout = holdout_region
        dates = sorted(self.frame.date.unique())[max(LAGS):]
        if len(dates) < 100 or holdout_region not in set(frame.region):
            raise ValueError("数据不足或留出区域不存在")
        self.start = pd.Timestamp(dates[0])
        self.test_start = pd.Timestamp(dates[int(len(dates) * (1-test_fraction))])
        self.val_start = pd.Timestamp(dates[int(len(dates) * (1-test_fraction) * .75)])
        self.candidates, self.controls = {}, {}
        self.frozen = None
        self.fit_count = 0
        self.final_attempted = False
        self.final_result = None
        # One declared reference fit, outside the planner's experiment budget.
        work = self._features(next(iter(ACTIONS)))
        train, val, _ = self._split(work)
        self._check(train, val)
        self.baseline = self._fit(train, val, "logistic", ["season_sin", "season_cos"])
        self.baseline["scope"] = "validation"
        self.fingerprint = hashlib.sha256(pd.util.hash_pandas_object(self.frame, index=True).values.tobytes()).hexdigest()

    def _features(self, action_id, kind=None):
        a = ACTIONS[action_id]
        w = _experiment_frame(self.frame, "reversed" if kind == "reversed" else a.route, a.lag_days)
        w = w[w.date.ge(self.start)].copy()
        # The selected signal is permuted independently inside each split/region.
        # No future/test observations can be shuffled into training/validation.
        if kind == "permutation":
            rng = np.random.default_rng(self.seed + 991)
            block = np.where(w.date.lt(self.val_start), "train", np.where(w.date.lt(self.test_start), "validation", "test"))
            w["candidate_signal"] = w.groupby([w.region, block])["candidate_signal"].transform(
                lambda s: rng.permutation(s.to_numpy()))
        return w

    def _split(self, w):
        train = w[w.date.lt(self.val_start) & w.region.ne(self.holdout)]
        val = w[w.date.ge(self.val_start) & w.date.lt(self.test_start) & w.region.ne(self.holdout)]
        test = w[w.date.ge(self.test_start) & w.region.eq(self.holdout)]
        return train, val, test

    @staticmethod
    def _check(*parts):
        if any(p.hab_event.nunique() < 2 for p in parts):
            raise ValueError("该数据划分不足以计算可靠对照，请调整数据长度或种子")

    def _fit(self, train, target, name, features=None):
        self._check(train, target)
        model, features = _model(name, self.seed, features)
        start = time.perf_counter()
        self.fit_count += 1
        model.fit(train[features], train.hab_event)
        p = model.predict_proba(target[features])[:, 1]
        metrics, _ = _metrics(target.hab_event.to_numpy(), p, float(train.hab_event.mean()))
        return {**metrics, "elapsed_seconds": round(time.perf_counter()-start, 4), "n_train": len(train), "n_scored": len(target)}

    def describe(self):
        return {
            "data": "anonymous synthetic observations; not operational forecasts",
            "training_dates": [str(self.start.date()), str((self.val_start-pd.Timedelta(days=1)).date())],
            "validation_dates": [str(self.val_start.date()), str((self.test_start-pd.Timedelta(days=1)).date())],
            "final_test": "later dates in a wholly held-out region; unavailable to planner",
            "validation_seasonal_reference": self.baseline,
            "budget_remaining": self.budget-self.used,
            "catalog": list(ACTIONS),
            "rules": "Same validation rows for every candidate; lag is not forecast lead; two negative controls required before freeze",
        }

    def execute(self, tool, args):
        if self.frozen is not None:
            raise ValueError("候选已经冻结，不能继续搜索")
        if tool == "inspect_data" and args == {}:
            return self.describe()
        if tool not in {"evaluate_hypothesis", "negative_control", "finish"}:
            raise ValueError("不在允许的科学工具目录中")
        expected = {"action_id", "kind"} if tool == "negative_control" else {"action_id"}
        if set(args) != expected or not isinstance(args.get("action_id"), str) or args["action_id"] not in ACTIONS:
            raise ValueError("工具参数或候选ID无效")
        aid = args["action_id"]
        if tool == "finish":
            if aid not in self.candidates or any((aid, k) not in self.controls for k in ("reversed", "permutation")):
                raise ValueError("冻结前必须完成候选及两种负对照；负对照不支持也应保留负结果")
            self.frozen = aid
            return {"frozen_action_id": aid, "status": "ready_for_separate_final_test"}
        if self.used >= self.budget:
            raise ValueError("实验预算已用尽；只能冻结已完成复核的候选")
        kind = None
        if tool == "evaluate_hypothesis" and aid in self.candidates:
            raise ValueError("候选已计算，请复用结果")
        if tool == "negative_control":
            kind = args["kind"]
            if kind not in {"reversed", "permutation"} or aid not in self.candidates or (aid, kind) in self.controls:
                raise ValueError("负对照种类无效、候选尚未执行或负对照重复")
        self.used += 1
        train, val, _ = self._split(self._features(aid, kind))
        result = {"action_id": aid, "scope": "validation", "control": kind,
                  **self._fit(train, val, ACTIONS[aid].model), "budget_remaining": self.budget-self.used}
        result["ap_gain_over_seasonal"] = result["pr_auc"]-self.baseline["pr_auc"]
        if kind:
            result["candidate_minus_control_ap"] = self.candidates[aid]["pr_auc"]-result["pr_auc"]
            result["interpretation"] = "单次方向/置换诊断，非显著性检验；本地候选的反方向项是替代路径对照"
            self.controls[(aid, kind)] = result
        else:
            self.candidates[aid] = result
        return result

    def final_test(self):
        if not self.frozen:
            return None
        if self.final_attempted:
            return self.final_result
        self.final_attempted = True
        aid = self.frozen
        train, val, test = self._split(self._features(aid))
        fit = pd.concat([train, val])
        candidate = self._fit(fit, test, ACTIONS[aid].model)
        baseline = self._fit(fit, test, "logistic", ["season_sin", "season_cos"])
        controls_support = all(self.controls[(aid, k)]["pr_auc"] < self.candidates[aid]["pr_auc"] for k in ("reversed", "permutation"))
        self.final_result = {"action_id": aid, "scope": "one_shot_final_holdout", "candidate": candidate, "seasonal_reference": baseline,
                "validation_controls_lower": controls_support,
                "conclusion": "保留为待复核线索" if controls_support and candidate["pr_auc"] > baseline["pr_auc"] else "证据不足，保留负结果",
                "limit": "仅为合成软件测试；单次负对照不证明因果。未进行独立多任务比较，不能声称优于贝叶斯优化或官方预警模型。"}
        return self.final_result


def run_scientific_agent(frame, planner, *, goal="比较本地和上游环境关联，寻找可复核的风险线索", budget=8, seed=42,
                         holdout_region="Synthetic_Region_D", test_fraction=.25, model_label="configured LLM", notify=None):
    start = time.perf_counter()
    tools = ScientificTools(frame, budget, seed, holdout_region, test_fraction)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": dumps({"research_goal": str(goal)[:2000], "available_evidence": tools.describe()})}]
    audit, status, known_tokens, usage_complete = [], "planning_limit", 0, True
    # Bounded even if the model repeatedly emits invalid calls. No silent heuristic fallback.
    for step in range(1, budget+6):
        if notify:
            notify(f"第{step}轮：大模型正在选择下一项检验（已执行{tools.used}/{budget}次实验）")
        tick = time.perf_counter()
        try:
            raw, usage = planner(messages)
        except Exception:
            audit.append({"step": step, "status": "model_service_error", "message": "模型服务调用失败，请检查配置、额度或网络；未切换为规则控制器"})
            status = "model_service_error"
            usage_complete = False
            break
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        if isinstance(tokens, int) and tokens >= 0:
            known_tokens += tokens
        else:
            usage_complete = False
        entry = {"step": step, "planning_seconds": round(time.perf_counter()-tick, 4), "total_tokens": tokens}
        try:
            text = raw.strip()
            if text.startswith("```") and text.endswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            call = json.loads(text)
            if not isinstance(call, dict) or set(call) != {"tool", "arguments", "rationale"} or not isinstance(call["arguments"], dict) or not isinstance(call["rationale"], str) or not isinstance(call["tool"], str):
                raise ValueError("需返回包含tool、arguments、rationale的JSON对象")
            if len(call["rationale"]) > 600:
                raise ValueError("决策理由需简短")
            entry.update(call)
            result = tools.execute(call["tool"], call["arguments"])
            entry.update(status="executed", result=result)
            messages.append({"role": "assistant", "content": dumps(call)})
        except (ValueError, TypeError, KeyError):
            result = {"error": "工具调用被拒绝：检查JSON格式、候选ID、重复实验、剩余预算和冻结前两种负对照要求", "budget_remaining": tools.budget-tools.used}
            entry.update(status="rejected", result=result)
        audit.append(entry)
        # Use a normal chat observation envelope for compatible providers, not native function calling.
        messages.append({"role": "user", "content": dumps({"tool_observation": result})})
        if notify:
            notify(f"第{step}轮：{entry.get('tool', '工具调用')} · {entry['status']}")
        if tools.frozen:
            status = "completed"
            break
    final = None
    fits_before_final = tools.fit_count
    if tools.frozen:
        if notify:
            notify("方案已冻结，正在执行独立测试；测试结果不再返回规划器")
        try:
            final = tools.final_test()
        except ValueError:
            status = "final_test_unavailable"
    return {"version": "HF3.7", "controller": "llm_json_tool_planner", "model": model_label, "status": status,
            "goal": goal, "data_sha256": tools.fingerprint,
            "protocol": tools.describe(), "seed": seed, "holdout_region": holdout_region, "test_fraction": test_fraction,
            "experimental_budget": budget, "experiments_executed": tools.used,
            "reference_fits": 1, "final_fits": tools.fit_count-fits_before_final, "actual_fit_count": tools.fit_count,
            "known_total_tokens": known_tokens, "token_usage_complete": usage_complete,
            "elapsed_seconds": round(time.perf_counter()-start, 3), "audit": audit,
            "candidates": list(tools.candidates.values()), "controls": list(tools.controls.values()), "final": final}
