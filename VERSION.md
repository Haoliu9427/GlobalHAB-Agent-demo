# Version 4.1

## Experiment selection

- Added Bayesian Expected Improvement, Bayesian Information Gain and Thompson Sampling alongside the constrained heuristic and Random policy.
- All policies use the same 24 candidate experiments and the same budget.
- The synthetic 14-day reference is excluded from acquisition functions and used only after a trajectory is complete.

## Florida/Gulf retrospective analysis

- Added NOAA HABSOS `Karenia brevis` input.
- Added live HYCOM-TSIS GOMb0.04 Gulf reanalysis as the default retrospective current source; NOAA CoastWatch remains an online alternative, with CSV adapters for HYCOM/Copernicus/HF-radar products.
- Added flow-constrained, no-flow and reverse-flow lag comparisons.

## Field forward validation

- Added station-observation and current-field CSV templates.
- Added data-quality checks, training-period lag selection, later-period forward evaluation and next-sampling projections.
- Insufficient data return `DEFER`.


## LLM result interpretation workspace

- Added a third top-level workspace: `大模型结果解读`.
- Reads registered project results, real-training results, mainland validation, the latest in-session user analysis, or uploaded result files.
- Supports research, defense, paper-results and management-summary modes.
- Uses the existing user/server Chat Completions compatible configuration; credentials remain session-only.
- The LLM receives a bounded result summary only after explicit consent and cannot modify upstream metrics or predictions.

## Reproduction

```bash
python scripts/run_agent_policy_benchmark.py
python scripts/run_florida_sts_validation.py --online
python scripts/run_field_forward_validation.py --observations <csv> --currents <csv>
```


## Field visual screening workspace

- Added a fourth top-level workspace: `现场影像甄别 / Field Visual Screening`.
- Supports Streamlit camera capture or JPG/PNG upload, local image-quality checks, transparent colour/texture screening, and field metadata.
- The default backend is explicitly a heuristic visual baseline, not a trained HAB species/toxin classifier.
- Outputs a follow-up screening priority (`low / moderate / high / defer`) rather than an HAB probability.
- Stores the structured result in-session and can hand it to the LLM interpretation workspace; raw photos are not sent to the remote LLM by default.
- Added `VisualScreeningBackend` as a replaceable interface for a future calibrated project-specific vision model.


## Adaptive visual routing

- Upgraded the field visual workspace with quality-first adaptive routing.
- Added optional DINOv2, ConvNeXt-Tiny and EfficientNet-B0 frozen feature extractors.
- Added safe NumPy linear heads that fuse image embeddings with 12 field-metadata features.
- Added entropy, top-1/top-2 margin, multi-branch disagreement and OOD-aware DEFER.
- Deep branches activate only when both encoder assets and project-trained heads are available; otherwise the workspace falls back to the transparent heuristic baseline.
- Added `requirements-vision.txt`, training manifest template, lightweight-head training script and `VISION_ROUTER_GUIDE.md`.


## 2026-09-16 adaptive visual router operational update
- Removed the long internal-method sentence from the field-image hero.
- Replaced the default "all unconfigured" state with runtime-ready encoder/source reporting.
- Added a built-in visual-phenomenon prototype head; project-trained heads still take precedence.
- EfficientNet-B0 and ConvNeXt-Tiny now perform a real deep forward pass even in fully offline mode through a deterministic prototype-encoder fallback, explicitly labelled non-pretrained.
- Public pretrained weights are preferred and may be cached on first use; DINOv2-small joins routing when its real model assets are available.
- Adaptive mode now DEFERs when no deep branch actually executes; it no longer silently substitutes the heuristic baseline as the final deep result.
- Added encoder-source/head-type reporting to branch results and LLM summaries.

## 2026-09-16 continuous-learning field vision agent
- Split the field-vision workspace into three subpages: screening, user visual data, and model training/version management.
- Added a SHA256-deduplicated user image library with visual labels, evidence levels, training inclusion flags, metadata and active-learning scores.
- Added active-learning prioritisation for high-entropy, low-margin, cross-backbone disagreement, OOD and DEFER samples.
- Added lightweight user-head retraining on frozen EfficientNet / ConvNeXt / DINOv2 embeddings plus 12 field-metadata features.
- Added deterministic holdout diagnostics, same-holdout public-baseline comparison, candidate/active/rejected version states and explicit promotion gates.
- Added `vision_models/active_model.json`; adaptive inference now resolves the currently registered user head automatically.
- Added user visual-library ZIP export/import and model-version ZIP export.
- Added an auditable public visual source catalog plus optional Zenodo fetch and positive-domain embedding-adapter scripts; raw third-party images are not bundled in the normal source package.
- Added `CONTINUOUS_VISUAL_AGENT_GUIDE.md`, `PUBLIC_VISUAL_DATA_SOURCES.md`, `src/globalhab_demo/visual_learning.py`, and visual-learning tests.

## 2026-09-16 Case-driven cross-workspace evidence loop
- Added `src/globalhab_demo/case_manager.py` and a persistent `data/cases/cases.json` ledger.
- Research risk scenarios can create a field-verification Case carrying candidate region, risk score, Route, Lag, Top-k capacity/coverage and scenario context.
- Field visual screening reads the active Case, registers visual screening as a separate evidence layer, and can return to the research evidence ledger.
- Professional / microscopy / qPCR / toxin confirmation can be appended to the same Case; confirmed photos may be promoted into the visual training library with `case_id` provenance.
- The visual library now stores `case_id` and preserves the Case link through retraining data selection.
- The LLM workspace adds `当前完整Case（推荐）`, combining research, visual, field metadata and lab evidence for bounded interpretation.
- LLM interpretations can be saved back to the Case as explanation records without changing upstream metrics or evidence grades.
- Added `CASE_EVIDENCE_LOOP_GUIDE.md`, `scripts/check_case_loop.py`, and `tests/test_case_manager.py`.

## 2026-09-16 Case闭环界面一致性与证据链报错修复
- 修复 `app.py` 中生物响应说明表局部变量 `evidence_rows` 覆盖同名证据链函数的问题；该问题会在Streamlit整页执行tabs时导致研究与验证各页底部统一出现 `TypeError`。
- “从风险候选生成现场复核任务”改为与全站一致的四列科研KPI卡片，不再使用会截断长海区名称的原生metric布局。
- 现场影像甄别的当前Case摘要同步改为相同KPI卡片风格，候选海区、Case ID和Route/Lag允许自动换行，不再遮挡。
- 大模型结果解读Hero移除“让大模型解释已经计算完成的科学结果，而不是替代模型计算”一句，保留更简洁的工作区能力说明。
- 自有数据分析新增与其他一级工作区一致的深海蓝Hero宣传卡，统一导航后的视觉入口。
- 新增 `tests/test_case_ui_regression.py`，防止证据链函数重名覆盖、Case卡片样式和工作区Hero文案回归。

## 2026-09-16 Case task queue and batch-review update
- Upgraded the Case ledger to schema `1.1` with explicit lifecycle states: pending review, in progress, visual screened, visual DEFER, lab pending, confirmed, cancelled and archived.
- Added batch Case creation from the risk-candidate table, with duplicate protection for identical research candidates.
- Added a sidebar Case/task manager with queue counts, start/continue, cancel, archive, restore and guarded permanent deletion.
- Added audit checks before permanent deletion; linked visual samples can be removed from future training while prior trained model snapshots remain immutable.
- Added a field-vision task queue and `登记并处理下一个` workflow for sequential review of multiple Cases.
- Added risk-page task overview and dedicated regression tests for queue transitions, batch creation, deletion safeguards and UI controls.

## 2026-09-16 DeepSeek result-interpretation compatibility fix
- Fixed the LLM interpretation path for current DeepSeek V4 Chat Completions responses.
- DeepSeek result interpretation now defaults to a stable non-thinking request so the visible `content` field is not starved by reasoning tokens.
- Added optional low/high thinking modes; if a thinking request returns reasoning but no visible final answer or reaches the output limit, the app retries once in stable mode.
- The parser now accepts both string content and OpenAI-compatible text-block content, while never substituting `reasoning_content` for the final answer.
- Increased interpretation output budgets and added provider-specific diagnostics instead of the generic “模型返回格式无效” message.
- Remote probability scoring continues to require a valid probability JSON and now uses the stable DeepSeek path.

