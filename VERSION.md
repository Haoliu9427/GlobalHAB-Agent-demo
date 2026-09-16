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


## 2026-09-16 workspace visual consistency polish
- Removed numeric prefixes from the three field-vision subpages; navigation now reads `现场影像甄别 / 我的影像数据 / 模型训练与版本`.
- Replaced native Streamlit metric blocks in the visual library and model-version subpages with the project KPI card system, including wrapped long model-version names.
- Added keyed, ocean-palette cards for visual-library save/batch import and model baseline/data-check/training sections.
- Simplified the own-data workspace hero by removing the secondary implementation sentence.
- Redesigned the left workspace navigation with a compact ocean-brand header, clearer section hierarchy, card-style workspace options, hover/selected states and a more consistent Case-panel appearance.
- Added regression checks so tab labels, hero copy and workspace/sidebar styling do not revert.

## 2026-09-16 LLM connection-state fix
- Fixed a Streamlit state mismatch where a model name could be visible in the browser while the backend still treated `llm_api_model` as empty, leaving “测试连接” and “生成大模型解读” disabled.
- DeepSeek now has backend-effective defaults (`https://api.deepseek.com`, `deepseek-flash`) rather than UI-only values.
- After `/models` succeeds, the UI exposes an “实际调用模型” selector and uses that selected returned model for the actual request.
- Added explicit connection readiness messaging and exact blockers for the generate button.
- Added regression tests for provider defaults and discovered-model selection.

## 2026-09-16 workspace layout refinement
- Removed numbered prefixes from field-vision section headings and removed the extra scientific-boundary banner from the workspace landing area.
- Added Chinese camera-permission guidance and CSS localisation for Streamlit's standard camera permission helper where supported by the current frontend DOM.
- Standardised the LLM workspace Hero subtitle to the same tagline typography used by the other top-level workspaces.
- Rebalanced Own Data, Field Vision and LLM paired cards to equal column widths and stretch-to-row card heights on desktop, with responsive auto-height fallback on smaller screens.
- Unified card padding, heading scale, border radius and visual rhythm across these three workspaces.

## 2026-09-16 paired-card strict height alignment
- Fixed remaining desktop height mismatch in the Own Data and LLM workspaces by targeting Streamlit's keyed container wrappers directly rather than relying only on the outer column flex row.
- Compactified the remote-service card: API URL/model ID share a row, service actions share one row, returned model IDs and privacy details are collapsed into expanders.
- Increased the prediction-task description area slightly so the first Own Data pair uses the same visual rhythm.
- The LLM question box now uses a fixed compact height; paired result-source and interpretation-mode cards share the same desktop height floor.
- Mobile/tablet layouts below 900 px keep natural heights.

## 2026-09-16 card content density and presentation refinement
- Kept the paired-card equal-height layout while filling previously empty space with user-facing functions rather than decorative whitespace.
- LLM result selection now shows evidence scope, summary size and a read-only current-summary preview inside the source card.
- Replaced the internal-looking LLM question example with a general evidence-consistency and follow-up-review prompt suitable for end users.
- Own-data observation cards now show upload readiness, field groups and preprocessing expectations; task cards show the outputs implied by the selected validation/forecast mode.
- Remote-service cards now expose connection readiness and the exact remote data scope; model/run cards show validation strategy, selected model count, horizon/training budget and saved analysis records.
- No model, Case, visual-routing, continuous-learning or remote-LLM logic was removed.

## 2026-09-16 paired-card content balance refinement
- Reworked the Own Data first-row cards so `观测数据` and `预测任务` share a stricter two-column grid and matched visible bottoms on desktop.
- Replaced the unequal `下载字段模板 / 等待上传CSV` controls with two equal-width, equal-height action buttons.
- Added upload-side automatic checks (time/holdout and label/field checks) and task-side pre-run checks (validation split, unknown-label handling, forecast origin and saved outputs) so alignment no longer creates empty white space.
- Added LLM source-card send-readiness details: structured-summary readiness, raw-file default, evidence scope and remote character limit.
- Switched the Own Data and LLM top paired rows to a CSS-grid based equal-height strategy, which follows the taller card rather than relying on Streamlit's nested flex wrappers.

### 2026-09-16 · LLM 解读卡片最终平衡
- 大模型结果解读左右卡片改为更严格的等高布局。
- 右侧新增输出长度、重点关注、关键数字核对清单和当前解读配置，避免以空白换对齐。
- 摘要预览略收紧，使左右内容密度更接近；上述设置会真实进入大模型提示词与结果签名。

## 2026-09-16 visible-card ownership and final balance
- Moved the visible border/background of the Own Data observation/task pair and LLM source/mode pair from nested Streamlit containers to the equal-height parent columns, eliminating residual bottom-edge mismatch caused by Streamlit wrapper height inheritance.
- Kept the inner keyed containers for widgets and state only; they are now visually transparent.
- Enriched the Own Data observation card with recommended row organisation and the task card with explicit result-record contents.
- Enriched the LLM source card with source-specific explainable topics, downloadable structured summary and intended usage scenarios, so the equal-height layout is filled with real user-facing functions rather than blank space.


## 2026-09-16 · 卡片信息精简与留白优化
- 保留自有数据分析与大模型解读成对卡片的严格对齐，同时取消过大的固定高度下限，由同一行较高卡片自然决定高度。
- 精简重复说明：自有数据卡仅保留必要字段、自动检查、输出和验证规则；大模型解读卡仅保留输入摘要、发送设置、解读控制与输出结构。
- 留白优先分布在卡片内部内容区块之间，减少顶部和底部的无效空白。

- 2026-09-16：精简自有数据分析与大模型结果解读卡片的常驻文字；保留等宽等高布局，将字段要求、验证规则和远程调用说明收进折叠项，并用紧凑底部状态替代大段说明。

## 2026-09-16 remote model status strip fix
- Moved remote-service action feedback out of the three narrow action-button columns.
- Replaced squeezed Streamlit success/error alerts with a full-width compact horizontal status strip below the action row.
- Removed duplicate large connection-status messaging while preserving equal-width/equal-height paired card geometry.

- UI修复：自有数据分析的远程服务默认模型可直接测试连接；“模型与运行”主按钮固定在卡片底部；大模型解读左卡重排摘要区域以减少底部空白，同时保持成对卡片严格等宽等高。

- 2026-09-16：修复自有数据分析远程模型“测试连接”误禁用；远程服务/模型运行卡改为严格等高外层卡片，开始分析固定在右卡底部；大模型解读左卡扩大摘要预览并把留白分配到内容区块之间。
