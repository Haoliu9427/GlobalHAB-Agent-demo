"""Field visual screening workspace for GlobalHAB-Agent.

This module provides a deliberately bounded *screening* baseline for sea-surface
photos. It is not a trained HAB species classifier. The default backend uses
simple, transparent colour/texture/quality features plus user-entered field
context to decide whether a photo deserves higher-priority follow-up sampling.

The transparent baseline remains available as an explicit comparison mode. In
adaptive mode, the release attempts real EfficientNet / ConvNeXt / DINOv2 frozen
feature extraction and combines it with an internal visual-phenomenon prototype
head or an optional project-trained head. If no requested deep branch actually
runs, the final decision is DEFER rather than silently presenting the heuristic
baseline as a deep-model result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, time
from io import BytesIO
import json
import html
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageOps


MAX_IMAGE_BYTES = 12_000_000
MAX_ANALYSIS_SIDE = 768

VISUAL_CLASSES = (
    "正常/未见明显异常",
    "绿色水体异常",
    "红棕色水体异常",
    "高浑浊/泥沙样",
    "表层浮沫/漂浮物样",
    "不确定",
)


@dataclass
class ImageQuality:
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float
    underexposed_fraction: float
    overexposed_fraction: float
    glare_fraction: float
    quality_score: float
    suitable: bool
    warnings: list[str]


@dataclass
class VisualFeatures:
    green_fraction: float
    red_brown_fraction: float
    turbid_fraction: float
    white_low_saturation_fraction: float
    blue_cyan_fraction: float
    mean_saturation: float
    visual_anomaly_score: float
    category: str
    category_score: float
    possible_confounders: list[str]
    feature_notes: list[str]


class VisualScreeningBackend(Protocol):
    """Replaceable visual screening interface."""

    name: str

    def analyse(self, image: Image.Image) -> tuple[ImageQuality, VisualFeatures]:
        ...


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _resize_rgb(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    img = ImageOps.exif_transpose(image).convert("RGB")
    img.thumbnail((MAX_ANALYSIS_SIDE, MAX_ANALYSIS_SIDE), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return img, arr


def _quality(arr: np.ndarray, original_size: tuple[int, int]) -> ImageQuality:
    width, height = original_size
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mx = np.max(arr, axis=2)
    mn = np.min(arr, axis=2)
    sat = mx - mn

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    dx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    dy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    sharpness = float((dx + dy) / 2.0)
    under = float(np.mean(gray < 0.08))
    over = float(np.mean(gray > 0.94))
    glare = float(np.mean((gray > 0.86) & (sat < 0.10)))

    warnings: list[str] = []
    severe = 0
    if min(width, height) < 480:
        warnings.append("图像分辨率偏低；建议靠近水面后重新拍摄。")
        severe += 1 if min(width, height) < 300 else 0
    if brightness < 0.15 or under > 0.35:
        warnings.append("画面偏暗，水色判断可能不稳定。")
        severe += 1
    if brightness > 0.86 or over > 0.28:
        warnings.append("画面过曝，浅色水面与泡沫可能被高估。")
        severe += 1
    if glare > 0.18:
        warnings.append("强反光/浪花高亮区域较多，可能干扰浮沫判断。")
    if sharpness < 0.012:
        warnings.append("图像纹理信息较少或可能模糊；平静水面也可能出现该提示，建议结合原图判断。")
        if sharpness < 0.006 and contrast < 0.035:
            severe += 1
    if contrast < 0.045:
        warnings.append("画面对比度较低，颜色与纹理分离有限。")

    penalty = (
        0.25 * min(1.0, under / 0.35)
        + 0.25 * min(1.0, over / 0.28)
        + 0.18 * min(1.0, glare / 0.20)
        + 0.18 * max(0.0, (0.018 - sharpness) / 0.018)
        + 0.14 * (1.0 if min(width, height) < 480 else 0.0)
    )
    quality_score = _clip01(1.0 - penalty)
    suitable = severe == 0 and quality_score >= 0.45
    return ImageQuality(
        width=width,
        height=height,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        underexposed_fraction=under,
        overexposed_fraction=over,
        glare_fraction=glare,
        quality_score=quality_score,
        suitable=suitable,
        warnings=warnings,
    )


def _visual_features(arr: np.ndarray, q: ImageQuality) -> VisualFeatures:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.max(arr, axis=2)
    mn = np.min(arr, axis=2)
    sat = mx - mn
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Transparent, intentionally broad colour signatures. They are visual cues,
    # not taxonomic labels and not calibrated HAB probabilities.
    green_mask = (g > 0.16) & (g > r * 1.08) & (g > b * 1.06) & (sat > 0.07)
    red_brown_mask = (
        (r > 0.17)
        & (g > 0.10)
        & (r > g * 1.06)
        & (g > b * 1.05)
        & (r > b * 1.25)
        & (sat > 0.08)
    )
    turbid_mask = (
        (r > 0.16)
        & (g > 0.13)
        & (b < 0.30)
        & ((r + g) / 2 > b * 1.18)
        & (np.abs(r - g) < 0.24)
        & (sat < 0.34)
    )
    white_low_sat = (gray > 0.78) & (sat < 0.12)
    blue_cyan = (b > r * 1.05) & (g > r * 1.02) & (b > 0.18)

    green_fraction = float(np.mean(green_mask))
    red_brown_fraction = float(np.mean(red_brown_mask))
    turbid_fraction = float(np.mean(turbid_mask))
    white_fraction = float(np.mean(white_low_sat))
    blue_fraction = float(np.mean(blue_cyan))
    mean_saturation = float(np.mean(sat))

    green_score = _clip01((green_fraction - 0.035) / 0.30)
    red_score = _clip01((red_brown_fraction - 0.025) / 0.28)
    turbidity_score = _clip01((turbid_fraction - 0.06) / 0.40)
    foam_score = _clip01((white_fraction - 0.035) / 0.22)

    scores = {
        "绿色水体异常": green_score,
        "红棕色水体异常": red_score,
        "高浑浊/泥沙样": turbidity_score,
        "表层浮沫/漂浮物样": foam_score,
    }
    category, category_score = max(scores.items(), key=lambda kv: kv[1])
    anomaly = max(scores.values())

    if anomaly < 0.18:
        if blue_fraction > 0.16 and q.quality_score >= 0.45:
            category = "正常/未见明显异常"
            category_score = _clip01(0.45 + blue_fraction)
        else:
            category = "不确定"
            category_score = 1.0 - anomaly
    elif category_score < 0.34:
        category = "不确定"

    confounders: list[str] = []
    if q.glare_fraction > 0.08 or q.overexposed_fraction > 0.12:
        confounders.append("太阳反光、浪花或过曝可能形成亮白区域")
    if turbidity_score > 0.25 or red_score > 0.25:
        confounders.append("泥沙、河口径流或悬浮颗粒可形成黄褐/红棕色水体")
    if green_score > 0.25:
        confounders.append("浅水底质、海草/大型藻类或相机白平衡可能造成绿色偏移")
    if foam_score > 0.25:
        confounders.append("自然泡沫、船尾浪、油膜或漂浮碎屑可能与表层聚集混淆")
    if not confounders:
        confounders.append("普通照片仍受光照、白平衡、拍摄角度和水深影响")

    notes = [
        f"绿色视觉特征占比={green_fraction:.1%}",
        f"红棕视觉特征占比={red_brown_fraction:.1%}",
        f"浑浊样视觉特征占比={turbid_fraction:.1%}",
        f"亮白低饱和区域占比={white_fraction:.1%}",
    ]
    return VisualFeatures(
        green_fraction=green_fraction,
        red_brown_fraction=red_brown_fraction,
        turbid_fraction=turbid_fraction,
        white_low_saturation_fraction=white_fraction,
        blue_cyan_fraction=blue_fraction,
        mean_saturation=mean_saturation,
        visual_anomaly_score=float(anomaly),
        category=category,
        category_score=float(category_score),
        possible_confounders=confounders,
        feature_notes=notes,
    )


class HeuristicVisualBaseline:
    name = "规则 + 通用颜色/纹理特征演示基线"

    def analyse(self, image: Image.Image) -> tuple[ImageQuality, VisualFeatures]:
        rgb, arr = _resize_rgb(image)
        quality = _quality(arr, ImageOps.exif_transpose(image).size)
        features = _visual_features(arr, quality)
        return quality, features


DEFAULT_BACKEND: VisualScreeningBackend = HeuristicVisualBaseline()


def load_image(raw: bytes) -> Image.Image:
    if not raw:
        raise ValueError("没有读取到图像内容。")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("图片请控制在12 MB以内。")
    try:
        img = Image.open(BytesIO(raw))
        img.verify()
        img = Image.open(BytesIO(raw))
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        raise ValueError("无法解析图片，请上传JPG、JPEG或PNG。") from exc


def _context_flags(metadata: dict[str, Any]) -> tuple[list[str], int]:
    flags: list[str] = []
    if metadata.get("odor") == "有明显异味":
        flags.append("现场记录到明显异味")
    surface = set(metadata.get("surface_signs") or [])
    if {"泡沫", "浮膜", "漂浮物"} & surface:
        flags.append("现场记录到泡沫/浮膜/漂浮物")
    if metadata.get("recent_heat") == "是":
        flags.append("近期存在高温背景")
    color = metadata.get("water_color")
    if color in {"绿色", "黄绿色", "红棕色", "褐色", "乳白色"}:
        flags.append(f"现场记录水色为{color}")
    if metadata.get("mass_mortality") == "观察到":
        flags.append("现场观察到鱼贝异常/死亡，需要独立应急复核")
    return flags, len(flags)


def make_result(
    image: Image.Image,
    metadata: dict[str, Any],
    backend: VisualScreeningBackend = DEFAULT_BACKEND,
    root: Any | None = None,
    requested_mode: str = "自适应路由",
) -> dict[str, Any]:
    quality, features = backend.analyse(image)
    context_flags, n_flags = _context_flags(metadata)

    from globalhab_demo.adaptive_visual import run_adaptive_route
    adaptive = run_adaptive_route(
        image=image,
        metadata=metadata,
        quality=quality,
        visual=features,
        root=root,
        requested_mode=requested_mode,
    )

    effective_category = features.category
    effective_score = features.visual_anomaly_score
    if adaptive.get("active") and adaptive.get("ensemble"):
        effective_category = adaptive["ensemble"].get("predicted_class", effective_category)
        effective_score = float(adaptive["ensemble"].get("visual_anomaly_score", effective_score))

    if not metadata.get("sea_surface_confirmed", True):
        priority = "DEFER / 需重拍"
        reason = "用户未确认海面是照片主体。"
    elif not quality.suitable:
        priority = "DEFER / 需重拍"
        reason = "图像质量不足以支持稳定的水色/表层特征甄别。"
    elif requested_mode not in {"安全规则基线", "规则基线"} and not adaptive.get("active"):
        priority = "DEFER / 视觉引擎未就绪"
        detail = adaptive.get("fallback_reason") or "当前没有深度视觉分支成功执行"
        errors = adaptive.get("branch_errors") or []
        reason = detail + ("；" + "；".join(errors[:2]) if errors else "") + "。规则基线仅作辅助参考，不作为当前自适应模式的最终判定。"
    elif adaptive.get("active") and (adaptive.get("uncertainty") or {}).get("defer"):
        priority = "DEFER / 需人工复核"
        reasons = (adaptive.get("uncertainty") or {}).get("reasons") or ["深度视觉分支不确定性较高"]
        reason = "；".join(reasons) + "。不强制给出高/中/低判断。"
    else:
        score = effective_score
        if score >= 0.58 or (score >= 0.42 and n_flags >= 2):
            priority = "高"
        elif score >= 0.28 or n_flags >= 2:
            priority = "中"
        else:
            priority = "低"
        reason = "视觉异常强度与现场辅助线索共同决定复核优先级；该等级不是HAB发生概率。"

    follow_up = [
        "若条件允许，补采叶绿素a、显微镜/流式或分子(qPCR/eDNA)证据。",
        "对可能产毒类群，毒素检测必须与视觉筛查分开完成。",
        "记录温度、盐度、溶解氧并保留同位置/相邻位置的重复照片。",
    ]
    if effective_category in {"红棕色水体异常", "高浑浊/泥沙样"}:
        follow_up.append("优先补充浊度/悬浮颗粒或河口径流信息，以排除泥沙混淆。")
    if effective_category == "表层浮沫/漂浮物样" or quality.glare_fraction > 0.08:
        follow_up.append("换一个避开太阳反光的角度复拍，以区分泡沫、浪花和表层聚集。")

    if adaptive.get("active"):
        branch_names = [b.get("display_name", b.get("name", "")) for b in adaptive.get("branches", [])]
        backend_name = "自适应视觉路由 · " + " + ".join(x for x in branch_names if x)
    else:
        backend_name = "深度视觉分支未成功执行 · DEFER（规则基线仅供辅助参考）" if requested_mode not in {"安全规则基线", "规则基线"} else backend.name

    return {
        "schema": "globalhab-field-visual-screening-v3",
        "backend": backend_name,
        "baseline_backend": backend.name,
        "scientific_role": "现场影像辅助甄别；不是藻种/毒素确诊器",
        "quality": asdict(quality),
        "visual": asdict(features),
        "adaptive_visual": adaptive,
        "effective_visual_category": effective_category,
        "effective_visual_anomaly_score": float(effective_score),
        "field_metadata": metadata,
        "field_context_flags": context_flags,
        "screening_priority": priority,
        "priority_reason": reason,
        "recommended_follow_up": follow_up,
        "allowed_claims": [
            "可描述照片中的水色、浑浊、亮白表层和其他视觉异常线索。",
            "深度编码器实际执行时，可报告其视觉现象筛查结果、不确定性与跨分支一致性；若使用内置原型头，应明确其属于未校准的视觉现象筛查。",
            "可将视觉异常作为是否值得进一步采样复核的低成本现场证据。",
        ],
        "prohibited_claims": [
            "不能仅凭普通海面照片确诊具体藻种。",
            "不能仅凭普通海面照片判断是否产毒或毒素浓度。",
            "不能把视觉类别概率或本页复核优先级解释为HAB发生概率、监管阈值或业务预警。",
            "不能用视觉阴性排除肉眼不可见但具有生态风险的HAB。",
            "不能把DINOv2/ConvNeXt/EfficientNet的通用预训练表征或内置原型头表述为经过真实HAB现场照片验证的藻华分类器。",
        ],
    }

def result_summary(result: dict[str, Any] | None) -> str:
    if not result:
        return "## 最近一次影像识别\n- 当前会话还没有完成影像识别。"
    q = result.get("quality", {})
    v = result.get("visual", {})
    m = result.get("field_metadata", {})
    adaptive = result.get("adaptive_visual") or {}
    effective_category = result.get("effective_visual_category", v.get("category", "NA"))
    effective_score = result.get("effective_visual_anomaly_score", v.get("visual_anomaly_score", 0))
    lines = [
        "## 最近一次影像识别",
        f"- 方法：{result.get('backend','NA')}。",
        f"- 科学角色：{result.get('scientific_role','NA')}。",
        f"- 最终视觉类型：{effective_category}；视觉异常分数={float(effective_score):.3f}（用于现场复核，不是HAB概率）。",
        f"- 透明规则基线类型：{v.get('category','NA')}；基线异常特征强度={v.get('visual_anomaly_score',0):.3f}。",
        f"- 复核优先级：{result.get('screening_priority','NA')}。",
        f"- 图像质量分数={q.get('quality_score',0):.3f}；适合甄别={q.get('suitable',False)}；尺寸={q.get('width','NA')}×{q.get('height','NA')}。",
        f"- 现场位置/海域：{m.get('location_text') or '未填写'}；拍摄日期={m.get('capture_date') or '未填写'}；水色={m.get('water_color') or '未填写'}。",
        f"- 现场辅助线索：{'; '.join(result.get('field_context_flags') or ['无明确辅助线索'])}。",
        f"- 主要混淆因素：{'; '.join(v.get('possible_confounders') or [])}。",
        f"- 建议复核：{'; '.join(result.get('recommended_follow_up') or [])}",
    ]
    if adaptive:
        route = adaptive.get("route") or {}
        selected = route.get("selected") or []
        lines.append(f"- 自适应路由：{route.get('route_family','NA')}；选择分支={', '.join(selected) if selected else '无'}；原因={route.get('reason','NA')}。")
        if adaptive.get("active"):
            branch_bits = []
            for b in adaptive.get("branches") or []:
                branch_bits.append(f"{b.get('display_name','NA')}[{b.get('head_kind','NA')}；{b.get('encoder_source','NA')}]")
            if branch_bits:
                lines.append("- 实际视觉分支：" + "；".join(branch_bits) + "。")
        if adaptive.get("active") and adaptive.get("uncertainty"):
            u = adaptive["uncertainty"]
            lines.append(f"- 不确定性：entropy={u.get('entropy',0):.3f}；margin={u.get('margin',0):.3f}；disagreement={u.get('disagreement',0):.3f}；OOD={u.get('ood_flag',False)}；DEFER={u.get('defer',False)}。")
        elif adaptive.get("fallback_reason"):
            lines.append(f"- 深度视觉状态：{adaptive.get('fallback_reason')}。")
    lines.append("## 证据边界")
    lines.extend("- " + x for x in result.get("prohibited_claims", []))
    return "\n".join(lines)

def _parse_optional_float(text: str, label: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label}需要填写数字或留空。") from exc


def _render_context_kpis(items: list[tuple[str, str, str]]) -> None:
    """Render project-consistent four-card context summary without metric truncation."""
    from globalhab_demo.display_locale import st
    cards = "".join(
        '<div class="kpi">'
        f'<div class="kpi-label">{html.escape(str(label))}</div>'
        f'<div class="kpi-value">{html.escape(str(value))}</div>'
        f'<div class="kpi-note">{html.escape(str(note))}</div>'
        '</div>'
        for label, value, note in items
    )
    st.markdown(f'<div class="kpi-grid kpi-4">{cards}</div>', unsafe_allow_html=True)


def _render_screening_tab(root: Any = None) -> None:
    """Render the screening subpage with adaptive routing when assets exist."""
    from globalhab_demo.display_locale import st
    from globalhab_demo.adaptive_visual import compact_status_rows

    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    from globalhab_demo.case_manager import (
        LAB_METHODS, get_case, mark_case_in_progress, next_review_case, queue_cases,
        register_lab_evidence, register_visual_evidence, case_status_code,
    )

    active_case_id = st.session_state.get("active_case_id")
    active_case = get_case(active_case_id, root_path) if active_case_id else None

    batch_notice = st.session_state.pop("_case_batch_notice", None)
    if batch_notice:
        st.success(batch_notice)

    st.markdown("### 影像识别")
    st.caption("拍照 → 自适应路由 → 视觉筛查 → 不确定性门控 / DEFER → 现场复核。")
    task_queue = queue_cases(root_path)
    if task_queue:
        with st.expander(f"待复核任务队列 · {len(task_queue)} 个", expanded=False):
            queue_rows = []
            for case in task_queue[:20]:
                research = case.get("research") or {}
                queue_rows.append({
                    "当前": "●" if str(case.get("case_id")) == str(active_case_id) else "",
                    "Case": case.get("case_id"),
                    "海区": research.get("candidate_region", "NA"),
                    "风险": research.get("risk_score", "NA"),
                    "Route/Lag": f"{research.get('route','NA')} × {research.get('lag_days','NA')}d",
                    "状态": case.get("status", "NA"),
                })
            st.dataframe(queue_rows, hide_index=True, use_container_width=True)
            nxt = next_review_case(active_case_id, root_path)
            if nxt and st.button("切换到下一个待复核任务", use_container_width=True, key="vision_next_queue_case"):
                mark_case_in_progress(str(nxt.get("case_id")), root_path)
                st.session_state["active_case_id"] = str(nxt.get("case_id"))
                st.session_state["case_sidebar_select"] = str(nxt.get("case_id"))
                st.session_state.pop("field_visual_result", None)
                st.session_state.pop("field_visual_image_bytes", None)
                st.rerun()
    if active_case:
        research = active_case.get("research") or {}
        with st.container(border=True, key="vision_case_context"):
            risk = research.get("risk_score")
            _render_context_kpis([
                ("当前Case", str(active_case.get("case_id")), "研究、现场、实验室和解释共享同一Case"),
                ("研究候选区", str(research.get("candidate_region", "NA")), "从研究验证工作区自动带入"),
                ("Route / Lag", f"{research.get('route','NA')} / {research.get('lag_days','NA')}d", "研究候选的方向与时滞"),
                ("研究风险指数", f"{float(risk):.1f}/100" if isinstance(risk, (int,float)) else str(risk or "NA"), "仅用于安排现场复核优先级"),
            ])
            st.caption("已读取研究验证生成的现场复核任务。本页新增的视觉、现场与实验室证据会按证据层级登记到同一Case，不会覆盖研究模型结果。")
    else:
        st.caption("可直接上传影像；关联研究任务请先在风险研判中创建 Case。")

    input_col, meta_col = st.container(), st.container()
    with input_col, st.container(border=True, key="vision_input_card"):
        st.markdown("### 拍照或上传")
        source_mode = st.radio("图像来源", ["现场拍照", "上传图片"], horizontal=True, key="vision_source_mode")
        image_file = None
        native_camera = "GlobalHABAndroid/1.2" in st.context.headers.get("User-Agent", "")
        if source_mode == "现场拍照" and native_camera:
            st.caption("点击下方选择文件按钮，再选择“打开系统相机拍照”。拍摄确认后，照片会返回此处供识别。")
            image_file = st.file_uploader("拍照并添加照片", type=["jpg", "jpeg", "png"], key="vision_native_camera")
        elif source_mode == "现场拍照":
            st.markdown(
                '<div class="camera-permission-cn">需要使用摄像头。若浏览器尚未授权，请在站点权限中允许摄像头访问后再拍摄。</div>',
                unsafe_allow_html=True,
            )
            image_file = st.camera_input("对准海面拍摄", key="vision_camera")
        else:
            image_file = st.file_uploader("上传JPG / JPEG / PNG", type=["jpg", "jpeg", "png"], key="vision_upload")
        vision_mode = st.selectbox(
            "视觉推理模式",
            ["自适应路由（推荐）", "多模型一致性", "EfficientNet", "ConvNeXt", "DINOv2", "安全规则基线"],
            key="vision_inference_mode",
            help="自适应模式会按水色、纹理和不确定性选择深度视觉分支；若当前运行环境无法真正执行所选分支，则返回DEFER。",
        )
        sea_surface = st.checkbox("照片主体是海面/水体，而不是天空、岸边或人物", value=True, key="vision_surface_confirm")
        st.caption("建议避开逆光，尽量让海面占画面大部分；同一点位最好从不同角度拍2–3张。图像只在当前会话中分析，不默认上传到远程大模型。")
        if image_file is not None:
            import hashlib
            fingerprint = hashlib.sha256(image_file.getvalue()).hexdigest()
            if st.session_state.get('vision_file_fingerprint') != fingerprint:
                st.session_state['vision_file_fingerprint'] = fingerprint
                st.session_state.pop('field_visual_result', None)
            st.caption(f'图片已收到：{getattr(image_file, "name", "现场照片")}。请选择下面的识别方式。')
        run = st.button("开始影像识别", type="primary", use_container_width=True, key="vision_run")
        quick = st.button("快速规则筛查（无需下载模型）", use_container_width=True, key="vision_quick_run")
        st.caption('快速筛查分析水色、纹理与照片质量，不调用深度模型，也不能识别藻种或毒素。')
        feedback = st.container()

    with meta_col, st.expander("现场信息 · 可选补充", expanded=False):
        c1, c2 = st.columns(2)
        capture_date = c1.date_input("拍摄日期", value=date.today(), key="vision_date")
        capture_time = c2.time_input("拍摄时间", value=None, key="vision_time")
        default_location = str(((active_case or {}).get("research") or {}).get("candidate_region") or "")
        location_text = st.text_input("海域 / 位置（可写站点名或大致海域）", value=default_location, placeholder="例如：南海某近岸养殖区", key="vision_location")
        water_color = st.selectbox("现场肉眼水色", ["不确定", "常规蓝/蓝绿", "绿色", "黄绿色", "红棕色", "褐色", "乳白色"], key="vision_water_color")
        odor = st.selectbox("异味", ["未观察", "无明显异味", "有明显异味", "不确定"], key="vision_odor")
        surface_signs = st.multiselect("表面现象", ["泡沫", "浮膜", "漂浮物", "鱼贝聚集", "无明显表面现象", "不确定"], key="vision_surface_signs")
        recent_heat = st.radio("近期是否明显高温", ["未知", "否", "是"], horizontal=True, key="vision_recent_heat")
        mass_mortality = st.radio("鱼贝异常/死亡", ["未观察", "未见明显异常", "观察到", "不确定"], horizontal=True, key="vision_mortality")

        with st.expander("可选仪器/现场测量", expanded=False):
            t = st.text_input("水温 °C", key="vision_temp")
            sal = st.text_input("盐度", key="vision_salinity")
            do = st.text_input("溶解氧 mg/L", key="vision_do")
            chla = st.text_input("Chl-a（请同时在备注中写单位）", key="vision_chla")
            notes = st.text_area("现场备注", max_chars=1000, key="vision_notes")

    with st.expander("高级视觉模型信息", expanded=False):
        st.dataframe(compact_status_rows(root_path), use_container_width=True, hide_index=True)
        st.caption("默认允许在首次使用时获取并缓存公共预训练视觉编码器；已有本地权重时优先使用本地文件。项目训练头存在时优先使用，否则使用内置视觉现象原型头。原型头用于现场视觉筛查，不代表经过真实HAB照片校准的藻华分类器。")

    if run or quick:
        st.session_state.pop('field_visual_result', None)
        if image_file is None:
            feedback.error("请先拍照或上传一张海面图片。")
        else:
            mode = '安全规则基线' if quick else vision_mode
            with feedback:
                status = st.status('正在读取图片…', expanded=True)
            try:
                raw = image_file.getvalue()
                image = load_image(raw)
                metadata = {
                    "capture_date": capture_date.isoformat() if capture_date else None,
                    "capture_time": capture_time.isoformat(timespec="minutes") if capture_time else None,
                    "location_text": location_text.strip(),
                    "water_color": water_color,
                    "odor": odor,
                    "surface_signs": surface_signs,
                    "recent_heat": recent_heat,
                    "mass_mortality": mass_mortality,
                    "sea_surface_confirmed": bool(sea_surface),
                    "water_temp_c": _parse_optional_float(t, "水温"),
                    "salinity": _parse_optional_float(sal, "盐度"),
                    "dissolved_oxygen_mg_l": _parse_optional_float(do, "溶解氧"),
                    "chlorophyll_a": _parse_optional_float(chla, "Chl-a"),
                    "notes": notes.strip(),
                    "case_id": active_case_id,
                }
                status.write('图片已解析，正在分析质量、水色和纹理。')
                if mode != '安全规则基线':
                    status.write('正在加载服务器视觉模型并推理。首次使用可能需要下载权重；请保持页面打开。')
                result = make_result(image, metadata, root=root_path, requested_mode=mode)
                st.session_state["field_visual_result"] = result
                st.session_state["field_visual_image_bytes"] = raw
                st.session_state["field_visual_image_name"] = getattr(image_file, "name", "field_capture.jpg") or "field_capture.jpg"
                status.update(label='本次分析已完成', state='complete', expanded=False)
                feedback.success(f"结果：{result['effective_visual_category']}；复核优先级：{result['screening_priority']}。详细结果在下方。")
            except Exception as exc:
                status.update(label='本次分析未完成', state='error', expanded=True)
                feedback.error(str(exc) if isinstance(exc, ValueError) else '服务器未能完成本次识别。可以先使用快速规则筛查，再检查视觉模型状态。')
                with feedback.expander('查看错误信息'):
                    st.code(f'{type(exc).__name__}: {exc}')

    result = st.session_state.get("field_visual_result")
    if not result:
        st.caption("完成一次甄别后，本页会显示图像质量、主要视觉异常、混淆因素、现场复核优先级和后续采样建议。")
        return

    q = result["quality"]
    v = result["visual"]
    raw = st.session_state.get("field_visual_image_bytes")
    st.markdown("### 甄别结果")
    preview_col, result_col = st.container(), st.container()
    with preview_col, st.container(border=True, key="vision_preview_card"):
        if raw:
            st.image(raw, caption="本次分析照片", use_container_width=True)
        st.caption(f"分析后端：{result['backend']}")
        if q["warnings"]:
            for w in q["warnings"]:
                st.warning(w)
        else:
            st.success("未发现明显的图像质量限制。")

    with result_col, st.container(border=True, key="vision_result_card"):
        effective_category = result.get("effective_visual_category", v["category"])
        effective_score = float(result.get("effective_visual_anomaly_score", v["visual_anomaly_score"]))
        k1, k2, k3 = st.columns(3)
        k1.metric("主要视觉类型", effective_category)
        k2.metric("复核优先级", result["screening_priority"])
        k3.metric("图像质量", f"{q['quality_score']:.0%}")
        st.caption(result["priority_reason"])
        st.progress(min(1.0, effective_score), text=f"视觉异常特征强度 {effective_score:.0%}（不是HAB概率）")
        st.markdown("**视觉特征摘要**")
        st.write(" · ".join(v["feature_notes"]))
        st.markdown("**可能混淆因素**")
        for x in v["possible_confounders"]:
            st.write("- " + x)
        if result["field_context_flags"]:
            st.markdown("**现场辅助线索**")
            for x in result["field_context_flags"]:
                st.write("- " + x)

    adaptive = result.get("adaptive_visual") or {}
    st.markdown("### 04 · 自适应路由与不确定性")
    with st.container(border=True, key="vision_adaptive_card"):
        route = adaptive.get("route") or {}
        a1, a2, a3 = st.columns(3)
        a1.metric("路由类型", route.get("route_family", "规则基线"))
        a2.metric("已激活分支", str(len(adaptive.get("branches") or [])))
        a3.metric("深度视觉状态", "已执行" if adaptive.get("active") else "DEFER")
        st.write("**路由理由：** " + str(route.get("reason", adaptive.get("fallback_reason", "NA"))))
        if adaptive.get("active"):
            rows = []
            for b in adaptive.get("branches", []):
                rows.append({
                    "分支": b.get("display_name"),
                    "视觉类别": b.get("predicted_class"),
                    "筛查头": b.get("head_kind", "NA"),
                    "融合置信度": round(float(b.get("confidence", 0)), 3),
                    "entropy": round(float(b.get("entropy", 0)), 3),
                    "margin": round(float(b.get("margin", 0)), 3),
                    "OOD": bool(b.get("ood_flag", False)),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            u = adaptive.get("uncertainty") or {}
            st.caption(f"融合不确定性：entropy={u.get('entropy',0):.3f} · margin={u.get('margin',0):.3f} · disagreement={u.get('disagreement',0):.3f} · OOD={u.get('ood_flag',False)}。融合置信度是视觉现象筛查分数，不是HAB发生概率。")
            if u.get("defer"):
                st.warning("不确定性门控触发 DEFER：" + "；".join(u.get("reasons") or []))
        else:
            st.warning(adaptive.get("fallback_reason") or "当前没有深度视觉分支成功执行，因此自适应模式返回DEFER。")
            if adaptive.get("branch_errors"):
                with st.expander("查看视觉引擎诊断", expanded=False):
                    for err in adaptive.get("branch_errors"):
                        st.code(err)
            st.caption("透明规则基线仍会显示颜色/纹理辅助线索，但不会在自适应模式中冒充深度模型的最终结果。")

    st.markdown("### 05 · 下一步复核")
    follow_col, boundary_col = st.container(), st.container()
    with follow_col, st.container(border=True, key="vision_follow_card"):
        st.markdown("#### 推荐补充证据")
        for x in result["recommended_follow_up"]:
            st.write("- " + x)
    with boundary_col, st.container(border=True, key="vision_boundary_card"):
        st.markdown("#### 本页不能据此声称")
        for x in result["prohibited_claims"]:
            st.write("- " + x)

    st.markdown("### 06 · Case联动与确认")
    if active_case:
        with st.container(border=True, key="vision_case_actions"):
            st.markdown("#### 登记现场视觉证据")
            st.caption("视觉筛查登记为B级现场证据，不会把研究候选自动改成真实HAB事件。照片可同时保存到“我的影像数据”，但默认不进入监督训练。")
            def _register_current_visual() -> None:
                metadata = result.get("field_metadata") or {}
                raw_now = st.session_state.get("field_visual_image_bytes")
                sample_id = None
                if raw_now:
                    from globalhab_demo.visual_learning import save_image_sample
                    sample_id, _ = save_image_sample(
                        raw_now, st.session_state.get("field_visual_image_name"), metadata,
                        label="不确定", evidence_level="仅肉眼判断", include_in_training=False,
                        root=root_path, screening_result=result, source="Case现场视觉筛查",
                    )
                register_visual_evidence(active_case_id, result, metadata, sample_id=sample_id, root=root_path)

            ca1, ca2 = st.columns(2)
            if ca1.button("登记到当前研究Case", type="primary", use_container_width=True, key="vision_register_case"):
                try:
                    _register_current_visual()
                    st.success("现场视觉证据已登记到当前Case；照片已保存到影像库但不会自动进入训练集。")
                except Exception as exc:
                    st.error(str(exc))
            if ca2.button("登记并处理下一个", use_container_width=True, key="vision_register_next_case"):
                try:
                    _register_current_visual()
                    nxt = next_review_case(active_case_id, root_path)
                    if nxt:
                        next_id = str(nxt.get("case_id"))
                        mark_case_in_progress(next_id, root_path)
                        st.session_state["active_case_id"] = next_id
                        st.session_state["case_sidebar_select"] = next_id
                        st.session_state.pop("field_visual_result", None)
                        st.session_state.pop("field_visual_image_bytes", None)
                        st.session_state["_case_batch_notice"] = "当前Case已保存，已切换到下一个待复核任务。"
                        st.rerun()
                    else:
                        st.success("当前Case已保存；任务队列中没有其他待复核Case。")
                except Exception as exc:
                    st.error(str(exc))
            if st.button("返回研究验证查看证据链", use_container_width=True, key="vision_back_research"):
                st.session_state["_workspace_jump"] = "研究验证"
                st.rerun()

            st.markdown("#### 专业 / 实验室确认")
            l1, l2 = st.columns(2)
            method = l1.selectbox("确认方式", LAB_METHODS, key="vision_lab_method")
            confirmed_label = l2.selectbox("确认后的视觉现象标签", list(VISUAL_CLASSES), key="vision_lab_label")
            conclusion = st.text_input("确认结论", placeholder="例如：qPCR检出目标藻；显微镜未见目标藻；毒素未检出", key="vision_lab_conclusion")
            value_text = st.text_input("检测值/方法信息（可选）", placeholder="例如：Ct=24；细胞丰度=...；毒素=...", key="vision_lab_value")
            lab_notes = st.text_area("确认备注（可选）", max_chars=1000, key="vision_lab_notes")
            add_train = st.checkbox("将该确认照片写入视觉训练库并进入后续训练", value=True, key="vision_lab_to_train")
            if st.button("登记确认并更新证据链", use_container_width=True, key="vision_register_lab"):
                if not conclusion.strip():
                    st.warning("请填写确认结论。")
                else:
                    try:
                        raw_now = st.session_state.get("field_visual_image_bytes")
                        metadata = result.get("field_metadata") or {}
                        sample_id = None
                        if raw_now:
                            from globalhab_demo.visual_learning import save_image_sample
                            sample_id, _ = save_image_sample(
                                raw_now, st.session_state.get("field_visual_image_name"), metadata,
                                label=confirmed_label, evidence_level=method,
                                include_in_training=bool(add_train and confirmed_label != "不确定"),
                                root=root_path, screening_result=result, source=f"Case确认·{method}",
                            )
                        register_lab_evidence(
                            active_case_id, method, conclusion, confirmed_label, value_text, lab_notes,
                            sample_id=sample_id, root=root_path,
                        )
                        st.success("确认结果已进入研究证据链；如勾选训练，照片已进入视觉训练库。")
                    except Exception as exc:
                        st.error(str(exc))
    else:
        st.caption("未激活研究Case，因此本次甄别只保留在当前会话/影像库中。")

    d1, d2 = st.columns(2)
    d1.download_button(
        "下载本次甄别 JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="GlobalHAB_field_visual_screening.json",
        mime="application/json",
        use_container_width=True,
    )
    if d2.button("送入大模型综合解读", use_container_width=True, key="vision_to_llm"):
        st.session_state["_workspace_jump"] = "模型解读"
        st.session_state["_llm_source_jump"] = "当前完整Case（推荐）" if active_case else "最近一次影像识别"
        st.rerun()

    st.caption("隐私说明：照片本身不会因为点击“大模型综合解读”而发送给远程服务；默认仅发送结构化Case摘要，并且仍需在大模型工作区显式勾选授权。")


def render(root: Any = None) -> None:
    """Render the continuous-learning field visual workspace."""
    from globalhab_demo.display_locale import st
    from globalhab_demo.visual_learning import (
        ensure_visual_learning_store,
        render_library_tab,
        render_training_tab,
    )

    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    ensure_visual_learning_store(root_path)
    from globalhab_demo.ui_system import render_workspace_header
    render_workspace_header(
        "影像识别",
        "上传影像 · 筛查复核 · 保存证据",
        kicker="Field evidence",
    )
    with st.container(key="vision_workspace_tabs"):
        tab_screen, tab_example, tab_video, tab_data, tab_model = st.tabs([
            "影像识别",
            "示例体验",
            "视频筛查",
            "我的影像数据",
            "模型训练与版本",
        ])
    with tab_screen:
        _render_screening_tab(root_path)
    with tab_example:
        from globalhab_demo.visual_examples import render_examples
        render_examples(root_path)
    with tab_video:
        from globalhab_demo.visual_examples import render_video
        render_video(root_path)
    with tab_data:
        render_library_tab(root_path)
    with tab_model:
        render_training_tab(root_path)
