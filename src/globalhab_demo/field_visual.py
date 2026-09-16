"""Field visual screening workspace for GlobalHAB-Agent.

This module provides a deliberately bounded *screening* baseline for sea-surface
photos. It is not a trained HAB species classifier. The default backend uses
simple, transparent colour/texture/quality features plus user-entered field
context to decide whether a photo deserves higher-priority follow-up sampling.

The backend is intentionally replaceable: a calibrated project-specific vision
model can implement ``VisualScreeningBackend`` later without changing the UI or
result contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, time
from io import BytesIO
import json
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
) -> dict[str, Any]:
    quality, features = backend.analyse(image)
    context_flags, n_flags = _context_flags(metadata)

    if not metadata.get("sea_surface_confirmed", True):
        priority = "DEFER / 需重拍"
        reason = "用户未确认海面是照片主体。"
    elif not quality.suitable:
        priority = "DEFER / 需重拍"
        reason = "图像质量不足以支持稳定的水色/表层特征甄别。"
    else:
        score = features.visual_anomaly_score
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
    if features.category in {"红棕色水体异常", "高浑浊/泥沙样"}:
        follow_up.append("优先补充浊度/悬浮颗粒或河口径流信息，以排除泥沙混淆。")
    if features.category == "表层浮沫/漂浮物样" or quality.glare_fraction > 0.08:
        follow_up.append("换一个避开太阳反光的角度复拍，以区分泡沫、浪花和表层聚集。")

    return {
        "schema": "globalhab-field-visual-screening-v1",
        "backend": backend.name,
        "scientific_role": "现场影像辅助甄别；不是藻种/毒素确诊器",
        "quality": asdict(quality),
        "visual": asdict(features),
        "field_metadata": metadata,
        "field_context_flags": context_flags,
        "screening_priority": priority,
        "priority_reason": reason,
        "recommended_follow_up": follow_up,
        "allowed_claims": [
            "可描述照片中的水色、浑浊、亮白表层和其他视觉异常线索。",
            "可将视觉异常作为是否值得进一步采样复核的低成本现场证据。",
        ],
        "prohibited_claims": [
            "不能仅凭普通海面照片确诊具体藻种。",
            "不能仅凭普通海面照片判断是否产毒或毒素浓度。",
            "不能把本页复核优先级解释为HAB发生概率、监管阈值或业务预警。",
            "不能用视觉阴性排除肉眼不可见但具有生态风险的HAB。",
        ],
    }


def result_summary(result: dict[str, Any] | None) -> str:
    if not result:
        return "## 最近一次现场影像甄别\n- 当前会话还没有完成现场影像甄别。"
    q = result.get("quality", {})
    v = result.get("visual", {})
    m = result.get("field_metadata", {})
    lines = [
        "## 最近一次现场影像甄别",
        f"- 方法：{result.get('backend','NA')}。",
        f"- 科学角色：{result.get('scientific_role','NA')}。",
        f"- 主要视觉类型：{v.get('category','NA')}；视觉异常分数={v.get('visual_anomaly_score',0):.3f}（启发式特征强度，不是HAB概率）。",
        f"- 复核优先级：{result.get('screening_priority','NA')}。",
        f"- 图像质量分数={q.get('quality_score',0):.3f}；适合甄别={q.get('suitable',False)}；尺寸={q.get('width','NA')}×{q.get('height','NA')}。",
        f"- 现场位置/海域：{m.get('location_text') or '未填写'}；拍摄日期={m.get('capture_date') or '未填写'}；水色={m.get('water_color') or '未填写'}。",
        f"- 现场辅助线索：{'; '.join(result.get('field_context_flags') or ['无明确辅助线索'])}。",
        f"- 主要混淆因素：{'; '.join(v.get('possible_confounders') or [])}。",
        f"- 建议复核：{'; '.join(result.get('recommended_follow_up') or [])}",
        "## 证据边界",
    ]
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


def render(root: Any = None) -> None:
    """Render the Streamlit workspace. ``root`` is accepted for UI symmetry."""
    import streamlit as st

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow" style="color:#b5d9dc">FIELD VISUAL SCREENING</div>
          <h1>现场影像甄别</h1>
          <p class="tagline">手机拍摄海面照片，先做低成本视觉初筛，再决定是否值得进一步采样复核</p>
          <p class="value">默认使用透明的颜色/纹理启发式基线，不冒充已经训练好的藻华分类器；输出只表示视觉异常与复核优先级，不能替代藻种、毒素或实验室鉴定。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "科学边界：普通手机照片可以帮助发现水色、浑浊、泡沫/漂浮物等可疑表征，但不能可靠确诊Karenia、Alexandrium、Pseudo-nitzschia等具体藻种，也不能判断毒素浓度。肉眼看起来正常也不能排除HAB。"
    )

    input_col, meta_col = st.columns([1.05, 1], gap="large")
    with input_col, st.container(border=True, key="vision_input_card"):
        st.markdown("### 01 · 拍照或上传")
        source_mode = st.radio("图像来源", ["现场拍照", "上传图片"], horizontal=True, key="vision_source_mode")
        image_file = None
        if source_mode == "现场拍照":
            image_file = st.camera_input("对准海面拍摄", key="vision_camera")
        else:
            image_file = st.file_uploader("上传JPG / JPEG / PNG", type=["jpg", "jpeg", "png"], key="vision_upload")
        sea_surface = st.checkbox("照片主体是海面/水体，而不是天空、岸边或人物", value=True, key="vision_surface_confirm")
        st.caption("建议避开逆光，尽量让海面占画面大部分；同一点位最好从不同角度拍2–3张。图像只在当前会话中分析，不默认上传到远程大模型。")

    with meta_col, st.container(border=True, key="vision_meta_card"):
        st.markdown("### 02 · 现场信息")
        c1, c2 = st.columns(2)
        capture_date = c1.date_input("拍摄日期", value=date.today(), key="vision_date")
        capture_time = c2.time_input("拍摄时间", value=None, key="vision_time")
        location_text = st.text_input("海域 / 位置（可写站点名或大致海域）", placeholder="例如：南海某近岸养殖区", key="vision_location")
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

    run = st.button("开始现场影像甄别", type="primary", use_container_width=True, key="vision_run")
    if run:
        if image_file is None:
            st.error("请先拍照或上传一张海面图片。")
        else:
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
                }
                result = make_result(image, metadata)
                st.session_state["field_visual_result"] = result
                st.session_state["field_visual_image_bytes"] = raw
            except ValueError as exc:
                st.error(str(exc))

    result = st.session_state.get("field_visual_result")
    if not result:
        st.caption("完成一次甄别后，本页会显示图像质量、主要视觉异常、混淆因素、现场复核优先级和后续采样建议。")
        return

    q = result["quality"]
    v = result["visual"]
    raw = st.session_state.get("field_visual_image_bytes")
    st.markdown("### 03 · 甄别结果")
    preview_col, result_col = st.columns([1.05, 1.2], gap="large")
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
        k1, k2, k3 = st.columns(3)
        k1.metric("主要视觉类型", v["category"])
        k2.metric("复核优先级", result["screening_priority"])
        k3.metric("图像质量", f"{q['quality_score']:.0%}")
        st.caption(result["priority_reason"])
        st.progress(min(1.0, float(v["visual_anomaly_score"])), text=f"视觉异常特征强度 {v['visual_anomaly_score']:.0%}（不是HAB概率）")
        st.markdown("**视觉特征摘要**")
        st.write(" · ".join(v["feature_notes"]))
        st.markdown("**可能混淆因素**")
        for x in v["possible_confounders"]:
            st.write("- " + x)
        if result["field_context_flags"]:
            st.markdown("**现场辅助线索**")
            for x in result["field_context_flags"]:
                st.write("- " + x)

    st.markdown("### 04 · 下一步复核")
    follow_col, boundary_col = st.columns(2, gap="large")
    with follow_col, st.container(border=True, key="vision_follow_card"):
        st.markdown("#### 推荐补充证据")
        for x in result["recommended_follow_up"]:
            st.write("- " + x)
    with boundary_col, st.container(border=True, key="vision_boundary_card"):
        st.markdown("#### 本页不能据此声称")
        for x in result["prohibited_claims"]:
            st.write("- " + x)

    d1, d2 = st.columns(2)
    d1.download_button(
        "下载本次甄别 JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="GlobalHAB_field_visual_screening.json",
        mime="application/json",
        use_container_width=True,
    )
    if d2.button("送入大模型结果解读", use_container_width=True, key="vision_to_llm"):
        st.session_state["_workspace_jump"] = "大模型结果解读"
        st.session_state["_llm_source_jump"] = "最近一次现场影像甄别"
        st.rerun()

    st.caption("隐私说明：照片本身不会因为点击“大模型结果解读”而发送给远程服务；默认仅发送本页生成的结构化文字摘要，并且仍需在大模型工作区显式勾选授权。")
