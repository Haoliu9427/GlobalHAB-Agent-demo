"""Adaptive visual routing for the GlobalHAB-Agent field screening workspace.

The module adds a *framework* for DINOv2 / ConvNeXt / EfficientNet feature
extraction, a lightweight linear classifier head, field-metadata fusion and
uncertainty-aware DEFER.  It is intentionally offline-safe:

- the core Streamlit app does not require torch/torchvision/transformers;
- no pretrained weights or trained HAB heads are fabricated or silently
  downloaded;
- deep branches become active only when a compatible encoder and a trained
  ``.npz`` head bundle are available;
- otherwise the field workspace transparently falls back to the existing
  interpretable colour/texture baseline.

This separation lets the project demonstrate the adaptive routing architecture
without pretending that an untrained ImageNet/foundation encoder is already a
validated HAB classifier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import importlib.util
import math
import os
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps


VISUAL_CLASSES = (
    "正常/未见明显异常",
    "绿色水体异常",
    "红棕色水体异常",
    "高浑浊/泥沙样",
    "表层浮沫/漂浮物样",
    "不确定",
)

META_FEATURE_NAMES = (
    "odor_flag",
    "foam_flag",
    "film_flag",
    "debris_flag",
    "recent_heat_flag",
    "abnormal_color_flag",
    "mortality_flag",
    "water_temp_scaled",
    "salinity_scaled",
    "do_scaled",
    "chla_log_scaled",
    "instrument_missing_fraction",
)

MODEL_NAMES = ("efficientnet", "convnext", "dinov2")
DISPLAY_NAMES = {
    "efficientnet": "EfficientNet-B0",
    "convnext": "ConvNeXt-Tiny",
    "dinov2": "DINOv2",
}


@dataclass
class BackboneStatus:
    name: str
    display_name: str
    encoder_available: bool
    head_available: bool
    ready: bool
    encoder_source: str
    head_path: str | None
    note: str


@dataclass
class RouteDecision:
    requested_mode: str
    route_family: str
    preferred: list[str]
    selected: list[str]
    fallback_to_heuristic: bool
    reason: str


@dataclass
class BranchPrediction:
    name: str
    display_name: str
    class_probabilities: dict[str, float]
    predicted_class: str
    confidence: float
    entropy: float
    margin: float
    ood_score: float | None
    ood_threshold: float | None
    ood_flag: bool
    feature_dim: int


@dataclass
class UncertaintyState:
    entropy: float
    margin: float
    disagreement: float
    ood_flag: bool
    defer: bool
    reasons: list[str]


def _optional_import_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _allow_downloads() -> bool:
    return os.getenv("GLOBALHAB_ALLOW_MODEL_DOWNLOADS", "0").strip().lower() in {"1", "true", "yes", "on"}


def _root_path(root: Any | None) -> Path:
    if root is None:
        return Path(__file__).resolve().parents[2]
    return Path(root)


def _model_asset_paths(root: Path) -> dict[str, dict[str, Path | None]]:
    model_root = root / "vision_models"
    dino_env = os.getenv("GLOBALHAB_DINOV2_MODEL_DIR", "").strip()
    conv_env = os.getenv("GLOBALHAB_CONVNEXT_CHECKPOINT", "").strip()
    eff_env = os.getenv("GLOBALHAB_EFFICIENTNET_CHECKPOINT", "").strip()
    return {
        "dinov2": {
            "encoder": Path(dino_env) if dino_env else model_root / "dinov2_local",
            "head": model_root / "heads" / "dinov2.npz",
        },
        "convnext": {
            "encoder": Path(conv_env) if conv_env else model_root / "convnext_tiny.pth",
            "head": model_root / "heads" / "convnext.npz",
        },
        "efficientnet": {
            "encoder": Path(eff_env) if eff_env else model_root / "efficientnet_b0.pth",
            "head": model_root / "heads" / "efficientnet.npz",
        },
    }


def get_backbone_status(root: Any | None = None) -> dict[str, BackboneStatus]:
    root_path = _root_path(root)
    paths = _model_asset_paths(root_path)
    torch_ok = _optional_import_available("torch")
    torchvision_ok = _optional_import_available("torchvision")
    transformers_ok = _optional_import_available("transformers")
    allow_download = _allow_downloads()
    statuses: dict[str, BackboneStatus] = {}

    for name in MODEL_NAMES:
        encoder_path = paths[name]["encoder"]
        head_path = paths[name]["head"]
        head_available = bool(head_path and Path(head_path).is_file())
        if name == "dinov2":
            local_encoder = bool(encoder_path and Path(encoder_path).exists() and any(Path(encoder_path).iterdir())) if encoder_path and Path(encoder_path).is_dir() else False
            encoder_available = torch_ok and transformers_ok and (local_encoder or allow_download)
            source = str(encoder_path) if local_encoder else ("Hugging Face: facebook/dinov2-base (download enabled)" if allow_download else "not configured")
            dependency_note = "需要torch + transformers；默认只读取本地DINOv2目录。"
        else:
            local_encoder = bool(encoder_path and Path(encoder_path).is_file())
            encoder_available = torch_ok and torchvision_ok and (local_encoder or allow_download)
            source = str(encoder_path) if local_encoder else (f"torchvision {DISPLAY_NAMES[name]} pretrained weights (download enabled)" if allow_download else "not configured")
            dependency_note = "需要torch + torchvision；默认只读取本地checkpoint。"
        ready = encoder_available and head_available
        if ready:
            note = "编码器与训练好的轻量分类头均可用，可参与自适应路由。"
        elif encoder_available and not head_available:
            note = "编码器可用，但缺少项目标注数据训练得到的轻量分类头，因此不会把通用特征伪装成HAB分类结果。"
        elif not encoder_available and head_available:
            note = "检测到分类头，但编码器不可用；请安装可选视觉依赖并配置本地权重。"
        else:
            note = dependency_note + " 未配置时自动回退到可解释规则基线。"
        statuses[name] = BackboneStatus(
            name=name,
            display_name=DISPLAY_NAMES[name],
            encoder_available=encoder_available,
            head_available=head_available,
            ready=ready,
            encoder_source=source,
            head_path=str(head_path) if head_available else None,
            note=note,
        )
    return statuses


def metadata_vector(metadata: dict[str, Any]) -> np.ndarray:
    """Encode field metadata into a fixed, bounded numeric vector.

    The vector is deliberately small and transparent. Missing optional
    instrument values are represented both by zero-filled standardized values
    and by a missing-fraction feature.
    """
    surface = set(metadata.get("surface_signs") or [])
    water_color = str(metadata.get("water_color") or "")

    def fnum(key: str) -> float | None:
        try:
            value = metadata.get(key)
            if value is None or value == "":
                return None
            v = float(value)
            return v if math.isfinite(v) else None
        except Exception:
            return None

    temp = fnum("water_temp_c")
    sal = fnum("salinity")
    do = fnum("dissolved_oxygen_mg_l")
    chla = fnum("chlorophyll_a")
    instrument = [temp, sal, do, chla]
    missing = sum(v is None for v in instrument) / len(instrument)

    vec = np.asarray([
        1.0 if metadata.get("odor") == "有明显异味" else 0.0,
        1.0 if "泡沫" in surface else 0.0,
        1.0 if "浮膜" in surface else 0.0,
        1.0 if "漂浮物" in surface else 0.0,
        1.0 if metadata.get("recent_heat") == "是" else 0.0,
        1.0 if water_color in {"绿色", "黄绿色", "红棕色", "褐色", "乳白色"} else 0.0,
        1.0 if metadata.get("mass_mortality") == "观察到" else 0.0,
        0.0 if temp is None else float(np.clip((temp - 20.0) / 15.0, -2.0, 2.0)),
        0.0 if sal is None else float(np.clip((sal - 30.0) / 10.0, -3.0, 3.0)),
        0.0 if do is None else float(np.clip((do - 6.0) / 4.0, -2.0, 2.0)),
        0.0 if chla is None else float(np.clip(np.log1p(max(chla, 0.0)) / 4.0, 0.0, 3.0)),
        float(missing),
    ], dtype=np.float32)
    return vec


def _feature_value(obj: Any, key: str, default: float = 0.0) -> float:
    if isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        value = getattr(obj, key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _feature_text(obj: Any, key: str, default: str = "") -> str:
    if isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        value = getattr(obj, key, default)
    return str(value)


def plan_route(
    quality: Any,
    visual: Any,
    statuses: dict[str, BackboneStatus] | None = None,
    requested_mode: str = "自适应路由",
) -> RouteDecision:
    """Choose a backbone family using quality + scene cues, then availability."""
    statuses = statuses or {}
    ready = {k for k, v in statuses.items() if v.ready}
    qscore = _feature_value(quality, "quality_score")
    suitable = bool(quality.get("suitable", False)) if isinstance(quality, dict) else bool(getattr(quality, "suitable", False))
    category = _feature_text(visual, "category")
    anomaly = _feature_value(visual, "visual_anomaly_score")
    cat_score = _feature_value(visual, "category_score")
    foam = _feature_value(visual, "white_low_saturation_fraction")
    turbid = _feature_value(visual, "turbid_fraction")
    saturation = _feature_value(visual, "mean_saturation")

    mode_norm = requested_mode.replace("（推荐）", "").strip()
    if not suitable or qscore < 0.45:
        return RouteDecision(requested_mode, "quality_defer", [], [], True, "图像质量门控未通过，深度视觉分支不应继续给出强制分类。")

    explicit = {
        "DINOv2": "dinov2",
        "ConvNeXt": "convnext",
        "EfficientNet": "efficientnet",
    }
    if mode_norm in {"安全规则基线", "规则基线"}:
        return RouteDecision(requested_mode, "heuristic_only", [], [], True, "用户选择仅使用透明规则基线。")
    if mode_norm in explicit:
        preferred = [explicit[mode_norm]]
        selected = [x for x in preferred if x in ready]
        return RouteDecision(
            requested_mode,
            "manual",
            preferred,
            selected,
            not bool(selected),
            "指定视觉分支可用。" if selected else "指定分支缺少本地编码器或训练好的轻量分类头，安全回退到规则基线。",
        )
    if mode_norm == "多模型一致性":
        preferred = ["efficientnet", "convnext", "dinov2"]
        selected = [x for x in preferred if x in ready]
        return RouteDecision(
            requested_mode,
            "ensemble",
            preferred,
            selected,
            not bool(selected),
            "使用所有可用且已校准的视觉分支做一致性检查。" if selected else "没有同时具备编码器与训练头的深度视觉分支，回退规则基线。",
        )

    # Adaptive routing. EfficientNet is the light path for clear colour signals;
    # ConvNeXt handles texture/foam/turbidity; DINOv2 is the generalist path for
    # ambiguous or scene-complex inputs. Ambiguous cases may request two routes.
    if category in {"表层浮沫/漂浮物样", "高浑浊/泥沙样"} or foam > 0.08 or turbid > 0.18:
        preferred = ["convnext", "dinov2", "efficientnet"]
        family = "texture_route"
        reason = "表层纹理/浮沫/浑浊线索较强，优先ConvNeXt；DINOv2作为复杂场景复核。"
    elif category in {"绿色水体异常", "红棕色水体异常"} and anomaly >= 0.28 and cat_score >= 0.34:
        preferred = ["efficientnet", "dinov2", "convnext"]
        family = "colour_route"
        reason = "颜色异常较明确，优先轻量EfficientNet；若证据边界接近则用DINOv2复核。"
    else:
        preferred = ["dinov2", "efficientnet", "convnext"]
        family = "generalist_route"
        reason = "视觉类别不明确或场景更复杂，优先DINOv2通用表征。"

    selected: list[str] = []
    if preferred[0] in ready:
        selected.append(preferred[0])
    else:
        selected.extend([x for x in preferred[1:] if x in ready][:1])

    ambiguous = category == "不确定" or (0.16 <= anomaly <= 0.48) or cat_score < 0.42 or saturation < 0.08
    if ambiguous:
        for x in preferred:
            if x in ready and x not in selected:
                selected.append(x)
                break
        reason += " 当前输入接近不确定边界，若有第二个已校准分支则增加一致性复核。"

    return RouteDecision(
        requested_mode=requested_mode,
        route_family=family,
        preferred=preferred,
        selected=selected,
        fallback_to_heuristic=not bool(selected),
        reason=reason if selected else reason + " 当前包未配置可用深度分支，安全回退到透明规则基线。",
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / max(float(exp.sum()), 1e-12)


class LinearFusionHead:
    """Safe NumPy-only multinomial linear head loaded from a .npz bundle."""

    def __init__(self, path: Path):
        bundle = np.load(path, allow_pickle=False)
        self.path = path
        self.coef = np.asarray(bundle["coef"], dtype=np.float32)
        self.intercept = np.asarray(bundle["intercept"], dtype=np.float32)
        self.mean = np.asarray(bundle["mean"], dtype=np.float32)
        self.scale = np.asarray(bundle["scale"], dtype=np.float32)
        raw_classes = bundle["classes"]
        self.classes = [str(x) for x in raw_classes.tolist()]
        self.ood_threshold = float(bundle["ood_threshold"]) if "ood_threshold" in bundle.files else None
        if self.coef.ndim != 2 or self.intercept.ndim != 1:
            raise ValueError(f"Invalid head bundle: {path}")
        if self.coef.shape[0] != len(self.classes) or self.intercept.shape[0] != len(self.classes):
            raise ValueError(f"Class dimension mismatch in {path}")
        if self.coef.shape[1] != self.mean.shape[0] or self.mean.shape != self.scale.shape:
            raise ValueError(f"Feature dimension mismatch in {path}")

    def predict(self, embedding: np.ndarray, metadata: np.ndarray) -> tuple[np.ndarray, float | None]:
        x = np.concatenate([np.asarray(embedding, dtype=np.float32).reshape(-1), metadata.reshape(-1)])
        if x.shape[0] != self.mean.shape[0]:
            raise ValueError(f"Head expects {self.mean.shape[0]} features but received {x.shape[0]}")
        scale = np.where(np.abs(self.scale) < 1e-6, 1.0, self.scale)
        z = (x - self.mean) / scale
        logits = self.coef @ z + self.intercept
        probs = _softmax(logits)
        ood_score = float(np.sqrt(np.mean(np.square(z)))) if self.ood_threshold is not None else None
        return probs, ood_score


_ENCODER_CACHE: dict[tuple[str, str, bool], Any] = {}


def _load_state_dict(path: Path) -> dict[str, Any]:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    clean = {}
    for key, value in obj.items():
        k = str(key)
        for prefix in ("module.", "model.", "backbone."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        clean[k] = value
    return clean


def _tv_preprocess(image: Image.Image):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)


def _load_encoder(name: str, root: Path):
    allow_download = _allow_downloads()
    paths = _model_asset_paths(root)
    source = paths[name]["encoder"]
    cache_key = (name, str(source), allow_download)
    if cache_key in _ENCODER_CACHE:
        return _ENCODER_CACHE[cache_key]

    if name == "dinov2":
        import torch
        from transformers import AutoImageProcessor, AutoModel

        local_dir = Path(source) if source else None
        model_ref = str(local_dir) if local_dir and local_dir.is_dir() and any(local_dir.iterdir()) else "facebook/dinov2-base"
        local_only = not allow_download
        processor = AutoImageProcessor.from_pretrained(model_ref, local_files_only=local_only)
        model = AutoModel.from_pretrained(model_ref, local_files_only=local_only)
        model.eval()

        def encode(image: Image.Image) -> np.ndarray:
            inputs = processor(images=ImageOps.exif_transpose(image).convert("RGB"), return_tensors="pt")
            with torch.inference_mode():
                out = model(**inputs)
            if getattr(out, "pooler_output", None) is not None:
                feat = out.pooler_output[0]
            else:
                feat = out.last_hidden_state[0, 0]
            return feat.detach().cpu().numpy().astype(np.float32)

        encoder = encode
    elif name == "convnext":
        import torch
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        ckpt = Path(source) if source else None
        if ckpt and ckpt.is_file():
            model = convnext_tiny(weights=None)
            model.load_state_dict(_load_state_dict(ckpt), strict=False)
        else:
            model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if allow_download else None)
            if not allow_download:
                raise FileNotFoundError("ConvNeXt local checkpoint is not configured")
        model.eval()

        def encode(image: Image.Image) -> np.ndarray:
            x = _tv_preprocess(image)
            with torch.inference_mode():
                y = model.features(x)
                y = model.avgpool(y)
                feat = torch.flatten(y, 1)[0]
            return feat.detach().cpu().numpy().astype(np.float32)

        encoder = encode
    elif name == "efficientnet":
        import torch
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        ckpt = Path(source) if source else None
        if ckpt and ckpt.is_file():
            model = efficientnet_b0(weights=None)
            model.load_state_dict(_load_state_dict(ckpt), strict=False)
        else:
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if allow_download else None)
            if not allow_download:
                raise FileNotFoundError("EfficientNet local checkpoint is not configured")
        model.eval()

        def encode(image: Image.Image) -> np.ndarray:
            x = _tv_preprocess(image)
            with torch.inference_mode():
                y = model.features(x)
                y = model.avgpool(y)
                feat = torch.flatten(y, 1)[0]
            return feat.detach().cpu().numpy().astype(np.float32)

        encoder = encode
    else:
        raise ValueError(f"Unknown backbone: {name}")

    _ENCODER_CACHE[cache_key] = encoder
    return encoder


def _entropy(probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


def _margin(probs: np.ndarray) -> float:
    p = np.sort(np.asarray(probs, dtype=np.float64))[::-1]
    return float(p[0] - p[1]) if len(p) > 1 else float(p[0])


def run_adaptive_route(
    image: Image.Image,
    metadata: dict[str, Any],
    quality: Any,
    visual: Any,
    root: Any | None = None,
    requested_mode: str = "自适应路由",
) -> dict[str, Any]:
    """Run routed deep branches when calibrated assets are available.

    Returns a JSON-serialisable dictionary. Errors in optional branches are
    captured and reported; they never prevent the base workspace from running.
    """
    root_path = _root_path(root)
    statuses = get_backbone_status(root_path)
    route = plan_route(quality, visual, statuses, requested_mode=requested_mode)
    meta = metadata_vector(metadata)
    branches: list[BranchPrediction] = []
    branch_errors: list[str] = []

    for name in route.selected:
        try:
            encoder = _load_encoder(name, root_path)
            embedding = encoder(image)
            head_path = _model_asset_paths(root_path)[name]["head"]
            if head_path is None:
                raise FileNotFoundError("missing linear head path")
            head = LinearFusionHead(Path(head_path))
            probs_raw, ood_score = head.predict(embedding, meta)
            # Reorder to the project contract. Missing classes receive zero.
            mapping = {cls: float(p) for cls, p in zip(head.classes, probs_raw)}
            probs = np.asarray([mapping.get(cls, 0.0) for cls in VISUAL_CLASSES], dtype=np.float64)
            if probs.sum() <= 0:
                raise ValueError("head does not contain compatible project classes")
            probs /= probs.sum()
            idx = int(np.argmax(probs))
            ood_threshold = head.ood_threshold
            ood_flag = bool(ood_score is not None and ood_threshold is not None and ood_score > ood_threshold)
            branches.append(BranchPrediction(
                name=name,
                display_name=DISPLAY_NAMES[name],
                class_probabilities={cls: float(p) for cls, p in zip(VISUAL_CLASSES, probs)},
                predicted_class=VISUAL_CLASSES[idx],
                confidence=float(probs[idx]),
                entropy=_entropy(probs),
                margin=_margin(probs),
                ood_score=ood_score,
                ood_threshold=ood_threshold,
                ood_flag=ood_flag,
                feature_dim=int(np.asarray(embedding).size),
            ))
        except Exception as exc:  # optional branch must never crash the app
            branch_errors.append(f"{DISPLAY_NAMES.get(name, name)}: {type(exc).__name__}: {exc}")

    if not branches:
        return {
            "schema": "globalhab-adaptive-visual-v1",
            "active": False,
            "requested_mode": requested_mode,
            "route": asdict(route),
            "statuses": {k: asdict(v) for k, v in statuses.items()},
            "branches": [],
            "ensemble": None,
            "uncertainty": None,
            "branch_errors": branch_errors,
            "fallback_reason": route.reason if route.fallback_to_heuristic else "可选深度视觉分支运行失败，已安全回退到规则基线。",
            "scientific_note": "未使用未训练的深度分类头制造预测；当前结果仍由透明规则基线产生。",
        }

    prob_matrix = np.asarray([[b.class_probabilities[c] for c in VISUAL_CLASSES] for b in branches], dtype=np.float64)
    mean_probs = prob_matrix.mean(axis=0)
    mean_probs /= mean_probs.sum()
    idx = int(np.argmax(mean_probs))
    disagreement = float(np.mean(np.sum(np.abs(prob_matrix - mean_probs[None, :]), axis=1) / 2.0)) if len(branches) > 1 else 0.0
    entropy = _entropy(mean_probs)
    margin = _margin(mean_probs)
    any_ood = any(b.ood_flag for b in branches)
    reasons: list[str] = []
    if entropy > 0.72:
        reasons.append("预测分布熵较高")
    if margin < 0.12:
        reasons.append("第一与第二候选类别差距过小")
    if disagreement > 0.28:
        reasons.append("不同视觉分支之间的一致性不足")
    if any_ood:
        reasons.append("至少一个视觉分支提示输入偏离其训练特征分布")
    defer = bool(reasons)

    uncertainty = UncertaintyState(
        entropy=entropy,
        margin=margin,
        disagreement=disagreement,
        ood_flag=any_ood,
        defer=defer,
        reasons=reasons,
    )
    normal_prob = float(mean_probs[VISUAL_CLASSES.index("正常/未见明显异常")])
    uncertain_prob = float(mean_probs[VISUAL_CLASSES.index("不确定")])
    anomaly_score = float(np.clip(1.0 - normal_prob - 0.5 * uncertain_prob, 0.0, 1.0))
    return {
        "schema": "globalhab-adaptive-visual-v1",
        "active": True,
        "requested_mode": requested_mode,
        "route": asdict(route),
        "statuses": {k: asdict(v) for k, v in statuses.items()},
        "branches": [asdict(b) for b in branches],
        "ensemble": {
            "class_probabilities": {cls: float(p) for cls, p in zip(VISUAL_CLASSES, mean_probs)},
            "predicted_class": VISUAL_CLASSES[idx],
            "visual_class_confidence": float(mean_probs[idx]),
            "visual_anomaly_score": anomaly_score,
            "note": "这里的概率仅表示已校准视觉类别头的类别概率，不等同于HAB发生概率、藻种概率或毒素概率。",
        },
        "uncertainty": asdict(uncertainty),
        "branch_errors": branch_errors,
        "fallback_reason": None,
        "scientific_note": "深度分支只有在本地编码器与项目标注数据训练得到的轻量头同时可用时才参与结果。",
    }


def compact_status_rows(root: Any | None = None) -> list[dict[str, Any]]:
    statuses = get_backbone_status(root)
    return [
        {
            "视觉分支": s.display_name,
            "编码器": "可用" if s.encoder_available else "未配置",
            "轻量分类头": "可用" if s.head_available else "未配置",
            "可参与路由": "是" if s.ready else "否",
            "说明": s.note,
        }
        for s in statuses.values()
    ]
