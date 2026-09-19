"""Server-owned catalog. Credentials never enter widgets or experiment reports."""
import os
import hashlib
import json
from .real_training.remote_qwen import chat, ready


def catalog(secrets=None, environ=None):
    if secrets is None:
        from .display_locale import st
        try:
            secrets = st.secrets.to_dict()
        except FileNotFoundError:
            secrets = {}
    env = os.environ if environ is None else environ
    entries = []
    # Explicit platform catalog supports several providers and model choices.
    for index, row in enumerate(secrets.get("platform_models", [])):
        if not isinstance(row, dict) or not row.get("enabled", True):
            continue
        config = {k: str(row.get(k, "")).strip() for k in ("base_url", "api_key", "model")}
        if ready(config):
            entries.append({"id": "platform_"+str(index), "label": str(row.get("name", config["model"])), "config": config})
    # Preserve the website's existing Qwen configuration and environment variables.
    qwen = secrets.get("qwen", {})
    config = {k: str(qwen.get(k, env.get("HAB_QWEN_"+k.upper(), ""))).strip() for k in ("base_url", "api_key", "model")}
    model_ids = qwen.get("models", [])
    if not model_ids:
        try:
            model_ids = json.loads(env.get("HAB_QWEN_MODELS", "[]"))
        except (ValueError, TypeError):
            model_ids = []
    if not isinstance(model_ids, list):
        model_ids = []
    model_ids = [m for m in model_ids if isinstance(m, str) and m.strip()]
    if not model_ids:
        model_ids = [config["model"]]
    for model_id in model_ids:
        item = dict(config, model=model_id)
        if ready(item) and not any(e["config"] == item for e in entries):
            label = model_id.split("/")[-1] if len(model_ids)>1 else str(qwen.get("name", model_id.split("/")[-1]))
            entries.append({"id": "qwen_"+model_id, "label": label, "config": item})
    return entries


def connection_id(config):
    return hashlib.sha256("\0".join(config.get(k, "") for k in ("base_url", "model", "api_key")).encode()).hexdigest()


def probe(config):
    # This is a real, small generation request, not just an HTTP reachability test.
    text, usage = chat(config, [{"role": "user", "content": '请只回复 OK。'}], max_tokens=256)
    if text.strip().strip('.。') != "OK":
        raise ValueError("服务未返回预期的检查响应")
    return usage
