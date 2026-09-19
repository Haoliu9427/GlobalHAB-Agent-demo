"""Generate bounded tool plans using the demo's existing HF Qwen inventory."""
from functools import lru_cache
from threading import RLock
import importlib.util

from .model_registry import FOUNDATIONS

QWEN_MODELS = {k: v for k, v in FOUNDATIONS.items() if k.startswith("Qwen")}
_LOCK = RLock()


def missing_dependencies():
    return [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]


@lru_cache(maxsize=1)
def load_model(model_id):
    if model_id not in QWEN_MODELS.values():
        raise ValueError("不支持的内置Qwen模型")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, trust_remote_code=False,
    ).eval()
    return tokenizer, model


def chat_local(messages, model_id="Qwen/Qwen2.5-0.5B-Instruct"):
    # Model load and generation share the lock across Streamlit sessions.
    # No API key or remote inference request; HF may download missing weights.
    with _LOCK:
        import torch
        tokenizer, model = load_model(model_id)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        count = inputs["input_ids"].shape[-1]
        if count > 24000:
            raise ValueError("规划上下文过长，请减少实验预算")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=512, max_time=90,
                do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = output[0, count:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise ValueError("内置Qwen未生成有效工具方案")
        return text, {"prompt_tokens": int(count), "completion_tokens": int(len(generated)),
                      "total_tokens": int(count+len(generated))}
