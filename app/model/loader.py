import os
import torch
from typing import Optional, Dict, Any
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from app.core.config import settings, Quant, DType

processor = None
model = None

def _torch_dtype(dtype: DType):
    return torch.bfloat16 if dtype == "bfloat16" else torch.float16

def _make_quant(q: Quant):
    if q == "none":
        return None
    if q == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if q == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    raise ValueError(q)

def _make_max_memory(gpu_count: int, per_gpu: str, cpu: str) -> Dict[Any, str]:
    mm = {"cpu": cpu}
    for i in range(max(0, gpu_count)):
        mm[i] = per_gpu
    return mm

def ensure_loaded():
    global processor, model
    if model is not None:
        return processor, model

    token = os.getenv(settings.model.hf_token_env)
    if not token:
        raise RuntimeError(f"{settings.model.hf_token_env} not set")

    processor = AutoProcessor.from_pretrained(
        settings.model.id,
        token=token,
        trust_remote_code=settings.model.trust_remote_code,
    )

    kwargs = dict(
        token=token,
        device_map=settings.model.device_map,
        max_memory=_make_max_memory(
            settings.model.gpu_count,
            settings.model.max_memory_per_gpu,
            settings.model.max_memory_cpu,
        ),
        torch_dtype=_torch_dtype(settings.model.dtype),
        offload_folder=settings.model.offload_folder,
        trust_remote_code=settings.model.trust_remote_code,
    )

    quant = _make_quant(settings.model.quantization)
    if quant:
        kwargs["quantization_config"] = quant

    model = AutoModelForImageTextToText.from_pretrained(settings.model.id, **kwargs)
    model.eval()
    return processor, model
