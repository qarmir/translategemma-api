import asyncio
import threading
import torch
from transformers import TextIteratorStreamer
from app.core.config import settings
from app.model.loader import ensure_loaded

@torch.inference_mode()
def generate_full(messages, max_new_tokens: int) -> str:
    processor, model = ensure_loaded()

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_len = inputs["input_ids"].shape[-1]
    out = model.generate(
        **inputs,
        do_sample=settings.generation.do_sample,
        max_new_tokens=max_new_tokens,
        temperature=settings.generation.temperature if settings.generation.do_sample else None,
        top_p=settings.generation.top_p if settings.generation.do_sample else None,
    )

    gen = out[0][input_len:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def stream_worker(messages, max_new_tokens: int, q, loop):
    processor, model = ensure_loaded()
    try:
        streamer = TextIteratorStreamer(
            tokenizer=processor.tokenizer,
            skip_special_tokens=True,
        )

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        kwargs = dict(
            **inputs,
            streamer=streamer,
            do_sample=settings.generation.do_sample,
            max_new_tokens=max_new_tokens,
        )
        if settings.generation.do_sample:
            kwargs["temperature"] = settings.generation.temperature
            kwargs["top_p"] = settings.generation.top_p

        t = threading.Thread(target=model.generate, kwargs=kwargs, daemon=True)
        t.start()

        for piece in streamer:
            asyncio.run_coroutine_threadsafe(q.put(piece), loop)

        t.join()
        asyncio.run_coroutine_threadsafe(q.put(None), loop)

    except Exception as e:
        asyncio.run_coroutine_threadsafe(q.put(e), loop)
