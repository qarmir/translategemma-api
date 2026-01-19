import asyncio
import json
import threading
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.core.config import settings
from app.model.generator import generate_full, stream_worker

router = APIRouter()
_sem = asyncio.Semaphore(max(1, settings.service.max_concurrent))
_LOOP = None

def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

@router.on_event("startup")
def _startup():
    global _LOOP
    _LOOP = asyncio.get_event_loop()

@router.get("/health")
def health():
    return {"ok": True}

@router.post("/translate")
async def translate(req: dict):
    max_new = req.get("max_new_tokens") or settings.generation.default_max_new_tokens
    if max_new > settings.generation.max_new_tokens_limit:
        raise HTTPException(400, "max_new_tokens too large")

    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": req["source_lang_code"],
            "target_lang_code": req["target_lang_code"],
            "text": req["text"],
        }],
    }]

    async with _sem:
        out = await asyncio.to_thread(generate_full, messages, max_new)
    return {"translated_text": out}

@router.post("/translate/stream")
async def translate_stream(req: dict):
    max_new = req.get("max_new_tokens") or settings.generation.default_max_new_tokens
    if max_new > settings.generation.max_new_tokens_limit:
        raise HTTPException(400, "max_new_tokens too large")

    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": req["source_lang_code"],
            "target_lang_code": req["target_lang_code"],
            "text": req["text"],
        }],
    }]

    async def gen():
        q = asyncio.Queue()
        async with _sem:
            threading.Thread(
                target=stream_worker,
                args=(messages, max_new, q, _LOOP),
                daemon=True,
            ).start()

            yield _sse("ready", {"ok": True})
            while True:
                item = await q.get()
                if item is None:
                    yield _sse("done", {"ok": True})
                    break
                if isinstance(item, Exception):
                    yield _sse("error", {"error": str(item)})
                    break
                yield _sse("token", {"text": item})

    return StreamingResponse(gen(), media_type="text/event-stream")
