# TranslateGemma FastAPI Service

FastAPI service for **Google TranslateGemma** (text → text and image → text/translation) with support for:

- 🧠 **GPU sharding (1+ GPUs)**
- ⚡ **Quantization**: none / int8 / int4 (bitsandbytes)
- 🌊 **SSE token streaming**
- 🧩 **YAML + .env configuration** (pydantic-settings)
- 🖼 **Text translation from images via URL**
- 🔒 Production-friendly design (rate limiting, metrics, proxies, restarts)

---

## Features

- `/translate` — standard text translation
- `/translate/stream` — translation with **SSE token streaming**
- `/translate_image_url` — translate text from an image (URL)
- `/translate_image_url/stream` — streaming translation from an image
- `/health` — health check + GPU/config diagnostics

Supports:
- **Single GPU / Multi-GPU / CPU offload**
- **BF16 / FP16**
- **Gated Hugging Face models (via HF token)**

---

## Requirements

### Minimum
- Python **3.11+**
- Linux x86_64
- Hugging Face account + model access
- NVIDIA GPU (optional, but recommended)

### Recommended
- CUDA 12.x
- RTX 3090 / A100 / L40 / H100
- 64GB+ RAM (if using CPU offload)

---

## Supported Models

Default:
- `google/translategemma-12b-it`
- `google/translategemma-27b-it`

You can use any compatible **image-text-to-text** model from Hugging Face.

---

## Installation

### 1. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Clone repository
```bash
git clone https://github.com/qarmir/translategemma-api.git
cd translategemma-api
```

### 3. Install Python 3.11 (or higher)
```bash
sudo apt install -y python3.11 python3.11-venv
poetry env use python3.11
```

### 4. Install dependencies
```bash
poetry install
```

---

## Hugging Face Access
TranslateGemma models are gated.
You must accept the license and generate a token on Hugging Face:

https://huggingface.co/google/translategemma-12b-it

---

## Configuration
```bash
# Model
TG_MODEL__ID=google/translategemma-12b-it
TG_MODEL__GPU_COUNT=1         # How many GPUs to use
TG_MODEL__QUANTIZATION=int8   # None for no quantization
TG_MODEL__MAX_MEMORY_PER_GPU=23GiB
TG_MODEL__MAX_MEMORY_CPU=128GiB

# Service
TG_SERVICE__MAX_CONCURRENT=1

# Generation
TG_GENERATION__DEFAULT_MAX_NEW_TOKENS=200
TG_GENERATION__MAX_NEW_TOKENS_LIMIT=1024
```

---

## Running the service
```bash
export HF_TOKEN=hf_token_here
poetry run serve
```

## Docker Compose / Provider deployment
The repository includes:
- a root `compose.yaml` for local Docker Compose runs
- `.provider/compose.yml` and `.provider/deployment.yml` for Provider deployment analysis and rollout

Requirements:
- `HF_TOKEN` must be set at deploy time.
- The application listens on port `8000`.
- Provider should be configured with `application_port=8000`.

Local validation:
```bash
export HF_TOKEN=hf_token_here
docker compose up --build
```

Health check:
```bash
curl http://localhost:8000/health | jq .
```

---

## Usage
### Standard Translation
```bash
curl http://localhost:8000/translate \
  -H "content-type: application/json" \
  -d '{
    "text": "Hello world",
    "source_lang_code": "en",
    "target_lang_code": "de"
  }'
```

---

## SSE Streaming
### Text
```bash
curl -N http://localhost:8000/translate/stream \
  -H "content-type: application/json" \
  -d '{
    "text": "Translate into Russian: I will be late for the meeting.",
    "source_lang_code": "en",
    "target_lang_code": "ru"
  }'
```

### Image URL

```bash
curl -N http://localhost:8000/translate_image_url/stream \
  -H "content-type: application/json" \
  -d '{
    "image_url": "https://example.com/text_image.png",
    "source_lang_code": "en",
    "target_lang_code": "fr"
  }'
```

---

## GPU Modes

| Mode               | VRAM Usage | Recommendation  |
| ------------------ | ---------- | --------------- |
| `none` (bf16/fp16) | 🔴 High    | Multi-GPU only  |
| `int8`             | 🟡 Medium  | Best balance    |
| `int4`             | 🟢 Low     | Maximum density |

### 12B Model

* 1×3090 → `int8` / `int4`
* 2×3090 → `bf16` or `int8`

### 27B Model

* 2×3090 → `int4` / `int8`
* A100 80GB → `bf16`

---

## License

Model: Google TranslateGemma — see license on Hugging Face
Service code: MIT / Apache-2.0 (your choice)

---
