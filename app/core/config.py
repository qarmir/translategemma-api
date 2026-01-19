from typing import Literal
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

Quant = Literal["none", "int8", "int4"]
DType = Literal["float16", "bfloat16"]


# -----------------------
# Settings (YAML + ENV) без ручного парсинга
# -----------------------
class ModelCfg(BaseModel):
    id: str = "google/translategemma-12b-it"
    hf_token_env: str = "HF_TOKEN"
    dtype: DType = "bfloat16"
    quantization: Quant = "none"
    device_map: str = "auto"
    gpu_count: int = 1
    max_memory_per_gpu: str = "23GiB"
    max_memory_cpu: str = "64GiB"
    offload_folder: str = "./offload"
    trust_remote_code: bool = False


class ServiceCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent: int = 1
    log_level: str = "info"


class GenerationCfg(BaseModel):
    default_max_new_tokens: int = 200
    max_new_tokens_limit: int = 1024
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0


class Settings(BaseSettings):
    config_path: str = "config.yaml"

    model: ModelCfg = ModelCfg()
    service: ServiceCfg = ServiceCfg()
    generation: GenerationCfg = GenerationCfg()

    # ENV overrides: MODEL.id -> TG_MODEL__ID и т.п.
    model_config = SettingsConfigDict(
        env_prefix="TG_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls,
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
    ):
        # YAML как источник конфигурации (без ручного парсинга)
        # Порядок: init > env > dotenv > yaml > secrets
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

settings = Settings()
