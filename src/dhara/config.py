"""Runtime settings (from environment / `.env`) and hard compliance constants."""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Latest OSM snapshot allowed as a pipeline input (problem statement: pre-event, ≤ 27 July 2026).
OSM_SNAPSHOT_MAX = date(2026, 7, 27)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="", extra="ignore", populate_by_name=True
    )

    cdse_client_id: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("CDSE_SENTINEL_HUB_CLIENT_ID", "CDSE_CLIENT_ID")
    )
    cdse_client_secret: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("CDSE_SENTINEL_HUB_CLIENT_SECRET", "CDSE_CLIENT_SECRET"),
    )
    # Optional CDSE S3 (eodata) keys: preferred route for Copernicus DEM GLO-30 COG tiles.
    cdse_s3_access_key: SecretStr | None = None
    cdse_s3_secret_key: SecretStr | None = None
    # ohsome API v2 key (free, https://account.heigit.org/signup). Without it: slow v1 fallback.
    ohsome_api_key: SecretStr | None = None
    # Assistant language model. "auto" (default) uses the first provider with a key (Groq, Gemini,
    # OpenAI) and "off" never calls one. Only the resolved provider's host passes the allowlist,
    # and any model answer with a number not found in the job's results is discarded.
    assistant_llm: Literal["auto", "off", "groq", "gemini", "openai"] = "auto"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    gemini_model: str = "gemini-1.5-flash"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"

    dhara_data_dir: Path = Path("data")
    dhara_max_aoi_km2: float = 2500.0
    # Search window (days) either side of the event for S1 scene selection.
    dhara_s1_search_days: int = 24
    # Minimum fraction of the AOI a S1 pass must cover to be usable.
    dhara_min_s1_coverage: float = 0.5
    dhara_http_timeout_s: float = 120.0

    # Gray area (PLAN.md §3.3): foundation-model weights trained on non-listed data. Default OFF.
    allow_foundation_weights: bool = False

    @property
    def llm_provider(self) -> str:
        """The assistant provider in use: "groq", "gemini", "openai" or "off"."""
        keys = {
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "openai": self.openai_api_key,
        }
        if self.assistant_llm == "auto":
            return next((k for k, v in keys.items() if v), "off")
        if self.assistant_llm == "off" or not keys[self.assistant_llm]:
            return "off"
        return self.assistant_llm

    @property
    def cache_dir(self) -> Path:
        return self.dhara_data_dir / "cache"

    @property
    def jobs_dir(self) -> Path:
        return self.dhara_data_dir / "jobs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
