"""Runtime settings (from environment / `.env`) and hard compliance constants."""

import os
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
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
    # Address people open the dashboard at (links inside report PDFs) and the port this API listens
    # on (the server prints its own report page for the PDF).
    public_url: str = Field(default="http://localhost:8000", validation_alias="DHARA_PUBLIC_URL")
    api_port: int = Field(default=8000, validation_alias="DHARA_API_PORT")
    # Optional CDSE S3 (eodata) keys: preferred route for Copernicus DEM GLO-30 COG tiles.
    cdse_s3_access_key: SecretStr | None = None
    cdse_s3_secret_key: SecretStr | None = None
    # ohsome API v2 key (free, https://account.heigit.org/signup). Without it: slow v1 fallback.
    ohsome_api_key: SecretStr | None = None
    # Assistant language model. "auto" (default) uses the first configured provider and "off"
    # never calls one. Only the resolved provider's host passes the allowlist,
    # and any model answer with a number not found in the job's results is discarded.
    assistant_llm: Literal["auto", "off", "groq", "gemini", "openai", "openrouter"] = "auto"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    gemini_model: str = "gemini-3.8-flash"
    # Tried in order on every account when the one before is overloaded or unavailable.
    gemini_fallback_models: str = "gemini-2.5-flash"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str = "nvidia/nemotron-3.5-lightning:free"

    dhara_data_dir: Path = Path("data")
    dhara_max_aoi_km2: float = 2500.0
    # Search window (days) either side of the event for S1 scene selection.
    dhara_s1_search_days: int = 24
    # Minimum fraction of the AOI a S1 pass must cover to be usable.
    dhara_min_s1_coverage: float = 0.5
    dhara_http_timeout_s: float = 120.0

    # Gray area: foundation-model weights trained on data outside the allowed sets. Default OFF.
    allow_foundation_weights: bool = False

    @property
    def gemini_api_keys(self) -> list[SecretStr]:
        """GEMINI_API_KEY, then GEMINI_API_KEY_ACC_1, _ACC_2, … (any number), from .env and the
        environment, in that order."""
        keys = _numbered_keys(self, "GEMINI_API_KEY")
        if not keys and self.gemini_api_key:
            return [self.gemini_api_key]
        return keys

    @property
    def groq_api_keys(self) -> list[SecretStr]:
        """Return the primary Groq key followed by any numbered account keys from the environment."""
        env_file = self.model_config.get("env_file", ".env")
        env_path = Path(env_file) if isinstance(env_file, str) else Path(".env")
        values = dict(dotenv_values(env_path)) if env_path.is_file() else {}
        values.update(os.environ)
        keys = [
            (name, value)
            for name, value in values.items()
            if isinstance(name, str)
            and isinstance(value, str)
            and value.strip()
            and re.fullmatch(r"GROQ_API_KEY(?:_ACC_\d+)?", name, re.IGNORECASE)
        ]

        def account_order(item: tuple[str, str]) -> tuple[bool, int]:
            name = item[0].upper()
            match = re.fullmatch(r"GROQ_API_KEY_ACC_(\d+)", name)
            return (name != "GROQ_API_KEY", int(match.group(1)) if match else 0)

        keys.sort(key=account_order)
        if not keys and self.groq_api_key:
            return [self.groq_api_key]
        return [SecretStr(value.strip()) for _, value in keys]

    @property
    def llm_provider(self) -> str:
        """The assistant provider in use: "groq", "gemini", "openai" or "off"."""
        keys = {
            "gemini": self.gemini_api_keys,  # primary when configured
            "groq": self.groq_api_keys,
            "openai": self.openai_api_key,
            "openrouter": self.openrouter_api_key,
        }
        if self.assistant_llm == "auto":
            return next((k for k, v in keys.items() if v), "off")
        if self.assistant_llm == "off" or not keys[self.assistant_llm]:
            return "off"
        return self.assistant_llm

    @property
    def llm_fallback_providers(self) -> list[str]:
        """Providers the assistant may fall back to after the active one (keys configured)."""
        if self.llm_provider == "gemini" and self.groq_api_keys:
            return ["groq"]
        return []

    @property
    def cache_dir(self) -> Path:
        return self.dhara_data_dir / "cache"

    @property
    def jobs_dir(self) -> Path:
        return self.dhara_data_dir / "jobs"


def _numbered_keys(settings: "Settings", prefix: str) -> list[SecretStr]:
    """`prefix` and `prefix`_ACC_<n> values from .env and the environment, unnumbered first."""
    env_file = settings.model_config.get("env_file", ".env")
    env_path = Path(env_file) if isinstance(env_file, str) else Path(".env")
    values = dict(dotenv_values(env_path)) if env_path.is_file() else {}
    values.update(os.environ)
    found = []
    for name, value in values.items():
        m = re.fullmatch(rf"{prefix}(?:_ACC_(\d+))?", str(name), re.IGNORECASE)
        if m and isinstance(value, str) and value.strip():
            found.append((int(m.group(1)) if m.group(1) else 0, value.strip()))
    return [SecretStr(v) for _, v in sorted(found)]


@lru_cache
def get_settings() -> Settings:
    return Settings()
