"""Runtime settings (from environment / `.env`) and hard compliance constants."""

from datetime import date
from functools import lru_cache
from pathlib import Path

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
    def cache_dir(self) -> Path:
        return self.dhara_data_dir / "cache"

    @property
    def jobs_dir(self) -> Path:
        return self.dhara_data_dir / "jobs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
