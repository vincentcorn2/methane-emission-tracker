"""
Centralized configuration management.
All credentials loaded from environment variables — never hardcoded.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Copernicus credentials
    copernicus_user: str
    copernicus_password: str

    # Hugging Face
    hf_token: str = ""

    # ERA5 / CDS
    cds_api_key: str = ""

    # API config
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Detection parameters (calibrated)
    detection_threshold: float = 0.18
    min_plume_pixels: int = 115

    # Copernicus API endpoints
    copernicus_token_url: str = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE/protocol/openid-connect/token"
    )
    copernicus_catalog_url: str = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    )

    class Config:
        env_file = "config/.env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings instance, cached after first load."""
    return Settings()
