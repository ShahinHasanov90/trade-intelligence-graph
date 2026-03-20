"""Configuration management for Trade Intelligence Graph.

Loads configuration from YAML files and environment variables, providing
typed access to all system settings through Pydantic models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class EntityResolutionConfig(BaseModel):
    """Settings for entity resolution and fuzzy matching."""

    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for entity matching",
    )
    match_fields: list[str] = Field(
        default=["name", "tax_id", "address", "phone"],
        description="Fields used for entity matching",
    )


class EdgeWeightConfig(BaseModel):
    """Settings for computing edge weights."""

    method: str = Field(
        default="composite",
        description="Weight computation method: frequency, volume, or composite",
    )
    components: dict[str, float] = Field(
        default={"frequency": 0.4, "volume": 0.3, "recency": 0.3},
        description="Weight components for composite method",
    )


class GraphConfig(BaseModel):
    """Graph construction configuration."""

    backend: str = Field(
        default="networkx",
        description="Graph backend: networkx or neo4j",
    )
    entity_resolution: EntityResolutionConfig = Field(
        default_factory=EntityResolutionConfig
    )
    edge_weights: EdgeWeightConfig = Field(default_factory=EdgeWeightConfig)


class CommunityConfig(BaseModel):
    """Community detection configuration."""

    algorithm: str = Field(
        default="louvain",
        description="Detection algorithm: louvain, leiden, label_propagation",
    )
    resolution: float = Field(
        default=1.0, ge=0.0, description="Resolution parameter for modularity"
    )
    min_community_size: int = Field(
        default=3, ge=1, description="Minimum nodes to form a community"
    )
    random_seed: int = Field(default=42, description="Random seed for reproducibility")


class PageRankConfig(BaseModel):
    """PageRank algorithm configuration."""

    alpha: float = Field(default=0.85, ge=0.0, le=1.0, description="Damping factor")
    max_iter: int = Field(default=100, ge=1, description="Maximum iterations")
    tol: float = Field(default=1e-6, gt=0, description="Convergence tolerance")


class BetweennessConfig(BaseModel):
    """Betweenness centrality configuration."""

    normalized: bool = Field(default=True, description="Whether to normalize scores")
    k: Optional[int] = Field(
        default=None, description="Sample size (None for exact computation)"
    )


class DegreeConfig(BaseModel):
    """Degree centrality configuration."""

    normalized: bool = Field(default=True, description="Whether to normalize scores")


class CentralityConfig(BaseModel):
    """Centrality analysis configuration."""

    pagerank: PageRankConfig = Field(default_factory=PageRankConfig)
    betweenness: BetweennessConfig = Field(default_factory=BetweennessConfig)
    degree: DegreeConfig = Field(default_factory=DegreeConfig)


class TemporalConfig(BaseModel):
    """Temporal analysis configuration."""

    window_size: int = Field(default=30, ge=1, description="Snapshot window in days")
    window_overlap: int = Field(
        default=7, ge=0, description="Overlap between windows in days"
    )
    structural_change_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Threshold for structural change alerts",
    )


class PropagationConfig(BaseModel):
    """Risk propagation configuration."""

    max_depth: int = Field(
        default=4, ge=1, description="Maximum propagation depth (hops)"
    )
    decay_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Multiplicative decay per hop",
    )
    min_propagation_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum risk score to continue propagation",
    )
    edge_type_weights: dict[str, float] = Field(
        default={
            "IMPORTS_FROM": 1.0,
            "REPRESENTED_BY": 0.8,
            "LOCATED_AT": 0.6,
            "CONTACTABLE_VIA": 0.5,
            "PAYS_THROUGH": 0.9,
            "CO_OCCURS_WITH": 0.3,
        },
        description="Propagation weight multipliers by edge type",
    )


class AnalysisConfig(BaseModel):
    """Analysis engine configuration."""

    community: CommunityConfig = Field(default_factory=CommunityConfig)
    centrality: CentralityConfig = Field(default_factory=CentralityConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    propagation: PropagationConfig = Field(default_factory=PropagationConfig)


class RingDetectionConfig(BaseModel):
    """Fraud ring detection configuration."""

    max_cycle_length: int = Field(
        default=6, ge=3, description="Maximum cycle length to search"
    )
    min_shared_attributes: int = Field(
        default=2, ge=1, description="Minimum shared attributes to flag"
    )


class ShellCompanyConfig(BaseModel):
    """Shell company detection configuration."""

    centrality_volume_ratio: float = Field(
        default=5.0,
        ge=1.0,
        description="Centrality/volume divergence threshold",
    )
    min_flagged_connections: int = Field(
        default=3, ge=1, description="Minimum connections to flagged entities"
    )
    free_trade_zones: list[str] = Field(
        default=["AE-FZ", "SG-FTZ", "PA-CFZ", "HK", "BH-FTZ"],
        description="ISO codes for free trade zones",
    )


class CarouselConfig(BaseModel):
    """Carousel fraud detection configuration."""

    min_cycle_value: float = Field(
        default=50000.0, ge=0, description="Minimum cycle value for detection"
    )
    max_cycle_days: int = Field(
        default=90, ge=1, description="Maximum days between cycle transactions"
    )
    vat_countries: list[str] = Field(
        default=["GB", "DE", "FR", "IT", "ES", "NL", "BE", "PL"],
        description="Countries where VAT carousel is relevant",
    )


class DetectionConfig(BaseModel):
    """Detection engine configuration."""

    rings: RingDetectionConfig = Field(default_factory=RingDetectionConfig)
    shell_company: ShellCompanyConfig = Field(default_factory=ShellCompanyConfig)
    carousel: CarouselConfig = Field(default_factory=CarouselConfig)


class ApiConfig(BaseModel):
    """API server configuration."""

    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins",
    )
    graphql_path: str = Field(default="/graphql", description="GraphQL endpoint path")


class Neo4jConfig(BaseModel):
    """Neo4j connection configuration."""

    uri: str = Field(default="bolt://localhost:7687", description="Neo4j Bolt URI")
    user: str = Field(default="neo4j", description="Neo4j username")
    password: str = Field(default="changeme", description="Neo4j password")
    database: str = Field(default="tradegraph", description="Neo4j database name")
    max_connection_pool_size: int = Field(
        default=50, ge=1, description="Connection pool size"
    )


class SignalExportConfig(BaseModel):
    """Risk signal export configuration."""

    endpoint: str = Field(
        default="http://localhost:8080/api/v1/signals",
        description="Signal receiver endpoint",
    )
    batch_size: int = Field(default=100, ge=1, description="Batch size for export")
    retry_attempts: int = Field(default=3, ge=0, description="Number of retry attempts")
    retry_delay: int = Field(default=5, ge=0, description="Delay between retries (s)")


class ExportConfig(BaseModel):
    """Export configuration."""

    signals: SignalExportConfig = Field(default_factory=SignalExportConfig)


class Settings(BaseSettings):
    """Root configuration for Trade Intelligence Graph.

    Settings are loaded from a YAML configuration file, with environment
    variable overrides for deployment-specific values.
    """

    graph: GraphConfig = Field(default_factory=GraphConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Settings":
        """Load settings from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Settings instance populated from the file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Resolve environment variable references like ${NEO4J_PASSWORD}
        raw = _resolve_env_vars(raw)
        return cls(**raw)


def _resolve_env_vars(obj: object) -> object:
    """Recursively resolve ${VAR} references in configuration values.

    Args:
        obj: Configuration object (dict, list, or scalar).

    Returns:
        Object with environment variable references resolved.
    """
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var_name = obj[2:-1]
        return os.environ.get(var_name, obj)
    return obj


# Module-level singleton for convenient access
_settings: Optional[Settings] = None


def get_settings(config_path: str | Path | None = None) -> Settings:
    """Get or create the global settings instance.

    Args:
        config_path: Optional path to YAML configuration file.
            If None and no settings exist, returns defaults.

    Returns:
        The global Settings instance.
    """
    global _settings
    if _settings is None:
        if config_path is not None:
            _settings = Settings.from_yaml(config_path)
        else:
            _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance. Primarily used in testing."""
    global _settings
    _settings = None
