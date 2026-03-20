"""Shell company identification in trade entity graphs.

Detects potential shell companies and pass-through entities using
structural graph heuristics. Shell companies in trade fraud typically
exhibit:
- High centrality but low actual trade volume (structural importance
  disproportionate to economic activity)
- Registration in known free-trade zones or offshore jurisdictions
- Many connections to flagged or high-risk entities
- Recent registration dates relative to trade volume
- Minimal physical footprint (no distinct address patterns)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import networkx as nx
import numpy as np
import structlog

from graph_intel.config import Settings, ShellCompanyConfig, get_settings
from graph_intel.graph.schema import NodeType

logger = structlog.get_logger(__name__)


@dataclass
class ShellCompanyCandidate:
    """A potential shell company identified by structural analysis.

    Attributes:
        node_id: The entity node identifier.
        entity_name: Human-readable name of the entity.
        entity_type: Type of entity (IMPORTER, EXPORTER, AGENT).
        confidence: Confidence that this is a shell company (0.0 to 1.0).
        indicators: List of indicators contributing to the assessment.
        centrality_score: Combined centrality score.
        trade_volume: Total trade volume through this entity.
        centrality_volume_ratio: Ratio of structural importance to economic activity.
        connection_count: Total number of connections.
        flagged_connection_count: Number of connections to flagged entities.
        country: Registration country.
        is_free_trade_zone: Whether registered in a known FTZ.
        registration_age_days: Days since registration.
        connected_entities: List of connected entity IDs.
    """

    node_id: str
    entity_name: str
    entity_type: str
    confidence: float
    indicators: list[str] = field(default_factory=list)
    centrality_score: float = 0.0
    trade_volume: float = 0.0
    centrality_volume_ratio: float = 0.0
    connection_count: int = 0
    flagged_connection_count: int = 0
    country: str = ""
    is_free_trade_zone: bool = False
    registration_age_days: int = 0
    connected_entities: list[str] = field(default_factory=list)


class ShellCompanyDetector:
    """Identifies potential shell companies using graph structure analysis.

    Uses a multi-signal approach combining centrality metrics, trade volume
    analysis, jurisdictional indicators, and network neighborhood analysis
    to score entities on their likelihood of being shell companies.

    Args:
        graph: The trade entity graph to analyze.
        config: Optional shell company detection configuration.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        config: Optional[ShellCompanyConfig] = None,
    ) -> None:
        """Initialize the shell company detector.

        Args:
            graph: The trade entity graph (NetworkX MultiDiGraph).
            config: Optional shell company detection configuration.
        """
        self._graph = graph
        self._config = config or get_settings().detection.shell_company
        self._centrality_cache: dict[str, float] = {}

    def detect(
        self,
        min_confidence: float = 0.5,
        node_types: Optional[list[NodeType]] = None,
    ) -> list[ShellCompanyCandidate]:
        """Run shell company detection across all eligible entities.

        Evaluates each entity node against multiple shell company indicators
        and produces a ranked list of candidates.

        Args:
            min_confidence: Minimum confidence score to include.
            node_types: Entity types to evaluate. Defaults to IMPORTER,
                EXPORTER, and AGENT.

        Returns:
            List of ShellCompanyCandidate objects, sorted by confidence.
        """
        if node_types is None:
            node_types = [NodeType.IMPORTER, NodeType.EXPORTER, NodeType.AGENT]

        type_values = {nt.value for nt in node_types}

        # Precompute centrality scores
        self._compute_centrality()

        # Precompute trade volumes
        node_volumes = self._compute_node_volumes()

        # Normalize centrality and volume for comparison
        max_centrality = (
            max(self._centrality_cache.values())
            if self._centrality_cache
            else 1.0
        )
        max_volume = max(node_volumes.values()) if node_volumes else 1.0

        candidates: list[ShellCompanyCandidate] = []

        for node_id, data in self._graph.nodes(data=True):
            node_type = data.get("node_type", "")
            if node_type not in type_values:
                continue

            indicators: list[str] = []
            confidence_components: list[float] = []

            # --- Indicator 1: Centrality-Volume Divergence ---
            centrality = self._centrality_cache.get(node_id, 0.0)
            volume = node_volumes.get(node_id, 0.0)

            centrality_norm = centrality / max_centrality if max_centrality > 0 else 0.0
            volume_norm = volume / max_volume if max_volume > 0 else 0.0

            cv_ratio = 0.0
            if volume_norm > 0:
                cv_ratio = centrality_norm / volume_norm
            elif centrality_norm > 0.1:
                cv_ratio = float("inf")

            if cv_ratio > self._config.centrality_volume_ratio:
                indicators.append("high_centrality_low_volume")
                confidence_components.append(
                    min(cv_ratio / (self._config.centrality_volume_ratio * 2), 0.3)
                )

            # --- Indicator 2: Free Trade Zone Registration ---
            country = data.get("country", "")
            is_ftz = country in self._config.free_trade_zones
            if is_ftz:
                indicators.append("free_trade_zone_registration")
                confidence_components.append(0.2)

            # --- Indicator 3: Connections to Flagged Entities ---
            flagged_connections = self._count_flagged_connections(node_id)
            if flagged_connections >= self._config.min_flagged_connections:
                indicators.append("many_flagged_connections")
                confidence_components.append(
                    min(flagged_connections / 10.0, 0.25)
                )

            # --- Indicator 4: Recent Registration + High Activity ---
            reg_date = data.get("registration_date", "")
            reg_age_days = self._compute_registration_age(reg_date)
            total_connections = self._graph.degree(node_id)

            if reg_age_days > 0 and reg_age_days < 365 and total_connections > 10:
                indicators.append("recently_registered_high_activity")
                confidence_components.append(0.15)

            # --- Indicator 5: Many Connections, Few Unique Commodities ---
            unique_commodities = self._count_unique_commodities(node_id)
            if total_connections > 5 and unique_commodities <= 1:
                indicators.append("single_commodity_many_connections")
                confidence_components.append(0.1)

            # --- Indicator 6: Asymmetric Degree ---
            in_deg = self._graph.in_degree(node_id)
            out_deg = self._graph.out_degree(node_id)
            if in_deg > 0 and out_deg > 0:
                asymmetry = abs(in_deg - out_deg) / max(in_deg, out_deg)
                if asymmetry > 0.9:
                    indicators.append("highly_asymmetric_connections")
                    confidence_components.append(0.1)

            # Compute overall confidence
            confidence = sum(confidence_components)
            confidence = min(confidence, 1.0)

            if confidence >= min_confidence:
                connected = [
                    n
                    for n in nx.all_neighbors(self._graph, node_id)
                ]

                candidate = ShellCompanyCandidate(
                    node_id=node_id,
                    entity_name=data.get("name", "Unknown"),
                    entity_type=node_type,
                    confidence=confidence,
                    indicators=indicators,
                    centrality_score=centrality,
                    trade_volume=volume,
                    centrality_volume_ratio=cv_ratio
                    if cv_ratio != float("inf")
                    else 999.9,
                    connection_count=total_connections,
                    flagged_connection_count=flagged_connections,
                    country=country,
                    is_free_trade_zone=is_ftz,
                    registration_age_days=reg_age_days,
                    connected_entities=connected[:50],  # Limit for output
                )
                candidates.append(candidate)

        candidates.sort(key=lambda c: c.confidence, reverse=True)

        logger.info(
            "shell_company_detection_complete",
            candidates_found=len(candidates),
            total_evaluated=sum(
                1
                for _, d in self._graph.nodes(data=True)
                if d.get("node_type") in type_values
            ),
        )

        return candidates

    def score_entity(self, node_id: str) -> Optional[ShellCompanyCandidate]:
        """Score a single entity for shell company indicators.

        Args:
            node_id: The node identifier to evaluate.

        Returns:
            ShellCompanyCandidate if the entity exists, None otherwise.
        """
        if node_id not in self._graph:
            return None

        results = self.detect(min_confidence=0.0)
        for candidate in results:
            if candidate.node_id == node_id:
                return candidate

        return None

    def _compute_centrality(self) -> None:
        """Precompute combined centrality scores for all nodes.

        Uses a combination of degree centrality and betweenness centrality
        to produce a single centrality measure.
        """
        if self._centrality_cache:
            return

        if self._graph.number_of_nodes() == 0:
            return

        # Compute degree centrality
        degree = nx.degree_centrality(self._graph)

        # Compute betweenness (approximate for large graphs)
        n = self._graph.number_of_nodes()
        k = min(n, 500)  # Sample for large graphs
        simple = nx.DiGraph(self._graph)
        betweenness = nx.betweenness_centrality(simple, normalized=True, k=k)

        # Combined score (weighted average)
        for node_id in self._graph.nodes():
            self._centrality_cache[node_id] = (
                0.4 * degree.get(node_id, 0.0)
                + 0.6 * betweenness.get(node_id, 0.0)
            )

    def _compute_node_volumes(self) -> dict[str, float]:
        """Compute total trade volume through each node.

        Returns:
            Mapping of node ID to total trade volume.
        """
        volumes: dict[str, float] = defaultdict(float)

        for u, v, data in self._graph.edges(data=True):
            val = data.get("total_value", 0.0)
            volumes[u] += val
            volumes[v] += val

        return dict(volumes)

    def _count_flagged_connections(self, node_id: str) -> int:
        """Count the number of flagged entities connected to a node.

        Args:
            node_id: The node to check.

        Returns:
            Number of directly connected flagged entities.
        """
        count = 0
        for neighbor in nx.all_neighbors(self._graph, node_id):
            if self._graph.nodes[neighbor].get("flagged", False):
                count += 1
            elif self._graph.nodes[neighbor].get("risk_score", 0.0) >= 0.6:
                count += 1
        return count

    def _count_unique_commodities(self, node_id: str) -> int:
        """Count the number of unique commodities traded by an entity.

        Traverses through declarations to find connected commodity nodes.

        Args:
            node_id: The entity node to check.

        Returns:
            Number of unique commodity nodes connected through declarations.
        """
        commodities: set[str] = set()

        # Entity -> Declaration -> Commodity
        for _, decl_id, data in self._graph.out_edges(node_id, data=True):
            if data.get("edge_type") == "DECLARES":
                for _, comm_id, comm_data in self._graph.out_edges(
                    decl_id, data=True
                ):
                    if comm_data.get("edge_type") == "TRADES_COMMODITY":
                        commodities.add(comm_id)

        return len(commodities)

    @staticmethod
    def _compute_registration_age(reg_date: str) -> int:
        """Compute the age in days since registration.

        Args:
            reg_date: Registration date as ISO format string.

        Returns:
            Number of days since registration, or -1 if date is invalid.
        """
        if not reg_date:
            return -1

        try:
            reg = datetime.fromisoformat(reg_date)
            return (datetime.now() - reg).days
        except (ValueError, TypeError):
            return -1
