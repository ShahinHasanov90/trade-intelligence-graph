"""Carousel fraud detection in trade entity graphs.

Carousel fraud (also known as missing trader fraud) exploits VAT refund
mechanisms across borders. Goods are repeatedly imported and exported
through a chain of companies, collecting VAT refunds at each step.
The goods eventually return to their origin, completing the carousel.

Detection approach:
1. Find directed cycles in the trade graph
2. Filter for cycles that cross VAT-relevant borders
3. Check for temporal feasibility (transactions within plausible timeframes)
4. Identify the "missing trader" — the entity that disappears with collected VAT
5. Score cycles by value, velocity, and structural indicators
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import networkx as nx
import numpy as np
import structlog

from graph_intel.config import CarouselConfig, Settings, get_settings
from graph_intel.graph.schema import EdgeType, NodeType

logger = structlog.get_logger(__name__)


@dataclass
class CarouselPattern:
    """A detected carousel fraud pattern.

    Attributes:
        pattern_id: Unique identifier for this pattern.
        cycle: Ordered list of entity node IDs forming the carousel.
        cycle_length: Number of entities in the carousel.
        total_value: Total declared value across all carousel transactions.
        estimated_vat_exposure: Estimated VAT at risk.
        countries: Countries involved in the carousel.
        border_crossings: Number of cross-border edges in the cycle.
        temporal_span_days: Number of days from first to last transaction.
        velocity: Transactions per day within the carousel.
        missing_trader_candidates: Entity IDs likely to be missing traders.
        confidence: Detection confidence (0.0 to 1.0).
        evidence: Supporting evidence for the detection.
        transaction_dates: Dates of transactions in the carousel.
    """

    pattern_id: str
    cycle: list[str]
    cycle_length: int
    total_value: float = 0.0
    estimated_vat_exposure: float = 0.0
    countries: list[str] = field(default_factory=list)
    border_crossings: int = 0
    temporal_span_days: int = 0
    velocity: float = 0.0
    missing_trader_candidates: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    transaction_dates: list[str] = field(default_factory=list)


class CarouselDetector:
    """Detects carousel fraud patterns in trade entity graphs.

    Combines cycle detection with border-crossing analysis, temporal
    feasibility checks, and missing trader identification to find
    carousel fraud patterns.

    Args:
        graph: The trade entity graph to analyze.
        config: Optional carousel detection configuration.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        config: Optional[CarouselConfig] = None,
    ) -> None:
        """Initialize the carousel fraud detector.

        Args:
            graph: The trade entity graph (NetworkX MultiDiGraph).
            config: Optional carousel detection configuration.
        """
        self._graph = graph
        self._config = config or get_settings().detection.carousel
        self._pattern_counter = 0

    def detect(
        self,
        min_confidence: float = 0.5,
        max_cycle_length: int = 6,
    ) -> list[CarouselPattern]:
        """Run carousel fraud detection.

        Finds directed cycles involving cross-border trade in VAT-relevant
        countries, then scores each cycle based on multiple indicators.

        Args:
            min_confidence: Minimum confidence score to include.
            max_cycle_length: Maximum cycle length to search.

        Returns:
            List of CarouselPattern objects, sorted by confidence.
        """
        patterns: list[CarouselPattern] = []

        # Step 1: Find trade cycles
        cycles = self._find_trade_cycles(max_cycle_length)

        for cycle in cycles:
            # Step 2: Analyze each cycle
            pattern = self._analyze_cycle(cycle)
            if pattern is None:
                continue

            # Step 3: Score the pattern
            pattern.confidence = self._compute_confidence(pattern)

            if pattern.confidence >= min_confidence:
                patterns.append(pattern)

        patterns.sort(key=lambda p: p.confidence, reverse=True)

        logger.info(
            "carousel_detection_complete",
            cycles_examined=len(cycles),
            patterns_found=len(patterns),
        )

        return patterns

    def find_missing_traders(
        self,
        pattern: CarouselPattern,
    ) -> list[dict[str, Any]]:
        """Identify potential missing traders within a carousel pattern.

        The missing trader is typically the entity that:
        - Has the shortest lifespan in the network
        - Collects VAT on imports but never remits it
        - Disappears after a burst of trading activity
        - Has minimal assets or physical presence

        Args:
            pattern: The carousel pattern to analyze.

        Returns:
            List of candidate missing traders with scoring details.
        """
        candidates = []

        for node_id in pattern.cycle:
            if node_id not in self._graph:
                continue

            data = self._graph.nodes[node_id]
            indicators: list[str] = []
            score = 0.0

            # Check for recent registration
            reg_date = data.get("registration_date", "")
            if reg_date:
                try:
                    days_since = (datetime.now() - datetime.fromisoformat(reg_date)).days
                    if days_since < 180:
                        indicators.append("recently_registered")
                        score += 0.3
                except (ValueError, TypeError):
                    pass

            # Check trade asymmetry (imports >> exports or vice versa)
            in_value = sum(
                d.get("total_value", 0.0)
                for _, _, d in self._graph.in_edges(node_id, data=True)
                if d.get("edge_type") == EdgeType.IMPORTS_FROM.value
            )
            out_value = sum(
                d.get("total_value", 0.0)
                for _, _, d in self._graph.out_edges(node_id, data=True)
                if d.get("edge_type") == EdgeType.IMPORTS_FROM.value
            )

            total = in_value + out_value
            if total > 0:
                asymmetry = abs(in_value - out_value) / total
                if asymmetry > 0.8:
                    indicators.append("trade_asymmetry")
                    score += 0.2

            # Check for FTZ registration
            country = data.get("country", "")
            if country in self._config.vat_countries:
                # Entities in VAT countries that import but don't export
                # are potential missing traders
                if in_value > out_value * 2:
                    indicators.append("vat_country_importer")
                    score += 0.2

            # Check for low degree (minimal other relationships)
            degree = self._graph.degree(node_id)
            cycle_edges = 2  # Each cycle member has at least 2 cycle edges
            if degree <= cycle_edges + 2:
                indicators.append("minimal_external_connections")
                score += 0.15

            # Check for high risk score from propagation
            risk = data.get("risk_score", 0.0)
            if risk >= 0.5:
                indicators.append("elevated_risk_score")
                score += 0.15

            candidates.append(
                {
                    "node_id": node_id,
                    "entity_name": data.get("name", "Unknown"),
                    "entity_type": data.get("node_type", "UNKNOWN"),
                    "country": country,
                    "missing_trader_score": min(score, 1.0),
                    "indicators": indicators,
                    "import_value": in_value,
                    "export_value": out_value,
                }
            )

        candidates.sort(key=lambda c: c["missing_trader_score"], reverse=True)
        return candidates

    def _find_trade_cycles(self, max_length: int) -> list[list[str]]:
        """Find directed trade cycles in the graph.

        Builds a filtered subgraph of trade relationships (IMPORTS_FROM)
        between entity nodes and finds simple cycles.

        Args:
            max_length: Maximum cycle length to search.

        Returns:
            List of cycles (each cycle is a list of node IDs).
        """
        entity_types = {
            NodeType.IMPORTER.value,
            NodeType.EXPORTER.value,
        }

        # Build filtered subgraph
        trade_nodes = [
            n
            for n, d in self._graph.nodes(data=True)
            if d.get("node_type") in entity_types
        ]

        simple = nx.DiGraph()
        for u, v, data in self._graph.edges(data=True):
            if data.get("edge_type") == EdgeType.IMPORTS_FROM.value:
                if u in trade_nodes and v in trade_nodes:
                    if not simple.has_edge(u, v):
                        simple.add_edge(u, v, **data)
                    else:
                        # Aggregate values
                        simple[u][v]["total_value"] = simple[u][v].get(
                            "total_value", 0.0
                        ) + data.get("total_value", 0.0)

        # Find simple cycles
        cycles = []
        try:
            for cycle in nx.simple_cycles(simple, length_bound=max_length):
                if len(cycle) >= 3:
                    cycles.append(cycle)
        except Exception:
            # Fallback for older NetworkX versions
            for cycle in self._bounded_simple_cycles(simple, max_length):
                cycles.append(cycle)

        return cycles

    def _analyze_cycle(self, cycle: list[str]) -> Optional[CarouselPattern]:
        """Analyze a detected cycle for carousel fraud indicators.

        Args:
            cycle: Ordered list of node IDs forming the cycle.

        Returns:
            CarouselPattern if the cycle meets minimum criteria, None otherwise.
        """
        self._pattern_counter += 1

        countries: list[str] = []
        border_crossings = 0
        total_value = 0.0
        transaction_dates: list[str] = []

        for i in range(len(cycle)):
            src = cycle[i]
            tgt = cycle[(i + 1) % len(cycle)]

            # Get country for each node
            if src in self._graph:
                src_country = self._graph.nodes[src].get("country", "")
                if src_country and src_country not in countries:
                    countries.append(src_country)

            # Get edge data
            for _, _, data in self._graph.edges(src, data=True):
                if data.get("edge_type") == EdgeType.IMPORTS_FROM.value:
                    total_value += data.get("total_value", 0.0)
                    last_seen = data.get("last_seen", "")
                    if last_seen:
                        transaction_dates.append(last_seen)

            # Count border crossings
            if src in self._graph and tgt in self._graph:
                src_c = self._graph.nodes[src].get("country", "")
                tgt_c = self._graph.nodes[tgt].get("country", "")
                if src_c and tgt_c and src_c != tgt_c:
                    border_crossings += 1

        # Filter: must cross at least one border
        if border_crossings == 0:
            return None

        # Filter: must meet minimum value
        if total_value < self._config.min_cycle_value:
            return None

        # Filter: at least one VAT-relevant country
        vat_countries_involved = [
            c for c in countries if c in self._config.vat_countries
        ]
        if not vat_countries_involved:
            return None

        # Compute temporal span
        temporal_span = 0
        if len(transaction_dates) >= 2:
            try:
                dates = sorted(
                    datetime.fromisoformat(d) for d in transaction_dates if d
                )
                if len(dates) >= 2:
                    temporal_span = (dates[-1] - dates[0]).days
            except (ValueError, TypeError):
                pass

        # Check temporal feasibility
        if temporal_span > self._config.max_cycle_days:
            return None

        # Estimate VAT exposure (simplified: 20% average VAT rate)
        vat_rate = 0.20
        estimated_vat = total_value * vat_rate * border_crossings

        # Velocity (transactions per day)
        velocity = len(cycle) / max(temporal_span, 1)

        # Identify missing trader candidates
        missing_trader_ids = []
        for node_id in cycle:
            if node_id in self._graph:
                data = self._graph.nodes[node_id]
                # Simple heuristic: node with fewest connections beyond the cycle
                degree = self._graph.degree(node_id)
                if degree <= 4:  # Minimal connections
                    missing_trader_ids.append(node_id)

        evidence = [
            {
                "type": "carousel_cycle",
                "cycle_entities": cycle,
                "border_crossings": border_crossings,
                "countries": countries,
                "vat_countries": vat_countries_involved,
                "total_value": total_value,
                "temporal_span_days": temporal_span,
            }
        ]

        return CarouselPattern(
            pattern_id=f"CAROUSEL-{self._pattern_counter:04d}",
            cycle=cycle,
            cycle_length=len(cycle),
            total_value=total_value,
            estimated_vat_exposure=estimated_vat,
            countries=countries,
            border_crossings=border_crossings,
            temporal_span_days=temporal_span,
            velocity=velocity,
            missing_trader_candidates=missing_trader_ids,
            evidence=evidence,
            transaction_dates=transaction_dates,
        )

    def _compute_confidence(self, pattern: CarouselPattern) -> float:
        """Compute confidence score for a carousel pattern.

        Factors:
        - Multiple border crossings increase confidence
        - Higher value increases confidence
        - Shorter temporal span (faster cycling) increases confidence
        - Presence of missing trader candidates increases confidence
        - Shared attributes between cycle members increase confidence

        Args:
            pattern: The carousel pattern to score.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        score = 0.0

        # Border crossings (multiple crossings = more suspicious)
        if pattern.border_crossings >= 3:
            score += 0.25
        elif pattern.border_crossings >= 2:
            score += 0.2
        else:
            score += 0.1

        # Value
        if pattern.total_value >= 500000:
            score += 0.2
        elif pattern.total_value >= 100000:
            score += 0.15
        elif pattern.total_value >= 50000:
            score += 0.1

        # Temporal velocity (faster cycling = more suspicious)
        if pattern.velocity > 0.5:
            score += 0.15
        elif pattern.velocity > 0.1:
            score += 0.1

        # Missing trader candidates
        if pattern.missing_trader_candidates:
            score += 0.15

        # Short cycle (3-4 entities is classic carousel)
        if pattern.cycle_length <= 4:
            score += 0.15
        elif pattern.cycle_length <= 5:
            score += 0.1

        # Check for shared attributes between cycle members
        if self._cycle_has_shared_attributes(pattern.cycle):
            score += 0.1

        return min(score, 1.0)

    def _cycle_has_shared_attributes(self, cycle: list[str]) -> bool:
        """Check if cycle members share addresses, phones, or bank accounts.

        Args:
            cycle: List of node IDs forming the cycle.

        Returns:
            True if any attribute is shared between cycle members.
        """
        attr_types = {"ADDRESS", "PHONE", "BANK_ACCOUNT"}
        attr_connections: dict[str, set[str]] = defaultdict(set)

        for node_id in cycle:
            if node_id in self._graph:
                for _, target, data in self._graph.out_edges(node_id, data=True):
                    if target in self._graph:
                        if self._graph.nodes[target].get("node_type") in attr_types:
                            attr_connections[target].add(node_id)

        return any(len(members) > 1 for members in attr_connections.values())

    @staticmethod
    def _bounded_simple_cycles(
        graph: nx.DiGraph, max_length: int
    ) -> list[list[str]]:
        """Find simple cycles with bounded length using DFS.

        Fallback for environments where nx.simple_cycles doesn't
        support length_bound.

        Args:
            graph: Directed graph to search.
            max_length: Maximum cycle length.

        Returns:
            List of cycles found.
        """
        cycles: list[list[str]] = []
        seen: set[frozenset[str]] = set()

        for start in graph.nodes():
            stack: list[tuple[str, list[str]]] = [(start, [start])]
            while stack:
                current, path = stack.pop()
                if len(path) > max_length:
                    continue
                for neighbor in graph.successors(current):
                    if neighbor == start and len(path) >= 3:
                        key = frozenset(path)
                        if key not in seen:
                            seen.add(key)
                            cycles.append(list(path))
                    elif neighbor not in path and len(path) < max_length:
                        stack.append((neighbor, path + [neighbor]))

        return cycles
