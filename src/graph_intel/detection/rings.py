"""Fraud ring detection in trade entity graphs.

Identifies coordinated fraud patterns by detecting:
- Circular trade paths (A -> B -> C -> A) indicating round-tripping
- Shared-attribute clusters (entities sharing addresses, phones, bank accounts)
- Synchronized behavioral patterns (simultaneous filings, uniform values)

Fraud rings typically manifest as tightly connected subgraphs where
multiple entities collude to exploit customs processes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx
import numpy as np
import structlog

from graph_intel.config import RingDetectionConfig, Settings, get_settings
from graph_intel.graph.schema import EdgeType, NodeType

logger = structlog.get_logger(__name__)


@dataclass
class FraudRing:
    """Represents a detected potential fraud ring.

    Attributes:
        ring_id: Unique identifier for this ring.
        ring_type: Category of ring pattern (circular, shared_attribute, behavioral).
        members: Set of node IDs participating in the ring.
        evidence: List of evidence items supporting the detection.
        confidence: Confidence score (0.0 to 1.0).
        total_value: Total declared value involved in the ring.
        pattern_description: Human-readable description of the detected pattern.
        subgraph: The induced subgraph of ring members.
    """

    ring_id: str
    ring_type: str
    members: set[str]
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    total_value: float = 0.0
    pattern_description: str = ""
    subgraph: Optional[nx.MultiDiGraph] = field(default=None, repr=False)

    @property
    def size(self) -> int:
        """Number of entities in the ring."""
        return len(self.members)


@dataclass
class CircularTradePattern:
    """A detected circular trade path in the graph.

    Attributes:
        cycle: Ordered list of node IDs forming the cycle.
        length: Number of edges in the cycle.
        total_value: Sum of declared values along cycle edges.
        edge_types: List of edge types along the cycle.
        countries: Set of countries involved in the cycle.
        involves_shared_attributes: Whether cycle members share attributes.
    """

    cycle: list[str]
    length: int
    total_value: float = 0.0
    edge_types: list[str] = field(default_factory=list)
    countries: set[str] = field(default_factory=set)
    involves_shared_attributes: bool = False


@dataclass
class SharedAttributeCluster:
    """A cluster of entities sharing common attributes.

    Attributes:
        shared_node_id: The attribute node shared by cluster members.
        shared_node_type: Type of the shared attribute (ADDRESS, PHONE, etc.).
        connected_entities: Entity nodes connected through this attribute.
        entity_count: Number of entities sharing this attribute.
        attribute_value: The actual shared attribute value.
    """

    shared_node_id: str
    shared_node_type: str
    connected_entities: set[str]
    entity_count: int = 0
    attribute_value: str = ""

    def __post_init__(self) -> None:
        """Compute entity count from connected entities."""
        if self.entity_count == 0:
            self.entity_count = len(self.connected_entities)


class FraudRingDetector:
    """Detects fraud ring patterns in trade entity graphs.

    Combines multiple detection strategies:
    1. Circular trade detection (DFS-based cycle finding)
    2. Shared attribute clustering (co-location, shared contacts)
    3. Behavioral synchronization (filing patterns, value uniformity)

    Args:
        graph: The trade entity graph to analyze.
        config: Optional ring detection configuration.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        config: Optional[RingDetectionConfig] = None,
    ) -> None:
        """Initialize the fraud ring detector.

        Args:
            graph: The trade entity graph (NetworkX MultiDiGraph).
            config: Optional ring detection configuration.
        """
        self._graph = graph
        self._config = config or get_settings().detection.rings
        self._ring_counter = 0

    def find_circular_trade(
        self,
        max_depth: Optional[int] = None,
        node_types: Optional[list[NodeType]] = None,
        edge_types: Optional[list[EdgeType]] = None,
    ) -> list[CircularTradePattern]:
        """Find circular trade patterns (round-tripping) in the graph.

        Searches for directed cycles where goods flow from A -> B -> C -> A,
        potentially indicating carousel fraud or origin laundering. The search
        is bounded by depth to maintain tractability.

        Args:
            max_depth: Maximum cycle length to search for. Uses config if None.
            node_types: Optional filter for node types to consider in cycles.
                Default considers IMPORTER and EXPORTER nodes.
            edge_types: Optional filter for edge types to traverse.
                Default considers IMPORTS_FROM edges.

        Returns:
            List of CircularTradePattern objects, sorted by total value.
        """
        max_depth = max_depth or self._config.max_cycle_length

        if node_types is None:
            node_types = [NodeType.IMPORTER, NodeType.EXPORTER]
        if edge_types is None:
            edge_types = [EdgeType.IMPORTS_FROM]

        type_values = {nt.value for nt in node_types}
        edge_type_values = {et.value for et in edge_types}

        # Build a filtered subgraph for cycle detection
        filtered_nodes = [
            n
            for n, d in self._graph.nodes(data=True)
            if d.get("node_type") in type_values
        ]
        subgraph = self._graph.subgraph(filtered_nodes).copy()

        # Remove edges not matching the filter
        edges_to_remove = []
        for u, v, key, data in subgraph.edges(keys=True, data=True):
            if data.get("edge_type") not in edge_type_values:
                edges_to_remove.append((u, v, key))
        for u, v, key in edges_to_remove:
            subgraph.remove_edge(u, v, key=key)

        # Convert to simple DiGraph for cycle detection
        simple = nx.DiGraph()
        edge_data_map: dict[tuple[str, str], dict[str, Any]] = {}
        for u, v, data in subgraph.edges(data=True):
            if not simple.has_edge(u, v):
                simple.add_edge(u, v)
                edge_data_map[(u, v)] = data
            else:
                # Aggregate if multiple edges
                existing = edge_data_map[(u, v)]
                existing["total_value"] = existing.get("total_value", 0) + data.get(
                    "total_value", 0
                )

        # Find all simple cycles up to max_depth
        cycles = []
        try:
            for cycle in nx.simple_cycles(simple, length_bound=max_depth):
                if len(cycle) >= 3:
                    cycles.append(cycle)
        except Exception:
            # Fallback: DFS-based bounded cycle detection
            cycles = self._dfs_find_cycles(simple, max_depth)

        # Build CircularTradePattern objects
        patterns = []
        seen_cycles: set[frozenset[str]] = set()

        for cycle in cycles:
            # Deduplicate rotations of the same cycle
            cycle_set = frozenset(cycle)
            if cycle_set in seen_cycles:
                continue
            seen_cycles.add(cycle_set)

            # Compute cycle metrics
            total_value = 0.0
            edge_types_in_cycle = []
            countries: set[str] = set()

            for i in range(len(cycle)):
                src = cycle[i]
                tgt = cycle[(i + 1) % len(cycle)]

                if (src, tgt) in edge_data_map:
                    data = edge_data_map[(src, tgt)]
                    total_value += data.get("total_value", 0.0)
                    edge_types_in_cycle.append(data.get("edge_type", ""))

                # Collect countries
                if src in self._graph:
                    countries.add(
                        self._graph.nodes[src].get("country", "")
                    )

            # Check if cycle members share attributes
            has_shared = self._cycle_has_shared_attributes(cycle)

            pattern = CircularTradePattern(
                cycle=cycle,
                length=len(cycle),
                total_value=total_value,
                edge_types=edge_types_in_cycle,
                countries=countries - {""},
                involves_shared_attributes=has_shared,
            )
            patterns.append(pattern)

        patterns.sort(key=lambda p: p.total_value, reverse=True)

        logger.info(
            "circular_trade_detected",
            cycles_found=len(patterns),
            max_depth=max_depth,
        )

        return patterns

    def find_shared_attribute_clusters(
        self,
        attribute_types: Optional[list[NodeType]] = None,
        min_shared: int = 2,
    ) -> list[SharedAttributeCluster]:
        """Find clusters of entities sharing common attributes.

        Identifies groups of importers/exporters that share addresses,
        phone numbers, or bank accounts — a strong indicator of
        coordinated activity or common beneficial ownership.

        Args:
            attribute_types: Node types to check for sharing.
                Default: ADDRESS, PHONE, BANK_ACCOUNT.
            min_shared: Minimum number of entities sharing an attribute.

        Returns:
            List of SharedAttributeCluster objects.
        """
        if attribute_types is None:
            attribute_types = [NodeType.ADDRESS, NodeType.PHONE, NodeType.BANK_ACCOUNT]

        attr_type_values = {nt.value for nt in attribute_types}
        entity_types = {NodeType.IMPORTER.value, NodeType.EXPORTER.value, NodeType.AGENT.value}

        clusters: list[SharedAttributeCluster] = []

        # Find all attribute nodes
        for node_id, data in self._graph.nodes(data=True):
            if data.get("node_type") not in attr_type_values:
                continue

            # Find all entity nodes connected to this attribute
            connected = set()
            for source, _, edge_data in self._graph.in_edges(node_id, data=True):
                if source in self._graph:
                    source_type = self._graph.nodes[source].get("node_type")
                    if source_type in entity_types:
                        connected.add(source)

            if len(connected) >= min_shared:
                attr_value = data.get(
                    "normalized_address",
                    data.get(
                        "normalized_number",
                        data.get("account_hash", node_id),
                    ),
                )
                cluster = SharedAttributeCluster(
                    shared_node_id=node_id,
                    shared_node_type=data.get("node_type", "UNKNOWN"),
                    connected_entities=connected,
                    attribute_value=str(attr_value),
                )
                clusters.append(cluster)

        clusters.sort(key=lambda c: c.entity_count, reverse=True)

        logger.info(
            "shared_attribute_clusters_found",
            clusters=len(clusters),
            max_entities=clusters[0].entity_count if clusters else 0,
        )

        return clusters

    def detect_fraud_rings(
        self,
        min_confidence: float = 0.5,
    ) -> list[FraudRing]:
        """Run comprehensive fraud ring detection combining all strategies.

        Executes all detection methods and consolidates results into
        FraudRing objects with evidence and confidence scores.

        Args:
            min_confidence: Minimum confidence score to include a ring.

        Returns:
            List of FraudRing objects, sorted by confidence.
        """
        rings: list[FraudRing] = []

        # Strategy 1: Circular trade patterns
        circular = self.find_circular_trade()
        for pattern in circular:
            confidence = self._compute_circular_confidence(pattern)
            if confidence >= min_confidence:
                ring = self._pattern_to_ring(pattern, confidence)
                rings.append(ring)

        # Strategy 2: Shared attribute clusters
        clusters = self.find_shared_attribute_clusters()
        cluster_groups = self._merge_overlapping_clusters(clusters)

        for group_members, group_clusters in cluster_groups:
            confidence = self._compute_cluster_confidence(group_clusters)
            if confidence >= min_confidence:
                ring = self._cluster_group_to_ring(
                    group_members, group_clusters, confidence
                )
                rings.append(ring)

        # Strategy 3: Behavioral synchronization
        behavioral_rings = self._detect_behavioral_synchronization()
        for beh_ring in behavioral_rings:
            if beh_ring.confidence >= min_confidence:
                rings.append(beh_ring)

        rings.sort(key=lambda r: r.confidence, reverse=True)

        logger.info(
            "fraud_rings_detected",
            total_rings=len(rings),
            by_type={
                "circular": sum(1 for r in rings if r.ring_type == "circular"),
                "shared_attribute": sum(
                    1 for r in rings if r.ring_type == "shared_attribute"
                ),
                "behavioral": sum(1 for r in rings if r.ring_type == "behavioral"),
            },
        )

        return rings

    def _dfs_find_cycles(
        self, graph: nx.DiGraph, max_depth: int
    ) -> list[list[str]]:
        """Find cycles using bounded DFS traversal.

        Fallback method when simple_cycles is not available or too slow.

        Args:
            graph: Simple directed graph to search.
            max_depth: Maximum cycle length.

        Returns:
            List of cycles (each cycle is a list of node IDs).
        """
        cycles: list[list[str]] = []
        visited_global: set[frozenset[str]] = set()

        for start_node in graph.nodes():
            # DFS from each node
            stack: list[tuple[str, list[str]]] = [(start_node, [start_node])]

            while stack:
                current, path = stack.pop()

                if len(path) > max_depth:
                    continue

                for neighbor in graph.successors(current):
                    if neighbor == start_node and len(path) >= 3:
                        cycle_set = frozenset(path)
                        if cycle_set not in visited_global:
                            visited_global.add(cycle_set)
                            cycles.append(list(path))
                    elif neighbor not in path and len(path) < max_depth:
                        stack.append((neighbor, path + [neighbor]))

        return cycles

    def _cycle_has_shared_attributes(self, cycle: list[str]) -> bool:
        """Check if any nodes in the cycle share attribute nodes.

        Args:
            cycle: List of node IDs forming the cycle.

        Returns:
            True if any pair of cycle members shares an attribute node.
        """
        attr_types = {"ADDRESS", "PHONE", "BANK_ACCOUNT"}
        attribute_neighbors: dict[str, set[str]] = defaultdict(set)

        for node_id in cycle:
            if node_id in self._graph:
                for _, target, data in self._graph.out_edges(node_id, data=True):
                    if target in self._graph:
                        target_type = self._graph.nodes[target].get("node_type")
                        if target_type in attr_types:
                            attribute_neighbors[target].add(node_id)

        # Check if any attribute is shared by multiple cycle members
        return any(len(members) > 1 for members in attribute_neighbors.values())

    def _compute_circular_confidence(self, pattern: CircularTradePattern) -> float:
        """Compute confidence score for a circular trade pattern.

        Factors:
        - Shorter cycles are more suspicious
        - Shared attributes increase confidence
        - Higher value increases confidence
        - Cross-border cycles are more suspicious

        Args:
            pattern: The circular trade pattern.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        score = 0.0

        # Shorter cycles are more suspicious
        if pattern.length <= 3:
            score += 0.3
        elif pattern.length <= 4:
            score += 0.2
        else:
            score += 0.1

        # Shared attributes
        if pattern.involves_shared_attributes:
            score += 0.3

        # Cross-border (multiple countries)
        if len(pattern.countries) >= 2:
            score += 0.2

        # High value
        if pattern.total_value > 100000:
            score += 0.2
        elif pattern.total_value > 50000:
            score += 0.1

        return min(score, 1.0)

    def _compute_cluster_confidence(
        self, clusters: list[SharedAttributeCluster]
    ) -> float:
        """Compute confidence for a group of shared-attribute clusters.

        Args:
            clusters: The shared attribute clusters.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        score = 0.0

        # More shared attributes = higher confidence
        num_shared = len(clusters)
        score += min(num_shared * 0.2, 0.6)

        # Different types of shared attributes
        shared_types = set(c.shared_node_type for c in clusters)
        if len(shared_types) >= 2:
            score += 0.2

        # Large clusters
        max_entities = max(c.entity_count for c in clusters) if clusters else 0
        if max_entities >= 5:
            score += 0.2
        elif max_entities >= 3:
            score += 0.1

        return min(score, 1.0)

    def _pattern_to_ring(
        self,
        pattern: CircularTradePattern,
        confidence: float,
    ) -> FraudRing:
        """Convert a CircularTradePattern to a FraudRing.

        Args:
            pattern: The circular trade pattern.
            confidence: Pre-computed confidence score.

        Returns:
            FraudRing object.
        """
        self._ring_counter += 1
        members = set(pattern.cycle)

        evidence = [
            {
                "type": "circular_trade",
                "cycle": pattern.cycle,
                "length": pattern.length,
                "total_value": pattern.total_value,
                "countries": list(pattern.countries),
                "shared_attributes": pattern.involves_shared_attributes,
            }
        ]

        description = (
            f"Circular trade pattern: {' -> '.join(pattern.cycle[:5])}"
            f"{'...' if len(pattern.cycle) > 5 else ''} -> {pattern.cycle[0]}. "
            f"Total value: ${pattern.total_value:,.2f}. "
            f"Countries: {', '.join(pattern.countries)}."
        )

        return FraudRing(
            ring_id=f"RING-CIRC-{self._ring_counter:04d}",
            ring_type="circular",
            members=members,
            evidence=evidence,
            confidence=confidence,
            total_value=pattern.total_value,
            pattern_description=description,
            subgraph=self._graph.subgraph(members).copy(),
        )

    def _merge_overlapping_clusters(
        self, clusters: list[SharedAttributeCluster]
    ) -> list[tuple[set[str], list[SharedAttributeCluster]]]:
        """Merge clusters with overlapping entity members.

        Args:
            clusters: List of shared attribute clusters.

        Returns:
            List of (merged_members, component_clusters) tuples.
        """
        if not clusters:
            return []

        # Build a union-find structure based on shared entities
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union entities that appear in the same cluster
        for cluster in clusters:
            entities = list(cluster.connected_entities)
            for i in range(1, len(entities)):
                union(entities[0], entities[i])

        # Group clusters by their root entity
        groups: dict[str, tuple[set[str], list[SharedAttributeCluster]]] = {}
        for cluster in clusters:
            root = find(next(iter(cluster.connected_entities)))
            if root not in groups:
                groups[root] = (set(), [])
            groups[root][0].update(cluster.connected_entities)
            groups[root][1].append(cluster)

        # Filter groups by minimum shared attributes
        return [
            (members, group_clusters)
            for members, group_clusters in groups.values()
            if len(group_clusters) >= self._config.min_shared_attributes
        ]

    def _cluster_group_to_ring(
        self,
        members: set[str],
        clusters: list[SharedAttributeCluster],
        confidence: float,
    ) -> FraudRing:
        """Convert a merged cluster group to a FraudRing.

        Args:
            members: All entity members.
            clusters: Component shared attribute clusters.
            confidence: Pre-computed confidence score.

        Returns:
            FraudRing object.
        """
        self._ring_counter += 1

        evidence = [
            {
                "type": "shared_attribute",
                "attribute_type": c.shared_node_type,
                "attribute_value": c.attribute_value,
                "entity_count": c.entity_count,
            }
            for c in clusters
        ]

        shared_types = set(c.shared_node_type for c in clusters)
        description = (
            f"Shared attribute cluster: {len(members)} entities share "
            f"{len(clusters)} attributes ({', '.join(shared_types)}). "
            f"This pattern suggests common beneficial ownership or coordination."
        )

        # Compute total trade value for members
        total_value = 0.0
        for member in members:
            for _, _, data in self._graph.out_edges(member, data=True):
                total_value += data.get("total_value", 0.0)

        return FraudRing(
            ring_id=f"RING-ATTR-{self._ring_counter:04d}",
            ring_type="shared_attribute",
            members=members,
            evidence=evidence,
            confidence=confidence,
            total_value=total_value,
            pattern_description=description,
            subgraph=self._graph.subgraph(members).copy(),
        )

    def _detect_behavioral_synchronization(self) -> list[FraudRing]:
        """Detect entities with synchronized filing patterns.

        Identifies groups of importers that file declarations on the same
        dates, with the same values, for the same commodities — suggesting
        coordinated activity.

        Returns:
            List of FraudRing objects for behavioral patterns.
        """
        rings: list[FraudRing] = []

        # Group declarations by date
        date_groups: dict[str, list[str]] = defaultdict(list)
        for node_id, data in self._graph.nodes(data=True):
            if data.get("node_type") == "DECLARATION":
                date = data.get("date", "")
                if date:
                    date_groups[date].append(node_id)

        # Find dates with many declarations from different importers
        for date, decl_ids in date_groups.items():
            if len(decl_ids) < 3:
                continue

            # Get importers for these declarations
            importers_by_decl: dict[str, str] = {}
            values: list[float] = []

            for decl_id in decl_ids:
                for source, _, data in self._graph.in_edges(decl_id, data=True):
                    if data.get("edge_type") == EdgeType.DECLARES.value:
                        importers_by_decl[decl_id] = source
                        break
                decl_data = self._graph.nodes.get(decl_id, {})
                val = decl_data.get("value")
                if val is not None:
                    values.append(float(val))

            # Check for value uniformity
            unique_importers = set(importers_by_decl.values())
            if len(unique_importers) < 3:
                continue

            if len(values) >= 3:
                cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 1.0
                if cv < 0.05:  # Near-zero variance
                    self._ring_counter += 1
                    members = unique_importers | set(decl_ids)
                    confidence = 0.6 + (0.1 * min(len(unique_importers) - 3, 4))

                    rings.append(
                        FraudRing(
                            ring_id=f"RING-BEHV-{self._ring_counter:04d}",
                            ring_type="behavioral",
                            members=members,
                            evidence=[
                                {
                                    "type": "synchronized_filing",
                                    "date": date,
                                    "importer_count": len(unique_importers),
                                    "value_cv": float(cv),
                                    "mean_value": float(np.mean(values)),
                                }
                            ],
                            confidence=min(confidence, 1.0),
                            total_value=sum(values),
                            pattern_description=(
                                f"Synchronized filing: {len(unique_importers)} importers "
                                f"filed declarations on {date} with nearly identical values "
                                f"(CV={cv:.4f}, mean=${np.mean(values):,.2f})."
                            ),
                        )
                    )

        return rings
