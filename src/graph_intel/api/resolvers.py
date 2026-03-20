"""GraphQL resolvers for the Trade Intelligence Graph API.

Implements the query logic that connects GraphQL queries to the
underlying graph analysis engine. Each resolver translates a GraphQL
request into graph operations and returns typed response objects.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import networkx as nx
import strawberry

from graph_intel.analysis.centrality import CentralityAnalyzer
from graph_intel.analysis.community import CommunityDetector
from graph_intel.analysis.propagation import RiskPropagator
from graph_intel.api.graphql_schema import (
    AffectedNodeInfo,
    CommunityInfo,
    EdgeInfo,
    EdgeTypeCount,
    FraudRingInfo,
    GraphStatistics,
    NodeInfo,
    NodeTypeCount,
    PathInfo,
    PropagationResultInfo,
    RiskEntityInfo,
)
from graph_intel.detection.rings import FraudRingDetector
from graph_intel.graph.store import GraphStore, NetworkXStore


class GraphResolver:
    """Resolves GraphQL queries against the trade intelligence graph.

    Maintains a reference to the graph store and instantiates analysis
    components on demand.

    Args:
        store: The graph store to query against.
    """

    def __init__(self, store: GraphStore) -> None:
        """Initialize the resolver with a graph store.

        Args:
            store: The graph store providing data access.
        """
        self._store = store
        self._graph = store.get_networkx_graph()

    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """Resolve a single node by ID.

        Args:
            node_id: The node identifier.

        Returns:
            NodeInfo if found, None otherwise.
        """
        data = self._store.get_node(node_id)
        if data is None:
            return None

        return NodeInfo(
            node_id=data.get("node_id", node_id),
            node_type=data.get("node_type", "UNKNOWN"),
            risk_score=data.get("risk_score", 0.0),
            risk_level=data.get("risk_level", "LOW"),
            flagged=data.get("flagged", False),
            community_id=data.get("community_id"),
            name=data.get("name"),
            country=data.get("country"),
            tax_id=data.get("tax_id"),
        )

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "both",
        max_depth: int = 1,
    ) -> list[NodeInfo]:
        """Resolve neighbors of a node, optionally filtered.

        Args:
            node_id: The source node identifier.
            edge_type: Optional edge type filter.
            direction: Direction filter (out, in, both).
            max_depth: Maximum traversal depth.

        Returns:
            List of neighbor NodeInfo objects.
        """
        if max_depth == 1:
            from graph_intel.graph.schema import EdgeType

            et = None
            if edge_type:
                try:
                    et = EdgeType(edge_type)
                except ValueError:
                    pass

            neighbors = self._store.get_neighbors(node_id, et, direction)
            return [
                NodeInfo(
                    node_id=n.get("node_id", ""),
                    node_type=n.get("node_type", "UNKNOWN"),
                    risk_score=n.get("risk_score", 0.0),
                    risk_level=n.get("risk_level", "LOW"),
                    flagged=n.get("flagged", False),
                    community_id=n.get("community_id"),
                    name=n.get("name"),
                    country=n.get("country"),
                    tax_id=n.get("tax_id"),
                )
                for n in neighbors
            ]

        # Multi-hop traversal using BFS
        visited: set[str] = {node_id}
        current_level = {node_id}
        all_neighbors: list[dict[str, Any]] = []

        for _ in range(max_depth):
            next_level: set[str] = set()
            for nid in current_level:
                for neighbor in nx.all_neighbors(self._graph, nid):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
                        node_data = dict(self._graph.nodes[neighbor])
                        node_data["node_id"] = neighbor
                        all_neighbors.append(node_data)
            current_level = next_level

        return [
            NodeInfo(
                node_id=n.get("node_id", ""),
                node_type=n.get("node_type", "UNKNOWN"),
                risk_score=n.get("risk_score", 0.0),
                risk_level=n.get("risk_level", "LOW"),
                flagged=n.get("flagged", False),
                community_id=n.get("community_id"),
                name=n.get("name"),
                country=n.get("country"),
                tax_id=n.get("tax_id"),
            )
            for n in all_neighbors
        ]

    def get_shortest_path(
        self,
        source_id: str,
        target_id: str,
        weighted: bool = False,
    ) -> PathInfo:
        """Find the shortest path between two nodes.

        Args:
            source_id: Source node identifier.
            target_id: Target node identifier.
            weighted: Whether to use edge weights.

        Returns:
            PathInfo with the path and its length.
        """
        weight = "weight" if weighted else None
        path = self._store.shortest_path(source_id, target_id, weight)

        if path is None:
            return PathInfo(
                source_id=source_id,
                target_id=target_id,
                path=[],
                length=0,
                exists=False,
            )

        return PathInfo(
            source_id=source_id,
            target_id=target_id,
            path=path,
            length=len(path) - 1,
            exists=True,
        )

    def get_communities(
        self,
        algorithm: str = "louvain",
        resolution: float = 1.0,
        min_size: int = 3,
    ) -> list[CommunityInfo]:
        """Detect and return communities in the graph.

        Args:
            algorithm: Detection algorithm (louvain, leiden, label_propagation).
            resolution: Resolution parameter for modularity.
            min_size: Minimum community size.

        Returns:
            List of CommunityInfo objects.
        """
        detector = CommunityDetector(self._graph)
        result = detector.detect(algorithm=algorithm, resolution=resolution)

        communities = []
        for comm in result.communities:
            if comm.size < min_size:
                continue

            type_dist = [
                NodeTypeCount(node_type=nt, count=count)
                for nt, count in comm.node_types.items()
            ]

            communities.append(
                CommunityInfo(
                    community_id=comm.community_id,
                    size=comm.size,
                    density=comm.density,
                    internal_edges=comm.internal_edges,
                    external_edges=comm.external_edges,
                    risk_score=comm.risk_score,
                    members=list(comm.members)[:100],  # Limit for API
                    node_type_distribution=type_dist,
                    anomaly_indicators=comm.metadata.get(
                        "anomaly_indicators", []
                    ),
                )
            )

        return communities

    def get_high_risk_entities(
        self,
        min_risk_score: float = 0.5,
        limit: int = 50,
        node_type: Optional[str] = None,
    ) -> list[RiskEntityInfo]:
        """Get entities above a risk score threshold.

        Args:
            min_risk_score: Minimum risk score filter.
            limit: Maximum number of entities to return.
            node_type: Optional node type filter.

        Returns:
            List of RiskEntityInfo objects, sorted by risk score.
        """
        entities = []

        for nid, data in self._graph.nodes(data=True):
            score = data.get("risk_score", 0.0)
            if score < min_risk_score:
                continue
            if node_type and data.get("node_type") != node_type:
                continue

            neighbor_count = self._graph.degree(nid)

            indicators: list[str] = []
            if data.get("flagged"):
                indicators.append("directly_flagged")
            if score >= 0.8:
                indicators.append("critical_risk")
            if neighbor_count > 20:
                indicators.append("high_connectivity")

            entities.append(
                RiskEntityInfo(
                    node_id=nid,
                    node_type=data.get("node_type", "UNKNOWN"),
                    risk_score=score,
                    risk_level=data.get("risk_level", "LOW"),
                    flagged=data.get("flagged", False),
                    indicators=indicators,
                    neighbor_count=neighbor_count,
                )
            )

        entities.sort(key=lambda e: e.risk_score, reverse=True)
        return entities[:limit]

    def get_graph_statistics(self) -> GraphStatistics:
        """Get overall graph statistics.

        Returns:
            GraphStatistics with node/edge counts and distributions.
        """
        node_types: dict[str, int] = defaultdict(int)
        for _, data in self._graph.nodes(data=True):
            node_types[data.get("node_type", "UNKNOWN")] += 1

        edge_types: dict[str, int] = defaultdict(int)
        for _, _, data in self._graph.edges(data=True):
            edge_types[data.get("edge_type", "UNKNOWN")] += 1

        wcc = (
            nx.number_weakly_connected_components(self._graph)
            if self._graph.number_of_nodes() > 0
            else 0
        )

        return GraphStatistics(
            total_nodes=self._graph.number_of_nodes(),
            total_edges=self._graph.number_of_edges(),
            density=nx.density(self._graph),
            weakly_connected_components=wcc,
            node_type_distribution=[
                NodeTypeCount(node_type=nt, count=c) for nt, c in node_types.items()
            ],
            edge_type_distribution=[
                EdgeTypeCount(edge_type=et, count=c) for et, c in edge_types.items()
            ],
            declaration_count=self._graph.graph.get("declaration_count", 0),
        )

    def detect_fraud_rings(
        self,
        min_confidence: float = 0.5,
    ) -> list[FraudRingInfo]:
        """Run fraud ring detection and return results.

        Args:
            min_confidence: Minimum confidence threshold.

        Returns:
            List of FraudRingInfo objects.
        """
        detector = FraudRingDetector(self._graph)
        rings = detector.detect_fraud_rings(min_confidence=min_confidence)

        return [
            FraudRingInfo(
                ring_id=ring.ring_id,
                ring_type=ring.ring_type,
                size=ring.size,
                confidence=ring.confidence,
                total_value=ring.total_value,
                pattern_description=ring.pattern_description,
                members=list(ring.members)[:50],
                evidence_count=len(ring.evidence),
            )
            for ring in rings
        ]

    def propagate_risk(
        self,
        source_node: str,
        risk_score: float = 0.8,
        max_depth: int = 4,
    ) -> PropagationResultInfo:
        """Propagate risk from a source node.

        Args:
            source_node: The node to propagate from.
            risk_score: The risk score to propagate.
            max_depth: Maximum propagation depth.

        Returns:
            PropagationResultInfo with affected nodes.
        """
        propagator = RiskPropagator(self._graph)
        result = propagator.propagate_from_node(
            source_node=source_node,
            risk_score=risk_score,
            max_depth=max_depth,
        )

        # Get top affected nodes
        sorted_affected = sorted(
            result.affected_nodes.items(), key=lambda x: x[1], reverse=True
        )[:20]

        top_affected = [
            AffectedNodeInfo(
                node_id=nid,
                risk_score=score,
                path=result.propagation_paths.get(nid, []),
            )
            for nid, score in sorted_affected
        ]

        return PropagationResultInfo(
            source_node=result.source_node,
            source_risk=result.source_risk,
            affected_count=len(result.affected_nodes),
            max_depth_reached=result.max_depth_reached,
            total_risk_distributed=result.total_risk_distributed,
            top_affected=top_affected,
        )
