"""Strawberry GraphQL schema definitions for the Trade Intelligence Graph API.

Defines the GraphQL types that expose graph entities, relationships,
communities, and risk assessments to API consumers.
"""

from __future__ import annotations

from typing import Optional

import strawberry


@strawberry.type
class NodeInfo:
    """GraphQL type representing a graph node (entity)."""

    node_id: str
    node_type: str
    risk_score: float
    risk_level: str
    flagged: bool
    community_id: Optional[int] = None
    name: Optional[str] = None
    country: Optional[str] = None
    tax_id: Optional[str] = None


@strawberry.type
class EdgeInfo:
    """GraphQL type representing a graph edge (relationship)."""

    source_id: str
    target_id: str
    edge_type: str
    weight: float
    transaction_count: int
    total_value: float
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


@strawberry.type
class CommunityInfo:
    """GraphQL type representing a detected community."""

    community_id: int
    size: int
    density: float
    internal_edges: int
    external_edges: int
    risk_score: float
    members: list[str]
    node_type_distribution: list[NodeTypeCount]
    anomaly_indicators: list[str]


@strawberry.type
class NodeTypeCount:
    """Count of nodes by type within a community."""

    node_type: str
    count: int


@strawberry.type
class PathInfo:
    """GraphQL type representing a path between two nodes."""

    source_id: str
    target_id: str
    path: list[str]
    length: int
    exists: bool


@strawberry.type
class RiskEntityInfo:
    """GraphQL type for high-risk entity listings."""

    node_id: str
    node_type: str
    risk_score: float
    risk_level: str
    flagged: bool
    indicators: list[str]
    neighbor_count: int


@strawberry.type
class GraphStatistics:
    """GraphQL type for overall graph statistics."""

    total_nodes: int
    total_edges: int
    density: float
    weakly_connected_components: int
    node_type_distribution: list[NodeTypeCount]
    edge_type_distribution: list[EdgeTypeCount]
    declaration_count: int


@strawberry.type
class EdgeTypeCount:
    """Count of edges by type."""

    edge_type: str
    count: int


@strawberry.type
class FraudRingInfo:
    """GraphQL type for detected fraud rings."""

    ring_id: str
    ring_type: str
    size: int
    confidence: float
    total_value: float
    pattern_description: str
    members: list[str]
    evidence_count: int


@strawberry.type
class PropagationResultInfo:
    """GraphQL type for risk propagation results."""

    source_node: str
    source_risk: float
    affected_count: int
    max_depth_reached: int
    total_risk_distributed: float
    top_affected: list[AffectedNodeInfo]


@strawberry.type
class AffectedNodeInfo:
    """GraphQL type for a node affected by risk propagation."""

    node_id: str
    risk_score: float
    path: list[str]
