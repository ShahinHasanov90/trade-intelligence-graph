"""Graph schema definitions for trade intelligence networks.

Defines the ontology of node types (entities) and edge types (relationships)
that compose a trade graph. Each type carries semantic meaning used by
detection algorithms to identify fraud patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class NodeType(str, Enum):
    """Types of entities (nodes) in the trade graph.

    Each node type represents a distinct class of entity extracted
    from customs declaration data.
    """

    IMPORTER = "IMPORTER"
    EXPORTER = "EXPORTER"
    COMMODITY = "COMMODITY"
    ROUTE = "ROUTE"
    AGENT = "AGENT"
    ADDRESS = "ADDRESS"
    PHONE = "PHONE"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    DECLARATION = "DECLARATION"


class EdgeType(str, Enum):
    """Types of relationships (edges) in the trade graph.

    Each edge type represents a semantic relationship between two entities.
    Edges are directed and may carry weight attributes.
    """

    IMPORTS_FROM = "IMPORTS_FROM"
    DECLARES = "DECLARES"
    TRADES_COMMODITY = "TRADES_COMMODITY"
    USES_ROUTE = "USES_ROUTE"
    REPRESENTED_BY = "REPRESENTED_BY"
    LOCATED_AT = "LOCATED_AT"
    CONTACTABLE_VIA = "CONTACTABLE_VIA"
    PAYS_THROUGH = "PAYS_THROUGH"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"


class RiskLevel(str, Enum):
    """Risk classification levels for entities and relationships."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class NodeSchema:
    """Schema definition for a node type, specifying required and optional attributes.

    Attributes:
        node_type: The category of this node.
        required_fields: Field names that must be present on every node of this type.
        optional_fields: Field names that may be present.
        id_field: The field used as the unique identifier for this node type.
    """

    node_type: NodeType
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()
    id_field: str = "id"


@dataclass(frozen=True)
class EdgeSchema:
    """Schema definition for an edge type, specifying source/target node types.

    Attributes:
        edge_type: The category of this edge.
        source_type: Allowed node type(s) for the source end.
        target_type: Allowed node type(s) for the target end.
        weight_field: The attribute name that carries the edge weight.
        directed: Whether this edge type is inherently directed.
    """

    edge_type: EdgeType
    source_type: tuple[NodeType, ...]
    target_type: tuple[NodeType, ...]
    weight_field: str = "weight"
    directed: bool = True


@dataclass
class NodeData:
    """Data payload for a graph node.

    Attributes:
        node_id: Unique identifier for this node.
        node_type: The type/category of this node.
        attributes: Dictionary of node attributes.
        risk_score: Current risk score (0.0 = safe, 1.0 = highest risk).
        risk_level: Derived risk classification.
        flagged: Whether this node has been explicitly flagged.
        community_id: Community assignment (set by community detection).
    """

    node_id: str
    node_type: NodeType
    attributes: dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    flagged: bool = False
    community_id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize node data to a dictionary.

        Returns:
            Dictionary representation of the node.
        """
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "flagged": self.flagged,
            "community_id": self.community_id,
            **self.attributes,
        }


@dataclass
class EdgeData:
    """Data payload for a graph edge.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        edge_type: The type/category of this edge.
        weight: Edge weight (higher = stronger relationship).
        attributes: Dictionary of edge attributes.
        transaction_count: Number of transactions this edge represents.
        total_value: Cumulative declared value across transactions.
        first_seen: ISO date string of first transaction.
        last_seen: ISO date string of most recent transaction.
    """

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)
    transaction_count: int = 0
    total_value: float = 0.0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize edge data to a dictionary.

        Returns:
            Dictionary representation of the edge.
        """
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "transaction_count": self.transaction_count,
            "total_value": self.total_value,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            **self.attributes,
        }


# Canonical schema registry
# Maps each node type to its schema definition
NODE_SCHEMAS: dict[NodeType, NodeSchema] = {
    NodeType.IMPORTER: NodeSchema(
        node_type=NodeType.IMPORTER,
        required_fields=("tax_id", "name", "country"),
        optional_fields=("address", "registration_date", "phone"),
        id_field="tax_id",
    ),
    NodeType.EXPORTER: NodeSchema(
        node_type=NodeType.EXPORTER,
        required_fields=("tax_id", "name", "country"),
        optional_fields=("address", "phone"),
        id_field="tax_id",
    ),
    NodeType.COMMODITY: NodeSchema(
        node_type=NodeType.COMMODITY,
        required_fields=("hs_code",),
        optional_fields=("description", "unit_price_range"),
        id_field="hs_code",
    ),
    NodeType.ROUTE: NodeSchema(
        node_type=NodeType.ROUTE,
        required_fields=("origin", "destination"),
        optional_fields=("transit_points",),
        id_field="origin-destination",
    ),
    NodeType.AGENT: NodeSchema(
        node_type=NodeType.AGENT,
        required_fields=("license_id", "name"),
        optional_fields=("type",),
        id_field="license_id",
    ),
    NodeType.ADDRESS: NodeSchema(
        node_type=NodeType.ADDRESS,
        required_fields=("normalized_address",),
        optional_fields=("country", "postal_code"),
        id_field="normalized_address",
    ),
    NodeType.PHONE: NodeSchema(
        node_type=NodeType.PHONE,
        required_fields=("normalized_number",),
        optional_fields=(),
        id_field="normalized_number",
    ),
    NodeType.BANK_ACCOUNT: NodeSchema(
        node_type=NodeType.BANK_ACCOUNT,
        required_fields=("bank_name", "account_hash"),
        optional_fields=(),
        id_field="account_hash",
    ),
    NodeType.DECLARATION: NodeSchema(
        node_type=NodeType.DECLARATION,
        required_fields=("declaration_id", "date"),
        optional_fields=("value", "weight", "quantity"),
        id_field="declaration_id",
    ),
}

# Maps each edge type to its schema definition
EDGE_SCHEMAS: dict[EdgeType, EdgeSchema] = {
    EdgeType.IMPORTS_FROM: EdgeSchema(
        edge_type=EdgeType.IMPORTS_FROM,
        source_type=(NodeType.IMPORTER,),
        target_type=(NodeType.EXPORTER,),
        weight_field="trade_volume",
    ),
    EdgeType.DECLARES: EdgeSchema(
        edge_type=EdgeType.DECLARES,
        source_type=(NodeType.IMPORTER,),
        target_type=(NodeType.DECLARATION,),
    ),
    EdgeType.TRADES_COMMODITY: EdgeSchema(
        edge_type=EdgeType.TRADES_COMMODITY,
        source_type=(NodeType.DECLARATION,),
        target_type=(NodeType.COMMODITY,),
    ),
    EdgeType.USES_ROUTE: EdgeSchema(
        edge_type=EdgeType.USES_ROUTE,
        source_type=(NodeType.DECLARATION,),
        target_type=(NodeType.ROUTE,),
    ),
    EdgeType.REPRESENTED_BY: EdgeSchema(
        edge_type=EdgeType.REPRESENTED_BY,
        source_type=(NodeType.IMPORTER,),
        target_type=(NodeType.AGENT,),
    ),
    EdgeType.LOCATED_AT: EdgeSchema(
        edge_type=EdgeType.LOCATED_AT,
        source_type=(NodeType.IMPORTER, NodeType.EXPORTER, NodeType.AGENT),
        target_type=(NodeType.ADDRESS,),
    ),
    EdgeType.CONTACTABLE_VIA: EdgeSchema(
        edge_type=EdgeType.CONTACTABLE_VIA,
        source_type=(NodeType.IMPORTER, NodeType.EXPORTER, NodeType.AGENT),
        target_type=(NodeType.PHONE,),
    ),
    EdgeType.PAYS_THROUGH: EdgeSchema(
        edge_type=EdgeType.PAYS_THROUGH,
        source_type=(NodeType.IMPORTER, NodeType.EXPORTER),
        target_type=(NodeType.BANK_ACCOUNT,),
    ),
    EdgeType.CO_OCCURS_WITH: EdgeSchema(
        edge_type=EdgeType.CO_OCCURS_WITH,
        source_type=tuple(NodeType),
        target_type=tuple(NodeType),
        directed=False,
    ),
}
