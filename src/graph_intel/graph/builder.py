"""Trade graph construction from customs declaration data.

The TradeGraphBuilder is the primary entry point for converting raw declaration
records into a richly typed, weighted NetworkX MultiDiGraph. Each declaration
generates multiple nodes (importer, exporter, commodity, route, agent) and
edges representing their relationships.

Usage:
    builder = TradeGraphBuilder()
    builder.add_declaration({...})
    graph = builder.build()
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, Sequence

import networkx as nx
import structlog

from graph_intel.config import Settings, get_settings
from graph_intel.graph.schema import (
    EdgeData,
    EdgeType,
    NodeData,
    NodeType,
    RiskLevel,
)

logger = structlog.get_logger(__name__)


class TradeGraphBuilder:
    """Constructs a trade entity graph from customs declaration records.

    The builder accumulates declarations and produces a NetworkX MultiDiGraph
    where nodes represent trade entities and edges represent their relationships.
    Duplicate entities are resolved by identifier, and edge weights are
    aggregated across multiple transactions.

    Attributes:
        graph: The underlying NetworkX MultiDiGraph being constructed.
        settings: Configuration settings.
        _declaration_count: Number of declarations processed.
        _edge_accumulator: Tracks edge weights for aggregation.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the graph builder.

        Args:
            settings: Optional configuration. Uses global settings if None.
        """
        self.settings = settings or get_settings()
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._declaration_count: int = 0
        self._edge_accumulator: dict[tuple[str, str, str], EdgeData] = {}
        self._node_cache: dict[str, NodeData] = {}

    def add_declaration(self, declaration: dict[str, Any]) -> None:
        """Process a single customs declaration and add its entities to the graph.

        Extracts entities (importer, exporter, commodity, route, agent) from
        the declaration record and creates nodes and edges accordingly. If
        entities already exist, their attributes are updated and edge weights
        are incremented.

        Args:
            declaration: A dictionary representing a customs declaration with
                the following expected keys:
                - declaration_id (str): Unique declaration identifier
                - importer (dict): Importer entity with tax_id, name, country
                - exporter (dict): Exporter entity with tax_id, name, country
                - commodity (dict): Commodity with hs_code, description
                - route (dict): Route with origin, destination, transit
                - agent (dict, optional): Customs agent with license_id, name
                - value (float): Declared monetary value
                - weight (float): Declared weight in kg
                - date (str): Declaration date (ISO format)
                - address (str, optional): Shared address
                - phone (str, optional): Contact phone number
                - bank_account (dict, optional): Payment information

        Raises:
            ValueError: If required fields are missing from the declaration.
        """
        self._validate_declaration(declaration)
        self._declaration_count += 1

        decl_id = declaration["declaration_id"]
        decl_date = declaration.get("date", datetime.now().isoformat()[:10])
        decl_value = float(declaration.get("value", 0.0))
        decl_weight = float(declaration.get("weight", 0.0))

        # -- Create declaration node --
        decl_node = self._ensure_node(
            node_id=f"DECL:{decl_id}",
            node_type=NodeType.DECLARATION,
            attributes={
                "declaration_id": decl_id,
                "date": decl_date,
                "value": decl_value,
                "weight": decl_weight,
            },
        )

        # -- Create importer node --
        imp = declaration["importer"]
        imp_id = f"IMP:{imp['tax_id']}"
        self._ensure_node(
            node_id=imp_id,
            node_type=NodeType.IMPORTER,
            attributes={
                "tax_id": imp["tax_id"],
                "name": imp.get("name", ""),
                "country": imp.get("country", ""),
                "address": imp.get("address", ""),
                "registration_date": imp.get("registration_date", ""),
            },
        )

        # -- Create exporter node --
        exp = declaration["exporter"]
        exp_id = f"EXP:{exp['tax_id']}"
        self._ensure_node(
            node_id=exp_id,
            node_type=NodeType.EXPORTER,
            attributes={
                "tax_id": exp["tax_id"],
                "name": exp.get("name", ""),
                "country": exp.get("country", ""),
                "address": exp.get("address", ""),
            },
        )

        # -- Create commodity node --
        comm = declaration["commodity"]
        hs_code = comm["hs_code"]
        comm_id = f"COMM:{hs_code}"
        self._ensure_node(
            node_id=comm_id,
            node_type=NodeType.COMMODITY,
            attributes={
                "hs_code": hs_code,
                "description": comm.get("description", ""),
            },
        )

        # -- Create route node --
        route = declaration["route"]
        origin = route["origin"]
        destination = route["destination"]
        transit = route.get("transit", [])
        route_id = f"ROUTE:{origin}-{destination}"
        self._ensure_node(
            node_id=route_id,
            node_type=NodeType.ROUTE,
            attributes={
                "origin": origin,
                "destination": destination,
                "transit_points": transit,
            },
        )

        # -- Create agent node (optional) --
        agent_id: Optional[str] = None
        if "agent" in declaration and declaration["agent"]:
            agent = declaration["agent"]
            agent_id = f"AGENT:{agent['license_id']}"
            self._ensure_node(
                node_id=agent_id,
                node_type=NodeType.AGENT,
                attributes={
                    "license_id": agent["license_id"],
                    "name": agent.get("name", ""),
                    "type": agent.get("type", "broker"),
                },
            )

        # -- Create address node (optional) --
        address_id: Optional[str] = None
        imp_address = imp.get("address", "")
        if imp_address:
            normalized = self._normalize_address(imp_address)
            address_id = f"ADDR:{hashlib.md5(normalized.encode()).hexdigest()[:12]}"
            self._ensure_node(
                node_id=address_id,
                node_type=NodeType.ADDRESS,
                attributes={
                    "normalized_address": normalized,
                    "country": imp.get("country", ""),
                },
            )

        # -- Create phone node (optional) --
        phone_id: Optional[str] = None
        if "phone" in declaration and declaration["phone"]:
            normalized_phone = self._normalize_phone(declaration["phone"])
            phone_id = f"PHONE:{normalized_phone}"
            self._ensure_node(
                node_id=phone_id,
                node_type=NodeType.PHONE,
                attributes={"normalized_number": normalized_phone},
            )

        # -- Create bank account node (optional) --
        bank_id: Optional[str] = None
        if "bank_account" in declaration and declaration["bank_account"]:
            bank = declaration["bank_account"]
            account_hash = hashlib.sha256(
                bank.get("account_number", "").encode()
            ).hexdigest()[:16]
            bank_id = f"BANK:{account_hash}"
            self._ensure_node(
                node_id=bank_id,
                node_type=NodeType.BANK_ACCOUNT,
                attributes={
                    "bank_name": bank.get("bank_name", ""),
                    "account_hash": account_hash,
                },
            )

        # -- Create edges --
        # Importer -> Exporter (IMPORTS_FROM)
        self._add_or_update_edge(
            source_id=imp_id,
            target_id=exp_id,
            edge_type=EdgeType.IMPORTS_FROM,
            value=decl_value,
            date=decl_date,
        )

        # Importer -> Declaration (DECLARES)
        self._add_or_update_edge(
            source_id=imp_id,
            target_id=decl_node.node_id,
            edge_type=EdgeType.DECLARES,
            value=decl_value,
            date=decl_date,
        )

        # Declaration -> Commodity (TRADES_COMMODITY)
        self._add_or_update_edge(
            source_id=decl_node.node_id,
            target_id=comm_id,
            edge_type=EdgeType.TRADES_COMMODITY,
            value=decl_value,
            date=decl_date,
        )

        # Declaration -> Route (USES_ROUTE)
        self._add_or_update_edge(
            source_id=decl_node.node_id,
            target_id=route_id,
            edge_type=EdgeType.USES_ROUTE,
            value=decl_value,
            date=decl_date,
        )

        # Importer -> Agent (REPRESENTED_BY)
        if agent_id:
            self._add_or_update_edge(
                source_id=imp_id,
                target_id=agent_id,
                edge_type=EdgeType.REPRESENTED_BY,
                value=decl_value,
                date=decl_date,
            )

        # Importer -> Address (LOCATED_AT)
        if address_id:
            self._add_or_update_edge(
                source_id=imp_id,
                target_id=address_id,
                edge_type=EdgeType.LOCATED_AT,
                value=0,
                date=decl_date,
            )

        # Exporter -> Address (LOCATED_AT) if exporter has address
        exp_address = exp.get("address", "")
        if exp_address:
            exp_normalized = self._normalize_address(exp_address)
            exp_addr_id = (
                f"ADDR:{hashlib.md5(exp_normalized.encode()).hexdigest()[:12]}"
            )
            self._ensure_node(
                node_id=exp_addr_id,
                node_type=NodeType.ADDRESS,
                attributes={
                    "normalized_address": exp_normalized,
                    "country": exp.get("country", ""),
                },
            )
            self._add_or_update_edge(
                source_id=exp_id,
                target_id=exp_addr_id,
                edge_type=EdgeType.LOCATED_AT,
                value=0,
                date=decl_date,
            )

        # Entity -> Phone (CONTACTABLE_VIA)
        if phone_id:
            self._add_or_update_edge(
                source_id=imp_id,
                target_id=phone_id,
                edge_type=EdgeType.CONTACTABLE_VIA,
                value=0,
                date=decl_date,
            )

        # Entity -> Bank Account (PAYS_THROUGH)
        if bank_id:
            self._add_or_update_edge(
                source_id=imp_id,
                target_id=bank_id,
                edge_type=EdgeType.PAYS_THROUGH,
                value=decl_value,
                date=decl_date,
            )

        logger.debug(
            "declaration_processed",
            declaration_id=decl_id,
            nodes=self.graph.number_of_nodes(),
            edges=self.graph.number_of_edges(),
        )

    def add_declarations(self, declarations: Sequence[dict[str, Any]]) -> None:
        """Process multiple declarations in batch.

        Args:
            declarations: Sequence of declaration dictionaries.
        """
        for decl in declarations:
            self.add_declaration(decl)

        logger.info(
            "batch_processed",
            count=len(declarations),
            total_declarations=self._declaration_count,
            total_nodes=self.graph.number_of_nodes(),
            total_edges=self.graph.number_of_edges(),
        )

    def build(self) -> nx.MultiDiGraph:
        """Finalize the graph and return it.

        Computes final edge weights based on the configured weighting method
        and sets graph-level metadata.

        Returns:
            The constructed MultiDiGraph with all entities and relationships.
        """
        self._compute_final_weights()

        self.graph.graph["declaration_count"] = self._declaration_count
        self.graph.graph["node_count"] = self.graph.number_of_nodes()
        self.graph.graph["edge_count"] = self.graph.number_of_edges()
        self.graph.graph["built_at"] = datetime.now().isoformat()

        logger.info(
            "graph_built",
            declarations=self._declaration_count,
            nodes=self.graph.number_of_nodes(),
            edges=self.graph.number_of_edges(),
        )

        return self.graph

    def get_statistics(self) -> dict[str, Any]:
        """Return summary statistics about the constructed graph.

        Returns:
            Dictionary with node counts by type, edge counts by type,
            and overall graph metrics.
        """
        node_type_counts: dict[str, int] = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "UNKNOWN")
            node_type_counts[nt] += 1

        edge_type_counts: dict[str, int] = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            et = data.get("edge_type", "UNKNOWN")
            edge_type_counts[et] += 1

        return {
            "declaration_count": self._declaration_count,
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": dict(node_type_counts),
            "edge_types": dict(edge_type_counts),
            "density": nx.density(self.graph),
            "is_weakly_connected": nx.is_weakly_connected(self.graph)
            if self.graph.number_of_nodes() > 0
            else False,
            "weakly_connected_components": nx.number_weakly_connected_components(
                self.graph
            )
            if self.graph.number_of_nodes() > 0
            else 0,
        }

    def _ensure_node(
        self,
        node_id: str,
        node_type: NodeType,
        attributes: dict[str, Any],
    ) -> NodeData:
        """Add a node if it doesn't exist, or update its attributes.

        Args:
            node_id: Unique identifier for the node.
            node_type: The type of this entity.
            attributes: Dictionary of node attributes.

        Returns:
            The NodeData for this node.
        """
        if node_id in self._node_cache:
            # Update existing node attributes (merge, don't overwrite)
            existing = self._node_cache[node_id]
            for key, value in attributes.items():
                if value and (key not in existing.attributes or not existing.attributes[key]):
                    existing.attributes[key] = value
            self.graph.nodes[node_id].update(existing.attributes)
            return existing

        node_data = NodeData(
            node_id=node_id,
            node_type=node_type,
            attributes=attributes,
        )
        self._node_cache[node_id] = node_data

        self.graph.add_node(
            node_id,
            node_type=node_type.value,
            risk_score=0.0,
            risk_level=RiskLevel.LOW.value,
            flagged=False,
            **attributes,
        )
        return node_data

    def _add_or_update_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        value: float,
        date: str,
    ) -> None:
        """Add an edge or update its accumulated weight.

        If an edge of the same type already exists between source and target,
        increments the transaction count, adds to total value, and updates
        the date range. Otherwise, creates a new edge.

        Args:
            source_id: Source node identifier.
            target_id: Target node identifier.
            edge_type: The type of relationship.
            value: Monetary value of this transaction.
            date: Date of this transaction (ISO format).
        """
        key = (source_id, target_id, edge_type.value)

        if key in self._edge_accumulator:
            edge = self._edge_accumulator[key]
            edge.transaction_count += 1
            edge.total_value += value
            if date and (edge.first_seen is None or date < edge.first_seen):
                edge.first_seen = date
            if date and (edge.last_seen is None or date > edge.last_seen):
                edge.last_seen = date
        else:
            edge = EdgeData(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=1.0,
                transaction_count=1,
                total_value=value,
                first_seen=date,
                last_seen=date,
            )
            self._edge_accumulator[key] = edge

            self.graph.add_edge(
                source_id,
                target_id,
                key=edge_type.value,
                edge_type=edge_type.value,
                weight=1.0,
                transaction_count=1,
                total_value=value,
                first_seen=date,
                last_seen=date,
            )

    def _compute_final_weights(self) -> None:
        """Compute final edge weights based on the configured weighting method.

        Supports three methods:
        - frequency: weight = transaction count
        - volume: weight = total declared value (log-scaled)
        - composite: weighted combination of frequency, volume, and recency
        """
        import math

        method = self.settings.graph.edge_weights.method
        components = self.settings.graph.edge_weights.components

        for key, edge in self._edge_accumulator.items():
            source_id, target_id, edge_type_val = key

            if method == "frequency":
                weight = float(edge.transaction_count)
            elif method == "volume":
                weight = math.log1p(edge.total_value)
            else:
                # Composite weighting
                freq_weight = float(edge.transaction_count)
                vol_weight = math.log1p(edge.total_value)
                recency_weight = self._compute_recency_weight(edge.last_seen)

                # Normalize individual components
                freq_norm = min(freq_weight / 10.0, 1.0)
                vol_norm = min(vol_weight / 15.0, 1.0)

                weight = (
                    components.get("frequency", 0.4) * freq_norm
                    + components.get("volume", 0.3) * vol_norm
                    + components.get("recency", 0.3) * recency_weight
                )

            edge.weight = weight

            # Update the edge in the graph
            if self.graph.has_edge(source_id, target_id, key=edge_type_val):
                self.graph[source_id][target_id][edge_type_val]["weight"] = weight
                self.graph[source_id][target_id][edge_type_val][
                    "transaction_count"
                ] = edge.transaction_count
                self.graph[source_id][target_id][edge_type_val][
                    "total_value"
                ] = edge.total_value
                self.graph[source_id][target_id][edge_type_val][
                    "first_seen"
                ] = edge.first_seen
                self.graph[source_id][target_id][edge_type_val][
                    "last_seen"
                ] = edge.last_seen

    @staticmethod
    def _compute_recency_weight(last_seen: Optional[str]) -> float:
        """Compute a recency weight between 0 and 1 based on last-seen date.

        More recent dates produce higher weights. Edges not seen in the
        last 365 days receive a weight of 0.

        Args:
            last_seen: ISO date string of the most recent transaction.

        Returns:
            Recency weight between 0.0 and 1.0.
        """
        if not last_seen:
            return 0.0

        try:
            last_date = datetime.fromisoformat(last_seen)
            days_ago = (datetime.now() - last_date).days
            if days_ago < 0:
                return 1.0
            if days_ago > 365:
                return 0.0
            return 1.0 - (days_ago / 365.0)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _normalize_address(address: str) -> str:
        """Normalize an address string for entity resolution.

        Lowercases, strips whitespace, removes common abbreviations to
        improve matching across variations of the same physical address.

        Args:
            address: Raw address string.

        Returns:
            Normalized address string.
        """
        if not address:
            return ""

        normalized = address.lower().strip()

        # Common abbreviation normalization
        replacements = {
            " st.": " street",
            " st ": " street ",
            " rd.": " road",
            " rd ": " road ",
            " ave.": " avenue",
            " ave ": " avenue ",
            " blvd.": " boulevard",
            " blvd ": " boulevard ",
            " ste.": " suite",
            " ste ": " suite ",
            " apt.": " apartment",
            " apt ": " apartment ",
        }
        for abbr, full in replacements.items():
            normalized = normalized.replace(abbr, full)

        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        return normalized

    @staticmethod
    def _normalize_phone(phone: str) -> str:
        """Normalize a phone number by removing non-digit characters.

        Args:
            phone: Raw phone number string.

        Returns:
            Normalized phone number (digits only, with leading +).
        """
        if not phone:
            return ""
        digits = "".join(c for c in phone if c.isdigit() or c == "+")
        return digits

    @staticmethod
    def _validate_declaration(declaration: dict[str, Any]) -> None:
        """Validate that a declaration has all required fields.

        Args:
            declaration: The declaration dictionary to validate.

        Raises:
            ValueError: If required fields are missing.
        """
        required_fields = ["declaration_id", "importer", "exporter", "commodity", "route"]
        missing = [f for f in required_fields if f not in declaration]
        if missing:
            raise ValueError(f"Declaration missing required fields: {missing}")

        # Validate nested entities
        for entity_name, required_keys in [
            ("importer", ["tax_id"]),
            ("exporter", ["tax_id"]),
            ("commodity", ["hs_code"]),
            ("route", ["origin", "destination"]),
        ]:
            entity = declaration.get(entity_name, {})
            if not isinstance(entity, dict):
                raise ValueError(f"'{entity_name}' must be a dictionary")
            missing_keys = [k for k in required_keys if k not in entity]
            if missing_keys:
                raise ValueError(
                    f"'{entity_name}' missing required keys: {missing_keys}"
                )


if __name__ == "__main__":
    """Build a sample graph from example declaration data."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build trade intelligence graph")
    parser.add_argument(
        "--config",
        type=str,
        default="config/graph_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to JSON file with declaration data",
    )
    args = parser.parse_args()

    settings = Settings.from_yaml(args.config)
    builder = TradeGraphBuilder(settings=settings)

    if args.input:
        with open(args.input, "r") as f:
            declarations = json.load(f)
        builder.add_declarations(declarations)
    else:
        # Generate sample data for demonstration
        sample_declarations = [
            {
                "declaration_id": f"DEC-2025-{i:04d}",
                "importer": {
                    "tax_id": f"IMP{(i % 5):03d}",
                    "name": f"Importer {i % 5}",
                    "country": "US",
                    "address": "123 Trade Street",
                },
                "exporter": {
                    "tax_id": f"EXP{(i % 3):03d}",
                    "name": f"Exporter {i % 3}",
                    "country": "CN",
                },
                "commodity": {
                    "hs_code": "8471.30",
                    "description": "Portable digital computers",
                },
                "route": {"origin": "CN", "destination": "US", "transit": ["HK"]},
                "agent": {
                    "license_id": f"BRK-{i % 2:03d}",
                    "name": f"Broker {i % 2}",
                },
                "value": 100000 + (i * 1000),
                "weight": 500 + (i * 10),
                "date": f"2025-01-{(i % 28) + 1:02d}",
            }
            for i in range(20)
        ]
        builder.add_declarations(sample_declarations)

    graph = builder.build()
    stats = builder.get_statistics()
    print(json.dumps(stats, indent=2))
