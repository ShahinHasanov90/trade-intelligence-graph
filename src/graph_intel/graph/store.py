"""Graph persistence layer with pluggable backends.

Provides a unified interface for graph storage and retrieval, with
implementations for:
- NetworkX (in-memory, default)
- Neo4j (persistent, optional)

The abstraction allows the analysis and detection layers to operate
identically regardless of the underlying storage backend.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import networkx as nx
import structlog

from graph_intel.config import Neo4jConfig, Settings, get_settings
from graph_intel.graph.schema import EdgeType, NodeType

logger = structlog.get_logger(__name__)


class GraphStore(ABC):
    """Abstract base class for graph storage backends.

    Defines the interface that all graph stores must implement,
    covering node/edge CRUD, querying, and graph-level operations.
    """

    @abstractmethod
    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        attributes: dict[str, Any],
    ) -> None:
        """Add or update a node in the graph.

        Args:
            node_id: Unique identifier for the node.
            node_type: The type/category of the node.
            attributes: Dictionary of node attributes.
        """

    @abstractmethod
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        attributes: dict[str, Any],
    ) -> None:
        """Add or update an edge in the graph.

        Args:
            source_id: Source node identifier.
            target_id: Target node identifier.
            edge_type: The type/category of the edge.
            attributes: Dictionary of edge attributes.
        """

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a node by its identifier.

        Args:
            node_id: The node identifier.

        Returns:
            Node attributes dictionary, or None if not found.
        """

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "out",
    ) -> list[dict[str, Any]]:
        """Get neighboring nodes, optionally filtered by edge type and direction.

        Args:
            node_id: The source node identifier.
            edge_type: Optional filter for edge type.
            direction: 'out' for successors, 'in' for predecessors, 'both' for all.

        Returns:
            List of neighbor node attribute dictionaries.
        """

    @abstractmethod
    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[EdgeType] = None,
    ) -> list[dict[str, Any]]:
        """Query edges with optional filters.

        Args:
            source_id: Optional source node filter.
            target_id: Optional target node filter.
            edge_type: Optional edge type filter.

        Returns:
            List of edge attribute dictionaries.
        """

    @abstractmethod
    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        weight: Optional[str] = None,
    ) -> Optional[list[str]]:
        """Find the shortest path between two nodes.

        Args:
            source_id: Source node identifier.
            target_id: Target node identifier.
            weight: Optional edge attribute to use as weight.

        Returns:
            List of node IDs in the path, or None if no path exists.
        """

    @abstractmethod
    def get_nodes_by_type(self, node_type: NodeType) -> list[dict[str, Any]]:
        """Get all nodes of a specific type.

        Args:
            node_type: The node type to filter by.

        Returns:
            List of node attribute dictionaries.
        """

    @abstractmethod
    def get_subgraph(self, node_ids: Sequence[str]) -> nx.MultiDiGraph:
        """Extract a subgraph containing only the specified nodes.

        Args:
            node_ids: Node identifiers to include.

        Returns:
            A NetworkX MultiDiGraph subgraph.
        """

    @abstractmethod
    def node_count(self) -> int:
        """Return the total number of nodes."""

    @abstractmethod
    def edge_count(self) -> int:
        """Return the total number of edges."""

    @abstractmethod
    def get_networkx_graph(self) -> nx.MultiDiGraph:
        """Get the underlying graph as a NetworkX MultiDiGraph.

        For NetworkX backends, returns the graph directly.
        For Neo4j backends, materializes the full graph into NetworkX.

        Returns:
            The graph as a NetworkX MultiDiGraph.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all nodes and edges from the store."""


class NetworkXStore(GraphStore):
    """In-memory graph store backed by NetworkX MultiDiGraph.

    This is the default backend, suitable for graphs with up to ~1M nodes.
    All operations are performed in-memory with O(1) node/edge access.
    """

    def __init__(self, graph: Optional[nx.MultiDiGraph] = None) -> None:
        """Initialize the NetworkX store.

        Args:
            graph: Optional pre-built graph. Creates empty graph if None.
        """
        self._graph: nx.MultiDiGraph = graph or nx.MultiDiGraph()

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        attributes: dict[str, Any],
    ) -> None:
        """Add or update a node in the NetworkX graph."""
        self._graph.add_node(
            node_id,
            node_type=node_type.value,
            **attributes,
        )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        attributes: dict[str, Any],
    ) -> None:
        """Add or update an edge in the NetworkX graph."""
        self._graph.add_edge(
            source_id,
            target_id,
            key=edge_type.value,
            edge_type=edge_type.value,
            **attributes,
        )

    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a node by identifier from the NetworkX graph."""
        if node_id not in self._graph:
            return None
        data = dict(self._graph.nodes[node_id])
        data["node_id"] = node_id
        return data

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "out",
    ) -> list[dict[str, Any]]:
        """Get neighboring nodes from the NetworkX graph."""
        if node_id not in self._graph:
            return []

        neighbors: set[str] = set()

        if direction in ("out", "both"):
            for _, target, data in self._graph.out_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type.value:
                    neighbors.add(target)

        if direction in ("in", "both"):
            for source, _, data in self._graph.in_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type.value:
                    neighbors.add(source)

        result = []
        for nid in neighbors:
            node_data = dict(self._graph.nodes[nid])
            node_data["node_id"] = nid
            result.append(node_data)

        return result

    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[EdgeType] = None,
    ) -> list[dict[str, Any]]:
        """Query edges with optional filters from the NetworkX graph."""
        results = []
        for src, tgt, key, data in self._graph.edges(keys=True, data=True):
            if source_id is not None and src != source_id:
                continue
            if target_id is not None and tgt != target_id:
                continue
            if edge_type is not None and data.get("edge_type") != edge_type.value:
                continue

            edge_data = dict(data)
            edge_data["source_id"] = src
            edge_data["target_id"] = tgt
            results.append(edge_data)

        return results

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        weight: Optional[str] = None,
    ) -> Optional[list[str]]:
        """Find shortest path using Dijkstra's algorithm."""
        try:
            if weight:
                # For weighted shortest path, we need to invert weights
                # (higher weight = stronger connection = shorter path)
                path = nx.shortest_path(
                    self._graph, source_id, target_id, weight=weight
                )
            else:
                path = nx.shortest_path(self._graph, source_id, target_id)
            return list(path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_nodes_by_type(self, node_type: NodeType) -> list[dict[str, Any]]:
        """Get all nodes of a specific type from the NetworkX graph."""
        results = []
        for node_id, data in self._graph.nodes(data=True):
            if data.get("node_type") == node_type.value:
                node_data = dict(data)
                node_data["node_id"] = node_id
                results.append(node_data)
        return results

    def get_subgraph(self, node_ids: Sequence[str]) -> nx.MultiDiGraph:
        """Extract a subgraph containing only the specified nodes."""
        return self._graph.subgraph(node_ids).copy()

    def node_count(self) -> int:
        """Return the total number of nodes."""
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        """Return the total number of edges."""
        return self._graph.number_of_edges()

    def get_networkx_graph(self) -> nx.MultiDiGraph:
        """Return the underlying NetworkX graph directly."""
        return self._graph

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._graph.clear()


class Neo4jStore(GraphStore):
    """Persistent graph store backed by Neo4j.

    Provides the same interface as NetworkXStore but persists data in
    a Neo4j graph database. Suitable for large-scale deployments with
    10M+ nodes that require concurrent access and persistence.

    Note: Requires the `neo4j` Python driver to be installed.
    """

    def __init__(self, config: Optional[Neo4jConfig] = None) -> None:
        """Initialize the Neo4j store.

        Args:
            config: Neo4j connection configuration.
                Uses global settings if None.

        Raises:
            ImportError: If the neo4j Python driver is not installed.
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "Neo4j driver not installed. Install with: pip install neo4j"
            )

        self._config = config or get_settings().neo4j
        self._driver = GraphDatabase.driver(
            self._config.uri,
            auth=(self._config.user, self._config.password),
            max_connection_pool_size=self._config.max_connection_pool_size,
        )
        self._database = self._config.database

        logger.info(
            "neo4j_store_initialized",
            uri=self._config.uri,
            database=self._database,
        )

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self._driver.close()

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        attributes: dict[str, Any],
    ) -> None:
        """Add or update a node in Neo4j using MERGE."""
        query = (
            f"MERGE (n:{node_type.value} {{node_id: $node_id}}) "
            f"SET n += $attributes, n.node_type = $node_type"
        )
        with self._driver.session(database=self._database) as session:
            session.run(
                query,
                node_id=node_id,
                node_type=node_type.value,
                attributes=self._serialize_attributes(attributes),
            )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        attributes: dict[str, Any],
    ) -> None:
        """Add or update an edge in Neo4j using MERGE."""
        query = (
            f"MATCH (a {{node_id: $source_id}}), (b {{node_id: $target_id}}) "
            f"MERGE (a)-[r:{edge_type.value}]->(b) "
            f"SET r += $attributes, r.edge_type = $edge_type"
        )
        with self._driver.session(database=self._database) as session:
            session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type.value,
                attributes=self._serialize_attributes(attributes),
            )

    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a node by identifier from Neo4j."""
        query = "MATCH (n {node_id: $node_id}) RETURN n"
        with self._driver.session(database=self._database) as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            if record is None:
                return None
            node = record["n"]
            return dict(node)

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "out",
    ) -> list[dict[str, Any]]:
        """Get neighboring nodes from Neo4j."""
        rel_filter = f":{edge_type.value}" if edge_type else ""

        if direction == "out":
            query = (
                f"MATCH (n {{node_id: $node_id}})-[r{rel_filter}]->(m) "
                f"RETURN m"
            )
        elif direction == "in":
            query = (
                f"MATCH (n {{node_id: $node_id}})<-[r{rel_filter}]-(m) "
                f"RETURN m"
            )
        else:
            query = (
                f"MATCH (n {{node_id: $node_id}})-[r{rel_filter}]-(m) "
                f"RETURN DISTINCT m"
            )

        with self._driver.session(database=self._database) as session:
            result = session.run(query, node_id=node_id)
            return [dict(record["m"]) for record in result]

    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[EdgeType] = None,
    ) -> list[dict[str, Any]]:
        """Query edges from Neo4j with optional filters."""
        conditions = []
        params: dict[str, Any] = {}

        if source_id:
            conditions.append("a.node_id = $source_id")
            params["source_id"] = source_id
        if target_id:
            conditions.append("b.node_id = $target_id")
            params["target_id"] = target_id

        rel_filter = f":{edge_type.value}" if edge_type else ""
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = (
            f"MATCH (a)-[r{rel_filter}]->(b) "
            f"{where_clause} "
            f"RETURN a.node_id AS source_id, b.node_id AS target_id, "
            f"type(r) AS edge_type, properties(r) AS props"
        )

        with self._driver.session(database=self._database) as session:
            result = session.run(query, **params)
            edges = []
            for record in result:
                edge = dict(record["props"])
                edge["source_id"] = record["source_id"]
                edge["target_id"] = record["target_id"]
                edge["edge_type"] = record["edge_type"]
                edges.append(edge)
            return edges

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        weight: Optional[str] = None,
    ) -> Optional[list[str]]:
        """Find shortest path using Neo4j's built-in algorithm."""
        query = (
            "MATCH p = shortestPath("
            "(a {node_id: $source_id})-[*]-(b {node_id: $target_id})"
            ") RETURN [n IN nodes(p) | n.node_id] AS path"
        )
        with self._driver.session(database=self._database) as session:
            result = session.run(
                query, source_id=source_id, target_id=target_id
            )
            record = result.single()
            if record is None:
                return None
            return record["path"]

    def get_nodes_by_type(self, node_type: NodeType) -> list[dict[str, Any]]:
        """Get all nodes of a specific type from Neo4j."""
        query = f"MATCH (n:{node_type.value}) RETURN n"
        with self._driver.session(database=self._database) as session:
            result = session.run(query)
            return [dict(record["n"]) for record in result]

    def get_subgraph(self, node_ids: Sequence[str]) -> nx.MultiDiGraph:
        """Extract a subgraph from Neo4j and return as NetworkX graph."""
        query = (
            "MATCH (a)-[r]->(b) "
            "WHERE a.node_id IN $node_ids AND b.node_id IN $node_ids "
            "RETURN a.node_id AS src, properties(a) AS src_props, "
            "type(r) AS rel_type, properties(r) AS rel_props, "
            "b.node_id AS tgt, properties(b) AS tgt_props"
        )
        g = nx.MultiDiGraph()
        with self._driver.session(database=self._database) as session:
            result = session.run(query, node_ids=list(node_ids))
            for record in result:
                src = record["src"]
                tgt = record["tgt"]
                if src not in g:
                    g.add_node(src, **dict(record["src_props"]))
                if tgt not in g:
                    g.add_node(tgt, **dict(record["tgt_props"]))
                g.add_edge(
                    src,
                    tgt,
                    key=record["rel_type"],
                    **dict(record["rel_props"]),
                )
        return g

    def node_count(self) -> int:
        """Return the total number of nodes in Neo4j."""
        query = "MATCH (n) RETURN count(n) AS count"
        with self._driver.session(database=self._database) as session:
            result = session.run(query)
            return result.single()["count"]

    def edge_count(self) -> int:
        """Return the total number of edges in Neo4j."""
        query = "MATCH ()-[r]->() RETURN count(r) AS count"
        with self._driver.session(database=self._database) as session:
            result = session.run(query)
            return result.single()["count"]

    def get_networkx_graph(self) -> nx.MultiDiGraph:
        """Materialize the entire Neo4j graph as a NetworkX MultiDiGraph.

        Warning: This loads the entire graph into memory. Use with caution
        for large graphs.

        Returns:
            The full graph as a NetworkX MultiDiGraph.
        """
        query = (
            "MATCH (a)-[r]->(b) "
            "RETURN a.node_id AS src, properties(a) AS src_props, "
            "type(r) AS rel_type, properties(r) AS rel_props, "
            "b.node_id AS tgt, properties(b) AS tgt_props"
        )
        g = nx.MultiDiGraph()
        with self._driver.session(database=self._database) as session:
            result = session.run(query)
            for record in result:
                src = record["src"]
                tgt = record["tgt"]
                if src not in g:
                    g.add_node(src, **dict(record["src_props"]))
                if tgt not in g:
                    g.add_node(tgt, **dict(record["tgt_props"]))
                g.add_edge(
                    src,
                    tgt,
                    key=record["rel_type"],
                    **dict(record["rel_props"]),
                )

        # Also add isolated nodes
        query_isolated = (
            "MATCH (n) WHERE NOT (n)--() RETURN n.node_id AS nid, properties(n) AS props"
        )
        with self._driver.session(database=self._database) as session:
            result = session.run(query_isolated)
            for record in result:
                nid = record["nid"]
                if nid not in g:
                    g.add_node(nid, **dict(record["props"]))

        return g

    def clear(self) -> None:
        """Remove all nodes and edges from Neo4j."""
        query = "MATCH (n) DETACH DELETE n"
        with self._driver.session(database=self._database) as session:
            session.run(query)

    @staticmethod
    def _serialize_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
        """Serialize attributes for Neo4j compatibility.

        Neo4j doesn't support nested structures or certain Python types,
        so we flatten and convert as needed.

        Args:
            attributes: Raw attribute dictionary.

        Returns:
            Serialized attribute dictionary.
        """
        serialized = {}
        for key, value in attributes.items():
            if isinstance(value, (list, tuple)):
                # Neo4j supports arrays of primitives
                serialized[key] = list(value)
            elif isinstance(value, dict):
                # Flatten nested dicts with dot notation
                for sub_key, sub_value in value.items():
                    serialized[f"{key}_{sub_key}"] = sub_value
            else:
                serialized[key] = value
        return serialized


def create_store(settings: Optional[Settings] = None) -> GraphStore:
    """Factory function to create the appropriate graph store.

    Args:
        settings: Configuration settings. Uses global settings if None.

    Returns:
        A GraphStore instance (NetworkXStore or Neo4jStore).

    Raises:
        ValueError: If the configured backend is not recognized.
    """
    settings = settings or get_settings()
    backend = settings.graph.backend

    if backend == "networkx":
        logger.info("store_created", backend="networkx")
        return NetworkXStore()
    elif backend == "neo4j":
        logger.info("store_created", backend="neo4j")
        return Neo4jStore(config=settings.neo4j)
    else:
        raise ValueError(f"Unknown graph backend: {backend}")
