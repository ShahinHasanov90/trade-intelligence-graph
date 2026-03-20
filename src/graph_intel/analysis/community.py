"""Community detection algorithms for trade entity graphs.

Identifies tightly connected clusters of trade entities using multiple
algorithms. Communities often correspond to legitimate supply chains,
but anomalous community structures can indicate coordinated fraud networks.

Supported algorithms:
- Louvain: Fast, hierarchical, modularity-maximizing
- Leiden: Improved Louvain with guaranteed connected communities
- Label Propagation: Near-linear time, good for large graphs
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx
import numpy as np
import structlog

from graph_intel.config import CommunityConfig, Settings, get_settings

logger = structlog.get_logger(__name__)


@dataclass
class Community:
    """Represents a detected community of trade entities.

    Attributes:
        community_id: Unique identifier for this community.
        members: Set of node IDs belonging to this community.
        size: Number of nodes in the community.
        internal_edges: Number of edges within the community.
        external_edges: Number of edges crossing community boundaries.
        density: Internal edge density (0.0 to 1.0).
        modularity_contribution: This community's contribution to overall modularity.
        node_types: Distribution of node types in this community.
        risk_score: Aggregate risk score for the community.
        metadata: Additional computed properties.
    """

    community_id: int
    members: set[str]
    size: int = 0
    internal_edges: int = 0
    external_edges: int = 0
    density: float = 0.0
    modularity_contribution: float = 0.0
    node_types: dict[str, int] = field(default_factory=dict)
    risk_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute size from members if not explicitly set."""
        if self.size == 0:
            self.size = len(self.members)


@dataclass
class CommunityResult:
    """Result of a community detection run.

    Attributes:
        communities: List of detected communities.
        algorithm: Name of the algorithm used.
        modularity: Overall modularity score.
        num_communities: Total number of communities detected.
        partition: Mapping from node ID to community ID.
        resolution: Resolution parameter used.
    """

    communities: list[Community]
    algorithm: str
    modularity: float
    num_communities: int
    partition: dict[str, int]
    resolution: float = 1.0


class CommunityDetector:
    """Detects and analyzes communities in trade entity graphs.

    Provides multiple community detection algorithms and tools for
    analyzing the resulting community structure, including anomaly
    detection within communities.

    Args:
        graph: The trade entity graph to analyze.
        config: Optional community detection configuration.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        config: Optional[CommunityConfig] = None,
    ) -> None:
        """Initialize the community detector.

        Args:
            graph: The trade entity graph (NetworkX MultiDiGraph).
            config: Optional community detection configuration.
        """
        self._graph = graph
        self._config = config or get_settings().analysis.community
        self._undirected: Optional[nx.Graph] = None

    @property
    def undirected_graph(self) -> nx.Graph:
        """Lazily compute an undirected simple graph for community detection.

        Community detection algorithms typically operate on undirected graphs.
        This converts the MultiDiGraph to an undirected Graph, aggregating
        edge weights.

        Returns:
            Undirected simple graph with aggregated edge weights.
        """
        if self._undirected is None:
            self._undirected = self._to_undirected_weighted()
        return self._undirected

    def detect_louvain(
        self,
        resolution: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> CommunityResult:
        """Detect communities using the Louvain algorithm.

        The Louvain method is a greedy modularity optimization algorithm
        that iteratively merges nodes into communities to maximize the
        modularity function. It is fast (near-linear time) and produces
        hierarchical community structures.

        Args:
            resolution: Resolution parameter. Higher values produce more,
                smaller communities. Defaults to config value.
            random_state: Random seed for reproducibility.

        Returns:
            CommunityResult with detected communities and modularity score.
        """
        try:
            import community as community_louvain
        except ImportError:
            raise ImportError(
                "python-louvain not installed. Install with: pip install python-louvain"
            )

        resolution = resolution or self._config.resolution
        random_state = random_state or self._config.random_seed
        ug = self.undirected_graph

        if ug.number_of_nodes() == 0:
            return CommunityResult(
                communities=[],
                algorithm="louvain",
                modularity=0.0,
                num_communities=0,
                partition={},
                resolution=resolution,
            )

        # Run Louvain community detection
        partition = community_louvain.best_partition(
            ug,
            resolution=resolution,
            random_state=random_state,
            weight="weight",
        )

        # Compute modularity
        modularity = community_louvain.modularity(partition, ug, weight="weight")

        # Build Community objects
        communities = self._build_communities(partition, modularity)

        # Filter by minimum size
        communities = [
            c for c in communities if c.size >= self._config.min_community_size
        ]

        result = CommunityResult(
            communities=communities,
            algorithm="louvain",
            modularity=modularity,
            num_communities=len(communities),
            partition=partition,
            resolution=resolution,
        )

        logger.info(
            "community_detection_complete",
            algorithm="louvain",
            communities=len(communities),
            modularity=round(modularity, 4),
            resolution=resolution,
        )

        return result

    def detect_leiden(
        self,
        resolution: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> CommunityResult:
        """Detect communities using the Leiden algorithm.

        The Leiden algorithm is an improvement over Louvain that guarantees
        connected communities and provides better modularity scores. It
        uses a refinement phase that prevents poorly connected communities.

        Args:
            resolution: Resolution parameter for modularity optimization.
            random_state: Random seed for reproducibility.

        Returns:
            CommunityResult with detected communities and modularity score.

        Raises:
            ImportError: If leidenalg or igraph are not installed.
        """
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError(
                "leidenalg and igraph required. Install with: "
                "pip install leidenalg igraph"
            )

        resolution = resolution or self._config.resolution
        random_state = random_state or self._config.random_seed
        ug = self.undirected_graph

        if ug.number_of_nodes() == 0:
            return CommunityResult(
                communities=[],
                algorithm="leiden",
                modularity=0.0,
                num_communities=0,
                partition={},
                resolution=resolution,
            )

        # Convert NetworkX graph to igraph
        node_list = list(ug.nodes())
        node_index = {node: i for i, node in enumerate(node_list)}

        edges = []
        weights = []
        for u, v, data in ug.edges(data=True):
            edges.append((node_index[u], node_index[v]))
            weights.append(data.get("weight", 1.0))

        ig_graph = ig.Graph(n=len(node_list), edges=edges, directed=False)
        ig_graph.es["weight"] = weights

        # Run Leiden algorithm
        partition_result = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=resolution,
            seed=random_state,
        )

        # Convert back to node ID -> community ID mapping
        partition = {}
        for comm_id, members in enumerate(partition_result):
            for node_idx in members:
                partition[node_list[node_idx]] = comm_id

        modularity = partition_result.modularity

        communities = self._build_communities(partition, modularity)
        communities = [
            c for c in communities if c.size >= self._config.min_community_size
        ]

        result = CommunityResult(
            communities=communities,
            algorithm="leiden",
            modularity=modularity,
            num_communities=len(communities),
            partition=partition,
            resolution=resolution,
        )

        logger.info(
            "community_detection_complete",
            algorithm="leiden",
            communities=len(communities),
            modularity=round(modularity, 4),
        )

        return result

    def detect_label_propagation(self) -> CommunityResult:
        """Detect communities using label propagation.

        Label propagation is a near-linear time algorithm that assigns
        community labels based on majority voting among neighbors. It is
        non-deterministic but very fast for large graphs.

        Returns:
            CommunityResult with detected communities.
        """
        ug = self.undirected_graph

        if ug.number_of_nodes() == 0:
            return CommunityResult(
                communities=[],
                algorithm="label_propagation",
                modularity=0.0,
                num_communities=0,
                partition={},
            )

        # Run label propagation
        label_communities = nx.community.label_propagation_communities(ug)

        # Convert to partition format
        partition: dict[str, int] = {}
        for comm_id, members in enumerate(label_communities):
            for node_id in members:
                partition[node_id] = comm_id

        # Compute modularity
        community_sets = defaultdict(set)
        for node_id, comm_id in partition.items():
            community_sets[comm_id].add(node_id)

        modularity = nx.community.modularity(
            ug, list(community_sets.values()), weight="weight"
        )

        communities = self._build_communities(partition, modularity)
        communities = [
            c for c in communities if c.size >= self._config.min_community_size
        ]

        result = CommunityResult(
            communities=communities,
            algorithm="label_propagation",
            modularity=modularity,
            num_communities=len(communities),
            partition=partition,
        )

        logger.info(
            "community_detection_complete",
            algorithm="label_propagation",
            communities=len(communities),
            modularity=round(modularity, 4),
        )

        return result

    def detect(
        self,
        algorithm: Optional[str] = None,
        **kwargs: Any,
    ) -> CommunityResult:
        """Detect communities using the configured or specified algorithm.

        Args:
            algorithm: Algorithm name. Uses config default if None.
            **kwargs: Additional keyword arguments passed to the algorithm.

        Returns:
            CommunityResult from the selected algorithm.

        Raises:
            ValueError: If the algorithm name is not recognized.
        """
        algo = algorithm or self._config.algorithm

        if algo == "louvain":
            return self.detect_louvain(**kwargs)
        elif algo == "leiden":
            return self.detect_leiden(**kwargs)
        elif algo == "label_propagation":
            return self.detect_label_propagation(**kwargs)
        else:
            raise ValueError(
                f"Unknown algorithm: {algo}. "
                f"Supported: louvain, leiden, label_propagation"
            )

    def find_anomalous_communities(
        self,
        result: CommunityResult,
        value_variance_threshold: float = 0.1,
        min_risk_score: float = 0.5,
    ) -> list[Community]:
        """Identify communities with anomalous internal patterns.

        A community is considered anomalous if it exhibits:
        - Very low variance in declared values (coordinated undervaluation)
        - High aggregate risk scores from member nodes
        - Unusual ratio of node types (e.g., many importers, one agent)

        Args:
            result: CommunityResult from a detection run.
            value_variance_threshold: Maximum normalized value variance
                before a community is considered anomalous.
            min_risk_score: Minimum community risk score for flagging.

        Returns:
            List of anomalous communities, sorted by risk score.
        """
        anomalous = []

        for community in result.communities:
            anomaly_indicators = []

            # Check for coordinated valuation patterns
            values = self._get_community_declaration_values(community.members)
            if len(values) >= 3:
                cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                if cv < value_variance_threshold:
                    anomaly_indicators.append("low_value_variance")
                    community.metadata["value_cv"] = float(cv)

            # Check for single-agent communities
            node_types = community.node_types
            agent_count = node_types.get("AGENT", 0)
            importer_count = node_types.get("IMPORTER", 0)
            if agent_count == 1 and importer_count >= 3:
                anomaly_indicators.append("single_agent_cluster")
                community.metadata["agent_importer_ratio"] = (
                    agent_count / importer_count
                )

            # Check for shared attribute concentration
            shared_addresses = self._count_shared_attribute(
                community.members, "ADDRESS"
            )
            if shared_addresses > 0 and importer_count >= 2:
                anomaly_indicators.append("shared_address")
                community.metadata["shared_addresses"] = shared_addresses

            # Compute community risk score
            risk_scores = []
            for node_id in community.members:
                if node_id in self._graph:
                    rs = self._graph.nodes[node_id].get("risk_score", 0.0)
                    risk_scores.append(rs)
            if risk_scores:
                community.risk_score = float(np.mean(risk_scores))

            if anomaly_indicators or community.risk_score >= min_risk_score:
                community.metadata["anomaly_indicators"] = anomaly_indicators
                anomalous.append(community)

        anomalous.sort(key=lambda c: c.risk_score, reverse=True)

        logger.info(
            "anomalous_communities_found",
            total_communities=len(result.communities),
            anomalous=len(anomalous),
        )

        return anomalous

    def _build_communities(
        self,
        partition: dict[str, int],
        overall_modularity: float,
    ) -> list[Community]:
        """Build Community objects from a partition mapping.

        Args:
            partition: Mapping from node ID to community ID.
            overall_modularity: The overall modularity score.

        Returns:
            List of Community objects with computed metrics.
        """
        # Group nodes by community
        community_members: dict[int, set[str]] = defaultdict(set)
        for node_id, comm_id in partition.items():
            community_members[comm_id].add(node_id)

        communities = []
        for comm_id, members in community_members.items():
            # Count node types
            node_types: dict[str, int] = defaultdict(int)
            for node_id in members:
                if node_id in self._graph:
                    nt = self._graph.nodes[node_id].get("node_type", "UNKNOWN")
                    node_types[nt] += 1

            # Count internal and external edges
            internal = 0
            external = 0
            for node_id in members:
                if node_id in self._graph:
                    for _, target in self._graph.out_edges(node_id):
                        if target in members:
                            internal += 1
                        else:
                            external += 1

            # Compute density
            n = len(members)
            max_edges = n * (n - 1) if n > 1 else 1
            density = internal / max_edges if max_edges > 0 else 0.0

            community = Community(
                community_id=comm_id,
                members=members,
                size=n,
                internal_edges=internal,
                external_edges=external,
                density=density,
                node_types=dict(node_types),
            )
            communities.append(community)

        return communities

    def _to_undirected_weighted(self) -> nx.Graph:
        """Convert the MultiDiGraph to an undirected weighted simple graph.

        Aggregates edge weights from multiple directed edges between the
        same pair of nodes into a single undirected edge weight.

        Returns:
            Undirected simple graph with aggregated weights.
        """
        ug = nx.Graph()

        for node_id, data in self._graph.nodes(data=True):
            ug.add_node(node_id, **data)

        for u, v, data in self._graph.edges(data=True):
            weight = data.get("weight", 1.0)
            if ug.has_edge(u, v):
                ug[u][v]["weight"] += weight
            else:
                ug.add_edge(u, v, weight=weight)

        return ug

    def _get_community_declaration_values(
        self, members: set[str]
    ) -> list[float]:
        """Extract declared values from declaration nodes in a community.

        Args:
            members: Set of node IDs in the community.

        Returns:
            List of declared values from DECLARATION nodes.
        """
        values = []
        for node_id in members:
            if node_id in self._graph:
                node_data = self._graph.nodes[node_id]
                if node_data.get("node_type") == "DECLARATION":
                    val = node_data.get("value")
                    if val is not None:
                        values.append(float(val))
        return values

    def _count_shared_attribute(
        self, members: set[str], attribute_node_type: str
    ) -> int:
        """Count attribute nodes (e.g., addresses) shared by multiple community members.

        Args:
            members: Set of node IDs in the community.
            attribute_node_type: The node type to check for sharing.

        Returns:
            Number of attribute nodes connected to multiple community members.
        """
        attribute_connections: dict[str, int] = defaultdict(int)
        for node_id in members:
            if node_id in self._graph:
                for _, target, data in self._graph.out_edges(node_id, data=True):
                    if target in self._graph:
                        if (
                            self._graph.nodes[target].get("node_type")
                            == attribute_node_type
                        ):
                            attribute_connections[target] += 1

        return sum(1 for count in attribute_connections.values() if count > 1)
