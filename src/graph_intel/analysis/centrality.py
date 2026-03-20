"""Centrality analysis for trade entity graphs.

Computes various centrality measures to identify key entities in the
trade network — entities that bridge communities, facilitate trade flows,
or occupy structurally important positions that may indicate involvement
in fraud networks.

Key metrics:
- PageRank: Importance based on inbound connection quality
- Betweenness Centrality: Frequency on shortest paths (bridge nodes)
- Degree Centrality: Normalized connection count
- Hub/Authority Scores: HITS algorithm for hub-and-spoke patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx
import numpy as np
import structlog

from graph_intel.config import CentralityConfig, Settings, get_settings
from graph_intel.graph.schema import NodeType

logger = structlog.get_logger(__name__)


@dataclass
class CentralityScores:
    """Container for centrality analysis results.

    Attributes:
        metric_name: Name of the centrality metric.
        scores: Mapping from node ID to centrality score.
        top_k: List of (node_id, score) tuples for the top-k highest scoring nodes.
        statistics: Summary statistics (mean, std, max, min).
    """

    metric_name: str
    scores: dict[str, float]
    top_k: list[tuple[str, float]] = field(default_factory=list)
    statistics: dict[str, float] = field(default_factory=dict)


@dataclass
class FacilitatorProfile:
    """Profile of a potential facilitator entity.

    A facilitator is an entity with high betweenness centrality that
    bridges otherwise disconnected communities. In trade fraud networks,
    facilitators are often customs brokers, freight forwarders, or shell
    companies that connect multiple fraud rings.

    Attributes:
        node_id: The entity identifier.
        node_type: The type of entity.
        pagerank: PageRank score.
        betweenness: Betweenness centrality score.
        degree: Degree centrality score.
        in_degree: Number of incoming connections.
        out_degree: Number of outgoing connections.
        bridging_score: Composite score indicating bridging potential.
        connected_communities: Community IDs this entity connects.
        risk_indicators: List of structural risk indicators.
    """

    node_id: str
    node_type: str
    pagerank: float
    betweenness: float
    degree: float
    in_degree: int
    out_degree: int
    bridging_score: float
    connected_communities: list[int] = field(default_factory=list)
    risk_indicators: list[str] = field(default_factory=list)


class CentralityAnalyzer:
    """Computes and analyzes centrality metrics on trade entity graphs.

    Provides methods for computing individual centrality metrics as well
    as composite analysis that combines multiple metrics to identify
    structurally important entities.

    Args:
        graph: The trade entity graph to analyze.
        config: Optional centrality analysis configuration.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        config: Optional[CentralityConfig] = None,
    ) -> None:
        """Initialize the centrality analyzer.

        Args:
            graph: The trade entity graph (NetworkX MultiDiGraph).
            config: Optional centrality configuration.
        """
        self._graph = graph
        self._config = config or get_settings().analysis.centrality
        self._cache: dict[str, CentralityScores] = {}

    def compute_pagerank(
        self,
        alpha: Optional[float] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        weight: str = "weight",
    ) -> CentralityScores:
        """Compute PageRank centrality for all nodes.

        PageRank measures node importance based on the quality and quantity
        of inbound connections. In trade graphs, high PageRank entities
        are those that receive trade from many important exporters.

        Args:
            alpha: Damping factor (probability of following an edge).
            max_iter: Maximum number of power iteration steps.
            tol: Convergence tolerance.
            weight: Edge attribute to use as weight.

        Returns:
            CentralityScores with PageRank values for all nodes.
        """
        alpha = alpha or self._config.pagerank.alpha
        max_iter = max_iter or self._config.pagerank.max_iter
        tol = tol or self._config.pagerank.tol

        if self._graph.number_of_nodes() == 0:
            return CentralityScores(
                metric_name="pagerank", scores={}, statistics={}
            )

        # Convert to simple DiGraph for PageRank (aggregate multi-edges)
        simple = self._to_simple_digraph(weight)

        scores = nx.pagerank(
            simple,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            weight=weight,
        )

        result = self._build_scores("pagerank", scores)
        self._cache["pagerank"] = result

        logger.info(
            "centrality_computed",
            metric="pagerank",
            nodes=len(scores),
            max_score=result.statistics.get("max", 0),
        )

        return result

    def compute_betweenness(
        self,
        normalized: Optional[bool] = None,
        k: Optional[int] = None,
        weight: str = "weight",
    ) -> CentralityScores:
        """Compute betweenness centrality for all nodes.

        Betweenness centrality measures how often a node lies on the
        shortest path between other nodes. High-betweenness entities
        are structural bridges — removing them disconnects parts of the
        network. In trade fraud, these are often facilitators.

        Args:
            normalized: Whether to normalize by the number of node pairs.
            k: Sample size for approximate computation. None for exact.
            weight: Edge attribute to use as weight.

        Returns:
            CentralityScores with betweenness values for all nodes.
        """
        normalized = (
            normalized
            if normalized is not None
            else self._config.betweenness.normalized
        )
        k = k if k is not None else self._config.betweenness.k

        if self._graph.number_of_nodes() == 0:
            return CentralityScores(
                metric_name="betweenness", scores={}, statistics={}
            )

        simple = self._to_simple_digraph(weight)

        scores = nx.betweenness_centrality(
            simple,
            normalized=normalized,
            k=k,
            weight=weight,
        )

        result = self._build_scores("betweenness", scores)
        self._cache["betweenness"] = result

        logger.info(
            "centrality_computed",
            metric="betweenness",
            nodes=len(scores),
            max_score=result.statistics.get("max", 0),
        )

        return result

    def compute_degree_centrality(
        self,
        normalized: Optional[bool] = None,
    ) -> CentralityScores:
        """Compute degree centrality for all nodes.

        Degree centrality measures the fraction of nodes each node is
        connected to. High degree centrality indicates entities with
        many direct relationships.

        Args:
            normalized: Whether to normalize by (n-1).

        Returns:
            CentralityScores with degree centrality values.
        """
        normalized = (
            normalized
            if normalized is not None
            else self._config.degree.normalized
        )

        if self._graph.number_of_nodes() == 0:
            return CentralityScores(
                metric_name="degree", scores={}, statistics={}
            )

        if normalized:
            scores = nx.degree_centrality(self._graph)
        else:
            scores = {node: float(deg) for node, deg in self._graph.degree()}

        result = self._build_scores("degree", scores)
        self._cache["degree"] = result

        logger.info(
            "centrality_computed",
            metric="degree",
            nodes=len(scores),
            max_score=result.statistics.get("max", 0),
        )

        return result

    def compute_in_degree_centrality(self) -> CentralityScores:
        """Compute in-degree centrality for all nodes.

        In-degree centrality specifically measures incoming connections,
        useful for identifying entities that receive from many sources.

        Returns:
            CentralityScores with in-degree centrality values.
        """
        if self._graph.number_of_nodes() == 0:
            return CentralityScores(
                metric_name="in_degree", scores={}, statistics={}
            )

        scores = nx.in_degree_centrality(self._graph)
        result = self._build_scores("in_degree", scores)
        self._cache["in_degree"] = result
        return result

    def compute_out_degree_centrality(self) -> CentralityScores:
        """Compute out-degree centrality for all nodes.

        Out-degree centrality measures outgoing connections, useful for
        identifying entities that supply to many destinations.

        Returns:
            CentralityScores with out-degree centrality values.
        """
        if self._graph.number_of_nodes() == 0:
            return CentralityScores(
                metric_name="out_degree", scores={}, statistics={}
            )

        scores = nx.out_degree_centrality(self._graph)
        result = self._build_scores("out_degree", scores)
        self._cache["out_degree"] = result
        return result

    def compute_hits(
        self, max_iter: int = 100, tol: float = 1e-8
    ) -> tuple[CentralityScores, CentralityScores]:
        """Compute HITS hub and authority scores.

        The HITS algorithm identifies two types of important nodes:
        - **Hubs**: Nodes that point to many good authorities (e.g., brokers)
        - **Authorities**: Nodes pointed to by many good hubs (e.g., major exporters)

        Args:
            max_iter: Maximum iterations for convergence.
            tol: Convergence tolerance.

        Returns:
            Tuple of (hub_scores, authority_scores).
        """
        if self._graph.number_of_nodes() == 0:
            empty = CentralityScores(metric_name="", scores={}, statistics={})
            return empty, empty

        simple = self._to_simple_digraph("weight")
        hubs, authorities = nx.hits(simple, max_iter=max_iter, tol=tol)

        hub_result = self._build_scores("hub", hubs)
        auth_result = self._build_scores("authority", authorities)

        self._cache["hub"] = hub_result
        self._cache["authority"] = auth_result

        return hub_result, auth_result

    def find_facilitators(
        self,
        top_k: int = 10,
        node_types: Optional[list[NodeType]] = None,
        community_partition: Optional[dict[str, int]] = None,
    ) -> list[FacilitatorProfile]:
        """Identify potential facilitator entities in the trade network.

        Facilitators are entities with high betweenness centrality that
        bridge multiple communities. They are structurally critical —
        their removal would fragment the network. In fraud contexts,
        facilitators are often the orchestrators of multi-party schemes.

        This method computes a composite bridging score from:
        - Betweenness centrality (0.5 weight)
        - PageRank (0.3 weight)
        - Number of connected communities (0.2 weight)

        Args:
            top_k: Number of top facilitators to return.
            node_types: Optional filter for specific entity types.
            community_partition: Optional pre-computed community assignments.

        Returns:
            List of FacilitatorProfile objects, sorted by bridging score.
        """
        # Ensure centrality scores are computed
        if "pagerank" not in self._cache:
            self.compute_pagerank()
        if "betweenness" not in self._cache:
            self.compute_betweenness()
        if "degree" not in self._cache:
            self.compute_degree_centrality()

        pagerank = self._cache["pagerank"]
        betweenness = self._cache["betweenness"]
        degree = self._cache["degree"]

        # Normalize scores to [0, 1] for composite calculation
        pr_max = max(pagerank.scores.values()) if pagerank.scores else 1.0
        bt_max = max(betweenness.scores.values()) if betweenness.scores else 1.0

        facilitators = []

        for node_id in self._graph.nodes():
            node_data = self._graph.nodes[node_id]
            node_type = node_data.get("node_type", "UNKNOWN")

            # Apply node type filter
            if node_types:
                type_values = [nt.value for nt in node_types]
                if node_type not in type_values:
                    continue

            pr_score = pagerank.scores.get(node_id, 0.0)
            bt_score = betweenness.scores.get(node_id, 0.0)
            dg_score = degree.scores.get(node_id, 0.0)

            # Normalize
            pr_norm = pr_score / pr_max if pr_max > 0 else 0.0
            bt_norm = bt_score / bt_max if bt_max > 0 else 0.0

            # Count connected communities
            connected_comms: list[int] = []
            if community_partition:
                neighbor_comms = set()
                for neighbor in nx.all_neighbors(self._graph, node_id):
                    if neighbor in community_partition:
                        neighbor_comms.add(community_partition[neighbor])
                connected_comms = list(neighbor_comms)

            comm_score = min(len(connected_comms) / 5.0, 1.0)

            # Composite bridging score
            bridging_score = (
                0.5 * bt_norm + 0.3 * pr_norm + 0.2 * comm_score
            )

            # Risk indicators
            risk_indicators = []
            in_deg = self._graph.in_degree(node_id)
            out_deg = self._graph.out_degree(node_id)

            if bt_norm > 0.7:
                risk_indicators.append("high_betweenness")
            if in_deg > 0 and out_deg > 0 and abs(in_deg - out_deg) / max(in_deg, out_deg) > 0.8:
                risk_indicators.append("asymmetric_connections")
            if len(connected_comms) >= 3:
                risk_indicators.append("multi_community_bridge")

            profile = FacilitatorProfile(
                node_id=node_id,
                node_type=node_type,
                pagerank=pr_score,
                betweenness=bt_score,
                degree=dg_score,
                in_degree=in_deg,
                out_degree=out_deg,
                bridging_score=bridging_score,
                connected_communities=connected_comms,
                risk_indicators=risk_indicators,
            )
            facilitators.append(profile)

        # Sort by bridging score and return top-k
        facilitators.sort(key=lambda f: f.bridging_score, reverse=True)
        top_facilitators = facilitators[:top_k]

        logger.info(
            "facilitators_identified",
            total_candidates=len(facilitators),
            top_k=top_k,
            top_score=top_facilitators[0].bridging_score if top_facilitators else 0,
        )

        return top_facilitators

    def compute_centrality_volume_divergence(
        self,
        volume_attribute: str = "total_value",
    ) -> dict[str, float]:
        """Compute the divergence between centrality and trade volume.

        Entities with high centrality but low trade volume are potential
        shell companies or pass-through entities. This metric quantifies
        the structural anomaly.

        Args:
            volume_attribute: Node attribute containing trade volume.

        Returns:
            Dictionary mapping node ID to divergence score.
            Higher values indicate greater centrality-volume mismatch.
        """
        if "degree" not in self._cache:
            self.compute_degree_centrality()

        degree_scores = self._cache["degree"].scores
        dg_max = max(degree_scores.values()) if degree_scores else 1.0

        # Compute trade volume for each node from connected edges
        node_volumes: dict[str, float] = {}
        for node_id in self._graph.nodes():
            total = 0.0
            for _, _, data in self._graph.edges(node_id, data=True):
                total += data.get(volume_attribute, data.get("total_value", 0.0))
            node_volumes[node_id] = total

        vol_max = max(node_volumes.values()) if node_volumes else 1.0

        divergence: dict[str, float] = {}
        for node_id in self._graph.nodes():
            dg_norm = degree_scores.get(node_id, 0.0) / dg_max if dg_max > 0 else 0.0
            vol_norm = (
                node_volumes.get(node_id, 0.0) / vol_max if vol_max > 0 else 0.0
            )
            # Divergence: high centrality + low volume = high divergence
            if vol_norm > 0:
                divergence[node_id] = dg_norm / vol_norm
            elif dg_norm > 0:
                divergence[node_id] = float("inf")
            else:
                divergence[node_id] = 0.0

        return divergence

    def _to_simple_digraph(self, weight: str = "weight") -> nx.DiGraph:
        """Convert the MultiDiGraph to a simple DiGraph with aggregated weights.

        Args:
            weight: Edge attribute to aggregate.

        Returns:
            Simple DiGraph with aggregated edge weights.
        """
        simple = nx.DiGraph()
        for node_id, data in self._graph.nodes(data=True):
            simple.add_node(node_id, **data)

        for u, v, data in self._graph.edges(data=True):
            w = data.get(weight, 1.0)
            if simple.has_edge(u, v):
                simple[u][v][weight] = simple[u][v].get(weight, 0.0) + w
            else:
                simple.add_edge(u, v, **{weight: w})

        return simple

    @staticmethod
    def _build_scores(
        metric_name: str,
        raw_scores: dict[str, float],
        top_k: int = 20,
    ) -> CentralityScores:
        """Build a CentralityScores object from raw score dictionaries.

        Args:
            metric_name: Name of the centrality metric.
            raw_scores: Mapping from node ID to raw score.
            top_k: Number of top-scoring nodes to include.

        Returns:
            CentralityScores with statistics and top-k list.
        """
        if not raw_scores:
            return CentralityScores(
                metric_name=metric_name,
                scores={},
                statistics={"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0},
            )

        values = list(raw_scores.values())
        statistics = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
            "median": float(np.median(values)),
        }

        sorted_scores = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_list = sorted_scores[:top_k]

        return CentralityScores(
            metric_name=metric_name,
            scores=raw_scores,
            top_k=top_k_list,
            statistics=statistics,
        )
