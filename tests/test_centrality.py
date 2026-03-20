"""Tests for the centrality analysis module.

Validates PageRank, betweenness centrality, degree centrality,
facilitator detection, and centrality-volume divergence analysis.
"""

from __future__ import annotations

import networkx as nx
import pytest

from graph_intel.analysis.centrality import (
    CentralityAnalyzer,
    CentralityScores,
    FacilitatorProfile,
)
from graph_intel.graph.schema import NodeType


class TestCentralityAnalyzer:
    """Tests for CentralityAnalyzer."""

    def test_pagerank_returns_scores_for_all_nodes(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """PageRank should return a score for every node."""
        analyzer = CentralityAnalyzer(built_graph)
        result = analyzer.compute_pagerank()

        assert isinstance(result, CentralityScores)
        assert result.metric_name == "pagerank"
        assert len(result.scores) == built_graph.number_of_nodes()

    def test_pagerank_scores_sum_to_one(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """PageRank scores should sum to approximately 1.0."""
        analyzer = CentralityAnalyzer(built_graph)
        result = analyzer.compute_pagerank()

        total = sum(result.scores.values())
        assert abs(total - 1.0) < 0.01

    def test_pagerank_statistics_computed(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """PageRank result should include summary statistics."""
        analyzer = CentralityAnalyzer(built_graph)
        result = analyzer.compute_pagerank()

        assert "mean" in result.statistics
        assert "std" in result.statistics
        assert "max" in result.statistics
        assert "min" in result.statistics
        assert "median" in result.statistics

    def test_betweenness_returns_scores(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Betweenness centrality should return scores for all nodes."""
        analyzer = CentralityAnalyzer(built_graph)
        result = analyzer.compute_betweenness()

        assert result.metric_name == "betweenness"
        assert len(result.scores) == built_graph.number_of_nodes()

    def test_betweenness_scores_non_negative(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """All betweenness scores should be non-negative."""
        analyzer = CentralityAnalyzer(built_graph)
        result = analyzer.compute_betweenness()

        for score in result.scores.values():
            assert score >= 0.0

    def test_degree_centrality_returns_scores(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Degree centrality should return scores for all nodes."""
        analyzer = CentralityAnalyzer(built_graph)
        result = analyzer.compute_degree_centrality()

        assert result.metric_name == "degree"
        assert len(result.scores) == built_graph.number_of_nodes()

    def test_degree_centrality_normalized(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Normalized degree centrality scores should be in [0, 1]."""
        analyzer = CentralityAnalyzer(built_graph)
        result = analyzer.compute_degree_centrality(normalized=True)

        for score in result.scores.values():
            assert 0.0 <= score <= 1.0

    def test_in_degree_centrality(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """In-degree centrality should return valid scores."""
        analyzer = CentralityAnalyzer(built_graph)
        result = analyzer.compute_in_degree_centrality()

        assert result.metric_name == "in_degree"
        assert len(result.scores) > 0

    def test_out_degree_centrality(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Out-degree centrality should return valid scores."""
        analyzer = CentralityAnalyzer(built_graph)
        result = analyzer.compute_out_degree_centrality()

        assert result.metric_name == "out_degree"
        assert len(result.scores) > 0

    def test_hits_returns_hub_and_authority_scores(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """HITS should return both hub and authority scores."""
        analyzer = CentralityAnalyzer(built_graph)
        hubs, authorities = analyzer.compute_hits()

        assert hubs.metric_name == "hub"
        assert authorities.metric_name == "authority"
        assert len(hubs.scores) == built_graph.number_of_nodes()
        assert len(authorities.scores) == built_graph.number_of_nodes()

    def test_find_facilitators_returns_profiles(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """find_facilitators should return FacilitatorProfile objects."""
        analyzer = CentralityAnalyzer(built_graph)
        facilitators = analyzer.find_facilitators(top_k=5)

        assert isinstance(facilitators, list)
        assert len(facilitators) <= 5
        for f in facilitators:
            assert isinstance(f, FacilitatorProfile)
            assert f.bridging_score >= 0.0

    def test_facilitators_sorted_by_bridging_score(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Facilitators should be sorted by bridging score descending."""
        analyzer = CentralityAnalyzer(built_graph)
        facilitators = analyzer.find_facilitators(top_k=10)

        for i in range(1, len(facilitators)):
            assert facilitators[i - 1].bridging_score >= facilitators[i].bridging_score

    def test_facilitators_with_node_type_filter(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Node type filter should restrict facilitator search."""
        analyzer = CentralityAnalyzer(built_graph)
        facilitators = analyzer.find_facilitators(
            top_k=10, node_types=[NodeType.AGENT]
        )

        for f in facilitators:
            assert f.node_type == "AGENT"

    def test_centrality_volume_divergence(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Centrality-volume divergence should identify structural anomalies."""
        analyzer = CentralityAnalyzer(built_graph)
        divergence = analyzer.compute_centrality_volume_divergence()

        assert isinstance(divergence, dict)
        assert len(divergence) == built_graph.number_of_nodes()

    def test_top_k_in_scores(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """CentralityScores.top_k should contain the highest-scoring nodes."""
        analyzer = CentralityAnalyzer(built_graph)
        result = analyzer.compute_pagerank()

        if result.top_k:
            top_node, top_score = result.top_k[0]
            assert top_score == result.statistics["max"]

    def test_empty_graph_pagerank(self) -> None:
        """PageRank on empty graph should return empty scores."""
        g = nx.MultiDiGraph()
        analyzer = CentralityAnalyzer(g)
        result = analyzer.compute_pagerank()

        assert len(result.scores) == 0

    def test_caching_avoids_recomputation(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Centrality results should be cached for reuse by facilitator detection."""
        analyzer = CentralityAnalyzer(built_graph)
        analyzer.compute_pagerank()
        analyzer.compute_betweenness()
        analyzer.compute_degree_centrality()

        assert "pagerank" in analyzer._cache
        assert "betweenness" in analyzer._cache
        assert "degree" in analyzer._cache
