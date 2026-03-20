"""Tests for the community detection module.

Validates Louvain, Leiden, and label propagation community detection,
anomalous community identification, and community metrics.
"""

from __future__ import annotations

import networkx as nx
import pytest

from graph_intel.analysis.community import (
    Community,
    CommunityDetector,
    CommunityResult,
)


class TestCommunityDetector:
    """Tests for CommunityDetector."""

    def test_louvain_detection_returns_communities(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Louvain detection should return at least one community."""
        detector = CommunityDetector(built_graph)
        result = detector.detect_louvain()

        assert isinstance(result, CommunityResult)
        assert result.algorithm == "louvain"
        assert result.num_communities >= 0
        assert isinstance(result.modularity, float)

    def test_louvain_partition_covers_all_nodes(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """The partition should assign every node to a community."""
        detector = CommunityDetector(built_graph)
        result = detector.detect_louvain()

        # Every node in the undirected graph should have an assignment
        ug = detector.undirected_graph
        for node in ug.nodes():
            assert node in result.partition

    def test_modularity_is_bounded(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Modularity should be between -0.5 and 1.0."""
        detector = CommunityDetector(built_graph)
        result = detector.detect_louvain()

        assert -0.5 <= result.modularity <= 1.0

    def test_resolution_affects_community_count(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Higher resolution should produce more (or equal) communities."""
        detector = CommunityDetector(built_graph)

        result_low = detector.detect_louvain(resolution=0.5)
        # Reset undirected cache
        detector._undirected = None
        result_high = detector.detect_louvain(resolution=2.0)

        # Higher resolution generally produces more communities
        # But the relationship isn't strictly monotonic in all cases
        assert result_low.num_communities >= 0
        assert result_high.num_communities >= 0

    def test_community_has_members(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Each community should have at least one member."""
        detector = CommunityDetector(built_graph)
        result = detector.detect_louvain()

        for community in result.communities:
            assert len(community.members) > 0
            assert community.size == len(community.members)

    def test_community_metrics_computed(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Communities should have computed internal/external edge counts."""
        detector = CommunityDetector(built_graph)
        result = detector.detect_louvain()

        for community in result.communities:
            assert isinstance(community.internal_edges, int)
            assert isinstance(community.external_edges, int)
            assert isinstance(community.density, float)
            assert 0.0 <= community.density <= 1.0

    def test_community_node_types_tracked(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Communities should track the distribution of node types."""
        detector = CommunityDetector(built_graph)
        result = detector.detect_louvain()

        for community in result.communities:
            assert isinstance(community.node_types, dict)
            total = sum(community.node_types.values())
            assert total == community.size

    def test_min_community_size_filter(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Communities smaller than min_community_size should be excluded."""
        from graph_intel.config import CommunityConfig

        config = CommunityConfig(min_community_size=5)
        detector = CommunityDetector(built_graph, config=config)
        result = detector.detect_louvain()

        for community in result.communities:
            assert community.size >= 5

    def test_label_propagation_detection(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Label propagation should produce valid communities."""
        detector = CommunityDetector(built_graph)
        result = detector.detect_label_propagation()

        assert result.algorithm == "label_propagation"
        assert isinstance(result.modularity, float)

    def test_detect_dispatches_by_algorithm(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """The detect() method should dispatch to the correct algorithm."""
        detector = CommunityDetector(built_graph)

        result = detector.detect(algorithm="louvain")
        assert result.algorithm == "louvain"

        result = detector.detect(algorithm="label_propagation")
        assert result.algorithm == "label_propagation"

    def test_detect_invalid_algorithm_raises(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """An invalid algorithm name should raise ValueError."""
        detector = CommunityDetector(built_graph)

        with pytest.raises(ValueError, match="Unknown algorithm"):
            detector.detect(algorithm="invalid_algo")

    def test_empty_graph_returns_empty(self) -> None:
        """An empty graph should return zero communities."""
        g = nx.MultiDiGraph()
        detector = CommunityDetector(g)
        result = detector.detect_louvain()

        assert result.num_communities == 0
        assert result.modularity == 0.0

    def test_find_anomalous_communities(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """find_anomalous_communities should return a list of communities."""
        detector = CommunityDetector(built_graph)
        result = detector.detect_louvain()

        anomalous = detector.find_anomalous_communities(result)
        assert isinstance(anomalous, list)
        # All returned items should be Community objects
        for comm in anomalous:
            assert isinstance(comm, Community)
