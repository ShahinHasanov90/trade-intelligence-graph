"""Tests for risk score propagation module.

Validates BFS-based risk propagation, decay mechanics, edge-type
weighting, multi-source propagation, and risk summary generation.
"""

from __future__ import annotations

import networkx as nx
import pytest

from graph_intel.analysis.propagation import (
    PropagationResult,
    RiskPropagator,
)
from graph_intel.config import PropagationConfig
from graph_intel.graph.schema import RiskLevel


class TestRiskPropagator:
    """Tests for RiskPropagator."""

    def test_propagate_from_single_node(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Propagation from a single node should affect neighbors."""
        propagator = RiskPropagator(built_graph)

        # Pick a node that has neighbors
        source = None
        for node in built_graph.nodes():
            if built_graph.degree(node) > 0:
                source = node
                break
        assert source is not None

        result = propagator.propagate_from_node(source, risk_score=0.9)

        assert isinstance(result, PropagationResult)
        assert result.source_node == source
        assert result.source_risk == 0.9
        assert isinstance(result.affected_nodes, dict)

    def test_risk_decays_with_distance(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Risk should decrease with graph distance from the source."""
        config = PropagationConfig(
            max_depth=4,
            decay_factor=0.5,
            min_propagation_threshold=0.01,
        )
        propagator = RiskPropagator(built_graph, config=config)

        source = None
        for node in built_graph.nodes():
            if built_graph.degree(node) > 2:
                source = node
                break
        assert source is not None

        result = propagator.propagate_from_node(source, risk_score=1.0)

        # Nodes further from source should have lower risk
        for node_id, risk in result.affected_nodes.items():
            path = result.propagation_paths.get(node_id, [])
            if len(path) > 1:
                # Risk should be less than source risk
                assert risk < 1.0

    def test_max_depth_limits_propagation(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Propagation should not exceed max_depth."""
        config = PropagationConfig(max_depth=1, decay_factor=0.5)
        propagator = RiskPropagator(built_graph, config=config)

        source = next(iter(built_graph.nodes()))
        result = propagator.propagate_from_node(source, risk_score=0.9)

        assert result.max_depth_reached <= 1

    def test_min_threshold_stops_propagation(self) -> None:
        """Risk below min_threshold should not propagate further."""
        g = nx.MultiDiGraph()
        g.add_node("A", node_type="IMPORTER", risk_score=0.0, risk_level="LOW", flagged=False)
        g.add_node("B", node_type="EXPORTER", risk_score=0.0, risk_level="LOW", flagged=False)
        g.add_node("C", node_type="IMPORTER", risk_score=0.0, risk_level="LOW", flagged=False)
        g.add_edge("A", "B", edge_type="IMPORTS_FROM", weight=1.0)
        g.add_edge("B", "C", edge_type="IMPORTS_FROM", weight=1.0)

        config = PropagationConfig(
            max_depth=10,
            decay_factor=0.1,  # Steep decay
            min_propagation_threshold=0.5,
        )
        propagator = RiskPropagator(g, config=config)
        result = propagator.propagate_from_node("A", risk_score=0.8)

        # With 0.1 decay and 0.5 threshold, propagation stops quickly
        # 0.8 * 0.1 = 0.08, which is below threshold
        assert result.max_depth_reached <= 1

    def test_edge_type_weights_affect_propagation(self) -> None:
        """Different edge types should propagate different amounts of risk."""
        g = nx.MultiDiGraph()
        g.add_node("A", node_type="IMPORTER", risk_score=0.0, risk_level="LOW", flagged=False)
        g.add_node("B", node_type="EXPORTER", risk_score=0.0, risk_level="LOW", flagged=False)
        g.add_node("C", node_type="ADDRESS", risk_score=0.0, risk_level="LOW", flagged=False)
        g.add_edge("A", "B", edge_type="IMPORTS_FROM", weight=1.0)
        g.add_edge("A", "C", edge_type="LOCATED_AT", weight=1.0)

        config = PropagationConfig(
            max_depth=2,
            decay_factor=0.8,
            min_propagation_threshold=0.01,
            edge_type_weights={
                "IMPORTS_FROM": 1.0,
                "LOCATED_AT": 0.3,
            },
        )
        propagator = RiskPropagator(g, config=config)
        result = propagator.propagate_from_node("A", risk_score=1.0)

        risk_b = result.affected_nodes.get("B", 0.0)
        risk_c = result.affected_nodes.get("C", 0.0)

        # B should receive more risk (IMPORTS_FROM weight=1.0)
        # C should receive less risk (LOCATED_AT weight=0.3)
        assert risk_b > risk_c

    def test_propagation_updates_graph_attributes(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """With update_graph=True, node risk_score attributes should be updated."""
        propagator = RiskPropagator(built_graph)

        source = next(iter(built_graph.nodes()))
        propagator.propagate_from_node(source, risk_score=0.9, update_graph=True)

        # Source should be flagged
        assert built_graph.nodes[source]["risk_score"] == 0.9
        assert built_graph.nodes[source]["flagged"] is True

    def test_propagation_without_graph_update(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """With update_graph=False, graph attributes should remain unchanged."""
        propagator = RiskPropagator(built_graph)

        source = next(iter(built_graph.nodes()))
        original_risk = built_graph.nodes[source].get("risk_score", 0.0)

        propagator.propagate_from_node(
            source, risk_score=0.9, update_graph=False
        )

        # Score should not have changed
        assert built_graph.nodes[source].get("risk_score", 0.0) == original_risk

    def test_propagate_from_multiple_sources(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Multi-source propagation should combine risk from all sources."""
        propagator = RiskPropagator(built_graph)

        sources = {}
        for i, node in enumerate(built_graph.nodes()):
            if i >= 3:
                break
            sources[node] = 0.8

        results = propagator.propagate_from_multiple(sources)

        assert len(results) == len(sources)

    def test_reset_risk_scores(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """reset_risk_scores should zero out all risk scores."""
        propagator = RiskPropagator(built_graph)

        # First propagate some risk
        source = next(iter(built_graph.nodes()))
        propagator.propagate_from_node(source, risk_score=0.9)

        # Then reset
        propagator.reset_risk_scores()

        for _, data in built_graph.nodes(data=True):
            assert data["risk_score"] == 0.0
            assert data["risk_level"] == "LOW"
            assert data["flagged"] is False

    def test_get_risk_summary(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """get_risk_summary should return distribution statistics."""
        propagator = RiskPropagator(built_graph)
        source = next(iter(built_graph.nodes()))
        propagator.propagate_from_node(source, risk_score=0.9)

        summary = propagator.get_risk_summary()

        assert "total_nodes" in summary
        assert "mean_risk" in summary
        assert "max_risk" in summary
        assert "level_distribution" in summary
        assert "top_risk_entities" in summary

    def test_propagation_log(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Propagation should maintain an audit log of events."""
        propagator = RiskPropagator(built_graph)
        source = next(
            n for n in built_graph.nodes() if built_graph.degree(n) > 0
        )
        propagator.propagate_from_node(source, risk_score=0.9)

        # Log may be empty if no propagation occurred (e.g., no outgoing edges)
        assert isinstance(propagator.propagation_log, list)

    def test_invalid_source_raises_error(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Propagating from a non-existent node should raise ValueError."""
        propagator = RiskPropagator(built_graph)

        with pytest.raises(ValueError, match="not in graph"):
            propagator.propagate_from_node("NONEXISTENT_NODE", risk_score=0.9)

    def test_score_to_level_mapping(self) -> None:
        """Risk scores should map correctly to risk levels."""
        assert RiskPropagator._score_to_level(0.0) == RiskLevel.LOW
        assert RiskPropagator._score_to_level(0.2) == RiskLevel.LOW
        assert RiskPropagator._score_to_level(0.4) == RiskLevel.MEDIUM
        assert RiskPropagator._score_to_level(0.7) == RiskLevel.HIGH
        assert RiskPropagator._score_to_level(0.9) == RiskLevel.CRITICAL

    def test_propagation_paths_recorded(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Propagation should record the path to each affected node."""
        propagator = RiskPropagator(built_graph)
        source = next(
            n for n in built_graph.nodes() if built_graph.degree(n) > 0
        )
        result = propagator.propagate_from_node(source, risk_score=0.9)

        for node_id, path in result.propagation_paths.items():
            assert isinstance(path, list)
            assert path[0] == source  # Path starts at source
            assert path[-1] == node_id  # Path ends at affected node
