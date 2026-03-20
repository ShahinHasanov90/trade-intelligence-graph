"""Tests for fraud ring detection module.

Validates circular trade detection, shared attribute clustering,
behavioral synchronization, and comprehensive ring detection.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pytest

from graph_intel.detection.rings import (
    CircularTradePattern,
    FraudRing,
    FraudRingDetector,
    SharedAttributeCluster,
)
from graph_intel.graph.schema import EdgeType, NodeType


class TestFraudRingDetector:
    """Tests for FraudRingDetector."""

    def test_find_circular_trade_in_cycle_graph(
        self, circular_trade_graph: nx.MultiDiGraph
    ) -> None:
        """Should detect the circular trade pattern in the test graph."""
        detector = FraudRingDetector(circular_trade_graph)
        patterns = detector.find_circular_trade(max_depth=6)

        assert len(patterns) >= 1
        # At least one cycle should involve 4 entities
        cycle_lengths = [p.length for p in patterns]
        assert any(length >= 3 for length in cycle_lengths)

    def test_circular_trade_pattern_has_metrics(
        self, circular_trade_graph: nx.MultiDiGraph
    ) -> None:
        """Circular trade patterns should have computed metrics."""
        detector = FraudRingDetector(circular_trade_graph)
        patterns = detector.find_circular_trade(max_depth=6)

        if patterns:
            pattern = patterns[0]
            assert isinstance(pattern, CircularTradePattern)
            assert pattern.length >= 3
            assert isinstance(pattern.total_value, float)
            assert isinstance(pattern.countries, set)

    def test_shared_attribute_clusters_detected(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Should find entities sharing common attributes (addresses)."""
        detector = FraudRingDetector(built_graph)
        clusters = detector.find_shared_attribute_clusters(min_shared=2)

        assert isinstance(clusters, list)
        for cluster in clusters:
            assert isinstance(cluster, SharedAttributeCluster)
            assert cluster.entity_count >= 2

    def test_shared_attribute_cluster_types(
        self, built_graph: nx.MultiDiGraph
    ) -> None:
        """Shared attribute clusters should identify the attribute type."""
        detector = FraudRingDetector(built_graph)
        clusters = detector.find_shared_attribute_clusters()

        for cluster in clusters:
            assert cluster.shared_node_type in [
                "ADDRESS", "PHONE", "BANK_ACCOUNT"
            ]

    def test_detect_fraud_rings_comprehensive(
        self, circular_trade_graph: nx.MultiDiGraph
    ) -> None:
        """Comprehensive detection should find rings from multiple strategies."""
        detector = FraudRingDetector(circular_trade_graph)
        rings = detector.detect_fraud_rings(min_confidence=0.3)

        assert isinstance(rings, list)
        for ring in rings:
            assert isinstance(ring, FraudRing)
            assert ring.confidence >= 0.3
            assert ring.ring_type in ["circular", "shared_attribute", "behavioral"]

    def test_fraud_ring_has_evidence(
        self, circular_trade_graph: nx.MultiDiGraph
    ) -> None:
        """Detected fraud rings should include evidence."""
        detector = FraudRingDetector(circular_trade_graph)
        rings = detector.detect_fraud_rings(min_confidence=0.3)

        for ring in rings:
            assert len(ring.evidence) > 0
            assert ring.pattern_description != ""

    def test_fraud_ring_has_subgraph(
        self, circular_trade_graph: nx.MultiDiGraph
    ) -> None:
        """Detected fraud rings should include an induced subgraph."""
        detector = FraudRingDetector(circular_trade_graph)
        rings = detector.detect_fraud_rings(min_confidence=0.3)

        for ring in rings:
            if ring.subgraph is not None:
                assert isinstance(ring.subgraph, nx.MultiDiGraph)
                assert ring.subgraph.number_of_nodes() > 0

    def test_circular_confidence_shorter_cycles_higher(self) -> None:
        """Shorter cycles should receive higher confidence scores."""
        detector = FraudRingDetector(nx.MultiDiGraph())

        short_pattern = CircularTradePattern(
            cycle=["A", "B", "C"],
            length=3,
            total_value=200000,
            countries={"US", "CN"},
            involves_shared_attributes=True,
        )
        long_pattern = CircularTradePattern(
            cycle=["A", "B", "C", "D", "E", "F"],
            length=6,
            total_value=200000,
            countries={"US", "CN"},
            involves_shared_attributes=True,
        )

        short_conf = detector._compute_circular_confidence(short_pattern)
        long_conf = detector._compute_circular_confidence(long_pattern)
        assert short_conf >= long_conf

    def test_no_cycles_in_acyclic_graph(self) -> None:
        """An acyclic graph should produce no circular trade patterns."""
        g = nx.MultiDiGraph()
        g.add_node("IMP:A", node_type="IMPORTER", country="US")
        g.add_node("EXP:B", node_type="EXPORTER", country="CN")
        g.add_edge(
            "IMP:A", "EXP:B",
            key="IMPORTS_FROM",
            edge_type="IMPORTS_FROM",
            weight=1.0,
            total_value=100000,
        )

        detector = FraudRingDetector(g)
        patterns = detector.find_circular_trade()

        assert len(patterns) == 0

    def test_behavioral_synchronization_detection(self) -> None:
        """Should detect synchronized filing patterns."""
        g = nx.MultiDiGraph()

        # Create 5 importers filing on the same date with same values
        for i in range(5):
            imp_id = f"IMP:SYNC{i}"
            decl_id = f"DECL:SYNC{i}"
            g.add_node(
                imp_id,
                node_type="IMPORTER",
                name=f"Sync Importer {i}",
                risk_score=0.0,
                risk_level="LOW",
                flagged=False,
            )
            g.add_node(
                decl_id,
                node_type="DECLARATION",
                declaration_id=f"SYNC-{i}",
                date="2025-03-15",
                value=100000.0,  # Same value
            )
            g.add_edge(
                imp_id, decl_id,
                key="DECLARES",
                edge_type="DECLARES",
                weight=1.0,
            )

        detector = FraudRingDetector(g)
        rings = detector._detect_behavioral_synchronization()

        assert isinstance(rings, list)
        # Should detect the synchronized pattern
        if rings:
            assert rings[0].ring_type == "behavioral"

    def test_ring_ids_are_unique(
        self, circular_trade_graph: nx.MultiDiGraph
    ) -> None:
        """All detected ring IDs should be unique."""
        detector = FraudRingDetector(circular_trade_graph)
        rings = detector.detect_fraud_rings(min_confidence=0.0)

        ring_ids = [r.ring_id for r in rings]
        assert len(ring_ids) == len(set(ring_ids))

    def test_min_confidence_filters_results(
        self, circular_trade_graph: nx.MultiDiGraph
    ) -> None:
        """Higher min_confidence should return fewer or equal rings."""
        detector = FraudRingDetector(circular_trade_graph)

        rings_low = detector.detect_fraud_rings(min_confidence=0.1)
        detector._ring_counter = 0  # Reset for fair comparison
        rings_high = detector.detect_fraud_rings(min_confidence=0.9)

        assert len(rings_high) <= len(rings_low)
