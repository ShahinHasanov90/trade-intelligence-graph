"""Tests for the TradeGraphBuilder module.

Validates graph construction from declaration data, entity resolution,
edge weight computation, and graph statistics.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pytest

from graph_intel.graph.builder import TradeGraphBuilder
from graph_intel.graph.schema import EdgeType, NodeType


class TestTradeGraphBuilder:
    """Tests for TradeGraphBuilder."""

    def test_single_declaration_creates_expected_nodes(self) -> None:
        """A single declaration should create importer, exporter, commodity,
        route, declaration, and agent nodes."""
        builder = TradeGraphBuilder()
        builder.add_declaration({
            "declaration_id": "DEC-001",
            "importer": {"tax_id": "IMP001", "name": "Acme", "country": "US"},
            "exporter": {"tax_id": "EXP001", "name": "Global", "country": "CN"},
            "commodity": {"hs_code": "8471.30", "description": "Computers"},
            "route": {"origin": "CN", "destination": "US", "transit": ["HK"]},
            "agent": {"license_id": "BRK-001", "name": "FastClear"},
            "value": 100000.0,
            "weight": 500.0,
            "date": "2025-01-15",
        })
        graph = builder.build()

        assert graph.number_of_nodes() >= 5
        assert "IMP:IMP001" in graph
        assert "EXP:EXP001" in graph
        assert "COMM:8471.30" in graph
        assert "ROUTE:CN-US" in graph
        assert "AGENT:BRK-001" in graph

    def test_declaration_creates_expected_edges(self) -> None:
        """A declaration should create IMPORTS_FROM, DECLARES, TRADES_COMMODITY,
        USES_ROUTE, and REPRESENTED_BY edges."""
        builder = TradeGraphBuilder()
        builder.add_declaration({
            "declaration_id": "DEC-001",
            "importer": {"tax_id": "IMP001", "name": "Acme", "country": "US"},
            "exporter": {"tax_id": "EXP001", "name": "Global", "country": "CN"},
            "commodity": {"hs_code": "8471.30"},
            "route": {"origin": "CN", "destination": "US"},
            "agent": {"license_id": "BRK-001", "name": "FastClear"},
            "value": 100000.0,
            "date": "2025-01-15",
        })
        graph = builder.build()

        assert graph.has_edge("IMP:IMP001", "EXP:EXP001")
        assert graph.has_edge("IMP:IMP001", "DECL:DEC-001")
        assert graph.has_edge("DECL:DEC-001", "COMM:8471.30")
        assert graph.has_edge("DECL:DEC-001", "ROUTE:CN-US")
        assert graph.has_edge("IMP:IMP001", "AGENT:BRK-001")

    def test_duplicate_entities_are_merged(self) -> None:
        """Multiple declarations with the same importer should not create
        duplicate importer nodes."""
        builder = TradeGraphBuilder()
        for i in range(5):
            builder.add_declaration({
                "declaration_id": f"DEC-{i:03d}",
                "importer": {"tax_id": "IMP001", "name": "Acme", "country": "US"},
                "exporter": {"tax_id": f"EXP{i:03d}", "name": f"Exporter {i}", "country": "CN"},
                "commodity": {"hs_code": "8471.30"},
                "route": {"origin": "CN", "destination": "US"},
                "value": 100000.0,
                "date": f"2025-01-{i + 1:02d}",
            })
        graph = builder.build()

        # Should have only 1 importer node despite 5 declarations
        importer_count = sum(
            1 for _, d in graph.nodes(data=True) if d.get("node_type") == "IMPORTER"
        )
        assert importer_count == 1

    def test_edge_weights_aggregate_across_transactions(self) -> None:
        """Multiple transactions between the same pair should increase edge weight."""
        builder = TradeGraphBuilder()
        for i in range(3):
            builder.add_declaration({
                "declaration_id": f"DEC-{i:03d}",
                "importer": {"tax_id": "IMP001", "name": "Acme", "country": "US"},
                "exporter": {"tax_id": "EXP001", "name": "Global", "country": "CN"},
                "commodity": {"hs_code": "8471.30"},
                "route": {"origin": "CN", "destination": "US"},
                "value": 50000.0,
                "date": f"2025-01-{i + 1:02d}",
            })
        graph = builder.build()

        edge_data = graph["IMP:IMP001"]["EXP:EXP001"]["IMPORTS_FROM"]
        assert edge_data["transaction_count"] == 3
        assert edge_data["total_value"] == 150000.0

    def test_missing_required_fields_raises_error(self) -> None:
        """Declarations missing required fields should raise ValueError."""
        builder = TradeGraphBuilder()

        with pytest.raises(ValueError, match="missing required fields"):
            builder.add_declaration({"declaration_id": "DEC-001"})

    def test_missing_nested_fields_raises_error(self) -> None:
        """Declarations with incomplete nested entities should raise ValueError."""
        builder = TradeGraphBuilder()

        with pytest.raises(ValueError, match="missing required keys"):
            builder.add_declaration({
                "declaration_id": "DEC-001",
                "importer": {"name": "Acme"},  # Missing tax_id
                "exporter": {"tax_id": "EXP001"},
                "commodity": {"hs_code": "8471.30"},
                "route": {"origin": "CN", "destination": "US"},
            })

    def test_batch_processing(self, sample_declarations: list[dict[str, Any]]) -> None:
        """add_declarations should process multiple declarations in batch."""
        builder = TradeGraphBuilder()
        builder.add_declarations(sample_declarations)
        graph = builder.build()

        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        assert builder._declaration_count == len(sample_declarations)

    def test_get_statistics(self, sample_declarations: list[dict[str, Any]]) -> None:
        """get_statistics should return comprehensive graph metrics."""
        builder = TradeGraphBuilder()
        builder.add_declarations(sample_declarations)
        builder.build()

        stats = builder.get_statistics()
        assert stats["declaration_count"] == len(sample_declarations)
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0
        assert "node_types" in stats
        assert "edge_types" in stats
        assert stats["density"] > 0

    def test_address_normalization(self) -> None:
        """Addresses should be normalized for entity resolution."""
        result = TradeGraphBuilder._normalize_address("123 Trade St.")
        assert result == "123 trade street"

    def test_phone_normalization(self) -> None:
        """Phone numbers should be normalized to digits only."""
        result = TradeGraphBuilder._normalize_phone("+1 (555) 123-4567")
        assert result == "+15551234567"

    def test_address_node_created_for_importer(self) -> None:
        """Importers with addresses should get LOCATED_AT edges."""
        builder = TradeGraphBuilder()
        builder.add_declaration({
            "declaration_id": "DEC-001",
            "importer": {
                "tax_id": "IMP001",
                "name": "Acme",
                "country": "US",
                "address": "123 Trade Street",
            },
            "exporter": {"tax_id": "EXP001", "name": "Global", "country": "CN"},
            "commodity": {"hs_code": "8471.30"},
            "route": {"origin": "CN", "destination": "US"},
            "value": 100000.0,
            "date": "2025-01-15",
        })
        graph = builder.build()

        # Find address nodes
        addr_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("node_type") == "ADDRESS"
        ]
        assert len(addr_nodes) >= 1

        # Verify LOCATED_AT edge exists
        located_edges = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get("edge_type") == "LOCATED_AT"
        ]
        assert len(located_edges) >= 1

    def test_graph_metadata_set_on_build(self) -> None:
        """build() should set graph-level metadata."""
        builder = TradeGraphBuilder()
        builder.add_declaration({
            "declaration_id": "DEC-001",
            "importer": {"tax_id": "IMP001", "name": "Acme", "country": "US"},
            "exporter": {"tax_id": "EXP001", "name": "Global", "country": "CN"},
            "commodity": {"hs_code": "8471.30"},
            "route": {"origin": "CN", "destination": "US"},
            "value": 100000.0,
            "date": "2025-01-15",
        })
        graph = builder.build()

        assert graph.graph["declaration_count"] == 1
        assert graph.graph["node_count"] > 0
        assert graph.graph["edge_count"] > 0
        assert "built_at" in graph.graph
