"""Shared test fixtures for Trade Intelligence Graph test suite.

Provides reusable graph fixtures with realistic trade data patterns
for testing all analysis and detection modules.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pytest

from graph_intel.config import Settings, reset_settings
from graph_intel.graph.builder import TradeGraphBuilder
from graph_intel.graph.schema import EdgeType, NodeType, RiskLevel


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    """Reset global settings before each test."""
    reset_settings()


@pytest.fixture
def sample_declarations() -> list[dict[str, Any]]:
    """Provide a set of sample declarations for testing.

    Creates a realistic dataset with 20 declarations involving
    5 importers, 3 exporters, 2 agents, and overlapping attributes.
    """
    declarations = []
    for i in range(20):
        decl = {
            "declaration_id": f"DEC-TEST-{i:04d}",
            "importer": {
                "tax_id": f"IMP{(i % 5):03d}",
                "name": f"Test Importer {i % 5}",
                "country": ["US", "GB", "DE"][i % 3],
                "address": ["123 Trade St", "456 Commerce Ave"][i % 2],
            },
            "exporter": {
                "tax_id": f"EXP{(i % 3):03d}",
                "name": f"Test Exporter {i % 3}",
                "country": ["CN", "VN", "IN"][i % 3],
                "address": f"Export Zone {i % 3}",
            },
            "commodity": {
                "hs_code": ["8471.30", "6403.99"][i % 2],
                "description": ["Portable computers", "Footwear"][i % 2],
            },
            "route": {
                "origin": ["CN", "VN", "IN"][i % 3],
                "destination": ["US", "GB", "DE"][i % 3],
                "transit": [["HK"], ["SG"], []][i % 3],
            },
            "agent": {
                "license_id": f"BRK-{i % 2:03d}",
                "name": f"Test Broker {i % 2}",
            },
            "value": 100000 + (i * 5000),
            "weight": 500 + (i * 25),
            "date": f"2025-01-{(i % 28) + 1:02d}",
        }
        declarations.append(decl)
    return declarations


@pytest.fixture
def built_graph(sample_declarations: list[dict[str, Any]]) -> nx.MultiDiGraph:
    """Provide a fully constructed trade graph from sample declarations."""
    builder = TradeGraphBuilder()
    builder.add_declarations(sample_declarations)
    return builder.build()


@pytest.fixture
def builder(sample_declarations: list[dict[str, Any]]) -> TradeGraphBuilder:
    """Provide a TradeGraphBuilder with sample data loaded."""
    b = TradeGraphBuilder()
    b.add_declarations(sample_declarations)
    return b


@pytest.fixture
def circular_trade_graph() -> nx.MultiDiGraph:
    """Provide a graph with explicit circular trade patterns.

    Creates:
    - Cycle: IMP:A -> EXP:B -> IMP:C -> EXP:A (circular)
    - Shared address between A and C
    """
    g = nx.MultiDiGraph()

    # Add importer/exporter nodes
    for node_id, node_type, country, name in [
        ("IMP:A", "IMPORTER", "US", "Company A"),
        ("EXP:B", "EXPORTER", "CN", "Company B"),
        ("IMP:C", "IMPORTER", "GB", "Company C"),
        ("EXP:A", "EXPORTER", "US", "Company A Export"),
        ("IMP:D", "IMPORTER", "DE", "Company D"),
    ]:
        g.add_node(
            node_id,
            node_type=node_type,
            country=country,
            name=name,
            risk_score=0.0,
            risk_level="LOW",
            flagged=False,
        )

    # Circular trade: A -> B -> C -> A
    g.add_edge(
        "IMP:A", "EXP:B",
        key="IMPORTS_FROM",
        edge_type="IMPORTS_FROM",
        weight=1.0,
        total_value=200000,
        transaction_count=5,
        last_seen="2025-01-15",
    )
    g.add_edge(
        "EXP:B", "IMP:C",
        key="IMPORTS_FROM",
        edge_type="IMPORTS_FROM",
        weight=1.0,
        total_value=190000,
        transaction_count=4,
        last_seen="2025-01-20",
    )
    g.add_edge(
        "IMP:C", "EXP:A",
        key="IMPORTS_FROM",
        edge_type="IMPORTS_FROM",
        weight=1.0,
        total_value=185000,
        transaction_count=4,
        last_seen="2025-01-25",
    )
    g.add_edge(
        "EXP:A", "IMP:A",
        key="IMPORTS_FROM",
        edge_type="IMPORTS_FROM",
        weight=1.0,
        total_value=180000,
        transaction_count=3,
        last_seen="2025-01-28",
    )

    # Additional non-circular trade
    g.add_edge(
        "IMP:D", "EXP:B",
        key="IMPORTS_FROM",
        edge_type="IMPORTS_FROM",
        weight=0.5,
        total_value=50000,
        transaction_count=2,
        last_seen="2025-01-10",
    )

    # Shared address
    g.add_node(
        "ADDR:shared123",
        node_type="ADDRESS",
        normalized_address="123 trade street",
        country="US",
    )
    g.add_edge(
        "IMP:A", "ADDR:shared123",
        key="LOCATED_AT",
        edge_type="LOCATED_AT",
        weight=1.0,
    )
    g.add_edge(
        "EXP:A", "ADDR:shared123",
        key="LOCATED_AT",
        edge_type="LOCATED_AT",
        weight=1.0,
    )

    return g


@pytest.fixture
def shell_company_graph() -> nx.MultiDiGraph:
    """Provide a graph with shell company patterns.

    Creates an entity with high centrality, low volume, and
    connections to flagged entities in a free-trade zone.
    """
    g = nx.MultiDiGraph()

    # The shell company candidate: high connectivity, FTZ, low volume
    g.add_node(
        "IMP:SHELL",
        node_type="IMPORTER",
        country="HK",
        name="HK Trading Ltd",
        tax_id="SHELL001",
        registration_date="2025-06-01",
        risk_score=0.0,
        risk_level="LOW",
        flagged=False,
    )

    # Connected legitimate importers (some flagged)
    for i in range(8):
        flagged = i < 4
        g.add_node(
            f"EXP:LEGIT{i}",
            node_type="EXPORTER",
            country=["CN", "VN", "IN", "TH"][i % 4],
            name=f"Legitimate Exporter {i}",
            tax_id=f"LEG{i:03d}",
            risk_score=0.7 if flagged else 0.1,
            risk_level="HIGH" if flagged else "LOW",
            flagged=flagged,
        )
        # Shell company connects to many exporters with low volume
        g.add_edge(
            "IMP:SHELL", f"EXP:LEGIT{i}",
            key="IMPORTS_FROM",
            edge_type="IMPORTS_FROM",
            weight=0.1,
            total_value=1000,  # Very low volume
            transaction_count=1,
        )

    # A normal high-volume importer for contrast
    g.add_node(
        "IMP:NORMAL",
        node_type="IMPORTER",
        country="US",
        name="Normal Imports Inc",
        tax_id="NORM001",
        risk_score=0.1,
        risk_level="LOW",
        flagged=False,
    )
    g.add_edge(
        "IMP:NORMAL", "EXP:LEGIT0",
        key="IMPORTS_FROM",
        edge_type="IMPORTS_FROM",
        weight=1.0,
        total_value=500000,
        transaction_count=50,
    )

    # Add commodity for the shell entity
    g.add_node(
        "COMM:8471",
        node_type="COMMODITY",
        hs_code="8471.30",
    )
    g.add_node(
        "DECL:SHELL1",
        node_type="DECLARATION",
        declaration_id="SHELL-DEC-001",
        date="2025-07-01",
        value=1000,
    )
    g.add_edge(
        "IMP:SHELL", "DECL:SHELL1",
        key="DECLARES",
        edge_type="DECLARES",
        weight=1.0,
    )
    g.add_edge(
        "DECL:SHELL1", "COMM:8471",
        key="TRADES_COMMODITY",
        edge_type="TRADES_COMMODITY",
        weight=1.0,
    )

    return g
