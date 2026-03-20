# Trade Intelligence Graph

**Graph-based network analysis for trade fraud ring detection and customs intelligence**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Strategic Problem

International trade fraud rarely involves isolated actors. Undervaluation rings, carousel fraud, and origin-laundering schemes operate as **coordinated networks** spanning multiple importers, exporters, freight forwarders, and intermediaries. Traditional transaction-level scoring misses these network patterns because it evaluates declarations independently.

Consider a real scenario: a group of 12 importers, all registered at different addresses, all filing separate declarations for the same commodity from different declared origins. Transaction-level systems see 12 independent low-risk shipments. Network analysis reveals they share the same freight forwarder, the same customs broker, overlapping phone numbers, and a single beneficial owner registered in a free-trade zone — a textbook undervaluation ring.

**Trade Intelligence Graph** applies graph theory, community detection, and anomaly propagation to expose these hidden relationships. It transforms flat declaration data into a rich, queryable network where fraud patterns become structurally visible.

---

## System Architecture

```
Trade Declarations ──► Entity Extraction ──► Graph Construction ──► Analysis Engine
                       (NER + Resolution)    (NetworkX/Neo4j)      (Community Detection
                                                                     Centrality Analysis
                                                                     Temporal Patterns
                                                                     Anomaly Propagation)
                                                                          │
                                              GraphQL API ◄──────────────┘
                                                   │
                                              Risk Signals ──► Sovereign Risk Platform
```

The pipeline operates in four stages:

1. **Entity Extraction** — Declaration records are parsed to extract distinct entities (importers, exporters, agents, commodities, routes) and resolve duplicates using fuzzy matching and identifier normalization.

2. **Graph Construction** — Extracted entities become nodes; their relationships (trade links, shared attributes, co-occurrence patterns) become directed, weighted edges in a NetworkX `MultiDiGraph`.

3. **Analysis Engine** — Community detection algorithms identify clusters; centrality measures find key facilitators; temporal analysis tracks structural evolution; anomaly propagation spreads risk through the network.

4. **API & Signals** — A GraphQL API exposes the graph for interactive exploration. Risk signals are exported to downstream systems (e.g., the Sovereign Risk Platform) for operational decision-making.

---

## Key Capabilities

### 1. Entity Graph Construction
Build trade entity graphs from customs declaration data. Each declaration generates multiple nodes (importer, exporter, commodity, route, agent) and edges representing their relationships. Edge weights encode trade volume, frequency, and declared value.

### 2. Community Detection
Apply **Louvain** and **Leiden** algorithms to identify tightly connected trading clusters. Communities that exhibit anomalous internal patterns (uniform declared values, synchronized filing dates, shared attributes) are flagged for review.

### 3. Centrality Analysis
Compute **PageRank**, **betweenness centrality**, and **degree centrality** to identify key facilitators — entities that connect otherwise separate trading clusters. High-betweenness, low-volume entities are strong shell company indicators.

### 4. Temporal Graph Analysis
Maintain time-windowed graph snapshots and detect structural changes: new edges appearing between previously unconnected communities, nodes disappearing after enforcement actions, community splits and merges that signal network adaptation.

### 5. Fraud Ring Detection
Identify coordinated patterns using graph motifs: circular trade paths (A -> B -> C -> A), shared-attribute clusters (common addresses, phone numbers, bank accounts), and synchronized behavioral patterns.

### 6. Anomaly Propagation
When one node is flagged as high-risk, propagate attenuated risk scores through connected entities using configurable decay functions. A flagged exporter raises the risk score of all connected importers, weighted by trade volume and relationship strength.

### 7. GraphQL API
Flexible querying for complex relationship traversals. Query node details, neighbors at arbitrary depth, shortest paths between entities, community memberships, and high-risk subgraphs — all through a single endpoint.

### 8. Neo4j Integration
Optional persistent graph database backend for large-scale deployments (10M+ nodes). The system defaults to NetworkX for simplicity and switches to Neo4j when horizontal scaling is required.

---

## Graph Schema

### Entity Types (Nodes)

| Node Type        | Key Attributes                                    | Description                          |
|------------------|---------------------------------------------------|--------------------------------------|
| `IMPORTER`       | tax_id, name, address, country, registration_date | Importing entity                     |
| `EXPORTER`       | tax_id, name, address, country                    | Exporting entity                     |
| `COMMODITY`      | hs_code, description, unit_price_range            | Traded commodity (HS code level)     |
| `ROUTE`          | origin, destination, transit_points               | Trade route                          |
| `AGENT`          | license_id, name, type (broker/forwarder)         | Customs broker or freight forwarder  |
| `ADDRESS`        | normalized_address, country, postal_code          | Physical address (for co-location)   |
| `PHONE`          | normalized_number                                 | Phone number (for shared-contact)    |
| `BANK_ACCOUNT`   | bank_name, account_hash                           | Financial account (for payment links)|
| `DECLARATION`    | declaration_id, date, value, weight               | Individual customs declaration       |

### Relationship Types (Edges)

| Edge Type           | Source → Target           | Weight Basis            | Description                        |
|----------------------|---------------------------|-------------------------|------------------------------------|
| `IMPORTS_FROM`       | Importer → Exporter       | Trade volume/frequency  | Direct trade relationship          |
| `DECLARES`           | Importer → Declaration    | -                       | Declaration ownership              |
| `TRADES_COMMODITY`   | Declaration → Commodity   | Declared quantity       | Commodity in declaration           |
| `USES_ROUTE`         | Declaration → Route       | -                       | Shipping route used                |
| `REPRESENTED_BY`     | Importer → Agent          | Transaction count       | Agent representation               |
| `LOCATED_AT`         | Entity → Address          | -                       | Physical co-location               |
| `CONTACTABLE_VIA`    | Entity → Phone            | -                       | Shared contact number              |
| `PAYS_THROUGH`       | Entity → Bank Account     | Transaction value       | Financial relationship             |
| `CO_OCCURS_WITH`     | Entity → Entity           | Co-occurrence frequency | Entities appearing together        |

---

## Fraud Detection Scenarios

### Scenario 1: Undervaluation Ring via Shared Agent

A customs broker files declarations for 8 different importers, all declaring the same HS code (8471.30 — portable computers) from 3 different exporters. Each declaration values the goods at $120/unit — 60% below the statistical average of $300/unit.

**Graph signal:** A single `AGENT` node with high betweenness centrality connects a tight community of `IMPORTER` nodes. All `DECLARATION` nodes within this community share anomalously low `unit_value` attributes. The community's internal value distribution has near-zero variance — a strong indicator of coordinated undervaluation.

**Detection method:** Community detection + intra-community statistical analysis.

### Scenario 2: Carousel Fraud via Circular Trade

Company A in Country X exports goods to Company B in Country Y. Company B sells to Company C (also in Country Y), which then re-exports to Company D in Country X. Company D is registered at the same address as Company A. The goods have made a full circle, collecting VAT refunds at each border crossing.

**Graph signal:** A directed cycle `A → B → C → D → A` in the trade graph, with `A` and `D` sharing an `ADDRESS` node. The cycle involves cross-border edges where VAT recovery is possible.

**Detection method:** Cycle detection + shared-attribute analysis + cross-border edge filtering.

### Scenario 3: Shell Company Network for Origin Laundering

Goods manufactured in Country Z (subject to anti-dumping duties) are shipped to a free-trade zone where a shell company re-labels them as originating from Country W (duty-free). The shell company has minimal trade volume but connects to 15 different importers in the destination country.

**Graph signal:** A node with high degree centrality (many connections) but anomalously low trade volume and a registration address in a known free-trade zone. All connected importers trade the same HS code that is subject to anti-dumping measures from Country Z.

**Detection method:** Centrality-volume divergence analysis + origin verification.

---

## Performance

| Metric                     | Value                        | Configuration                   |
|----------------------------|------------------------------|---------------------------------|
| Graph construction         | ~50,000 declarations/sec     | NetworkX, single thread         |
| Community detection        | <2s for 100K nodes           | Louvain, resolution=1.0        |
| PageRank computation       | <1s for 100K nodes           | NetworkX, 100 iterations        |
| Cycle detection (depth=5)  | <5s for 100K nodes           | DFS-based, bounded depth        |
| Risk propagation           | <3s for 100K nodes           | BFS, max_depth=4                |
| GraphQL query (neighbors)  | <50ms p95                    | In-memory graph                 |
| GraphQL query (shortest)   | <200ms p95                   | Dijkstra, weighted              |
| Neo4j (large-scale)        | 10M+ nodes sustained         | Neo4j 5.x, 32GB heap           |

---

## Integration

### Sovereign Risk Platform

Trade Intelligence Graph integrates with the [Sovereign Risk Platform](https://github.com/ShahinHasanov90/sovereign-risk-platform) as a specialized intelligence source:

```python
from graph_intel.export.signals import SignalExporter

exporter = SignalExporter(
    endpoint="https://risk-platform.internal/api/v1/signals",
    api_key="...",
    source_system="trade-intelligence-graph"
)

# Export high-risk communities as risk signals
communities = analyzer.detect_communities(min_risk_score=0.7)
for community in communities:
    exporter.emit_community_signal(community)
```

Risk signals include:
- **Entity risk scores** with graph-derived evidence (centrality, community membership, propagation path)
- **Community alerts** when new fraud-pattern communities are detected
- **Network evolution alerts** when structural changes indicate network adaptation

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/ShahinHasanov90/trade-intelligence-graph.git
cd trade-intelligence-graph
make install

# Run tests
make test

# Build graph from sample data
make build-graph

# Start GraphQL API
make serve
# → GraphQL playground at http://localhost:8000/graphql

# With Neo4j (optional)
docker-compose up -d
```

---

## Usage Example

```python
from graph_intel.graph.builder import TradeGraphBuilder
from graph_intel.analysis.community import CommunityDetector
from graph_intel.analysis.centrality import CentralityAnalyzer
from graph_intel.detection.rings import FraudRingDetector

# Build the trade graph
builder = TradeGraphBuilder()
builder.add_declaration({
    "declaration_id": "DEC-2025-001",
    "importer": {"tax_id": "IMP001", "name": "Acme Imports Ltd"},
    "exporter": {"tax_id": "EXP001", "name": "Global Exports Co"},
    "commodity": {"hs_code": "8471.30", "description": "Portable computers"},
    "route": {"origin": "CN", "destination": "US", "transit": ["HK"]},
    "agent": {"license_id": "BRK-100", "name": "FastClear Customs"},
    "value": 120000.00,
    "weight": 1000.0,
    "date": "2025-01-15"
})

graph = builder.build()

# Detect communities
detector = CommunityDetector(graph)
communities = detector.detect_louvain(resolution=1.0)

# Analyze centrality
analyzer = CentralityAnalyzer(graph)
pagerank = analyzer.compute_pagerank()
facilitators = analyzer.find_facilitators(top_k=10)

# Detect fraud rings
ring_detector = FraudRingDetector(graph)
cycles = ring_detector.find_circular_trade(max_depth=5)
shared_attr = ring_detector.find_shared_attribute_clusters()
```

---

## Evolution Roadmap

| Phase | Capability | Description |
|-------|-----------|-------------|
| **v1.0** | Core Graph Analytics | NetworkX-based graph construction, community detection, centrality analysis, fraud ring detection |
| **v1.5** | Temporal Analysis | Time-windowed snapshots, structural change detection, network evolution tracking |
| **v2.0** | Graph Neural Networks | GNN-based link prediction for anticipating new fraud connections; node classification for entity risk scoring using GraphSAGE/GAT architectures |
| **v2.5** | Federated Graph Sharing | Privacy-preserving graph analytics across customs authorities using federated learning; share structural patterns without exposing entity-level data |
| **v3.0** | Real-time Streaming | Apache Kafka/Flink integration for real-time graph updates; streaming community detection; live anomaly propagation as declarations arrive |
| **v3.5** | Explainable AI | Graph attention visualization; natural language explanations of why a community was flagged; evidence chains for audit trails |

---

## Development

```bash
make install     # Install dependencies
make test        # Run test suite
make lint        # Run linters (ruff, mypy)
make format      # Auto-format code (black, isort)
make clean       # Clean build artifacts
make serve       # Start GraphQL API server
make build-graph # Build graph from sample data
```

---

## License

Copyright 2025 Shahin Hasanov

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
