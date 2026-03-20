# ADR 001: NetworkX as Default Graph Backend Over Neo4j

## Status

Accepted

## Date

2025-01-15

## Context

Trade Intelligence Graph requires a graph processing engine to construct, analyze, and query trade entity networks. The two primary options are:

1. **NetworkX** — Python-native in-memory graph library
2. **Neo4j** — Persistent graph database with Cypher query language

The system needs to support deployments ranging from small customs offices processing thousands of declarations per month to national-level agencies processing millions.

## Decision

Use **NetworkX** as the default graph backend for in-memory operations, with a **Neo4j adapter** available for persistent large-scale deployments.

The system implements a `GraphStore` abstraction that allows switching between backends via configuration without changing application code.

## Rationale

### Why NetworkX as default:

- **Zero infrastructure:** No database server required; reduces deployment complexity for small/medium installations
- **Fast iteration:** In-memory operations are significantly faster for graphs under 1M nodes
- **Python-native:** Full compatibility with NumPy/SciPy ecosystem; no serialization overhead
- **Algorithm availability:** NetworkX implements all required algorithms (community detection, centrality, shortest paths) natively
- **Testing simplicity:** Unit tests run without external dependencies

### Why Neo4j as optional:

- **Persistence:** Graph state survives process restarts
- **Scale:** Handles 10M+ nodes with disk-backed storage
- **Concurrent access:** Multiple API instances can query the same graph
- **Cypher:** Declarative query language for complex traversals
- **Visualization:** Built-in browser for graph exploration

### Trade-offs accepted:

- NetworkX graphs are lost on process restart (mitigated by rebuild-from-source capability)
- NetworkX is single-process (mitigated by adequate performance for target scale)
- Neo4j adds operational complexity (mitigated by making it optional)

## Consequences

- The `GraphStore` interface must abstract all graph operations
- All analysis algorithms must work with the `GraphStore` abstraction
- Neo4j-specific optimizations (e.g., native Cypher traversals) are available but not required
- Deployment documentation must cover both modes
- CI/CD tests run against NetworkX only; Neo4j integration tests are optional
