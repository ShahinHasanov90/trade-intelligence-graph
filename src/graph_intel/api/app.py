"""FastAPI application with Strawberry GraphQL integration.

Provides the HTTP server that exposes the Trade Intelligence Graph
through a GraphQL endpoint. Includes health checks, CORS configuration,
and optional Neo4j backend support.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import strawberry
import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from graph_intel.api.graphql_schema import (
    CommunityInfo,
    FraudRingInfo,
    GraphStatistics,
    NodeInfo,
    PathInfo,
    PropagationResultInfo,
    RiskEntityInfo,
)
from graph_intel.api.resolvers import GraphResolver
from graph_intel.config import Settings, get_settings
from graph_intel.graph.builder import TradeGraphBuilder
from graph_intel.graph.store import NetworkXStore

logger = structlog.get_logger(__name__)

# Module-level resolver reference (initialized in lifespan)
_resolver: Optional[GraphResolver] = None


def _build_sample_graph() -> NetworkXStore:
    """Build a sample graph for demonstration purposes.

    Returns:
        NetworkXStore populated with sample trade data.
    """
    builder = TradeGraphBuilder()

    # Create a realistic sample dataset
    sample_declarations = [
        {
            "declaration_id": f"DEC-2025-{i:04d}",
            "importer": {
                "tax_id": f"IMP{(i % 8):03d}",
                "name": f"Import Corp {i % 8}",
                "country": ["US", "GB", "DE", "FR"][i % 4],
                "address": ["123 Trade St", "456 Commerce Ave", "789 Import Blvd"][
                    i % 3
                ],
            },
            "exporter": {
                "tax_id": f"EXP{(i % 5):03d}",
                "name": f"Export Global {i % 5}",
                "country": ["CN", "VN", "IN"][i % 3],
            },
            "commodity": {
                "hs_code": ["8471.30", "6403.99", "8517.12"][i % 3],
                "description": ["Portable computers", "Footwear", "Smartphones"][
                    i % 3
                ],
            },
            "route": {
                "origin": ["CN", "VN", "IN"][i % 3],
                "destination": ["US", "GB", "DE", "FR"][i % 4],
                "transit": [["HK"], ["SG"], []][i % 3],
            },
            "agent": {
                "license_id": f"BRK-{i % 3:03d}",
                "name": f"Customs Broker {i % 3}",
            },
            "value": 50000 + (i * 2500),
            "weight": 200 + (i * 15),
            "date": f"2025-{((i % 12) + 1):02d}-{((i % 28) + 1):02d}",
        }
        for i in range(50)
    ]

    builder.add_declarations(sample_declarations)
    graph = builder.build()

    store = NetworkXStore(graph)
    return store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for initialization and cleanup.

    Initializes the graph store and resolver on startup, and
    performs cleanup on shutdown.
    """
    global _resolver

    logger.info("application_starting")

    # Initialize graph store
    settings = get_settings()
    store = _build_sample_graph()
    _resolver = GraphResolver(store)

    logger.info(
        "graph_loaded",
        nodes=store.node_count(),
        edges=store.edge_count(),
    )

    yield

    logger.info("application_shutting_down")


# Define the Strawberry GraphQL schema
@strawberry.type
class Query:
    """Root GraphQL query type for the Trade Intelligence Graph API."""

    @strawberry.field(description="Get details for a specific entity node")
    def node(self, node_id: str) -> Optional[NodeInfo]:
        """Resolve a single node by ID."""
        if _resolver is None:
            return None
        return _resolver.get_node(node_id)

    @strawberry.field(description="Get neighbors of a node with optional filters")
    def neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "both",
        max_depth: int = 1,
    ) -> list[NodeInfo]:
        """Resolve neighbors of a node."""
        if _resolver is None:
            return []
        return _resolver.get_neighbors(node_id, edge_type, direction, max_depth)

    @strawberry.field(description="Find the shortest path between two nodes")
    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        weighted: bool = False,
    ) -> PathInfo:
        """Find the shortest path between two nodes."""
        if _resolver is None:
            return PathInfo(
                source_id=source_id,
                target_id=target_id,
                path=[],
                length=0,
                exists=False,
            )
        return _resolver.get_shortest_path(source_id, target_id, weighted)

    @strawberry.field(description="Detect communities in the trade graph")
    def communities(
        self,
        algorithm: str = "louvain",
        resolution: float = 1.0,
        min_size: int = 3,
    ) -> list[CommunityInfo]:
        """Detect and return communities."""
        if _resolver is None:
            return []
        return _resolver.get_communities(algorithm, resolution, min_size)

    @strawberry.field(description="Get high-risk entities above a threshold")
    def high_risk_entities(
        self,
        min_risk_score: float = 0.5,
        limit: int = 50,
        node_type: Optional[str] = None,
    ) -> list[RiskEntityInfo]:
        """Get entities above a risk score threshold."""
        if _resolver is None:
            return []
        return _resolver.get_high_risk_entities(min_risk_score, limit, node_type)

    @strawberry.field(description="Get overall graph statistics")
    def statistics(self) -> Optional[GraphStatistics]:
        """Get graph statistics."""
        if _resolver is None:
            return None
        return _resolver.get_graph_statistics()

    @strawberry.field(description="Detect fraud rings in the graph")
    def fraud_rings(
        self,
        min_confidence: float = 0.5,
    ) -> list[FraudRingInfo]:
        """Detect fraud rings."""
        if _resolver is None:
            return []
        return _resolver.detect_fraud_rings(min_confidence)

    @strawberry.field(description="Propagate risk from a source node")
    def propagate_risk(
        self,
        source_node: str,
        risk_score: float = 0.8,
        max_depth: int = 4,
    ) -> Optional[PropagationResultInfo]:
        """Propagate risk from a source node."""
        if _resolver is None:
            return None
        try:
            return _resolver.propagate_risk(source_node, risk_score, max_depth)
        except ValueError:
            return None


schema = strawberry.Schema(query=Query)

# Create the FastAPI application
app = FastAPI(
    title="Trade Intelligence Graph API",
    description="GraphQL API for trade fraud ring detection and customs intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the GraphQL endpoint
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix=settings.api.graphql_path)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Dictionary with status and graph information.
    """
    if _resolver is None:
        return {"status": "starting", "graph": "not loaded"}

    return {
        "status": "healthy",
        "graph": "loaded",
    }


def main() -> None:
    """Entry point for running the API server."""
    port = int(os.environ.get("PORT", settings.api.port))
    host = os.environ.get("HOST", settings.api.host)

    uvicorn.run(
        "graph_intel.api.app:app",
        host=host,
        port=port,
        reload=True,
    )


if __name__ == "__main__":
    main()
