"""Trade Intelligence Graph - Graph-based network analysis for trade fraud detection.

This package provides tools for constructing trade entity graphs from customs
declaration data and applying graph analytics to detect fraud rings, shell
companies, carousel fraud, and other coordinated trade fraud patterns.

Modules:
    graph: Graph construction, schema definitions, and persistence
    analysis: Community detection, centrality analysis, temporal tracking
    detection: Fraud ring, shell company, and carousel fraud detection
    api: GraphQL API for graph querying
    export: Risk signal export to external systems
"""

__version__ = "1.0.0"
__author__ = "Shahin Hasanov"
