"""Risk score propagation through the trade entity graph.

When an entity is flagged as high-risk (e.g., by customs inspection,
intelligence tip, or anomaly detection), the risk signal should propagate
to connected entities. A flagged exporter raises the risk of all importers
that trade with it, weighted by trade volume and attenuated by graph distance.

The propagation model uses a BFS-based approach with configurable:
- Decay factor per hop (multiplicative attenuation)
- Edge-type-specific propagation weights
- Maximum propagation depth
- Minimum threshold to continue propagation
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx
import numpy as np
import structlog

from graph_intel.config import PropagationConfig, Settings, get_settings
from graph_intel.graph.schema import EdgeType, RiskLevel

logger = structlog.get_logger(__name__)


@dataclass
class PropagationResult:
    """Result of a risk propagation operation.

    Attributes:
        source_node: The node that initiated the propagation.
        source_risk: The risk score of the source node.
        affected_nodes: Mapping of affected node IDs to their new risk scores.
        propagation_paths: For each affected node, the path through which
            risk was propagated (list of node IDs).
        max_depth_reached: Maximum propagation depth actually reached.
        total_risk_distributed: Sum of all propagated risk deltas.
    """

    source_node: str
    source_risk: float
    affected_nodes: dict[str, float]
    propagation_paths: dict[str, list[str]] = field(default_factory=dict)
    max_depth_reached: int = 0
    total_risk_distributed: float = 0.0


@dataclass
class PropagationEvent:
    """A single propagation step for audit trail purposes.

    Attributes:
        from_node: The node propagating risk.
        to_node: The node receiving risk.
        edge_type: The relationship type used for propagation.
        risk_delta: The amount of risk propagated.
        depth: The depth (hop count) of this propagation step.
        decay_applied: The cumulative decay factor at this step.
    """

    from_node: str
    to_node: str
    edge_type: str
    risk_delta: float
    depth: int
    decay_applied: float


class RiskPropagator:
    """Propagates risk scores through the trade entity graph.

    Implements BFS-based risk propagation with configurable decay,
    edge-type weighting, and depth limits. Supports both single-source
    and multi-source propagation.

    Args:
        graph: The trade entity graph.
        config: Optional propagation configuration.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        config: Optional[PropagationConfig] = None,
    ) -> None:
        """Initialize the risk propagator.

        Args:
            graph: The trade entity graph (NetworkX MultiDiGraph).
            config: Optional propagation configuration.
        """
        self._graph = graph
        self._config = config or get_settings().analysis.propagation
        self._propagation_log: list[PropagationEvent] = []

    @property
    def propagation_log(self) -> list[PropagationEvent]:
        """Return the full propagation event log for audit purposes."""
        return self._propagation_log

    def propagate_from_node(
        self,
        source_node: str,
        risk_score: Optional[float] = None,
        max_depth: Optional[int] = None,
        decay_factor: Optional[float] = None,
        min_threshold: Optional[float] = None,
        update_graph: bool = True,
    ) -> PropagationResult:
        """Propagate risk from a single source node through the graph.

        Uses BFS traversal to spread risk through connected edges. At each
        hop, the propagated risk is attenuated by the decay factor and
        further modulated by the edge type weight.

        Risk at depth d through edge type e:
            risk_d = source_risk * decay^d * edge_type_weight[e] * edge_weight

        If a node receives risk from multiple paths, the maximum is taken
        (not summed) to avoid double-counting.

        Args:
            source_node: The node initiating risk propagation.
            risk_score: The risk score to propagate. If None, uses the
                node's current risk_score attribute.
            max_depth: Maximum number of hops. Uses config if None.
            decay_factor: Multiplicative decay per hop. Uses config if None.
            min_threshold: Minimum risk to continue. Uses config if None.
            update_graph: If True, updates node risk_score attributes in the graph.

        Returns:
            PropagationResult detailing all affected nodes and paths.

        Raises:
            ValueError: If source_node is not in the graph.
        """
        if source_node not in self._graph:
            raise ValueError(f"Source node not in graph: {source_node}")

        max_depth = max_depth if max_depth is not None else self._config.max_depth
        decay_factor = (
            decay_factor if decay_factor is not None else self._config.decay_factor
        )
        min_threshold = (
            min_threshold
            if min_threshold is not None
            else self._config.min_propagation_threshold
        )

        if risk_score is None:
            risk_score = self._graph.nodes[source_node].get("risk_score", 0.5)

        # BFS propagation
        affected: dict[str, float] = {}
        paths: dict[str, list[str]] = {}
        visited: set[str] = {source_node}
        max_depth_reached = 0

        # Queue entries: (current_node, current_risk, depth, path)
        queue: deque[tuple[str, float, int, list[str]]] = deque()
        queue.append((source_node, risk_score, 0, [source_node]))

        while queue:
            current_node, current_risk, depth, path = queue.popleft()

            if depth >= max_depth:
                continue

            # Propagate to all neighbors (both directions for undirected risk spread)
            neighbors = self._get_propagation_neighbors(current_node)

            for neighbor_id, edge_type, edge_weight in neighbors:
                if neighbor_id in visited:
                    continue

                # Compute propagated risk
                edge_type_weight = self._config.edge_type_weights.get(
                    edge_type, 0.3
                )
                propagated_risk = (
                    current_risk
                    * decay_factor
                    * edge_type_weight
                    * min(edge_weight, 1.0)
                )

                if propagated_risk < min_threshold:
                    continue

                # Take maximum if already affected via another path
                new_depth = depth + 1
                if neighbor_id not in affected or propagated_risk > affected[neighbor_id]:
                    affected[neighbor_id] = propagated_risk
                    paths[neighbor_id] = path + [neighbor_id]

                    # Log the propagation event
                    self._propagation_log.append(
                        PropagationEvent(
                            from_node=current_node,
                            to_node=neighbor_id,
                            edge_type=edge_type,
                            risk_delta=propagated_risk,
                            depth=new_depth,
                            decay_applied=decay_factor ** new_depth,
                        )
                    )

                max_depth_reached = max(max_depth_reached, new_depth)
                visited.add(neighbor_id)
                new_path = path + [neighbor_id]
                queue.append((neighbor_id, propagated_risk, new_depth, new_path))

        # Update graph node attributes if requested
        if update_graph:
            self._apply_risk_updates(source_node, risk_score, affected)

        total_distributed = sum(affected.values())

        result = PropagationResult(
            source_node=source_node,
            source_risk=risk_score,
            affected_nodes=affected,
            propagation_paths=paths,
            max_depth_reached=max_depth_reached,
            total_risk_distributed=total_distributed,
        )

        logger.info(
            "risk_propagated",
            source=source_node,
            source_risk=risk_score,
            affected_count=len(affected),
            max_depth=max_depth_reached,
            total_risk=round(total_distributed, 4),
        )

        return result

    def propagate_from_multiple(
        self,
        source_nodes: dict[str, float],
        max_depth: Optional[int] = None,
        update_graph: bool = True,
    ) -> list[PropagationResult]:
        """Propagate risk from multiple source nodes simultaneously.

        For nodes affected by multiple sources, the maximum risk score
        across all sources is retained.

        Args:
            source_nodes: Mapping of source node IDs to their risk scores.
            max_depth: Maximum propagation depth.
            update_graph: If True, updates graph attributes.

        Returns:
            List of PropagationResult objects, one per source.
        """
        results = []
        combined_risks: dict[str, float] = {}

        # First pass: propagate from each source without updating graph
        for source_id, risk_score in source_nodes.items():
            result = self.propagate_from_node(
                source_node=source_id,
                risk_score=risk_score,
                max_depth=max_depth,
                update_graph=False,
            )
            results.append(result)

            # Merge: take max risk for each affected node
            for node_id, score in result.affected_nodes.items():
                if node_id not in combined_risks or score > combined_risks[node_id]:
                    combined_risks[node_id] = score

        # Apply combined risk updates to graph
        if update_graph:
            for node_id, risk in combined_risks.items():
                if node_id in self._graph:
                    current = self._graph.nodes[node_id].get("risk_score", 0.0)
                    new_risk = max(current, risk)
                    self._graph.nodes[node_id]["risk_score"] = new_risk
                    self._graph.nodes[node_id]["risk_level"] = self._score_to_level(
                        new_risk
                    ).value

        logger.info(
            "multi_source_propagation_complete",
            sources=len(source_nodes),
            total_affected=len(combined_risks),
        )

        return results

    def reset_risk_scores(self) -> None:
        """Reset all node risk scores to zero.

        Useful for re-running propagation with different parameters.
        """
        for node_id in self._graph.nodes():
            self._graph.nodes[node_id]["risk_score"] = 0.0
            self._graph.nodes[node_id]["risk_level"] = RiskLevel.LOW.value
            self._graph.nodes[node_id]["flagged"] = False

        self._propagation_log.clear()
        logger.info("risk_scores_reset", nodes=self._graph.number_of_nodes())

    def get_risk_summary(self) -> dict[str, Any]:
        """Generate a summary of current risk distribution across the graph.

        Returns:
            Dictionary with risk distribution statistics, counts by level,
            and the highest-risk entities.
        """
        scores = []
        level_counts: dict[str, int] = defaultdict(int)
        flagged_count = 0

        for _, data in self._graph.nodes(data=True):
            score = data.get("risk_score", 0.0)
            scores.append(score)
            level = data.get("risk_level", "LOW")
            level_counts[level] += 1
            if data.get("flagged", False):
                flagged_count += 1

        if not scores:
            return {"total_nodes": 0}

        # Top risk entities
        risk_ranked = sorted(
            [
                (nid, data.get("risk_score", 0.0))
                for nid, data in self._graph.nodes(data=True)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "total_nodes": len(scores),
            "mean_risk": float(np.mean(scores)),
            "median_risk": float(np.median(scores)),
            "max_risk": float(np.max(scores)),
            "std_risk": float(np.std(scores)),
            "level_distribution": dict(level_counts),
            "flagged_count": flagged_count,
            "top_risk_entities": risk_ranked[:10],
            "nodes_above_threshold": sum(
                1 for s in scores if s >= self._config.min_propagation_threshold
            ),
        }

    def _get_propagation_neighbors(
        self, node_id: str
    ) -> list[tuple[str, str, float]]:
        """Get all neighbors eligible for risk propagation.

        Considers both outgoing and incoming edges, since risk can
        propagate in either direction (a risky importer affects its
        exporter's reputation, and vice versa).

        Args:
            node_id: The current node.

        Returns:
            List of (neighbor_id, edge_type, edge_weight) tuples.
        """
        neighbors = []

        # Outgoing edges
        for _, target, data in self._graph.out_edges(node_id, data=True):
            edge_type = data.get("edge_type", "UNKNOWN")
            edge_weight = data.get("weight", 1.0)
            neighbors.append((target, edge_type, edge_weight))

        # Incoming edges (risk propagates bidirectionally)
        for source, _, data in self._graph.in_edges(node_id, data=True):
            edge_type = data.get("edge_type", "UNKNOWN")
            edge_weight = data.get("weight", 1.0)
            neighbors.append((source, edge_type, edge_weight))

        return neighbors

    def _apply_risk_updates(
        self,
        source_node: str,
        source_risk: float,
        affected: dict[str, float],
    ) -> None:
        """Apply computed risk scores to graph node attributes.

        Uses max(current, propagated) to avoid reducing existing risk scores.

        Args:
            source_node: The source node (gets its risk set directly).
            source_risk: The source's risk score.
            affected: Mapping of affected nodes to propagated risk scores.
        """
        # Update source node
        self._graph.nodes[source_node]["risk_score"] = source_risk
        self._graph.nodes[source_node]["risk_level"] = self._score_to_level(
            source_risk
        ).value
        self._graph.nodes[source_node]["flagged"] = True

        # Update affected nodes
        for node_id, risk in affected.items():
            if node_id in self._graph:
                current = self._graph.nodes[node_id].get("risk_score", 0.0)
                new_risk = max(current, risk)
                self._graph.nodes[node_id]["risk_score"] = new_risk
                self._graph.nodes[node_id]["risk_level"] = self._score_to_level(
                    new_risk
                ).value

    @staticmethod
    def _score_to_level(score: float) -> RiskLevel:
        """Convert a numeric risk score to a categorical risk level.

        Args:
            score: Risk score between 0.0 and 1.0.

        Returns:
            Corresponding RiskLevel enum value.
        """
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
