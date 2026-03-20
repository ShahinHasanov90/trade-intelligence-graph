"""Temporal graph analysis for tracking network evolution.

Maintains time-windowed snapshots of the trade graph and detects
structural changes that may indicate network adaptation — such as
fraud rings restructuring after enforcement actions, new intermediaries
appearing, or communication channels shifting.

Key capabilities:
- Graph snapshot management with configurable time windows
- Structural change detection (new edges, disappearing nodes)
- Community evolution tracking (splits, merges, growth, contraction)
- Velocity metrics (rate of change in graph structure)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import networkx as nx
import numpy as np
import structlog

from graph_intel.config import Settings, TemporalConfig, get_settings

logger = structlog.get_logger(__name__)


@dataclass
class GraphSnapshot:
    """A point-in-time snapshot of the trade graph.

    Attributes:
        timestamp: The time this snapshot was taken.
        window_start: Start of the time window for this snapshot.
        window_end: End of the time window for this snapshot.
        node_count: Number of nodes in this snapshot.
        edge_count: Number of edges in this snapshot.
        node_ids: Set of all node IDs present.
        edge_set: Set of (source, target, edge_type) tuples.
        node_type_counts: Distribution of node types.
        density: Graph density at this point in time.
        components: Number of weakly connected components.
        graph: The actual NetworkX graph for this snapshot.
    """

    timestamp: str
    window_start: str
    window_end: str
    node_count: int = 0
    edge_count: int = 0
    node_ids: set[str] = field(default_factory=set)
    edge_set: set[tuple[str, str, str]] = field(default_factory=set)
    node_type_counts: dict[str, int] = field(default_factory=dict)
    density: float = 0.0
    components: int = 0
    graph: Optional[nx.MultiDiGraph] = field(default=None, repr=False)


@dataclass
class StructuralChange:
    """Represents a detected structural change between two snapshots.

    Attributes:
        change_type: Category of change (e.g., 'new_edges', 'disappeared_nodes').
        severity: Severity score (0.0 to 1.0).
        timestamp: When the change was detected.
        details: Specifics of the change.
        affected_entities: Node IDs affected by this change.
    """

    change_type: str
    severity: float
    timestamp: str
    details: dict[str, Any] = field(default_factory=dict)
    affected_entities: list[str] = field(default_factory=list)


@dataclass
class CommunityEvolution:
    """Tracks how a community changes across snapshots.

    Attributes:
        community_id: Identifier for the tracked community.
        event_type: Type of evolution event (split, merge, grow, shrink, stable, new, dissolved).
        timestamp: When the event occurred.
        previous_size: Community size in the previous snapshot.
        current_size: Community size in the current snapshot.
        members_added: Nodes that joined the community.
        members_removed: Nodes that left the community.
        jaccard_similarity: Overlap with the previous snapshot's community.
    """

    community_id: int
    event_type: str
    timestamp: str
    previous_size: int = 0
    current_size: int = 0
    members_added: set[str] = field(default_factory=set)
    members_removed: set[str] = field(default_factory=set)
    jaccard_similarity: float = 0.0


class TemporalAnalyzer:
    """Analyzes temporal evolution of trade entity graphs.

    Manages a series of time-windowed graph snapshots and provides
    methods for detecting structural changes, tracking community
    evolution, and computing velocity metrics.

    Args:
        graph: The full trade entity graph.
        config: Optional temporal analysis configuration.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        config: Optional[TemporalConfig] = None,
    ) -> None:
        """Initialize the temporal analyzer.

        Args:
            graph: The trade entity graph (NetworkX MultiDiGraph).
            config: Optional temporal analysis configuration.
        """
        self._graph = graph
        self._config = config or get_settings().analysis.temporal
        self._snapshots: list[GraphSnapshot] = []

    @property
    def snapshots(self) -> list[GraphSnapshot]:
        """Return the list of stored snapshots."""
        return self._snapshots

    def create_snapshot(
        self,
        window_start: str,
        window_end: str,
        date_field: str = "date",
    ) -> GraphSnapshot:
        """Create a graph snapshot for a specific time window.

        Extracts a subgraph containing only edges whose date attribute
        falls within the specified window, along with their incident nodes.

        Args:
            window_start: Start date (ISO format) for the window.
            window_end: End date (ISO format) for the window.
            date_field: Edge attribute containing the date.

        Returns:
            GraphSnapshot capturing the graph state within the time window.
        """
        start_dt = datetime.fromisoformat(window_start)
        end_dt = datetime.fromisoformat(window_end)

        # Build subgraph for this time window
        subgraph = nx.MultiDiGraph()
        edge_set: set[tuple[str, str, str]] = set()

        for u, v, key, data in self._graph.edges(keys=True, data=True):
            edge_date_str = data.get(date_field) or data.get("last_seen", "")
            if not edge_date_str:
                continue

            try:
                edge_date = datetime.fromisoformat(edge_date_str)
            except (ValueError, TypeError):
                continue

            if start_dt <= edge_date <= end_dt:
                # Add nodes with their attributes
                if u not in subgraph:
                    subgraph.add_node(u, **dict(self._graph.nodes[u]))
                if v not in subgraph:
                    subgraph.add_node(v, **dict(self._graph.nodes[v]))

                subgraph.add_edge(u, v, key=key, **data)
                edge_type = data.get("edge_type", key)
                edge_set.add((u, v, str(edge_type)))

        # Compute snapshot metrics
        node_ids = set(subgraph.nodes())
        node_type_counts: dict[str, int] = defaultdict(int)
        for _, data in subgraph.nodes(data=True):
            nt = data.get("node_type", "UNKNOWN")
            node_type_counts[nt] += 1

        density = nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0.0
        components = (
            nx.number_weakly_connected_components(subgraph)
            if subgraph.number_of_nodes() > 0
            else 0
        )

        snapshot = GraphSnapshot(
            timestamp=datetime.now().isoformat(),
            window_start=window_start,
            window_end=window_end,
            node_count=subgraph.number_of_nodes(),
            edge_count=subgraph.number_of_edges(),
            node_ids=node_ids,
            edge_set=edge_set,
            node_type_counts=dict(node_type_counts),
            density=density,
            components=components,
            graph=subgraph,
        )

        self._snapshots.append(snapshot)

        logger.info(
            "snapshot_created",
            window=f"{window_start} to {window_end}",
            nodes=snapshot.node_count,
            edges=snapshot.edge_count,
            total_snapshots=len(self._snapshots),
        )

        return snapshot

    def create_sliding_windows(
        self,
        start_date: str,
        end_date: str,
        window_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> list[GraphSnapshot]:
        """Create a series of overlapping snapshots across a date range.

        Args:
            start_date: Start of the overall period (ISO format).
            end_date: End of the overall period (ISO format).
            window_size: Window size in days. Uses config default if None.
            overlap: Overlap between windows in days. Uses config default if None.

        Returns:
            List of GraphSnapshot objects for each window.
        """
        window_size = window_size or self._config.window_size
        overlap = overlap or self._config.window_overlap
        step = window_size - overlap

        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        snapshots = []
        current = start

        while current + timedelta(days=window_size) <= end + timedelta(days=1):
            window_end = current + timedelta(days=window_size - 1)
            snapshot = self.create_snapshot(
                window_start=current.strftime("%Y-%m-%d"),
                window_end=window_end.strftime("%Y-%m-%d"),
            )
            snapshots.append(snapshot)
            current += timedelta(days=step)

        logger.info(
            "sliding_windows_created",
            total_windows=len(snapshots),
            window_size=window_size,
            overlap=overlap,
        )

        return snapshots

    def detect_structural_changes(
        self,
        snapshot_a: Optional[GraphSnapshot] = None,
        snapshot_b: Optional[GraphSnapshot] = None,
        threshold: Optional[float] = None,
    ) -> list[StructuralChange]:
        """Detect structural changes between two consecutive snapshots.

        If no snapshots are provided, compares the two most recent snapshots.

        Detects:
        - New edges (relationships that didn't exist before)
        - Disappeared nodes (entities that were present but are now absent)
        - New nodes (entities appearing for the first time)
        - Density changes (significant changes in graph connectivity)
        - Component changes (network fragmentation or consolidation)

        Args:
            snapshot_a: Earlier snapshot. Uses second-to-last if None.
            snapshot_b: Later snapshot. Uses most recent if None.
            threshold: Minimum severity for reporting. Uses config if None.

        Returns:
            List of StructuralChange objects, sorted by severity.
        """
        if snapshot_a is None or snapshot_b is None:
            if len(self._snapshots) < 2:
                logger.warning("insufficient_snapshots", count=len(self._snapshots))
                return []
            snapshot_a = self._snapshots[-2]
            snapshot_b = self._snapshots[-1]

        threshold = threshold or self._config.structural_change_threshold
        changes: list[StructuralChange] = []

        # Detect new edges
        new_edges = snapshot_b.edge_set - snapshot_a.edge_set
        if new_edges:
            severity = min(len(new_edges) / max(len(snapshot_a.edge_set), 1), 1.0)
            if severity >= threshold:
                affected = set()
                for src, tgt, _ in new_edges:
                    affected.add(src)
                    affected.add(tgt)
                changes.append(
                    StructuralChange(
                        change_type="new_edges",
                        severity=severity,
                        timestamp=snapshot_b.timestamp,
                        details={
                            "count": len(new_edges),
                            "edges": [
                                {"source": s, "target": t, "type": e}
                                for s, t, e in list(new_edges)[:20]
                            ],
                        },
                        affected_entities=list(affected),
                    )
                )

        # Detect disappeared edges
        disappeared_edges = snapshot_a.edge_set - snapshot_b.edge_set
        if disappeared_edges:
            severity = min(
                len(disappeared_edges) / max(len(snapshot_a.edge_set), 1), 1.0
            )
            if severity >= threshold:
                affected = set()
                for src, tgt, _ in disappeared_edges:
                    affected.add(src)
                    affected.add(tgt)
                changes.append(
                    StructuralChange(
                        change_type="disappeared_edges",
                        severity=severity,
                        timestamp=snapshot_b.timestamp,
                        details={"count": len(disappeared_edges)},
                        affected_entities=list(affected),
                    )
                )

        # Detect new nodes
        new_nodes = snapshot_b.node_ids - snapshot_a.node_ids
        if new_nodes:
            severity = min(len(new_nodes) / max(len(snapshot_a.node_ids), 1), 1.0)
            if severity >= threshold:
                changes.append(
                    StructuralChange(
                        change_type="new_nodes",
                        severity=severity,
                        timestamp=snapshot_b.timestamp,
                        details={"count": len(new_nodes)},
                        affected_entities=list(new_nodes),
                    )
                )

        # Detect disappeared nodes
        disappeared_nodes = snapshot_a.node_ids - snapshot_b.node_ids
        if disappeared_nodes:
            severity = min(
                len(disappeared_nodes) / max(len(snapshot_a.node_ids), 1), 1.0
            )
            if severity >= threshold:
                changes.append(
                    StructuralChange(
                        change_type="disappeared_nodes",
                        severity=severity,
                        timestamp=snapshot_b.timestamp,
                        details={"count": len(disappeared_nodes)},
                        affected_entities=list(disappeared_nodes),
                    )
                )

        # Detect density change
        density_change = abs(snapshot_b.density - snapshot_a.density)
        if density_change > 0:
            severity = min(
                density_change / max(snapshot_a.density, 0.01), 1.0
            )
            if severity >= threshold:
                changes.append(
                    StructuralChange(
                        change_type="density_change",
                        severity=severity,
                        timestamp=snapshot_b.timestamp,
                        details={
                            "previous_density": snapshot_a.density,
                            "current_density": snapshot_b.density,
                            "change": density_change,
                        },
                    )
                )

        # Detect component count change
        comp_change = abs(snapshot_b.components - snapshot_a.components)
        if comp_change > 0:
            severity = min(
                comp_change / max(snapshot_a.components, 1), 1.0
            )
            if severity >= threshold:
                direction = (
                    "fragmentation"
                    if snapshot_b.components > snapshot_a.components
                    else "consolidation"
                )
                changes.append(
                    StructuralChange(
                        change_type=f"component_{direction}",
                        severity=severity,
                        timestamp=snapshot_b.timestamp,
                        details={
                            "previous_components": snapshot_a.components,
                            "current_components": snapshot_b.components,
                            "direction": direction,
                        },
                    )
                )

        changes.sort(key=lambda c: c.severity, reverse=True)

        logger.info(
            "structural_changes_detected",
            total_changes=len(changes),
            max_severity=changes[0].severity if changes else 0.0,
        )

        return changes

    def track_community_evolution(
        self,
        previous_partition: dict[str, int],
        current_partition: dict[str, int],
        timestamp: Optional[str] = None,
    ) -> list[CommunityEvolution]:
        """Track how communities evolve between two time periods.

        Matches communities across time periods using Jaccard similarity
        and classifies their evolution as: stable, grow, shrink, split,
        merge, new, or dissolved.

        Args:
            previous_partition: Node-to-community mapping from earlier period.
            current_partition: Node-to-community mapping from current period.
            timestamp: Optional timestamp for the evolution event.

        Returns:
            List of CommunityEvolution events.
        """
        timestamp = timestamp or datetime.now().isoformat()

        # Build community member sets
        prev_communities: dict[int, set[str]] = defaultdict(set)
        for node_id, comm_id in previous_partition.items():
            prev_communities[comm_id].add(node_id)

        curr_communities: dict[int, set[str]] = defaultdict(set)
        for node_id, comm_id in current_partition.items():
            curr_communities[comm_id].add(node_id)

        events: list[CommunityEvolution] = []

        # Match current communities to previous ones using Jaccard similarity
        matched_prev: set[int] = set()
        matched_curr: set[int] = set()

        for curr_id, curr_members in curr_communities.items():
            best_jaccard = 0.0
            best_prev_id: Optional[int] = None

            for prev_id, prev_members in prev_communities.items():
                intersection = len(curr_members & prev_members)
                union = len(curr_members | prev_members)
                jaccard = intersection / union if union > 0 else 0.0

                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_prev_id = prev_id

            if best_prev_id is not None and best_jaccard > 0.3:
                # Matched — determine evolution type
                prev_members = prev_communities[best_prev_id]
                members_added = curr_members - prev_members
                members_removed = prev_members - curr_members

                size_ratio = len(curr_members) / max(len(prev_members), 1)

                if best_jaccard > 0.8:
                    event_type = "stable"
                elif size_ratio > 1.3:
                    event_type = "grow"
                elif size_ratio < 0.7:
                    event_type = "shrink"
                else:
                    event_type = "evolve"

                events.append(
                    CommunityEvolution(
                        community_id=curr_id,
                        event_type=event_type,
                        timestamp=timestamp,
                        previous_size=len(prev_members),
                        current_size=len(curr_members),
                        members_added=members_added,
                        members_removed=members_removed,
                        jaccard_similarity=best_jaccard,
                    )
                )

                matched_prev.add(best_prev_id)
                matched_curr.add(curr_id)

        # New communities (no match in previous)
        for curr_id in set(curr_communities.keys()) - matched_curr:
            events.append(
                CommunityEvolution(
                    community_id=curr_id,
                    event_type="new",
                    timestamp=timestamp,
                    current_size=len(curr_communities[curr_id]),
                    members_added=curr_communities[curr_id],
                )
            )

        # Dissolved communities (no match in current)
        for prev_id in set(prev_communities.keys()) - matched_prev:
            events.append(
                CommunityEvolution(
                    community_id=prev_id,
                    event_type="dissolved",
                    timestamp=timestamp,
                    previous_size=len(prev_communities[prev_id]),
                    members_removed=prev_communities[prev_id],
                )
            )

        logger.info(
            "community_evolution_tracked",
            total_events=len(events),
            new=sum(1 for e in events if e.event_type == "new"),
            dissolved=sum(1 for e in events if e.event_type == "dissolved"),
            stable=sum(1 for e in events if e.event_type == "stable"),
        )

        return events

    def compute_velocity_metrics(self) -> dict[str, Any]:
        """Compute rate-of-change metrics across all stored snapshots.

        Returns:
            Dictionary containing:
            - node_growth_rate: Average rate of node count change per window
            - edge_growth_rate: Average rate of edge count change per window
            - density_trend: Linear trend in graph density
            - turnover_rate: Average fraction of nodes that change between windows
        """
        if len(self._snapshots) < 2:
            return {
                "node_growth_rate": 0.0,
                "edge_growth_rate": 0.0,
                "density_trend": 0.0,
                "turnover_rate": 0.0,
                "snapshot_count": len(self._snapshots),
            }

        node_counts = [s.node_count for s in self._snapshots]
        edge_counts = [s.edge_count for s in self._snapshots]
        densities = [s.density for s in self._snapshots]

        # Growth rates (per window)
        node_deltas = [
            node_counts[i] - node_counts[i - 1] for i in range(1, len(node_counts))
        ]
        edge_deltas = [
            edge_counts[i] - edge_counts[i - 1] for i in range(1, len(edge_counts))
        ]

        # Turnover: fraction of nodes that appear or disappear between consecutive snapshots
        turnovers = []
        for i in range(1, len(self._snapshots)):
            prev_nodes = self._snapshots[i - 1].node_ids
            curr_nodes = self._snapshots[i].node_ids
            if prev_nodes or curr_nodes:
                changed = len(prev_nodes.symmetric_difference(curr_nodes))
                total = len(prev_nodes.union(curr_nodes))
                turnovers.append(changed / total if total > 0 else 0.0)

        # Linear trend for density
        if len(densities) >= 2:
            x = np.arange(len(densities))
            coeffs = np.polyfit(x, densities, 1)
            density_trend = float(coeffs[0])
        else:
            density_trend = 0.0

        return {
            "node_growth_rate": float(np.mean(node_deltas)) if node_deltas else 0.0,
            "edge_growth_rate": float(np.mean(edge_deltas)) if edge_deltas else 0.0,
            "density_trend": density_trend,
            "turnover_rate": float(np.mean(turnovers)) if turnovers else 0.0,
            "snapshot_count": len(self._snapshots),
            "node_count_series": node_counts,
            "edge_count_series": edge_counts,
        }
