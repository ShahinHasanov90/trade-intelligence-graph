"""Risk signal export to external systems.

Transforms graph analysis results (risk scores, community detections,
fraud ring findings) into standardized risk signals and exports them
to the Sovereign Risk Platform or other downstream consumers.

Signal types:
- Entity risk signals (individual entity risk assessments)
- Community alerts (anomalous community detections)
- Network evolution alerts (structural change notifications)
- Fraud ring alerts (coordinated fraud pattern detections)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

import structlog

from graph_intel.analysis.community import Community, CommunityResult
from graph_intel.analysis.propagation import PropagationResult
from graph_intel.config import Settings, SignalExportConfig, get_settings
from graph_intel.detection.rings import FraudRing

logger = structlog.get_logger(__name__)


@dataclass
class RiskSignal:
    """A standardized risk signal for export.

    Attributes:
        signal_id: Unique identifier for this signal.
        signal_type: Category of signal (entity_risk, community_alert, etc.).
        source_system: Originating system identifier.
        timestamp: ISO timestamp of signal generation.
        severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL).
        confidence: Detection confidence (0.0 to 1.0).
        entity_ids: Entity identifiers related to this signal.
        payload: Signal-specific data payload.
        metadata: Additional context.
    """

    signal_id: str
    signal_type: str
    source_system: str
    timestamp: str
    severity: str
    confidence: float
    entity_ids: list[str]
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert signal to dictionary for serialization.

        Returns:
            Dictionary representation of the signal.
        """
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "source_system": self.source_system,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "confidence": self.confidence,
            "entity_ids": self.entity_ids,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize signal to JSON string.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), default=str)


class SignalExporter:
    """Exports risk signals to external systems.

    Provides methods for converting analysis results to standardized
    risk signals and delivering them to configured endpoints.

    Args:
        config: Optional export configuration.
        source_system: Source system identifier.
    """

    def __init__(
        self,
        config: Optional[SignalExportConfig] = None,
        source_system: str = "trade-intelligence-graph",
    ) -> None:
        """Initialize the signal exporter.

        Args:
            config: Optional export configuration.
            source_system: Identifier for this system in signals.
        """
        self._config = config or get_settings().export.signals
        self._source_system = source_system
        self._signal_buffer: list[RiskSignal] = []
        self._export_count: int = 0

    @property
    def buffer_size(self) -> int:
        """Return the number of signals in the buffer."""
        return len(self._signal_buffer)

    def emit_entity_risk_signal(
        self,
        entity_id: str,
        risk_score: float,
        risk_level: str,
        evidence: list[dict[str, Any]],
        propagation_path: Optional[list[str]] = None,
    ) -> RiskSignal:
        """Create and buffer an entity risk signal.

        Args:
            entity_id: The entity identifier.
            risk_score: The entity's risk score.
            risk_level: The entity's risk level.
            evidence: List of evidence items.
            propagation_path: Optional path through which risk was propagated.

        Returns:
            The created RiskSignal.
        """
        signal = RiskSignal(
            signal_id=self._generate_signal_id(),
            signal_type="entity_risk",
            source_system=self._source_system,
            timestamp=datetime.now().isoformat(),
            severity=risk_level,
            confidence=min(risk_score, 1.0),
            entity_ids=[entity_id],
            payload={
                "risk_score": risk_score,
                "risk_level": risk_level,
                "evidence": evidence,
                "propagation_path": propagation_path or [],
            },
        )

        self._signal_buffer.append(signal)
        self._auto_flush_if_needed()
        return signal

    def emit_community_signal(
        self,
        community: Community,
        anomaly_indicators: Optional[list[str]] = None,
    ) -> RiskSignal:
        """Create and buffer a community alert signal.

        Args:
            community: The detected community.
            anomaly_indicators: Optional list of anomaly indicators.

        Returns:
            The created RiskSignal.
        """
        severity = "LOW"
        if community.risk_score >= 0.8:
            severity = "CRITICAL"
        elif community.risk_score >= 0.6:
            severity = "HIGH"
        elif community.risk_score >= 0.3:
            severity = "MEDIUM"

        signal = RiskSignal(
            signal_id=self._generate_signal_id(),
            signal_type="community_alert",
            source_system=self._source_system,
            timestamp=datetime.now().isoformat(),
            severity=severity,
            confidence=community.risk_score,
            entity_ids=list(community.members),
            payload={
                "community_id": community.community_id,
                "size": community.size,
                "density": community.density,
                "internal_edges": community.internal_edges,
                "external_edges": community.external_edges,
                "node_types": community.node_types,
                "anomaly_indicators": anomaly_indicators or [],
                "risk_score": community.risk_score,
            },
            metadata=community.metadata,
        )

        self._signal_buffer.append(signal)
        self._auto_flush_if_needed()
        return signal

    def emit_fraud_ring_signal(self, ring: FraudRing) -> RiskSignal:
        """Create and buffer a fraud ring alert signal.

        Args:
            ring: The detected fraud ring.

        Returns:
            The created RiskSignal.
        """
        severity = "HIGH"
        if ring.confidence >= 0.8:
            severity = "CRITICAL"
        elif ring.confidence >= 0.5:
            severity = "HIGH"
        else:
            severity = "MEDIUM"

        signal = RiskSignal(
            signal_id=self._generate_signal_id(),
            signal_type="fraud_ring_alert",
            source_system=self._source_system,
            timestamp=datetime.now().isoformat(),
            severity=severity,
            confidence=ring.confidence,
            entity_ids=list(ring.members),
            payload={
                "ring_id": ring.ring_id,
                "ring_type": ring.ring_type,
                "size": ring.size,
                "total_value": ring.total_value,
                "pattern_description": ring.pattern_description,
                "evidence": ring.evidence,
            },
        )

        self._signal_buffer.append(signal)
        self._auto_flush_if_needed()
        return signal

    def emit_network_evolution_signal(
        self,
        change_type: str,
        severity_score: float,
        affected_entities: list[str],
        details: dict[str, Any],
    ) -> RiskSignal:
        """Create and buffer a network evolution alert signal.

        Args:
            change_type: Type of structural change detected.
            severity_score: Severity of the change (0.0 to 1.0).
            affected_entities: Entity IDs affected by the change.
            details: Change-specific details.

        Returns:
            The created RiskSignal.
        """
        severity = "LOW"
        if severity_score >= 0.8:
            severity = "CRITICAL"
        elif severity_score >= 0.5:
            severity = "HIGH"
        elif severity_score >= 0.3:
            severity = "MEDIUM"

        signal = RiskSignal(
            signal_id=self._generate_signal_id(),
            signal_type="network_evolution",
            source_system=self._source_system,
            timestamp=datetime.now().isoformat(),
            severity=severity,
            confidence=severity_score,
            entity_ids=affected_entities,
            payload={
                "change_type": change_type,
                "severity_score": severity_score,
                "details": details,
            },
        )

        self._signal_buffer.append(signal)
        self._auto_flush_if_needed()
        return signal

    def emit_propagation_result(
        self,
        result: PropagationResult,
        min_risk_threshold: float = 0.3,
    ) -> list[RiskSignal]:
        """Convert a propagation result into multiple entity risk signals.

        Args:
            result: The propagation result to convert.
            min_risk_threshold: Minimum risk to generate a signal.

        Returns:
            List of generated RiskSignal objects.
        """
        signals = []

        for entity_id, risk_score in result.affected_nodes.items():
            if risk_score < min_risk_threshold:
                continue

            risk_level = "LOW"
            if risk_score >= 0.8:
                risk_level = "CRITICAL"
            elif risk_score >= 0.6:
                risk_level = "HIGH"
            elif risk_score >= 0.3:
                risk_level = "MEDIUM"

            path = result.propagation_paths.get(entity_id, [])

            signal = self.emit_entity_risk_signal(
                entity_id=entity_id,
                risk_score=risk_score,
                risk_level=risk_level,
                evidence=[
                    {
                        "type": "risk_propagation",
                        "source_node": result.source_node,
                        "source_risk": result.source_risk,
                        "propagation_depth": len(path) - 1 if path else 0,
                    }
                ],
                propagation_path=path,
            )
            signals.append(signal)

        return signals

    def flush(self) -> list[RiskSignal]:
        """Flush the signal buffer, returning all buffered signals.

        In a production implementation, this would send signals to the
        configured endpoint. Here it returns them for processing.

        Returns:
            List of all buffered signals.
        """
        signals = list(self._signal_buffer)
        self._signal_buffer.clear()
        self._export_count += len(signals)

        logger.info(
            "signals_flushed",
            count=len(signals),
            total_exported=self._export_count,
        )

        return signals

    def get_signals_as_json(self) -> str:
        """Get all buffered signals as a JSON array.

        Returns:
            JSON string of all buffered signals.
        """
        return json.dumps(
            [s.to_dict() for s in self._signal_buffer],
            default=str,
            indent=2,
        )

    def _auto_flush_if_needed(self) -> None:
        """Automatically flush when buffer reaches configured batch size."""
        if len(self._signal_buffer) >= self._config.batch_size:
            logger.info(
                "auto_flush_triggered",
                buffer_size=len(self._signal_buffer),
                batch_size=self._config.batch_size,
            )
            self.flush()

    @staticmethod
    def _generate_signal_id() -> str:
        """Generate a unique signal identifier.

        Returns:
            UUID-based signal ID string.
        """
        return f"SIG-{uuid.uuid4().hex[:12].upper()}"
