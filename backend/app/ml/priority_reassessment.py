"""Dynamic Priority Reassessment Service (Phase 2.18)
Municipal Complaint Intelligence System

Manages lifecycle and dynamic priority re-scoring when civic events occur:
1. New nearby complaints (cluster density increases)
2. Community upvotes / citizen backing confirmations
3. Field inspection severity updates or citizen evidence
4. Status transitions (e.g. submitted -> in_progress -> resolved -> closed)
5. Duplicate linking (merges community backing into parent without double counting)

Strict safeguards:
- Reuses Phase 2.17 PriorityEngine
- Idempotent: tracks event IDs to avoid double-processing repeated events
- Never auto-merges or deletes complaints
- Provides full audit trail logging (previous score, new score, delta, reason)
- Thread-safe and compatible with SQLAlchemy Complaint & ComplaintAnalysis models
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

from app.ml.priority_engine import PriorityEngine, PriorityEngineConfig, PriorityResult

logger = logging.getLogger("DynamicPriorityReassessor")


class ReassessmentEventType(str, Enum):
    NEW_NEARBY_COMPLAINT = "NEW_NEARBY_COMPLAINT"
    COMMUNITY_UPVOTE = "COMMUNITY_UPVOTE"
    SEVERITY_UPDATED = "SEVERITY_UPDATED"
    STATUS_CHANGED = "STATUS_CHANGED"
    DUPLICATE_LINKED = "DUPLICATE_LINKED"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"


@dataclass
class ReassessmentEvent:
    """An event triggering priority re-evaluation."""
    event_type: ReassessmentEventType
    complaint_id: Union[int, str]
    event_id: Optional[str] = None  # Unique ID for idempotency deduplication
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_payload: dict = field(default_factory=dict)


@dataclass
class ReassessmentRecord:
    """Audit record capturing priority state transition."""
    complaint_id: Union[int, str]
    previous_score: float
    previous_level: str
    new_score: float
    new_level: str
    score_delta: float
    triggering_event: str
    recalculated_at: datetime
    is_changed: bool
    explanation: str
    component_details: dict

    def to_dict(self) -> dict:
        return {
            "complaint_id": self.complaint_id,
            "previous_score": round(self.previous_score, 1),
            "previous_level": self.previous_level,
            "new_score": round(self.new_score, 1),
            "new_level": self.new_level,
            "score_delta": round(self.score_delta, 1),
            "triggering_event": self.triggering_event,
            "recalculated_at": self.recalculated_at.isoformat(),
            "is_changed": self.is_changed,
            "explanation": self.explanation,
            "component_details": self.component_details,
        }


class DynamicPriorityReassessor:
    """Production service for event-driven priority reassessment and audit logging."""

    def __init__(self, engine: Optional[PriorityEngine] = None):
        self.engine = engine or PriorityEngine()
        self._processed_event_ids: Set[str] = set()
        self._audit_history: List[ReassessmentRecord] = []
        self._lock = threading.Lock()

    def process_event(
        self,
        event: ReassessmentEvent,
        current_state: dict,
    ) -> ReassessmentRecord:
        """Processes an event, calculates updated priority, and returns audit record.
        
        Args:
            event: ReassessmentEvent detailing trigger and payload.
            current_state: dict containing current complaint state:
                - 'complaint_id': ID
                - 'priority_score': current float score
                - 'priority_level': current level
                - 'subcategory': str
                - 'category': str
                - 'complaint_text': str
                - 'landmark_or_address': str
                - 'nearby_complaints_count': int
                - 'upvote_count': int
                - 'status': str ('submitted', 'in_progress', 'resolved', 'closed', 'duplicate')
                - 'is_sensitive_facility': bool
                - 'is_arterial_road': bool
        """
        with self._lock:
            # 1. Idempotency check: if event_id was already processed, return no-op record
            if event.event_id and event.event_id in self._processed_event_ids:
                prev_s = float(current_state.get("priority_score", 0.0))
                prev_l = str(current_state.get("priority_level", "Medium"))
                return ReassessmentRecord(
                    complaint_id=event.complaint_id,
                    previous_score=prev_s,
                    previous_level=prev_l,
                    new_score=prev_s,
                    new_level=prev_l,
                    score_delta=0.0,
                    triggering_event=f"{event.event_type.value} [IDEMPOTENT_SKIPPED]",
                    recalculated_at=datetime.now(timezone.utc),
                    is_changed=False,
                    explanation="Event was previously processed; skipped to maintain idempotence.",
                    component_details={},
                )

            # Copy state to avoid mutating caller's dict
            state = dict(current_state)
            prev_score = float(state.get("priority_score", 0.0))
            prev_level = str(state.get("priority_level", "Medium"))

            # 2. Apply event updates to state
            payload = event.event_payload or {}
            event_type = event.event_type

            if event_type == ReassessmentEventType.NEW_NEARBY_COMPLAINT:
                increment = int(payload.get("count_increment", 1))
                state["nearby_complaints_count"] = max(0, int(state.get("nearby_complaints_count", 0)) + increment)

            elif event_type == ReassessmentEventType.COMMUNITY_UPVOTE:
                increment = int(payload.get("upvote_increment", 1))
                state["upvote_count"] = max(0, int(state.get("upvote_count", 0)) + increment)

            elif event_type == ReassessmentEventType.SEVERITY_UPDATED:
                if "subcategory" in payload:
                    state["subcategory"] = payload["subcategory"]
                if "complaint_text" in payload:
                    state["complaint_text"] = payload["complaint_text"]
                if "is_sensitive_facility" in payload:
                    state["is_sensitive_facility"] = payload["is_sensitive_facility"]

            elif event_type == ReassessmentEventType.STATUS_CHANGED:
                state["status"] = str(payload.get("new_status", state.get("status", "submitted"))).lower()

            elif event_type == ReassessmentEventType.DUPLICATE_LINKED:
                # Merge duplicate's upvotes into parent without double-counting severity
                dup_upvotes = int(payload.get("duplicate_upvotes", 0))
                state["upvote_count"] = max(0, int(state.get("upvote_count", 0)) + dup_upvotes)

            # 3. Status Rule Override
            # If complaint has been resolved or closed, active operational priority drops to 0.0
            curr_status = str(state.get("status", "submitted")).lower()
            if curr_status in ["resolved", "closed"]:
                new_score = 0.0
                new_level = "Resolved"
                explanation = f"Complaint status is '{curr_status}'. Operational priority set to 0.0."
                comp_details = {"status_override": curr_status, "effective_score": 0.0}
            else:
                # 4. Re-calculate through Phase 2.17 Priority Engine
                calc_result = self.engine.calculate_priority(
                    subcategory=state.get("subcategory"),
                    category=state.get("category"),
                    complaint_text=state.get("complaint_text"),
                    landmark_or_address=state.get("landmark_or_address"),
                    nearby_complaints_count=int(state.get("nearby_complaints_count", 0)),
                    upvote_count=int(state.get("upvote_count", 0)),
                    is_arterial_road=bool(state.get("is_arterial_road", False)),
                    is_sensitive_facility=bool(state.get("is_sensitive_facility", False)),
                )
                new_score = calc_result.priority_score
                new_level = calc_result.priority_level
                explanation = f"Event '{event_type.value}' triggered recalculation: {calc_result.explanation}"
                comp_details = calc_result.component_details

            # 5. Record state transition
            delta = new_score - prev_score
            is_changed = abs(delta) >= 0.1 or (new_level != prev_level)

            record = ReassessmentRecord(
                complaint_id=event.complaint_id,
                previous_score=prev_score,
                previous_level=prev_level,
                new_score=new_score,
                new_level=new_level,
                score_delta=delta,
                triggering_event=event_type.value,
                recalculated_at=datetime.now(timezone.utc),
                is_changed=is_changed,
                explanation=explanation,
                component_details=comp_details,
            )

            # Mark event_id as processed
            if event.event_id:
                self._processed_event_ids.add(event.event_id)
            self._audit_history.append(record)

            logger.info(
                f"[Reassessment] Complaint #{event.complaint_id} via {event_type.value}: "
                f"{prev_score:.1f} ({prev_level}) -> {new_score:.1f} ({new_level}), Delta: {delta:+.1f}"
            )
            return record

    def get_audit_history(self, complaint_id: Optional[Union[int, str]] = None) -> List[ReassessmentRecord]:
        """Returns audit history, optionally filtered by complaint ID."""
        with self._lock:
            if complaint_id is not None:
                return [r for r in self._audit_history if str(r.complaint_id) == str(complaint_id)]
            return list(self._audit_history)
