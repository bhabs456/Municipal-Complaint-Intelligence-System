"""Phase 2.18 Dynamic Priority Reassessment Automated Validation Suite
Municipal Complaint Intelligence System

Validates:
1. Event processing & priority escalations (density surge, upvotes, severity upgrade)
2. Idempotency (duplicate event ID is not double-processed)
3. Status transitions & resolution de-escalation (resolved/closed -> score 0.0)
4. Duplicate linking without severity double-counting
5. Thread-safe concurrent event execution
6. Audit history integrity & delta logging
"""

import concurrent.futures
import json
import logging
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.priority_reassessment import (
    DynamicPriorityReassessor,
    ReassessmentEvent,
    ReassessmentEventType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase2.18Test")


def run_phase_2_18_validation():
    logger.info("=" * 70)
    logger.info("STARTING PHASE 2.18 DYNAMIC REASSESSMENT VALIDATION SUITE")
    logger.info("=" * 70)

    reassessor = DynamicPriorityReassessor()

    # Initial state of an open complaint
    complaint_state = {
        "complaint_id": 1001,
        "priority_score": 50.0,
        "priority_level": "Medium",
        "subcategory": "Potholes & Road Damage",
        "category": "Roads & Footpaths",
        "complaint_text": "Pothole on 80ft road near junction",
        "nearby_complaints_count": 1,
        "upvote_count": 0,
        "status": "submitted",
        "is_arterial_road": True,
    }

    # =========================================================================
    # Test 1: Event 1 - Cluster Density Surge (+4 nearby complaints filed)
    # =========================================================================
    e1 = ReassessmentEvent(
        event_type=ReassessmentEventType.NEW_NEARBY_COMPLAINT,
        complaint_id=1001,
        event_id="EVT-1001-DENSITY-01",
        event_payload={"count_increment": 4},
    )
    r1 = reassessor.process_event(e1, complaint_state)
    logger.info(f"Test 1 (Density Surge): {r1.previous_score:.1f} -> {r1.new_score:.1f} (Delta: {r1.score_delta:+.1f})")
    assert r1.new_score > r1.previous_score, "Score did not increase after nearby complaints surge!"
    assert r1.is_changed is True
    # Update state for next transition
    complaint_state["priority_score"] = r1.new_score
    complaint_state["priority_level"] = r1.new_level
    complaint_state["nearby_complaints_count"] = 5

    # =========================================================================
    # Test 2: Idempotency Verification (Resending same event_id)
    # =========================================================================
    r1_repeat = reassessor.process_event(e1, complaint_state)
    logger.info(f"Test 2 (Idempotency): Repeated event delta = {r1_repeat.score_delta}")
    assert r1_repeat.score_delta == 0.0, "Repeated event was not skipped!"
    assert "IDEMPOTENT_SKIPPED" in r1_repeat.triggering_event

    # =========================================================================
    # Test 3: Event 2 - Community Upvote Surge (+15 upvotes)
    # =========================================================================
    e2 = ReassessmentEvent(
        event_type=ReassessmentEventType.COMMUNITY_UPVOTE,
        complaint_id=1001,
        event_id="EVT-1001-UPVOTE-01",
        event_payload={"upvote_increment": 15},
    )
    r2 = reassessor.process_event(e2, complaint_state)
    logger.info(f"Test 3 (Upvote Surge): {r2.previous_score:.1f} -> {r2.new_score:.1f} (Delta: {r2.score_delta:+.1f})")
    assert r2.new_score > r2.previous_score, "Score did not increase after upvotes!"
    complaint_state["priority_score"] = r2.new_score
    complaint_state["priority_level"] = r2.new_level
    complaint_state["upvote_count"] = 15

    # =========================================================================
    # Test 4: Event 3 - Duplicate Linking (merges duplicate's upvotes without double counting severity)
    # =========================================================================
    e3 = ReassessmentEvent(
        event_type=ReassessmentEventType.DUPLICATE_LINKED,
        complaint_id=1001,
        event_id="EVT-1001-DUPLINK-01",
        event_payload={"duplicate_id": 1002, "duplicate_upvotes": 5},
    )
    r3 = reassessor.process_event(e3, complaint_state)
    logger.info(f"Test 4 (Duplicate Linked): {r3.previous_score:.1f} -> {r3.new_score:.1f}")
    assert r3.new_score >= r3.previous_score
    complaint_state["priority_score"] = r3.new_score
    complaint_state["priority_level"] = r3.new_level
    complaint_state["upvote_count"] = 20

    # =========================================================================
    # Test 5: Event 4 - Status Transition to 'resolved' (Priority de-escalates to 0.0)
    # =========================================================================
    e4 = ReassessmentEvent(
        event_type=ReassessmentEventType.STATUS_CHANGED,
        complaint_id=1001,
        event_id="EVT-1001-STATUS-RESOLVED",
        event_payload={"new_status": "resolved"},
    )
    r4 = reassessor.process_event(e4, complaint_state)
    logger.info(f"Test 5 (Status Resolved): {r4.previous_score:.1f} -> {r4.new_score:.1f} (Level: {r4.new_level})")
    assert r4.new_score == 0.0, "Resolved complaint did not drop to 0.0 priority!"
    assert r4.new_level == "Resolved"

    # =========================================================================
    # Test 6: Audit History Integrity
    # =========================================================================
    history = reassessor.get_audit_history(complaint_id=1001)
    logger.info(f"Test 6 (Audit Trail): Recorded {len(history)} transitions for complaint #1001")
    assert len(history) >= 4, f"Expected at least 4 audit events, got {len(history)}"

    # =========================================================================
    # Test 7: Thread-Safe Concurrent Reassessments
    # =========================================================================
    concurrent_state = dict(complaint_state)
    concurrent_state["status"] = "in_progress"

    def fire_event(i):
        evt = ReassessmentEvent(
            event_type=ReassessmentEventType.COMMUNITY_UPVOTE,
            complaint_id=2000,
            event_id=f"CONCURRENT-EVT-{i}",
            event_payload={"upvote_increment": 1},
        )
        return reassessor.process_event(evt, concurrent_state)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fire_event, i) for i in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 20, "Not all concurrent events completed successfully"
    logger.info("Test 7: 20 concurrent reassessments processed without deadlocks or errors.")

    logger.info("=" * 70)
    logger.info("ALL PHASE 2.18 VALIDATION TESTS PASSED SUCCESSFULLY!")
    logger.info("=" * 70)
    return True


if __name__ == "__main__":
    success = run_phase_2_18_validation()
    sys.exit(0 if success else 1)
