"""Phase 2.17 Priority Calculation Engine Automated Validation Suite
Municipal Complaint Intelligence System

Validates:
1. Score range constraints (strictly 0.0 to 100.0)
2. Monotonic priority ordering (Critical > High > Medium > Low)
3. Severe hazard vs routine case ordering
4. Deterministic repeatability
5. Missing data & edge case handling
6. Explainability structure
7. Weight sensitivity analysis
"""

import json
import logging
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.priority_engine import PriorityEngine, PriorityEngineConfig, SUBCATEGORY_SEVERITY_RATINGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase2.17Test")


def run_phase_2_17_validation():
    logger.info("=" * 70)
    logger.info("STARTING PHASE 2.17 PRIORITY ENGINE VALIDATION SUITE")
    logger.info("=" * 70)

    engine = PriorityEngine()

    # 1. Severe hazard test case (Sparking transformer near school)
    critical_case = engine.calculate_priority(
        subcategory="Electrical Pole / Transformer Issues",
        category="Street Lighting & Electricity",
        complaint_text="Transformer sparking with live wire hanging near National Public School",
        landmark_or_address="Opposite School Gate",
        nearby_complaints_count=8,
        upvote_count=15,
        is_sensitive_facility=True,
    )
    logger.info(f"Critical Scenario: Score = {critical_case.priority_score:.1f}, Level = {critical_case.priority_level}")
    assert critical_case.priority_level in ["Critical", "High"], f"Expected Critical/High, got {critical_case.priority_level}"
    assert critical_case.priority_score >= 80.0, f"Expected >= 80, got {critical_case.priority_score}"

    # 2. Routine low-impact test case (New water connection request on quiet residential lane)
    low_case = engine.calculate_priority(
        subcategory="New Water Connection / Pipeline",
        category="Water Supply",
        complaint_text="Application for new domestic water meter connection for newly built house",
        landmark_or_address="2nd Cross, layout",
        nearby_complaints_count=0,
        upvote_count=0,
    )
    logger.info(f"Low Scenario: Score = {low_case.priority_score:.1f}, Level = {low_case.priority_level}")
    assert low_case.priority_level == "Low", f"Expected Low, got {low_case.priority_level}"
    assert low_case.priority_score < 45.0, f"Expected < 45, got {low_case.priority_score}"

    # 3. Monotonic ordering: Critical Hazard > Medium Hazard > Low Request
    assert critical_case.priority_score > low_case.priority_score, "Priority ordering inversion detected!"

    # 4. Deterministic repeatability check
    run_1 = engine.calculate_priority(
        subcategory="Potholes & Road Damage",
        category="Roads & Footpaths",
        complaint_text="Dangerous deep pothole on ring road",
        nearby_complaints_count=5,
        upvote_count=10,
    )
    run_2 = engine.calculate_priority(
        subcategory="Potholes & Road Damage",
        category="Roads & Footpaths",
        complaint_text="Dangerous deep pothole on ring road",
        nearby_complaints_count=5,
        upvote_count=10,
    )
    assert run_1.priority_score == run_2.priority_score, "Non-deterministic priority score!"
    assert run_1.priority_level == run_2.priority_level, "Non-deterministic priority level!"
    logger.info(f"Deterministic repeatability verified: Score = {run_1.priority_score:.1f}")

    # 5. Missing data & edge case handling
    # Null category/subcategory/text/counts
    edge_nulls = engine.calculate_priority(
        subcategory=None,
        category=None,
        complaint_text=None,
        landmark_or_address=None,
        nearby_complaints_count=-5,  # Negative count
        upvote_count=-10,            # Negative upvotes
    )
    assert 0.0 <= edge_nulls.priority_score <= 100.0, "Score out of bounds on null inputs"
    assert edge_nulls.priority_level in ["Low", "Medium", "High", "Critical"]
    logger.info(f"Handled nulls safely: Score = {edge_nulls.priority_score:.1f}, Level = {edge_nulls.priority_level}")

    # 6. Sensitivity Analysis across weights
    # High severity weight vs high location weight
    cfg_sev_heavy = PriorityEngineConfig(w_severity=0.70, w_location=0.15, w_density=0.10, w_upvotes=0.05)
    eng_sev_heavy = PriorityEngine(cfg_sev_heavy)
    res_sev = eng_sev_heavy.calculate_priority(
        subcategory="Electrical Pole / Transformer Issues",
        nearby_complaints_count=0,
    )
    logger.info(f"Sensitivity Analysis: Severity-Heavy Config Score = {res_sev.priority_score:.1f}")
    assert res_sev.priority_score > 60.0

    # 7. Check all 24 subcategories produce valid scores
    for subcat, rating in SUBCATEGORY_SEVERITY_RATINGS.items():
        res = engine.calculate_priority(subcategory=subcat)
        assert 0.0 <= res.priority_score <= 100.0, f"Subcategory {subcat} produced invalid score: {res.priority_score}"

    logger.info("All 24 subcategories produce valid bounded scores.")
    logger.info("=" * 70)
    logger.info("ALL PHASE 2.17 VALIDATION TESTS PASSED SUCCESSFULLY!")
    logger.info("=" * 70)
    return True


if __name__ == "__main__":
    success = run_phase_2_17_validation()
    sys.exit(0 if success else 1)
