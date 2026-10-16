"""Phase 2.16 Duplicate Detection Benchmark & Automated Validation Suite
Municipal Complaint Intelligence System

Validates:
1. Exact duplicates detection (known duplicate rows)
2. Spatial hard negatives (identical text, distant coordinates > 5km)
3. Semantic hard negatives (colocated complaints with different categories)
4. Recurring issues (colocated complaints separated by > 90 days)
5. Candidate retrieval latency (< 5ms)
6. Edge case handling (missing coordinates, missing text, zero coordinates)
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.duplicate_detector import DuplicateDetector, DuplicateDetectorConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase2.16Test")


def run_phase_2_16_validation():
    logger.info("=" * 70)
    logger.info("STARTING PHASE 2.16 DUPLICATE DETECTION VALIDATION & BENCHMARK")
    logger.info("=" * 70)

    # 1. Initialize detector with pre-fitted TF-IDF vectorizer
    model_path = BACKEND_DIR / "app/ml/saved_models/batch_3/task2_category_classifier.joblib"
    master_csv_path = BACKEND_DIR / "app/data/processed/batch_2/prepared_master_dataset.csv"
    assert model_path.exists(), f"Model pipeline missing at {model_path}"
    assert master_csv_path.exists(), f"Master dataset missing at {master_csv_path}"

    detector = DuplicateDetector.from_saved_model(model_path)
    df = pd.read_csv(master_csv_path)

    # Build index on the master dataset
    start_t = time.time()
    detector.build_index(df)
    index_time = time.time() - start_t
    logger.info(f"Built BallTree & TF-IDF index on {len(df):,} complaints in {index_time*1000:.2f} ms")

    benchmark_results = {}

    # =========================================================================
    # Test Tier 1: Known Exact Duplicates Detection (N = 59 rows)
    # =========================================================================
    exact_dups = df[df["is_exact_duplicate_row"] == True]
    assert len(exact_dups) == 59, f"Expected 59 exact duplicate rows, got {len(exact_dups)}"

    tier1_found = 0
    tier1_scores = []
    for idx, row in exact_dups.head(20).iterrows():
        candidates = detector.find_candidates(
            query_text=row["combined_text"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            query_id=row["source_row_id"],
            category=row["target_category"],
            subcategory=row["target_subcategory"],
            created_at=row["clean_created_at_iso"],
            top_k=5,
        )
        # Check if an exact twin is in candidates
        matching_twins = [c for c in candidates if c.classification == "duplicate" and c.distance_meters <= 1.0]
        if len(matching_twins) > 0:
            tier1_found += 1
            tier1_scores.append(matching_twins[0].composite_score)

    tier1_recall = tier1_found / 20.0
    avg_score = float(np.mean(tier1_scores)) if tier1_scores else 0.0
    logger.info(f"Tier 1 Exact Duplicate Recall: {tier1_recall*100:.1f}% (Avg duplicate score: {avg_score:.4f})")
    assert tier1_recall >= 0.95, f"Tier 1 recall failed: {tier1_recall}"
    benchmark_results["tier1_exact_duplicates_recall"] = tier1_recall

    # =========================================================================
    # Test Tier 2: Spatial Hard Negatives (Identical text, distant coordinates > 5km)
    # =========================================================================
    # Take a common complaint text from Koramangala (12.935, 77.624) and query from Whitefield (12.970, 77.750) ~ 14km away
    sample_row = df[df["target_category"] == "Roads & Footpaths"].iloc[0]
    whitefield_lat = 12.9700
    whitefield_lon = 77.7500
    distant_cands = detector.find_candidates(
        query_text=sample_row["combined_text"],
        latitude=whitefield_lat,
        longitude=whitefield_lon,
        query_id=-999,
        category=sample_row["target_category"],
        top_k=5,
    )
    # Distance is > 5km; spatial gating should either return 0 candidates or strictly 0 duplicates
    distant_dups = [c for c in distant_cands if c.classification == "duplicate"]
    logger.info(f"Tier 2 Spatial Hard Negatives: {len(distant_dups)} false duplicate detections out of {len(distant_cands)} spatial neighbors")
    assert len(distant_dups) == 0, "Spatial hard negative test failed: distant complaint marked as duplicate!"
    benchmark_results["tier2_spatial_false_positive_rate"] = 0.0

    # =========================================================================
    # Test Tier 3: Semantic Hard Negatives (Colocated, totally different categories)
    # =========================================================================
    # Query at exact same coordinate as sample_row, but with completely different category & text
    colocated_diff_cands = detector.find_candidates(
        query_text="Street light bulb completely blown out and dark at night",
        latitude=sample_row["latitude"],
        longitude=sample_row["longitude"],
        query_id=-999,
        category="Street Lighting & Electricity",
        subcategory="Street Light Not Working / Repair",
        top_k=5,
    )
    # Even though distance is 0m, category is different (Roads vs Street Lighting) and text similarity is low
    false_semantic_dups = [c for c in colocated_diff_cands if c.candidate_id == sample_row["source_row_id"] and c.classification == "duplicate"]
    logger.info(f"Tier 3 Semantic Hard Negatives: {len(false_semantic_dups)} false duplicate detections")
    assert len(false_semantic_dups) == 0, "Semantic hard negative test failed: different category marked as duplicate!"
    benchmark_results["tier3_semantic_false_positive_rate"] = 0.0

    # =========================================================================
    # Test Tier 4: Recurring Issues Detection (Colocated, same category, > 90 days gap)
    # =========================================================================
    # Query at same coordinate and category, but timestamp is 200 days later
    sample_dt = pd.to_datetime(sample_row["clean_created_at_iso"])
    future_dt = sample_dt + pd.Timedelta(days=200)
    recurring_cands = detector.find_candidates(
        query_text=sample_row["combined_text"],
        latitude=sample_row["latitude"],
        longitude=sample_row["longitude"],
        query_id=-999,
        category=sample_row["target_category"],
        subcategory=sample_row["target_subcategory"],
        created_at=future_dt.strftime("%Y-%m-%d %H:%M:%S"),
        top_k=5,
    )
    matching_recurring = [c for c in recurring_cands if c.candidate_id == sample_row["source_row_id"] and c.classification == "recurring_issue"]
    logger.info(f"Tier 4 Recurring Issue Test: Found {len(matching_recurring)} recurring flags for historical candidate")
    assert len(matching_recurring) == 1, "Recurring issue test failed: not classified as recurring_issue!"
    benchmark_results["tier4_recurring_issue_accuracy"] = 1.0

    # =========================================================================
    # Test Tier 5: Candidate Retrieval Latency Benchmark
    # =========================================================================
    latencies = []
    test_sample = df.sample(50, random_state=42)
    for idx, r in test_sample.iterrows():
        t0 = time.time()
        detector.find_candidates(
            query_text=r["combined_text"],
            latitude=r["latitude"],
            longitude=r["longitude"],
            query_id=r["source_row_id"],
            category=r["target_category"],
            subcategory=r["target_subcategory"],
            top_k=5,
        )
        latencies.append((time.time() - t0) * 1000)

    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    logger.info(f"Tier 5 Retrieval Latency: Mean = {avg_lat:.2f} ms, P95 = {p95_lat:.2f} ms")
    assert avg_lat < 5.0, f"Query latency too high: {avg_lat} ms >= 5.0 ms"
    benchmark_results["avg_query_latency_ms"] = avg_lat
    benchmark_results["p95_query_latency_ms"] = p95_lat

    # =========================================================================
    # Test Tier 6: Edge Case & Missing Field Safety
    # =========================================================================
    # 1. Invalid / zero coordinates -> gracefully returns empty list
    edge_zero = detector.find_candidates(query_text="Some text", latitude=0.0, longitude=0.0)
    assert len(edge_zero) == 0, "Failed to handle zero coordinates safely"

    # 2. Out of bounds coordinates -> gracefully returns empty list
    edge_oob = detector.find_candidates(query_text="Some text", latitude=999.0, longitude=-999.0)
    assert len(edge_oob) == 0, "Failed to handle out-of-bounds coordinates safely"

    # 3. Empty text -> handled safely without crash
    edge_notext = detector.find_candidates(query_text="", latitude=sample_row["latitude"], longitude=sample_row["longitude"])
    assert isinstance(edge_notext, list), "Failed to handle empty text safely"
    logger.info("Tier 6 Edge cases and missing fields handled safely.")

    # Save benchmark deliverables
    out_dir = BACKEND_DIR / "app/data/processed/batch_3"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase_2_16_duplicate_benchmark.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    logger.info("=" * 70)
    logger.info("ALL PHASE 2.16 VALIDATION TESTS PASSED SUCCESSFULLY!")
    logger.info("=" * 70)
    return True


if __name__ == "__main__":
    success = run_phase_2_16_validation()
    sys.exit(0 if success else 1)
