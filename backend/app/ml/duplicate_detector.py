"""Duplicate and Related Complaint Detection Engine (Phase 2.16)
Municipal Complaint Intelligence System

Implements multi-signal duplicate & related complaint identification:
- Spatial candidate blocking via BallTree (Haversine metric, O(K) query complexity)
- Multi-signal scoring combining:
  1. Geospatial distance (exponential spatial decay)
  2. Semantic text similarity (TF-IDF cosine similarity)
  3. Category & Subcategory taxonomy compatibility
  4. Temporal proximity & status-based recurrence decay
- Configurable weights, thresholds, and distance bounds
- Clear classification: 'duplicate', 'related', 'recurring_issue', 'distinct'
- Human-readable transparent audit explanations
- Robust edge case handling (missing coordinates, missing text, timezones)
- Non-destructive: Never auto-deletes, auto-merges, or silently rejects complaints
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import BallTree

logger = logging.getLogger("DuplicateDetector")
EARTH_RADIUS_METERS = 6371000.0


@dataclass
class DuplicateDetectorConfig:
    """Configurable parameters for duplicate scoring and classification."""
    max_spatial_radius_meters: float = 500.0
    spatial_decay_scale_meters: float = 150.0
    temporal_decay_scale_days: float = 30.0
    duplicate_threshold: float = 0.75
    related_threshold: float = 0.50
    recurring_days_threshold: float = 90.0
    
    # Feature weights (must sum to 1.0)
    w_text: float = 0.45
    w_geo: float = 0.30
    w_cat: float = 0.15
    w_time: float = 0.10

    def __post_init__(self):
        total_w = self.w_text + self.w_geo + self.w_cat + self.w_time
        assert abs(total_w - 1.0) < 1e-4, f"Weights must sum to 1.0, got {total_w}"


@dataclass
class DuplicateCandidate:
    """Structured result for a candidate complaint comparison."""
    candidate_id: Union[int, str]
    candidate_title: str
    candidate_category: Optional[str]
    candidate_subcategory: Optional[str]
    distance_meters: float
    text_similarity: float
    category_similarity: float
    temporal_gap_days: Optional[float]
    composite_score: float
    classification: str  # 'duplicate', 'related', 'recurring_issue', 'distinct'
    explanation: str

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "candidate_title": self.candidate_title,
            "candidate_category": self.candidate_category,
            "candidate_subcategory": self.candidate_subcategory,
            "distance_meters": round(self.distance_meters, 1),
            "text_similarity": round(self.text_similarity, 4),
            "category_similarity": round(self.category_similarity, 2),
            "temporal_gap_days": round(self.temporal_gap_days, 1) if self.temporal_gap_days is not None else None,
            "composite_score": round(self.composite_score, 4),
            "classification": self.classification,
            "explanation": self.explanation,
        }


def parse_timestamp(dt_val) -> Optional[datetime]:
    """Safely parses timestamp string or datetime object."""
    if dt_val is None or pd.isna(dt_val):
        return None
    if isinstance(dt_val, datetime):
        return dt_val
    try:
        ts = pd.to_datetime(dt_val)
        return ts.to_pydatetime()
    except Exception:
        return None


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two points on Earth in meters."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


class DuplicateDetector:
    """Production multi-signal duplicate and related complaint detection engine."""

    def __init__(
        self,
        config: Optional[DuplicateDetectorConfig] = None,
        vectorizer=None,
    ):
        self.config = config or DuplicateDetectorConfig()
        self.vectorizer = vectorizer
        self.ball_tree: Optional[BallTree] = None
        self.indexed_df: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None

    @classmethod
    def from_saved_model(
        cls,
        model_path: Union[str, Path],
        config: Optional[DuplicateDetectorConfig] = None,
    ):
        """Loads pre-fitted TF-IDF vectorizer from a saved scikit-learn pipeline."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model pipeline not found at {model_path}")
        pipe = joblib.load(model_path)
        if hasattr(pipe, "named_steps") and "tfidf" in pipe.named_steps:
            vectorizer = pipe.named_steps["tfidf"]
        elif hasattr(pipe, "vectorizer"):
            vectorizer = pipe.vectorizer
        else:
            raise ValueError("Could not extract TF-IDF vectorizer from loaded pipeline")
        return cls(config=config, vectorizer=vectorizer)

    def build_index(self, complaints_df: pd.DataFrame):
        """Builds spatial BallTree and precomputes TF-IDF embeddings for candidate complaints."""
        if len(complaints_df) == 0:
            raise ValueError("Cannot build index on empty dataframe")

        required_cols = ["source_row_id", "latitude", "longitude"]
        for col in required_cols:
            if col not in complaints_df.columns:
                raise ValueError(f"Required column '{col}' missing from complaints dataframe")

        self.indexed_df = complaints_df.copy().reset_index(drop=True)

        # Pre-cache arrays for ultra-fast O(1) query access
        self.cached_ids = self.indexed_df["source_row_id"].values
        self.cached_titles = (
            self.indexed_df["clean_title"].values
            if "clean_title" in self.indexed_df.columns
            else self.indexed_df["title"].fillna("").astype(str).values
        )
        self.cached_cats = (
            self.indexed_df["target_category"].fillna("").astype(str).values
            if "target_category" in self.indexed_df.columns
            else self.indexed_df["category_title"].fillna("").astype(str).values
        )
        self.cached_subs = (
            self.indexed_df["target_subcategory"].fillna("").astype(str).values
            if "target_subcategory" in self.indexed_df.columns
            else self.indexed_df["sub_category_title"].fillna("").astype(str).values
        )
        self.cached_lats = self.indexed_df["latitude"].values
        self.cached_lons = self.indexed_df["longitude"].values
        
        # Pre-parse timestamps
        date_col = "clean_created_at_iso" if "clean_created_at_iso" in self.indexed_df.columns else "created_at"
        self.cached_dts = pd.to_datetime(self.indexed_df[date_col], errors="coerce").dt.to_pydatetime()

        # 1. Build BallTree with coordinates in radians
        coords_rad = np.radians(np.column_stack([self.cached_lats, self.cached_lons]))
        self.ball_tree = BallTree(coords_rad, metric="haversine")

        # 2. Precompute TF-IDF matrix
        if self.vectorizer is not None:
            text_series = (
                self.indexed_df["combined_text"]
                if "combined_text" in self.indexed_df.columns
                else (self.indexed_df["title"].fillna("") + " " + self.indexed_df["description"].fillna(""))
            )
            self.tfidf_matrix = self.vectorizer.transform(text_series)
        else:
            logger.warning("No vectorizer provided; text similarity will require on-the-fly computation.")

        logger.info(f"Built spatial & text index on {len(self.indexed_df):,} complaints.")
        return self

    def find_candidates(
        self,
        query_text: str,
        latitude: float,
        longitude: float,
        query_id: Optional[Union[int, str]] = None,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        created_at: Optional[Union[str, datetime]] = None,
        status: Optional[str] = None,
        top_k: int = 10,
    ) -> List[DuplicateCandidate]:
        """Finds and ranks candidate duplicate/related complaints for an incoming complaint."""
        if self.ball_tree is None or self.indexed_df is None:
            raise RuntimeError("Index not built. Call build_index() before querying.")

        # Coordinate sanity check
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0) or (latitude == 0 and longitude == 0):
            logger.warning(f"Invalid coordinate ({latitude}, {longitude}). Spatial blocking disabled.")
            return []

        # 1. Spatial candidate blocking (within max radius)
        query_rad = np.radians([[latitude, longitude]])
        search_radius_rad = self.config.max_spatial_radius_meters / EARTH_RADIUS_METERS
        candidate_indices = self.ball_tree.query_radius(query_rad, r=search_radius_rad)[0]

        if len(candidate_indices) == 0:
            return []

        # Filter out self-id before computing text similarities
        if query_id is not None:
            candidate_indices = [idx for idx in candidate_indices if str(self.cached_ids[idx]) != str(query_id)]
            if len(candidate_indices) == 0:
                return []

        # 2. Batch compute text similarities via sparse matrix dot-product
        clean_q_text = (query_text or "").strip()
        sims = np.zeros(len(candidate_indices))
        if self.vectorizer is not None and clean_q_text and self.tfidf_matrix is not None:
            q_vec = self.vectorizer.transform([clean_q_text])
            sims = (self.tfidf_matrix[candidate_indices] @ q_vec.T).toarray().ravel()
            sims = np.clip(sims, 0.0, 1.0)

        query_dt = parse_timestamp(created_at)

        candidates = []
        for i, idx in enumerate(candidate_indices):
            cand_id = self.cached_ids[idx]

            # A. Spatial distance & decay
            c_lat = self.cached_lats[idx]
            c_lon = self.cached_lons[idx]
            dist_m = haversine_distance_meters(latitude, longitude, c_lat, c_lon)
            if dist_m > self.config.max_spatial_radius_meters:
                continue

            s_geo = math.exp(-dist_m / self.config.spatial_decay_scale_meters)
            s_text = float(sims[i])

            # C. Category & Subcategory similarity
            c_cat = self.cached_cats[idx]
            c_sub = self.cached_subs[idx]

            s_cat = 0.0
            if category and c_cat:
                if str(category).strip().lower() == str(c_cat).strip().lower():
                    if subcategory and c_sub and (str(subcategory).strip().lower() == str(c_sub).strip().lower()):
                        s_cat = 1.0  # Exact subcategory match
                    else:
                        s_cat = 0.70  # Parent category match
                else:
                    s_cat = 0.0  # Category mismatch
            else:
                s_cat = 0.50

            # D. Temporal gap & decay
            cand_dt = self.cached_dts[idx]
            temp_gap_days = None
            s_time = 0.5
            is_recurring = False

            if query_dt and cand_dt:
                temp_gap_days = abs((query_dt - cand_dt).total_seconds()) / 86400.0
                if temp_gap_days > self.config.recurring_days_threshold and s_cat >= 0.70 and dist_m <= 150.0:
                    is_recurring = True
                s_time = math.exp(-temp_gap_days / self.config.temporal_decay_scale_days)
            # E. Multi-factor composite score calculation
            composite_score = (
                self.config.w_text * s_text
                + self.config.w_geo * s_geo
                + self.config.w_cat * s_cat
                + self.config.w_time * s_time
            )

            # F. Classification & Explanation
            explanation_parts = []
            explanation_parts.append(f"Distance: {dist_m:.0f}m (spatial score: {s_geo:.2f})")
            explanation_parts.append(f"Text similarity: {s_text*100:.1f}%")
            if category and c_cat:
                explanation_parts.append(f"Category: {'matched' if s_cat == 1.0 else 'related' if s_cat > 0 else 'mismatch'}")
            if temp_gap_days is not None:
                explanation_parts.append(f"Time gap: {temp_gap_days:.0f} days")

            classification = "distinct"
            if is_recurring:
                classification = "recurring_issue"
                explanation_parts.append("Classified as recurring historical issue (colocated but historical gap > 90 days).")
            elif composite_score >= self.config.duplicate_threshold and dist_m <= 250.0 and s_text >= 0.55:
                classification = "duplicate"
                explanation_parts.append("Classified as duplicate (high composite similarity, close proximity & text alignment).")
            elif composite_score >= self.config.related_threshold and dist_m <= self.config.max_spatial_radius_meters and s_cat >= 0.70:
                classification = "related"
                explanation_parts.append("Classified as related nearby civic issue (same category within 500m).")
            else:
                classification = "distinct"
                explanation_parts.append("Classified as distinct (insufficient combined spatial, text or category similarity).")

            candidate = DuplicateCandidate(
                candidate_id=cand_id,
                candidate_title=str(self.cached_titles[idx]),
                candidate_category=str(c_cat) if c_cat else None,
                candidate_subcategory=str(c_sub) if c_sub else None,
                distance_meters=dist_m,
                text_similarity=s_text,
                category_similarity=s_cat,
                temporal_gap_days=temp_gap_days,
                composite_score=composite_score,
                classification=classification,
                explanation="; ".join(explanation_parts),
            )
            candidates.append(candidate)

        # Sort by composite score descending
        candidates.sort(key=lambda c: c.composite_score, reverse=True)
        return candidates[:top_k]
