"""Complaint Priority Calculation Engine (Phase 2.17)
Municipal Complaint Intelligence System

Calculates objective, explainable civic grievance priority based on:
1. Problem seriousness & severity (inherent hazard + emergency text modifiers)
2. Location impact (public vulnerability, arterial transit, sensitive zones)
3. Complaint density (cluster concentration of unresolved nearby issues)
4. Optional community validation (citizen upvotes / backing)

Strict safeguards:
- Does NOT treat duplicate submissions as independent severity evidence
- Fully deterministic and independent of the web API
- Transparent formula with sensitivity documentation
- Uses locked 6-category, 24-subcategory taxonomy
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

# Inherent severity ratings for all 24 subcategories (scale 0.0 - 1.0)
# Derived from municipal civil safety standards, hazard risk, and public health impact
SUBCATEGORY_SEVERITY_RATINGS: Dict[str, float] = {
    # Electrical Hazards & Fire Risks (Immediate Danger)
    "Electrical Pole / Transformer Issues": 0.95,
    "Electricity Supply": 0.65,
    "Street Light Not Working / Repair": 0.60,
    "Street Light Installation": 0.40,
    
    # Public Health & Biohazards (Epidemic & Sewage Risks)
    "Sewer Blockage / Overflow": 0.90,
    "Water Pollution": 0.90,
    "Drain Blockage / Desilting": 0.75,
    "Sewer Line Repair / Maintenance": 0.75,
    "Garbage Dumping": 0.65,
    "Street / Public Area Cleaning": 0.45,
    "Garbage Collection / Removal": 0.45,
    "Waste Disposal & Segregation": 0.40,
    "Land / Waste Pollution": 0.40,
    
    # Critical Water Supply Interruption
    "Water Supply Interruption": 0.85,
    "Water Leakage": 0.70,
    "Water Pipeline Damage / Maintenance": 0.70,
    "New Water Connection / Pipeline": 0.30,
    
    # Traffic Safety & Road Infrastructure
    "Road Waterlogging": 0.85,
    "Storm Water Drainage / Waterlogging": 0.85,
    "Potholes & Road Damage": 0.80,
    "Footpath Damage / Repair": 0.55,
    "Road Repair / Construction": 0.50,
    
    # Environmental & Nuisance
    "Air Pollution": 0.65,
    "Noise Pollution": 0.45,
}

# Emergency and high-vulnerability text keyword modifiers
EMERGENCY_KEYWORDS = [
    r"\b(spark|sparking|fire|live wire|electrocution|short circuit)\b",
    r"\b(hospital|clinic|ambulance|emergency|school|college|pedestrian)\b",
    r"\b(accident|fatal|death|died|injured|injury|caved in|sinkhole)\b",
    r"\b(senior citizen|elderly|children|disabled|wheelchair)\b",
    r"\b(flooding inside home|sewage entering house|drinking water contaminated)\b",
]
EMERGENCY_REGEX = re.compile("|".join(EMERGENCY_KEYWORDS), re.IGNORECASE)

# High-impact location terms
SENSITIVE_LOCATIONS = re.compile(
    r"\b(hospital|school|college|bus stand|bus stop|metro station|railway|market|main road|arterial|ring road|flyover)\b",
    re.IGNORECASE,
)


@dataclass
class PriorityEngineConfig:
    """Configurable weights and threshold boundaries for priority scoring."""
    # Weights for primary signals (must sum to 1.0)
    w_severity: float = 0.50
    w_location: float = 0.25
    w_density: float = 0.15
    w_upvotes: float = 0.10
    
    # Priority Level Boundaries (0 - 100 scale)
    critical_threshold: float = 80.0
    high_threshold: float = 65.0
    medium_threshold: float = 45.0
    
    # Saturation caps
    max_density_neighbors: int = 10
    max_upvotes_scale: int = 25

    def __post_init__(self):
        total_w = self.w_severity + self.w_location + self.w_density + self.w_upvotes
        assert abs(total_w - 1.0) < 1e-4, f"Priority weights must sum to 1.0, got {total_w}"


@dataclass
class PriorityResult:
    """Detailed explainable priority assessment."""
    priority_score: float  # 0.0 to 100.0
    priority_level: str    # 'Low', 'Medium', 'High', 'Critical'
    severity_score: float  # 0.0 to 1.0
    location_score: float  # 0.0 to 1.0
    density_score: float   # 0.0 to 1.0
    upvote_score: float    # 0.0 to 1.0
    explanation: str
    component_details: dict

    def to_dict(self) -> dict:
        return {
            "priority_score": round(self.priority_score, 1),
            "priority_level": self.priority_level,
            "severity_score": round(self.severity_score, 3),
            "location_score": round(self.location_score, 3),
            "density_score": round(self.density_score, 3),
            "upvote_score": round(self.upvote_score, 3),
            "explanation": self.explanation,
            "component_details": self.component_details,
        }


class PriorityEngine:
    """Standalone, explainable priority calculation engine."""

    def __init__(self, config: Optional[PriorityEngineConfig] = None):
        self.config = config or PriorityEngineConfig()

    def calculate_priority(
        self,
        subcategory: Optional[str] = None,
        category: Optional[str] = None,
        complaint_text: Optional[str] = None,
        landmark_or_address: Optional[str] = None,
        nearby_complaints_count: int = 0,
        upvote_count: int = 0,
        is_arterial_road: bool = False,
        is_sensitive_facility: bool = False,
    ) -> PriorityResult:
        """Calculates normalized multi-factor priority score and level."""
        text = (complaint_text or "").strip()
        address = (landmark_or_address or "").strip()

        # ---------------------------------------------------------------------
        # 1. Problem Seriousness / Severity (0.0 to 1.0)
        # ---------------------------------------------------------------------
        base_sev = 0.50  # Default neutral severity
        if subcategory and subcategory in SUBCATEGORY_SEVERITY_RATINGS:
            base_sev = SUBCATEGORY_SEVERITY_RATINGS[subcategory]
        elif category:
            # Fallback by parent category
            cat_defaults = {
                "Street Lighting & Electricity": 0.65,
                "Sewage & Drainage": 0.80,
                "Water Supply": 0.75,
                "Roads & Footpaths": 0.70,
                "Pollution": 0.60,
                "Waste & Sanitation": 0.55,
            }
            base_sev = cat_defaults.get(category, 0.50)

        # Text-based emergency modifier (+0.10 if high-hazard keyword present)
        has_emergency_keyword = bool(EMERGENCY_REGEX.search(text))
        emergency_boost = 0.10 if has_emergency_keyword else 0.0
        severity_score = min(1.0, base_sev + emergency_boost)

        # ---------------------------------------------------------------------
        # 2. Location Impact (0.0 to 1.0)
        # ---------------------------------------------------------------------
        # Evaluates sensitive facilities, arterial corridors, and public hubs
        has_location_keyword = bool(SENSITIVE_LOCATIONS.search(text)) or bool(SENSITIVE_LOCATIONS.search(address))
        loc_val = 0.30  # Baseline residential street
        if is_sensitive_facility or has_emergency_keyword:
            loc_val = 0.90  # Immediate hospital, school, clinic zone
        elif is_arterial_road or has_location_keyword:
            loc_val = 0.70  # Major arterial / transit corridor
        location_score = min(1.0, max(0.0, loc_val))

        # ---------------------------------------------------------------------
        # 3. Complaint Density (0.0 to 1.0)
        # ---------------------------------------------------------------------
        # Logarithmic saturation curve over nearby unresolved complaints
        safe_nearby = max(0, int(nearby_complaints_count))
        # log(1 + N) / log(1 + max_neighbors)
        density_score = min(
            1.0,
            math.log1p(safe_nearby) / math.log1p(self.config.max_density_neighbors),
        )

        # ---------------------------------------------------------------------
        # 4. Community Upvotes / Citizen Backing (0.0 to 1.0)
        # ---------------------------------------------------------------------
        safe_upvotes = max(0, int(upvote_count))
        upvote_score = min(
            1.0,
            math.log1p(safe_upvotes) / math.log1p(self.config.max_upvotes_scale),
        )

        # ---------------------------------------------------------------------
        # 5. Composite Score & Priority Level
        # ---------------------------------------------------------------------
        raw_score = (
            self.config.w_severity * severity_score
            + self.config.w_location * location_score
            + self.config.w_density * density_score
            + self.config.w_upvotes * upvote_score
        )
        total_score = min(100.0, max(0.0, raw_score * 100.0))

        if total_score >= self.config.critical_threshold:
            level = "Critical"
        elif total_score >= self.config.high_threshold:
            level = "High"
        elif total_score >= self.config.medium_threshold:
            level = "Medium"
        else:
            level = "Low"

        # Explanation generation
        reasons = []
        reasons.append(f"Severity ({self.config.w_severity*100:.0f}% wt): {severity_score:.2f} based on '{subcategory or category or 'general'}'")
        if emergency_boost > 0:
            reasons.append("Emergency hazard modifier triggered by text")
        reasons.append(f"Location Impact ({self.config.w_location*100:.0f}% wt): {location_score:.2f}")
        reasons.append(f"Complaint Density ({self.config.w_density*100:.0f}% wt): {density_score:.2f} ({safe_nearby} nearby issues)")
        if safe_upvotes > 0:
            reasons.append(f"Community Backing ({self.config.w_upvotes*100:.0f}% wt): {upvote_score:.2f} ({safe_upvotes} upvotes)")

        explanation = f"Assigned {level} priority (Score: {total_score:.1f}/100) — " + "; ".join(reasons)

        component_details = {
            "base_severity": round(base_sev, 3),
            "emergency_boost": round(emergency_boost, 3),
            "effective_severity": round(severity_score, 3),
            "location_impact": round(location_score, 3),
            "density_unresolved_count": safe_nearby,
            "density_score": round(density_score, 3),
            "upvote_count": safe_upvotes,
            "upvote_score": round(upvote_score, 3),
            "weights": {
                "w_severity": self.config.w_severity,
                "w_location": self.config.w_location,
                "w_density": self.config.w_density,
                "w_upvotes": self.config.w_upvotes,
            },
        }

        return PriorityResult(
            priority_score=total_score,
            priority_level=level,
            severity_score=severity_score,
            location_score=location_score,
            density_score=density_score,
            upvote_score=upvote_score,
            explanation=explanation,
            component_details=component_details,
        )
