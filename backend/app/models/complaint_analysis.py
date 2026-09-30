from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.database import Base


class ComplaintAnalysis(Base):
    __tablename__ = "complaint_analysis"

    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(Integer, ForeignKey("complaints.id", ondelete="CASCADE"), unique=True, nullable=False)
    predicted_category_id = Column(Integer, ForeignKey("categories.id", ondelete="SET NULL"), nullable=True)
    predicted_subcategory_id = Column(Integer, ForeignKey("subcategories.id", ondelete="SET NULL"), nullable=True)
    classification_confidence = Column(Float, nullable=True)  # Classification confidence (0.0 - 1.0)
    is_civic_issue = Column(Boolean, default=True, nullable=False)  # Civic classification flag
    duplicate_score = Column(Float, default=0.0, nullable=False)  # Duplicate similarity score (0.0 - 1.0)
    urgency_score = Column(Float, default=0.0, nullable=False)  # Priority component
    severity_score = Column(Float, default=0.0, nullable=False)  # Priority component
    impact_score = Column(Float, default=0.0, nullable=False)  # Priority component
    priority_score = Column(Float, default=0.0, index=True, nullable=False)  # Calculated total priority
    priority_level = Column(String(50), default="Medium", index=True, nullable=False)  # Low, Medium, High, Critical
    model_version = Column(String(50), default="v1.0.0", nullable=False)
    analyzed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    complaint = relationship("Complaint", back_populates="analysis")
    category = relationship("Category", back_populates="analyses")
    subcategory = relationship("Subcategory", back_populates="analyses")
