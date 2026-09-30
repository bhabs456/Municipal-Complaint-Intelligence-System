from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


class Complaint(Base):
    __tablename__ = "complaints"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    full_complaint = Column(Text, nullable=False)  # Complete original citizen complaint
    status = Column(String(50), default="submitted", index=True, nullable=False)  # submitted, in_progress, resolved, rejected, duplicate
    category_id = Column(Integer, ForeignKey("categories.id", ondelete="SET NULL"), nullable=True)
    subcategory_id = Column(Integer, ForeignKey("subcategories.id", ondelete="SET NULL"), nullable=True)
    duplicate_complaint_id = Column(Integer, ForeignKey("complaints.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="complaints")
    category = relationship("Category", back_populates="complaints")
    subcategory = relationship("Subcategory", back_populates="complaints")
    location = relationship("ComplaintLocation", back_populates="complaint", uselist=False, cascade="all, delete-orphan")
    analysis = relationship("ComplaintAnalysis", back_populates="complaint", uselist=False, cascade="all, delete-orphan")
    interactions = relationship("ComplaintInteraction", back_populates="complaint", cascade="all, delete-orphan")
    history_records = relationship("ComplaintHistory", back_populates="complaint", cascade="all, delete-orphan")

    # Self-referential relationship for duplicate complaint referencing
    duplicate_of = relationship("Complaint", remote_side=[id], backref="duplicates")
