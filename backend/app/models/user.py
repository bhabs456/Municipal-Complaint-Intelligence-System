from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    phone_number = Column(String(50), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="citizen", nullable=False)  # citizen, official, admin
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    complaints = relationship("Complaint", back_populates="user", cascade="all, delete-orphan")
    interactions = relationship("ComplaintInteraction", back_populates="user")
    history_records = relationship("ComplaintHistory", back_populates="user")
