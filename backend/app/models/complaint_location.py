from datetime import datetime
from geoalchemy2 import Geometry
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.database import Base


class ComplaintLocation(Base):
    __tablename__ = "complaint_locations"

    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(Integer, ForeignKey("complaints.id", ondelete="CASCADE"), unique=True, nullable=False)
    address = Column(String(500), nullable=True)
    ward_number = Column(String(50), index=True, nullable=True)
    zone = Column(String(100), index=True, nullable=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    geom = Column(Geometry(geometry_type="POINT", srid=4326), index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # 1-to-1 Relationship with Complaint
    complaint = relationship("Complaint", back_populates="location")
