from sqlalchemy import Column, Integer, String, DateTime
from aegis.database.database import Base


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, index=True)
    label = Column(String)
    description = Column(String)
    timestamp = Column(DateTime)
