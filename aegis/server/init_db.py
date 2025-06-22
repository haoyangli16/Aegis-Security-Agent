from database.database import Base, engine
from database.models import DetectionResult

Base.metadata.create_all(bind=engine)
print("[Test Code]DB tables created")