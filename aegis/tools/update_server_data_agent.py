from datetime import datetime

from aegis.database.database import SessionLocal
from aegis.database.models import DetectionResult


def update_detection_result(camera_id: int, result: dict):
    db = SessionLocal()
    try:
        record = DetectionResult(
            camera_id=camera_id,
            label=result.get("label"),
            description=result.get("description"),
            timestamp=datetime.utcnow(),
        )
        db.add(record)
        db.commit()
        # print(f"[updated] DB Updated for Camera {camera_id}")
    except Exception as e:
        print(f"[error] DB Error: {e}")
        db.rollback()
    finally:
        db.close()
