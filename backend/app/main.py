from fastapi import Depends, FastAPI, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import engine, get_db
# Import all 8 models so they are registered with SQLAlchemy metadata
import app.models  # noqa: F401

app = FastAPI(
    title="Municipal Complaint Intelligence System",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "municipal-complaint-backend",
    }


@app.get("/health/database")
def health_database(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT 1")).scalar()
        if result == 1:
            return {
                "status": "ok",
                "database": "reachable",
            }
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "database": "unexpected response",
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "database": "unreachable",
                "details": str(exc),
            },
        )
