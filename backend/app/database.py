from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

# Create the SQLAlchemy engine using the database URL
engine = create_engine(
    settings.get_database_url(),
    pool_pre_ping=True,
)

# Factory for creating new database Session objects
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Declarative base class for the upcoming SQLAlchemy models
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a database session per request

    and safely closes it when the request completes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
