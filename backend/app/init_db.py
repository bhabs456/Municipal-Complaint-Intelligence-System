from app.database import Base, engine
import app.models  # noqa: F401 - ensure all 8 models are registered on Base.metadata


def create_tables():
    print("Connecting to database and creating tables...")
    Base.metadata.create_all(bind=engine)
    print("Successfully created all 8 database tables:")
    for table_name in sorted(Base.metadata.tables.keys()):
        print(f" - {table_name}")


if __name__ == "__main__":
    create_tables()
