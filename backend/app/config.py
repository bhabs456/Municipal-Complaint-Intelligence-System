from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str = "postgresql+psycopg2://postgres:12345@localhost:5432/municipal_db"

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_database_url(self) -> str:
        url = self.DATABASE_URL
        # Normalize protocol to postgresql+psycopg2
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        # If running outside docker on host machine, replace container host 'database:5432' with 'localhost:<POSTGRES_PORT>'
        import os
        if not os.path.exists("/.dockerenv") and "@database:" in url:
            url = url.replace("@database:5432", f"@localhost:{self.POSTGRES_PORT}", 1)
            url = url.replace("@database:", f"@localhost:{self.POSTGRES_PORT}:", 1)
        return url


settings = Settings()
