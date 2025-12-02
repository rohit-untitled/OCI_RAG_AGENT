from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Redaction Agent"
    ENVIRONMENT: str = "dev"
    DATABASE_URL: str = ""

    class Config:
        env_file = ".env"

settings = Settings()
