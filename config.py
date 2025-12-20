from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql://user:password@localhost:5432/sentiment_db"
    model_name: str = "talgat/bert-multilingual-sentiment-qlora"
    app_title: str = "Sentiment Analysis API"
    app_version: str = "1.0.0"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
