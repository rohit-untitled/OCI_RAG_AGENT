from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class FileLog(Base):
    __tablename__ = "file_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String(255), nullable=False)
    prefix = Column(String(255), nullable=False)
    local_path = Column(String(500), nullable=False)
    downloaded_at = Column(DateTime, default=datetime.utcnow)
