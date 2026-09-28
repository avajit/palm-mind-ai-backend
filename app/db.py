from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum as SQLEnum
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import enum
from .config import settings

# Setup standard synchronous SQLite engine
engine = create_engine(
    settings.database_url, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ChunkingStrategy(str, enum.Enum):
    fixed = "fixed"
    sentence = "sentence"

class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_type = Column(String)  # .pdf or .txt
    strategy = Column(SQLEnum(ChunkingStrategy))
    chunk_count = Column(Integer)
    char_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class BookingRecord(Base):
    __tablename__ = "bookings"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String)
    date = Column(String)  # Storing as string for simplicity (YYYY-MM-DD)
    time = Column(String)  # Storing as string (HH:MM)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    """Creates all tables in the database if they don't exist."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency injection for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
