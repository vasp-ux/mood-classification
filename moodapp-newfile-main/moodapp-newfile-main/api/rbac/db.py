from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import RBAC_DB_URL

engine = create_engine(RBAC_DB_URL, future=True, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
