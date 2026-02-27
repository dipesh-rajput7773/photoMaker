from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = f"mysql+aiomysql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"

engine = create_async_engine(
    DATABASE_URL,
    echo=True,            # development mein True, production mein False
    pool_pre_ping=True,   # connection alive check
    pool_size=10,         # max 10 connections pool mein
    max_overflow=20       # busy time pe 20 extra allowed
)

async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with async_session() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"DB Session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()