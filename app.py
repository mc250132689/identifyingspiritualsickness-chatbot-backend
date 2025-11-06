import os
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, Text, String, DateTime, select, insert, delete

# ===========================================================
# 1) DATABASE CONFIG (IMPORTANT: MUST BE asyncpg)
# ===========================================================
# Example:
# postgresql+asyncpg://username:password@host:port/database_name

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:1234@localhost:5432/mydb")

engine = create_async_engine(DATABASE_URL, echo=False, future=True)

async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

# ===========================================================
# 2) DATABASE MODEL
# ===========================================================
class TrainingData(Base):
    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    lang = Column(String(10), nullable=False, default="en")
    created_at = Column(DateTime, default=datetime.utcnow)


# ===========================================================
# 3) REQUEST SCHEMA
# ===========================================================
class TrainingDataCreate(BaseModel):
    question: str
    answer: str
    lang: str = "en"


# ===========================================================
# 4) FASTAPI APP
# ===========================================================
app = FastAPI()


# ===========================================================
# 5) CREATE TABLES IF NOT EXISTS
# ===========================================================
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ===========================================================
# 6) ROUTES (CRUD)
# ===========================================================

@app.get("/training", summary="Get all training data")
async def get_training():
    async with async_session() as session:
        result = await session.execute(select(TrainingData))
        data = result.scalars().all()
        return data


@app.post("/training", summary="Insert new training data")
async def add_training(entry: TrainingDataCreate):
    async with async_session() as session:
        stmt = insert(TrainingData).values(
            question=entry.question,
            answer=entry.answer,
            lang=entry.lang
        )
        await session.execute(stmt)
        await session.commit()
        return {"status": "success", "message": "Data inserted successfully"}


@app.delete("/training/{id}", summary="Delete training data by ID")
async def delete_training(id: int):
    async with async_session() as session:
        stmt = delete(TrainingData).where(TrainingData.id == id)
        result = await session.execute(stmt)
        await session.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="ID not found")

        return {"status": "success", "message": f"Deleted ID {id}"}


# ===========================================================
# 7) ROOT CHECK
# ===========================================================
@app.get("/")
async def root():
    return {"message": "Training Data API Running"}
