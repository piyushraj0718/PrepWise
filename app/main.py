from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.logging import configure_logging
from app.db.base import Base
from app.db.session import engine
import app.models  # Ensures all SQLAlchemy models are registered before create_all.


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="PrepWise", lifespan=lifespan)
app.include_router(router)
