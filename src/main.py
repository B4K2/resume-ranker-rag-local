from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.core.config import settings
from src.core.logger import app_logger
from src.api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info("Service is starting up...")
    yield
    app_logger.info("Service is shutting down...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)