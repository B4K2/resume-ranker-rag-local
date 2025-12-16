import sys
from loguru import logger
from src.core.config import settings

# Configure Logging
def setup_logging():
    # Remove default handler
    logger.remove()
    
    # Add Console Handler (for Docker/Dev)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    
    # Add File Handler (Rotation every 10MB, Retention 10 days)
    log_file_path = settings.BASE_DIR / "logs" / "app.log"
    logger.add(
        log_file_path,
        rotation="10 MB",
        retention="10 days",
        level="DEBUG",
        compression="zip"
    )

    return logger

# Initialize
app_logger = setup_logging()