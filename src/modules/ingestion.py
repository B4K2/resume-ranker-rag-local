import uuid
import shutil
import zipfile
from pathlib import Path
from fastapi import UploadFile, HTTPException
from src.core.config import settings
from src.core.logger import app_logger

class IngestionService:
    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR

    async def process_zip(self, file: UploadFile) -> Path:
        """
        Validates and extracts a Zip file to a unique temporary directory.
        Returns the path to the extracted folder.
        """
        session_id = str(uuid.uuid4())
        session_dir = self.upload_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        zip_path = session_dir / "input.zip"

        try:
            with open(zip_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            shutil.rmtree(session_dir)
            app_logger.error(f"Failed to save upload: {e}")
            raise HTTPException(status_code=500, detail="File upload failed")
        
        try:
            self._validate_zip(zip_path)
        except HTTPException as e:
            shutil.rmtree(session_dir)
            raise e

        extract_path = session_dir / "extracted"
        extract_path.mkdir()
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        except Exception as e:
            shutil.rmtree(session_dir)
            app_logger.error(f"Extraction failed: {e}")
            raise HTTPException(status_code=400, detail="Failed to extract zip file")

        app_logger.info(f"Successfully ingested session {session_id}")
        return extract_path

    def _validate_zip(self, zip_path: Path):
        """
        Checks for Zip Bombs (Size, Ratio, File Count).
        """
        if not zipfile.is_zipfile(zip_path):
            raise HTTPException(status_code=400, detail="File is not a valid zip archive")

        total_size = 0
        total_files = 0

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for info in zip_ref.infolist():
                total_files += 1
                total_size += info.file_size

                if total_files > settings.MAX_FILE_COUNT:
                    raise HTTPException(status_code=400, detail=f"Too many files in zip (Limit: {settings.MAX_FILE_COUNT})")

                if total_size > settings.MAX_EXTRACTED_SIZE_BYTES:
                     raise HTTPException(status_code=400, detail="Total extracted size exceeds limit")


                if info.compress_size > 0:
                    ratio = info.file_size / info.compress_size
                    if ratio > 100:  # Threshold: 100x compression
                        app_logger.warning(f"Zip bomb detected: {info.filename} ratio {ratio:.2f}")
                        raise HTTPException(status_code=400, detail="Suspicious compression ratio detected")

ingestion_service = IngestionService()