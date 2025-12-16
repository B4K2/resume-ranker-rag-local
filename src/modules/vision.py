import cv2
import numpy as np
import pytesseract
from pathlib import Path
from src.core.logger import app_logger

class VisionEngine:
    def __init__(self):
        self.custom_config = r'--oem 3 --psm 4'

    def is_valid_ocr(self, text: str) -> bool:
        if len(text) < 100:
            return False
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        return alpha_ratio > 0.4


    def process_directory(self, directory: Path) -> list[dict]:
        results = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        for file_path in directory.rglob("*"):
            if file_path.suffix.lower() in valid_extensions:
                try:
                    text = self._process_single_image(file_path)
                    if not self.is_valid_ocr(text):
                        app_logger.warning(f"OCR rejected (noise): {file_path.name}")
                        continue
                    clean_chars = sum(c.isalnum() for c in text)
                    if clean_chars > 10:  
                        results.append({
                            "filename": file_path.name,
                            "text": text,
                            "path": str(file_path)
                        })
                except Exception as e:
                    app_logger.warning(f"Failed to process {file_path.name}: {e}")
        
        return results

    def _process_single_image(self, image_path: Path) -> str:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("Could not read image")

        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        extract_text = pytesseract.image_to_string(gray)

        return extract_text

vision_engine = VisionEngine()