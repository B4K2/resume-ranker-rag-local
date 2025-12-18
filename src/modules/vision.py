import cv2
import numpy as np
import pytesseract
from pathlib import Path
from pdf2image import convert_from_path
from src.core.logger import app_logger

class VisionEngine:
    def __init__(self):
        self.custom_config = r'--oem 3 --psm 4'

    def process_directory(self, directory: Path) -> list[dict]:
        results = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        
        for file_path in directory.rglob("*"):
            if file_path.name.startswith("."): 
                continue
            
            suffix = file_path.suffix.lower()
            text = ""
            
            try:
                if suffix == ".pdf":
                    try:
                        pages = convert_from_path(str(file_path), dpi=300)
                        
                        for page in pages:
                            img_array = np.array(page)
                            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                            
                            page_text = self._process_cv2_image(img_bgr)
                            text += page_text + "\n"
                            
                    except Exception as e:
                        app_logger.warning(f"Could not convert PDF {file_path.name}: {e}")
                        continue

                elif suffix in image_extensions:
                    img = cv2.imread(str(file_path))
                    if img is None:
                        continue
                    text = self._process_cv2_image(img)

                else:
                    continue

                if self._is_valid_ocr(text):
                    results.append({
                        "filename": file_path.name,
                        "text": text,
                        "path": str(file_path)
                    })
                else:
                    app_logger.warning(f"Rejected file (low text quality): {file_path.name}")

            except Exception as e:
                app_logger.error(f"Error processing {file_path.name}: {e}")

        return results

    def _process_cv2_image(self, img: np.ndarray) -> str:
        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return pytesseract.image_to_string(gray, config=self.custom_config)

    def _is_valid_ocr(self, text: str) -> bool:
        if not text or len(text.strip()) < 50:
            return False

        alnum_count = sum(c.isalnum() for c in text)
        total_chars = len(text.strip())
        
        if total_chars == 0: return False
        
        if (alnum_count / total_chars) < 0.4:
            return False
            
        return True

vision_engine = VisionEngine()