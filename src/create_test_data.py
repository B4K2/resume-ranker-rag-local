import os
import random
import zipfile
import tempfile
from pathlib import Path
from pdf2image import convert_from_path

# --- CONFIGURATION ---
SOURCE_DATASET_PATH = Path("./archive/data/data/") # UPDATE THIS to your actual path
OUTPUT_DIR = Path(".")

def create_zip_dataset(zip_name, categories, files_per_category):
    """
    Creates a zip file containing converted images from PDFs.
    """
    print(f"\n📦 Creating {zip_name}...")
    
    zip_path = OUTPUT_DIR / zip_name
    total_processed = 0
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Filter valid folders
        available_folders = [f for f in SOURCE_DATASET_PATH.iterdir() if f.is_dir()]
        
        # Select random categories if we don't want all
        if categories != "all":
            selected_folders = random.sample(available_folders, min(len(available_folders), categories))
        else:
            selected_folders = available_folders

        for folder in selected_folders:
            category_name = folder.name
            pdfs = list(folder.glob("*.pdf"))
            
            if not pdfs:
                continue
                
            # Select random PDFs from this category
            selected_pdfs = random.sample(pdfs, min(len(pdfs), files_per_category))
            
            print(f"   Processing {category_name}: {len(selected_pdfs)} files...")
            
            for pdf_path in selected_pdfs:
                try:
                    # Convert PDF First Page to Image (in memory/temp)
                    # We use 200 DPI for good OCR quality
                    images = convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=200)
                    
                    if images:
                        img = images[0]
                        # Save as JPEG inside the ZIP
                        # Structure: category/filename.jpg
                        arcname = f"{category_name}/{pdf_path.stem}.jpg"
                        
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                            img.save(tmp.name, 'JPEG')
                            zf.write(tmp.name, arcname)
                            total_processed += 1
                except Exception as e:
                    print(f"   ⚠️ Failed to convert {pdf_path.name}: {e}")

    print(f"✅ Finished {zip_name}. Total images: {total_processed}")

# --- EXECUTION ---
if __name__ == "__main__":
    if not SOURCE_DATASET_PATH.exists():
        print(f"❌ Error: Path not found: {SOURCE_DATASET_PATH}")
        print("Please edit the SOURCE_DATASET_PATH in the script.")
        exit(1)

    # 1. Create Small Dataset (Rapid Testing)
    # Picks 3 random categories, 2 files each = ~6 images
    create_zip_dataset(
        "resume_samples_small.zip", 
        categories=3, 
        files_per_category=2
    )

    # 2. Create Large Dataset (Stress Testing)
    # Picks ALL categories, 5 files each = ~100+ images
    create_zip_dataset(
        "resume_samples_large.zip", 
        categories="all", 
        files_per_category=5
    )