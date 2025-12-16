# 📄 Resume Ranker RAG (Local GPU Accelerated)

A privacy-first, GPU-accelerated microservice that accepts a Zip file of resume images (scans/screenshots), performs OCR, and uses a **Two-Stage RAG + LLM** pipeline to semantically rank candidates against a Job Description.

Built with **FastAPI**, **Qwen-3**, **FAISS**, and **OpenCV**.

---

## 🚀 Key Features

### 🛡️ Secure Ingestion (Anti-Zip Bomb)
Processing user-uploaded Zip files is dangerous. This service implements a robust **Ingestion Guard** before extraction:
*   **Compression Ratio Check:** Rejects archives with suspicious compression ratios (e.g., >100x) to prevent "Zip Bombs" (Decompression Bombs).
*   **Size Limits:** Enforces strict limits on uncompressed file size and file counts.
*   **Nested Zip Detection:** Blocks recursive archives to prevent resource exhaustion.

### 👁️ Computer Vision Pipeline
Handles raw images, screenshots, and scans—not just text PDFs.
*   **Preprocessing:** Grayscale conversion, adaptive thresholding (for lighting correction), and 2x upscaling.
*   **OCR:** Uses **Tesseract 5** with LSTM engines to extract text from noisy images.
*   **Context Preservation:** Maps raw text back to the LLM to ensure headers (Name/Contact) aren't lost during chunking.

### 🧠 Two-Stage AI Ranking
Unlike simple keyword counters, this system uses a "Judge" architecture:
1.  **Stage 1 (Parallel Extraction):** The LLM analyzes every resume independently to extract verified skills and practical experience chunks retrieved via **RAG (FAISS)**.
2.  **Stage 2 (The Judge):** A final comparative pass where the LLM views *all* candidates simultaneously to assign relative rankings (0-100) based on the specific Job Description.

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Client Zip Upload] -->|FastAPI| B(Zip Guard & Extractor)
    B --> C{Image Preprocessing}
    C -->|OpenCV Upscale/Denoise| D[Tesseract OCR]
    D --> E[Full Text Map]
    D --> F[Chunking & FAISS Vector Index]
    G[Job Description] -->|Semantic Search| F
    F -->|Top Contexts| H[LLM Stage 1: Extraction]
    H -->|Candidate Profiles| I[LLM Stage 2: The Judge]
    I -->|JSON| J[Final Ranked List]