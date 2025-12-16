import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from src.api.schemas import RankingResponse
from src.modules.ingestion import ingestion_service
from src.modules.vision import vision_engine
from src.modules.rag import rag_engine
from src.modules.analysis import llm_ranker
from src.core.logger import app_logger

router = APIRouter()

@router.post("/rank", response_model=RankingResponse)
async def rank_resumes(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_description: str = Form(...),
    top_k: int = Form(5)
):
    job_id = str(uuid.uuid4())
    app_logger.info(f"Starting Job {job_id} | JD Preview: {job_description[:50]}...")

    extract_path = await ingestion_service.process_zip(file)
    
    ocr_results = vision_engine.process_directory(extract_path)
    
    if not ocr_results:
        shutil.rmtree(extract_path.parent)
        raise HTTPException(status_code=400, detail="No readable text found in the uploaded resumes")

    app_logger.info(f"Job {job_id}: Successfully OCR'd {len(ocr_results)} documents.")

    full_text_map = {item['filename']: item['text'] for item in ocr_results}

    index, chunk_metadata = rag_engine.create_index(ocr_results)
    
    relevant_chunks = rag_engine.search(index, chunk_metadata, query=job_description, k=top_k * 5)
    

    final_results = await llm_ranker.rank_candidates(
        job_description, 
        relevant_chunks, 
        full_docs_map=full_text_map
    )

    background_tasks.add_task(shutil.rmtree, extract_path.parent)

    return RankingResponse(
        job_id=job_id,
        candidates=final_results[:top_k]
    )