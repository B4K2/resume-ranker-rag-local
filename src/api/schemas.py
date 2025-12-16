from pydantic import BaseModel, Field
from typing import List, Optional

class JobDescription(BaseModel):
    title: str = Field(..., description="Job Title, e.g. 'Backend Engineer'")
    description: str = Field(..., description="Full text of the Job Description")
    top_k: int = Field(5, description="Number of top candidates to return")

class CandidateResult(BaseModel):
    rank: int
    filename: str
    score: float
    reasoning: str = Field(..., description="Why this candidate was chosen")
    extracted_skills: List[str]
    relevant_experience: List[str]

class RankingResponse(BaseModel):
    job_id: str
    candidates: List[CandidateResult]