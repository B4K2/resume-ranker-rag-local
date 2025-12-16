import torch
import json
import re
import asyncio
from typing import List, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.core.config import settings
from src.core.logger import app_logger
from src.api.schemas import CandidateResult

class LLMRanker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        app_logger.info(f"Loading LLM on {self.device} (Lazy Loading)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_MODEL_ID, 
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

    async def rank_candidates(self, job_description: str, retrieved_chunks: list, full_docs_map: dict = None) -> list[CandidateResult]:
        """
        Two-Stage Ranking:
        1. Parallel Extraction: Get details for every candidate.
        2. The Judge: Compare all candidates together to decide the final rank.
        """
        candidates_data = {}
        
        # 1. Group chunks by filename
        for item in retrieved_chunks:
            fname = item['chunk']['filename']
            if fname not in candidates_data: candidates_data[fname] = []
            candidates_data[fname].append(item['chunk']['content'])

        # 2. Prepare Tasks for Extraction
        extraction_tasks = []
        filenames = []
        
        for filename, contexts in candidates_data.items():
            header_context = ""
            if full_docs_map and filename in full_docs_map:
                header_context = full_docs_map[filename][:800]
            
            rag_context = "\n... ".join(contexts[:3])
            combined_context = f"--- HEADER ---\n{header_context}\n\n--- EXPERIENCE ---\n{rag_context}"
            
            filenames.append(filename)
            # We wrap the sync function in asyncio.to_thread to allow concurrent execution flow
            extraction_tasks.append(
                asyncio.to_thread(self._analyze_single_candidate, job_description, combined_context)
            )

        app_logger.info(f"Stage 1: Extracting data for {len(filenames)} candidates...")
        
        # 3. Execute Extraction in Parallel (Logic parallel, GPU is still serial but queued)
        extracted_results = await asyncio.gather(*extraction_tasks)
        
        # 4. Create Intermediate Result Objects
        candidates = []
        for fname, analysis in zip(filenames, extracted_results):
            candidates.append(CandidateResult(
                rank=0,
                filename=fname,
                score=0.0,
                reasoning=analysis.get("reasoning", "Analysis failed"),
                extracted_skills=analysis.get("skills", []),
                relevant_experience=analysis.get("experience", [])
            ))

        # 5. Stage 2: The Tournament (Judge)
        # We send a summary of ALL candidates to the LLM to get a comparative ranking.
        if len(candidates) > 1:
            app_logger.info("Stage 2: Running Comparative Judging...")
            candidates = await asyncio.to_thread(self._judge_tournament, job_description, candidates)
        
        # Final Sort
        candidates.sort(key=lambda x: x.score, reverse=True)
        for i, res in enumerate(candidates): res.rank = i + 1
            
        return candidates

    def _judge_tournament(self, jd: str, candidates: List[CandidateResult]) -> List[CandidateResult]:
        """
        Takes all extracted profiles and asks the LLM to re-score them
        relative to each other using a structured chat prompt.
        """

        # Build roster text
        roster_lines = []
        for i, c in enumerate(candidates):
            skills = ", ".join(c.extracted_skills[:5]) if c.extracted_skills else "N/A"
            exp = "; ".join(c.relevant_experience[:2]) if c.relevant_experience else "N/A"

            roster_lines.append(
                f"Candidate {i+1}:\n"
                f"Filename: {c.filename}\n"
                f"Skills: {skills}\n"
                f"Experience: {exp}"
            )

        roster = "\n\n".join(roster_lines)

        messages = [
    {
        "role": "system",
        "content": (
            "You are a strict technical hiring manager evaluating candidates "
            "ONLY for the given job description.\n\n"

            "IMPORTANT RULES:\n"
            "- Strongly prioritize DEMONSTRATED, HANDS-ON experience.\n"
            "- Mentioning a skill WITHOUT real experience must be heavily penalized.\n"
            "- Candidates missing core JD requirements must receive LOW scores.\n"
            "- Use the FULL score range (0–100).\n"
            "- Large gaps are expected between strong and weak matches.\n\n"

            "Return ONLY valid JSON. No markdown. No explanations."
        )
    },
    {
        "role": "user",
        "content": f"""
Job Description (Primary Role):
Python Developer

Evaluation Criteria:
1. Demonstrated Python development experience (MOST IMPORTANT)
2. Real-world projects, production systems, or long-term usage
3. Supporting skills (data, SQL, analytics, etc.)
4. Leadership or unrelated experience should NOT increase score

Candidates:
{roster}

Return JSON in EXACT format:
{{
  "rankings": [
    {{
      "filename": "<filename>",
      "skill_match": 0-100,
      "experience_match": 0-100,
      "final_score": 0-100,
      "reason": "<1–2 sentence justification>"
    }}
  ]
}}

Scoring Rules:
- If Python is only mentioned but NOT demonstrated → experience_match ≤ 40
- If no Python at all → final_score ≤ 30
- Final score must reflect EXPERIENCE more than SKILLS
- Rank candidates from best to worst
"""
    }
]


        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50000,
                do_sample=False,
                repetition_penalty=1.1
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        import pdb; pdb.set_trace()
        response_text = self.tokenizer.decode(generated, skip_special_tokens=True)

        parsed_output = self._clean_and_parse_json(response_text)

        # 2. Normalize Output (Wrap List in Dict if necessary)
        if isinstance(parsed_output, list):
            ranking_data = {"rankings": parsed_output}
        elif isinstance(parsed_output, dict) and "rankings" in parsed_output:
            ranking_data = parsed_output
        else:
            app_logger.error(f"Judge returned unknown schema: {parsed_output}")
            return candidates

        # 3. Apply Scores
        rank_map = {r["filename"]: r for r in ranking_data["rankings"] if "filename" in r}
        
        for cand in candidates:
            if cand.filename in rank_map:
                match = rank_map[cand.filename]
                cand.score = float(match.get("final_score", 0)) / 100.0
                cand.reasoning = match.get("reason", cand.reasoning)

        return candidates

    def _analyze_single_candidate(self, jd: str, context: str) -> dict:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a recruiter.\n"
                    "Return ONLY a valid JSON object.\n"
                    "No explanations. No markdown."
                )
            },
            {
                "role": "user",
                "content": f"""
    Job Description:
    {jd}

    Resume Context:
    {context}

    Return JSON in this exact format:
    {{
    "skills": ["<technical skills>"],
    "experience": ["<specific experience>"],
    "score": 0.0,
    "reasoning": "<short reason>"
    }}
    """
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                repetition_penalty=1.1
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        app_logger.debug(f"RAW JUDGE OUTPUT:\n{response_text}") 

        return self._clean_and_parse_json(response_text)

    def _clean_and_parse_json(self, text: str) -> Union[dict, list]:
        """
        Robust parser handling both Objects {} and Arrays []
        """
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        text = text.replace("```json", "").replace("```", "").strip()

        match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)

        if not match:
            app_logger.error(f"No JSON found. Raw output: {text[:50]}...")
            return {}

        json_str = match.group(0)

        if json_str.startswith("{"):
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            if close_braces < open_braces: json_str += '}' * (open_braces - close_braces)
        elif json_str.startswith("["):
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            if close_brackets < open_brackets: json_str += ']' * (open_brackets - close_brackets)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            try:
                return json.loads(json_str)
            except:
                app_logger.error("Final JSON decode failed.")
                return {}

llm_ranker = LLMRanker()