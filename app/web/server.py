import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from app.core.agent import run_agent, load_metadata_cache
from app.services.database import init_pgvector
from app.utils.logger import get_logger
import os
import json

logger = get_logger(__name__)

app = FastAPI(title="RAG API", version="1.0.0")

# CORS (Java 서버에서 호출 가능하게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 요청/응답 스펙 ──────────────────────────
class RagRequest(BaseModel):
    query: str                          # 사용자 원본 질의
    mcp_context: Optional[str] = None  # MCP 응답 내용
    source: Optional[str] = "tableau"  # "tableau" | "db" | "all"
    session_id: Optional[str] = None

class RagResponse(BaseModel):
    answer: str
    sources: list[str]
    elapsed: float
    status: str

# ── 앱 시작 시 초기화 ───────────────────────
@app.on_event("startup")
async def startup():
    logger.info("서버 시작 - 초기화 중...")
    init_pgvector()

    META_CACHE_FILE = "./data/metadata_cache.json"
    import app.core.agent as agent_module
    if os.path.exists(META_CACHE_FILE):
        with open(META_CACHE_FILE, "r", encoding="utf-8") as f:
            agent_module._metadata_cache = json.load(f)
        logger.info(f"메타데이터 캐시 로딩 완료: {len(agent_module._metadata_cache)}개 워크북")
    else:
        load_metadata_cache()
        os.makedirs("./data", exist_ok=True)
        with open(META_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(agent_module._metadata_cache, f, ensure_ascii=False, indent=2)
    logger.info("서버 준비 완료")

# ── 메인 RAG 엔드포인트 ─────────────────────
@app.post("/api/rag", response_model=RagResponse)
async def rag_endpoint(req: RagRequest):
    logger.info(f"요청 수신: {req.query}, {req.mcp_context} / source={req.source}")
    start = time.time()

    try:
        answer = run_agent(
            query=req.query,
            mcp_context=req.mcp_context  
        )
        elapsed = round(time.time() - start, 2)
        return RagResponse(
            answer=answer,
            sources=[],
            elapsed=elapsed,
            status="success"
        )
    except Exception as e:
        logger.error(f"오류: {e}")
        elapsed = round(time.time() - start, 2)
        return RagResponse(
            answer=f"오류가 발생했습니다: {str(e)}",
            sources=[],
            elapsed=elapsed,
            status="error"
        )

# ── 헬스체크 ────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}

# ── 메타데이터 조회 ─────────────────────────
@app.get("/api/metadata")
async def get_metadata():
    import app.core.agent as agent_module
    return {"workbooks": agent_module._metadata_cache}