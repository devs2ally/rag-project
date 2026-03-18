import os
import psycopg2
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from app.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

_embeddings_cache = None
_collections_cache: list[str] = []


# 컬렉션별 관련 키워드 매핑
COLLECTION_KEYWORDS = {
    "csv_mv_zmm_m08_mart": [
        "입고", "입고계획", "내수", "수입", "h/c", "cr", "egi", "gi", "gl",
        "al외", "월집계", "수량", "금액", "비율", "동국cm", "동국 cm"
    ],
    "db_sales_data": [
        "강남점", "홍대점", "신촌점", "커피", "스무디", "매출", "판매수량"
    ],
    "db_company_test": [
        "건강검진", "유연근무", "자기계발", "경조사", "식당",
        "보안", "근속", "워크샵", "복장", "주차", "복지", "규정"
    ]
}


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache
    model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    logger.info(f"임베딩 모델 로드: {model_name}")
    _embeddings_cache = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return _embeddings_cache


def get_vector_store(collection_name: str = None) -> PGVector:
    collection = collection_name or os.getenv("COLLECTION_NAME", "rag_test_v1")
    connection_string = os.getenv("DB_URL").replace(
        "postgresql+psycopg://", "postgresql+psycopg2://"
    )
    store = PGVector(
        collection_name=collection,
        connection_string=connection_string,
        embedding_function=get_embeddings(),
    )
    logger.info(f"pgvector 연결: {collection}")
    return store


def get_vector_collections() -> list[str]:
    global _collections_cache
    if _collections_cache:
        return _collections_cache

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM langchain_pg_collection")
            rows = cur.fetchall()
            _collections_cache = [row[0] for row in rows]
            logger.info(f"벡터DB 컬렉션 목록: {_collections_cache}")
            return _collections_cache
    except Exception as e:
        logger.warning(f"컬렉션 조회 실패: {e}")
        return []
    finally:
        conn.close()


def init_pgvector():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            logger.info("pgvector 익스텐션 활성화 완료")
    finally:
        conn.close()