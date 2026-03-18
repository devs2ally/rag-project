from langchain_core.documents import Document
from langchain_community.vectorstores import PGVector
from app.utils.logger import get_logger

logger = get_logger(__name__)

_cross_encoder_cache = None


def get_cross_encoder():
    global _cross_encoder_cache
    if _cross_encoder_cache is not None:
        return _cross_encoder_cache
    from sentence_transformers import CrossEncoder
    logger.info("CrossEncoder 모델 로드 중...")
    _cross_encoder_cache = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    logger.info("CrossEncoder 모델 로드 완료")
    return _cross_encoder_cache


def get_hybrid_retriever_docs(
    vector_store: PGVector,
    docs: list[Document],
    query: str,
    k: int = 10
) -> list[Document]:
    """벡터 검색만 사용 (BM25 타입 오류 이슈로 제외)"""
    try:
        results = vector_store.similarity_search(str(query), k=k)
        logger.info(f"벡터 검색 완료: {len(results)}개")
        return results
    except Exception as e:
        logger.warning(f"벡터 검색 실패: {e}")
        return []


def rerank_documents(
    query: str,
    docs: list[Document],
    top_k: int = 5
) -> list[Document]:
    if not docs:
        return docs
    try:
        model = get_cross_encoder()
        pairs = [(str(query), str(doc.page_content)) for doc in docs]
        scores = model.predict(pairs)
        scored_docs = sorted(
            zip(scores, docs),
            key=lambda x: float(x[0]),
            reverse=True
        )
        reranked = [doc for _, doc in scored_docs[:top_k]]
        logger.info(f"리랭킹 완료: {len(docs)}개 → 상위 {len(reranked)}개")
        return reranked
    except Exception as e:
        logger.warning(f"리랭킹 실패: {e}")
        return docs[:top_k]