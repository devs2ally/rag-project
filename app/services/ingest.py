import os
import time
import hashlib
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.database import get_vector_store
from app.services.tableau import TableauService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_cache: dict = {}
TTL = 3600


def _cache_key(workbook: str, view: str, filters: dict) -> str:
    raw = f"{workbook}:{view}:{sorted(filters.items())}"
    return hashlib.md5(raw.encode()).hexdigest()


def df_to_documents(df: pd.DataFrame, meta: dict) -> list[Document]:
    docs = []
    for i, row in df.iterrows():
        text = " | ".join(
            f"{col}: {val}"
            for col, val in row.items()
            if pd.notna(val)
        )
        docs.append(Document(page_content=text, metadata={**meta, "row": i}))
    return docs


def ingest_tableau_data(
    workbook_name: str,
    view_name: str,
    filters: dict | None = None,
    project_name: str | None = None,
    collection_name: str = None,
    force_refresh: bool = False
):
    filters = filters or {}
    # 프로젝트별 컬렉션 분리
    collection = collection_name or (
        f"tableau_{project_name or workbook_name}"
        .replace(" ", "_")
        .replace("/", "_")
        .lower()
    )
    key = _cache_key(workbook_name, view_name, filters)

    if not force_refresh and key in _cache:
        if time.time() - _cache[key] < TTL:
            logger.info(f"[Cache HIT] {workbook_name}/{view_name}")
            return get_vector_store(collection)

    logger.info(f"[Ingest 시작] {project_name}/{workbook_name}/{view_name} → 컬렉션: {collection}")

    tableau = TableauService()
    df = tableau.get_view_dataframe(
        workbook_name, view_name, filters, project_name
    )

    docs = df_to_documents(
        df,
        meta={"project": project_name, "workbook": workbook_name, "view": view_name}
    )

    if len(docs) > 200:
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        docs = splitter.split_documents(docs)

    store = get_vector_store(collection)
    store.add_documents(docs)
    _cache[key] = time.time()

    logger.info(f"[Ingest 완료] {len(docs)}개 청크 → {collection}")
    return store


def ingest_dataframe(
    df: pd.DataFrame,
    meta: dict,
    collection_name: str = None,
):
    # 소스별 컬렉션 분리
    source = meta.get("source", "default")
    table = meta.get("table", "")
    collection = collection_name or (
        f"db_{table or source}"
        .replace(" ", "_")
        .lower()
    )

    docs = df_to_documents(df, meta=meta)

    if len(docs) > 200:
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        docs = splitter.split_documents(docs)

    store = get_vector_store(collection)
    store.add_documents(docs)
    logger.info(f"[DB Ingest 완료] {len(docs)}개 청크 → {collection}")
    return store