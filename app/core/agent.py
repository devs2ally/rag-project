import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.tableau import TableauService
from app.services.ingest import ingest_tableau_data, ingest_dataframe
from app.services.db_query import DBQueryService
from app.services.retriever import get_hybrid_retriever_docs, rerank_documents
from app.utils.logger import get_logger
from app.services.database import get_vector_store, get_vector_collections

load_dotenv()
logger = get_logger(__name__)

_metadata_cache: list[dict] = []
_schema_cache: str = ""


def load_metadata_cache():
    global _metadata_cache
    tableau = TableauService()
    _metadata_cache = tableau.get_all_metadata()
    logger.info(f"메타데이터 캐시 로딩 완료: {len(_metadata_cache)}개 워크북")


def load_schema_cache():
    global _schema_cache
    db = DBQueryService()
    _schema_cache = db.get_schema_info()
    logger.info(f"DB 스키마 캐시 로딩 완료")


def get_metadata_as_text() -> str:
    lines = []
    for item in _metadata_cache:
        views_str = ", ".join(item["views"]) if item["views"] else "뷰 없음"
        lines.append(
            f"- 프로젝트: {item['project']} | "
            f"워크북: {item['workbook']} | "
            f"뷰 목록: {views_str}"
        )
    return "\n".join(lines)


SOURCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
당신은 데이터 소스 라우팅 전문가입니다.

규칙:
1. DB 테이블 목록에 관련 데이터가 있으면 반드시 "db"
2. Tableau에만 있는 시각화/대시보드 데이터면 "tableau"
3. DB를 항상 우선으로 판단

[DB 테이블 목록]
{schema}

[Tableau 데이터 목록]
{metadata}

JSON만 응답. 다른 텍스트 금지.
{{"source": "db" 또는 "tableau", "reason": "한 줄 이유"}}
"""),
    ("human", "질의: {query}")
])

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
당신은 Tableau 데이터 라우팅 전문가입니다.
오늘 날짜: {today}

[사용 가능한 Tableau 데이터 목록]
{metadata}

반드시 아래 JSON 형식으로만 응답하세요.
{{
  "project": "프로젝트명",
  "workbook": "워크북명",
  "view": "뷰명",
  "filters": {{}},
  "intent": "사용자가 원하는 것 한 줄 요약"
}}
"""),
    ("human", "질의: {query}")
])

SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
당신은 SQL 전문가입니다.
오늘 날짜: {today}
현재 년도: {year}, 현재 월: {month}, 현재 일: {day}

[사용 가능한 DB 테이블]
{schema}

[날짜 추론 규칙]  
- "이번달" → WHERE 년 = {year} AND 월 = {month}
- "저번달" → 월이 1이면 WHERE 년 = {year}-1 AND 월 = 12, 아니면 WHERE 년 = {year} AND 월 = {month}-1
- "올해" → WHERE 년 = {year}
- "작년" → WHERE 년 = {year}-1

[집계 추론 규칙]  
- "연도별" → GROUP BY 년 ORDER BY 년
- "월별" → GROUP BY 년, 월 ORDER BY 년, 월
- "지점별" → GROUP BY 지점명
- "합계", "총합" → SUM()
- "평균" → AVG()
- "순위" → ORDER BY 매출액 DESC

반드시 아래 JSON 형식으로만 응답하세요.
{{
  "sql": "SELECT ...",
  "intent": "사용자가 원하는 것 한 줄 요약"
}}

주의사항:
- SELECT만 허용 (INSERT/UPDATE/DELETE 금지)
- 결과는 최대 1000행으로 제한 (LIMIT 1000)
- 날짜 필터가 있으면 반드시 적용
"""),
    ("human", "질의: {query}")
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
당신은 비즈니스 데이터 분석 전문가입니다.
아래 [데이터]를 분석하여 사용자 질문에 한국어로 명확하게 답하세요.

규칙:
- 데이터에 있는 수치는 반드시 포함하여 답변
- 데이터를 그룹별로 정리하여 비교 형태로 제시
- 없는 데이터는 "해당 데이터 없음"으로 표기
- 수치는 단위 포함하여 읽기 쉽게 표현 (TON, %, 원 등)
- 비교 질문이면 표 형태로 정리
- "데이터가 없습니다", "정보가 포함되어 있지 않습니다" 같은 부정적 표현 금지
- 있는 데이터를 최대한 활용해서 인사이트 제공
- db 조회시 생성일은 언급하지 말 것

[MCP 분석 결과]
{mcp_context}

[데이터]
{context}
"""),
    ("human", "{question}")
])


# def get_llm(temperature: float = 0.1) -> ChatGroq:
#     return ChatGroq(
#         model="llama-3.1-8b-instant",
#         temperature=temperature,
#         api_key=os.getenv("GROQ_API_KEY")
#     )

def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",  
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY")
    )


def parse_json(raw: str) -> dict:
    try:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error(f"JSON 파싱 실패: {raw}")
        raise ValueError(f"JSON 파싱 실패: {raw}")


def detect_source(query: str) -> str:
    query_lower = query.lower()

    # ── 1순위: 벡터DB 컬렉션 키워드 매칭 ──
    from app.services.database import get_vector_collections, COLLECTION_KEYWORDS
    vector_collections = get_vector_collections()

    for col in vector_collections:
        if col.startswith("tableau_"):
            continue

        # 등록된 키워드 먼저 확인
        registered_keywords = COLLECTION_KEYWORDS.get(col, [])
        for kw in registered_keywords:
            if kw in query_lower:
                logger.info(f"[소스판단] 컬렉션 키워드 매칭 → vector ({col}, {kw})")
                return f"vector:{col}"

        # 등록 안 된 컬렉션은 컬렉션명으로 매칭
        col_clean = col.lower()
        for prefix in ["csv_", "db_", "tableau_"]:
            col_clean = col_clean.replace(prefix, "")
        keywords = [k for k in col_clean.split("_") if len(k) > 1]
        for kw in keywords:
            if kw in query_lower:
                logger.info(f"[소스판단] 컬렉션명 매칭 → vector ({col})")
                return f"vector:{col}"

    # ── 2순위: DB 테이블명 직접 언급 ──
    schema_table_names = []
    for line in _schema_cache.split("\n"):
        if line.strip().startswith("-"):
            table_name = line.strip().lstrip("-").strip().split(":")[0].split("--")[0].strip()
            if table_name:
                schema_table_names.append(table_name.lower())

    for table in schema_table_names:
        if table in query_lower:
            logger.info(f"[소스판단] 테이블명 직접 언급 → db ({table})")
            return "db"

    # ── 3순위: DB 전용 키워드 ──
    db_keywords = [
        "건강검진", "유연근무", "자기계발", "경조사", "식당",
        "보안", "근속", "워크샵", "복장", "주차",
        "강남점", "홍대점", "신촌점", "커피", "스무디",
        "연도별", "월별", "일별", "판매수량"
    ]
    for kw in db_keywords:
        if kw in query_lower:
            logger.info(f"[소스판단] DB 키워드 감지 → db ({kw})")
            return "db"

    # ── 4순위: Tableau 전용 키워드 ──
    tableau_keywords = [
        "대시보드", "dashboard", "워크북", "tableau", "태블로"
    ]
    for kw in tableau_keywords:
        if kw in query_lower:
            logger.info(f"[소스판단] Tableau 키워드 감지 → tableau ({kw})")
            return "tableau"

    # ── 5순위: LLM 판단 ──
    logger.info("[소스판단] 키워드 불명확 → LLM 판단")
    if not _schema_cache or _schema_cache == "사용 가능한 테이블 없음":
        return "tableau"

    llm = get_llm(temperature=0)
    chain = SOURCE_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "query": query,
        "metadata": get_metadata_as_text(),
        "schema": _schema_cache
    })
    result = parse_json(raw)
    source = result.get("source", "tableau")
    logger.info(f"[소스판단] LLM 판단: {source} / 이유: {result.get('reason')}")
    return source


def analyze_query(query: str) -> dict:
    llm = get_llm(temperature=0)
    chain = ANALYSIS_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "query": query,
        "metadata": get_metadata_as_text(),
        "today": datetime.now().strftime("%Y-%m-%d")
    })
    result = parse_json(raw)
    logger.info(f"Tableau 라우팅: {result}")
    return result


def generate_sql(query: str) -> dict:
    now = datetime.now() 
    llm = get_llm(temperature=0)
    chain = SQL_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "query": query,
        "schema": _schema_cache,
        "today": now.strftime("%Y-%m-%d"),
        "year": now.year,   
        "month": now.month, 
        "day": now.day      
    })
    result = parse_json(raw)
    logger.info(f"SQL 생성: {result['sql']}")
    return result


def run_agent(query: str, mcp_context: str = None) -> str:
    total_start = time.time()
    logger.info(f"=== 질의 수신: {query} ===")

    meta_keywords = ["어떤", "뭐가", "무슨", "조회 가능", "데이터 목록"]
    if any(kw in query for kw in meta_keywords):
        return f"현재 조회 가능한 데이터 목록입니다:\n\n{get_metadata_as_text()}"

    # ① 소스 판단
    t = time.time()
    source = detect_source(query)
    logger.info(f"[①완료] 소스 판단: {source} ({time.time()-t:.1f}초)")

    # ② 데이터 조회 + 벡터화
    t = time.time()
    if source == "db":
        sql_result = generate_sql(query)
        sql = sql_result["sql"]
        intent = sql_result.get("intent", query)
        logger.info(f"실행 SQL: {sql}")

        db = DBQueryService()
        df = db.execute_query(sql)
        logger.info(f"DB 조회 결과: {len(df)}행")
        logger.info(f"컬럼: {list(df.columns)}")

        if df.empty:
            logger.warning("SQL 결과 없음")
            return "조회된 데이터가 없습니다. 질의 내용을 다시 확인해주세요."

        # ← DB는 벡터 검색 스킵하고 바로 컨텍스트 생성
        context = df.to_string(index=False)
        logger.info(f"DB 컨텍스트:\n{context}")

        # ⑤ 바로 답변 생성
        t = time.time()
        llm = get_llm(temperature=0.1)
        prompt = ANSWER_PROMPT.invoke({
            "context": context,
            "question": intent,
            "mcp_context": mcp_context or "없음"
        })
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info(f"[완료] 답변 생성 ({time.time()-t:.1f}초)")
        logger.info(f"=== 응답 완료 (총 {time.time()-total_start:.1f}초) ===")
        return answer
    elif source.startswith("vector:"):
        # 벡터DB 직접 검색
        collection_name = source.split("vector:")[1]
        logger.info(f"벡터DB 직접 검색: {collection_name}")
        vector_store = get_vector_store(collection_name)
        intent = query

        # 바로 검색 → 답변
        fresh_docs = vector_store.similarity_search(intent, k=20)
        context = "\n\n".join(d.page_content for d in fresh_docs)
        logger.info(f"[②완료] 벡터DB 검색 ({time.time()-t:.1f}초)")

        t = time.time()
        llm = get_llm(temperature=0.1)
        prompt = ANSWER_PROMPT.invoke({
            "context": context,
            "question": intent,
            "mcp_context": mcp_context or "없음"
        })
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info(f"[완료] 답변 생성 ({time.time()-t:.1f}초)")
        logger.info(f"=== 응답 완료 (총 {time.time()-total_start:.1f}초) ===")
        return answer
    else:
        routing = analyze_query(query)
        workbook = routing["workbook"]
        view = routing["view"]
        filters = routing.get("filters", {})
        intent = routing.get("intent", query)

        # 필터 유효성 검증
        INVALID_FILTER_VALUES = [
            "도시명", "지역명", "날짜", "값", "value", "name",
            "필터값", "filter", "string", "text", "해당없음"
        ]
        filters = {
            k: v for k, v in filters.items()
            if v and v not in INVALID_FILTER_VALUES
        }

        # 프로젝트+워크북으로 뷰 검증
        project = routing.get("project")
        matched = next(
            (item for item in _metadata_cache
             if item["workbook"] == workbook and item["project"] == project),
            None
        )
        if not matched:
            matched = next(
                (item for item in _metadata_cache if item["workbook"] == workbook), None
            )
        if matched and view not in matched["views"]:
            view = matched["views"][0]
            logger.warning(f"뷰 대체: {view}")

        vector_store = ingest_tableau_data(
            workbook_name=workbook,
            view_name=view,
            filters=filters,
            project_name=project
        )
    logger.info(f"[②완료] 데이터 조회+벡터화 ({time.time()-t:.1f}초)")

    # ③ 하이브리드 검색 (현재 컬렉션에서만)
    t = time.time()
    fresh_docs = vector_store.similarity_search(intent, k=20)
    logger.info(f"검색 문서 수: {len(fresh_docs)}개")
    logger.info(f"상위 문서: {[d.page_content[:50] for d in fresh_docs[:3]]}")

    raw_docs = get_hybrid_retriever_docs(
        vector_store=vector_store,
        docs=fresh_docs,
        query=intent,
        k=10
    )
    logger.info(f"[③완료] 하이브리드 검색 ({time.time()-t:.1f}초)")

    # ④ 상위 5개 슬라이싱 (한국어라 CrossEncoder 스킵)
    t = time.time()
    reranked_docs = raw_docs[:5]
    context = "\n\n".join(d.page_content for d in reranked_docs)
    logger.info(f"[④완료] 상위 {len(reranked_docs)}개 선택 ({time.time()-t:.1f}초)")
    logger.info(f"최종 컨텍스트:\n{context[:300]}")

    # ⑤ 답변 생성
    t = time.time()
    llm = get_llm(temperature=0.1)
    prompt = ANSWER_PROMPT.invoke({
        "context": context,
        "question": intent,
        "mcp_context": mcp_context or "없음"
    })
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    logger.info(f"[⑤완료] 답변 생성 ({time.time()-t:.1f}초)")
    logger.info(f"=== 응답 완료 (총 {time.time()-total_start:.1f}초) ===")
    return answer