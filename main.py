import os
import sys
import json
from dotenv import load_dotenv
from app.services.database import init_pgvector
from app.core.agent import load_metadata_cache, load_schema_cache, run_agent
from app.utils.logger import get_logger

load_dotenv()
logger = get_logger("main")
META_CACHE_FILE = "./data/metadata_cache.json"

def load_or_fetch_metadata():
    import app.core.agent as agent_module
    if os.path.exists(META_CACHE_FILE):
        logger.info("메타데이터 파일 캐시 로딩...")
        with open(META_CACHE_FILE, "r", encoding="utf-8") as f:
            agent_module._metadata_cache = json.load(f)
        logger.info(f"캐시 로딩 완료: {len(agent_module._metadata_cache)}개 워크북")
    else:
        logger.info("Tableau 메타데이터 최초 로딩 중...")
        load_metadata_cache()
        os.makedirs("./data", exist_ok=True)
        with open(META_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(agent_module._metadata_cache, f, ensure_ascii=False, indent=2)
        logger.info("메타데이터 파일 캐시 저장 완료")
        
    load_schema_cache()

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cli"

    init_pgvector()
    load_or_fetch_metadata()

    # ← 모델 미리 워밍업
    logger.info("모델 워밍업 중...")
    from app.services.database import get_embeddings
    from app.services.retriever import get_cross_encoder
    get_embeddings()
    get_cross_encoder()
    logger.info("모델 워밍업 완료")

    # ── API 서버 모드 ──
    if mode == "server":
        import uvicorn
        logger.info("API 서버 모드로 시작")
        uvicorn.run(
            "app.web.server:app",
            host="0.0.0.0",
            port=8000,
            reload=False
        )

    # ── CLI 모드 ──
    else:
        print("\nRAG 시스템 준비 완료!")
        print("💬 질문을 입력하세요. 종료하려면 'exit' 또는 'quit' 입력")
        print("💡 메타데이터 새로고침: 'refresh' 입력\n")

        while True:
            try:
                query = input("질문 > ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit", "종료"):
                    print("종료합니다.")
                    break
                if query.lower() == "refresh":
                    os.remove(META_CACHE_FILE)
                    load_or_fetch_metadata()
                    print("메타데이터 새로고침 완료\n")
                    continue
                answer = run_agent(query)
                print(f"\n답변:\n{answer}\n")
            except KeyboardInterrupt:
                print("\n종료합니다.")
                break
            except Exception as e:
                print(f"\n오류: {e}\n")