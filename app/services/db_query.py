import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from app.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class DBQueryService:
    def __init__(self):
        self.conn_params = {
            "host": os.getenv("POSTGRES_HOST"),
            "port": os.getenv("POSTGRES_PORT"),
            "dbname": os.getenv("POSTGRES_DB"),
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD"),
        }

    def get_schema_info(self) -> str:
        """DB 전체 테이블/컬럼 스키마 자동 수집 → LLM 주입용 텍스트"""
        conn = psycopg2.connect(**self.conn_params)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        t.table_name,
                        c.column_name,
                        c.data_type,
                        obj_description(
                            ('"' || t.table_name || '"')::regclass, 'pg_class'
                        ) AS table_comment
                    FROM information_schema.tables t
                    JOIN information_schema.columns c
                        ON t.table_name = c.table_name
                    WHERE t.table_schema = 'public'
                        AND t.table_type = 'BASE TABLE'
                        AND t.table_name NOT IN (
                            'langchain_pg_embedding',
                            'langchain_pg_collection'
                        )
                    ORDER BY t.table_name, c.ordinal_position
                """)
                rows = cur.fetchall()

            if not rows:
                return "사용 가능한 테이블 없음"

            # 테이블별로 그룹핑
            schema = {}
            for row in rows:
                tbl = row["table_name"]
                if tbl not in schema:
                    schema[tbl] = {
                        "comment": row["table_comment"] or "",
                        "columns": []
                    }
                schema[tbl]["columns"].append(
                    f"{row['column_name']}({row['data_type']})"
                )

            # LLM 주입용 텍스트 생성
            lines = []
            for tbl, info in schema.items():
                comment = f" -- {info['comment']}" if info["comment"] else ""
                cols = ", ".join(info["columns"])
                lines.append(f"- {tbl}{comment}: {cols}")

            logger.info(f"스키마 수집 완료: {len(schema)}개 테이블")
            return "\n".join(lines)

        finally:
            conn.close()

    def execute_query(self, sql: str) -> pd.DataFrame:
        """SQL 실행 → DataFrame 반환"""
        logger.info(f"SQL 실행: {sql}")
        conn = psycopg2.connect(**self.conn_params)
        try:
            df = pd.read_sql_query(sql, conn)
            logger.info(f"DB 조회 완료: {len(df)}행 × {len(df.columns)}열")
            return df
        finally:
            conn.close()