"""
CSV 파일 벡터화 스크립트
사용법:
  # data/ 폴더 전체 CSV 벡터화
  python scripts/ingest_csv.py --all

  # 특정 파일만
  python scripts/ingest_csv.py --file data/sales.csv

  # 특정 파일 + 컬렉션 지정
  python scripts/ingest_csv.py --file data/sales.csv --collection my_collection
"""

import sys
import os
import argparse
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from app.services.ingest import ingest_dataframe
from app.services.database import init_pgvector
from app.utils.logger import get_logger

logger = get_logger("ingest_csv")


def ingest_csv_file(filepath: str, collection_name: str = None):
    filename = os.path.basename(filepath)
    logger.info(f"CSV 로드 중: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="euc-kr")

    logger.info(f"로드 완료: {len(df)}행 × {len(df.columns)}열")
    logger.info(f"컬럼: {list(df.columns)}")

    df = df.dropna(how="all")

    if df.empty:
        logger.warning(f"데이터 없음 스킵: {filename}")
        return

    # 파일명 기반 컬렉션명 생성
    col_name = collection_name or (
        "csv_" + os.path.splitext(filename)[0].lower().replace(" ", "_")
    )
    logger.info(f"컬렉션명: {col_name}")

    ingest_dataframe(
        df=df,
        meta={"source": "csv", "filename": filename},
        collection_name=col_name  # ← 여기가 핵심
    )
    logger.info(f"✅ 완료: {filename} → {col_name}")


def ingest_all_csv(data_dir: str = "./data", collection_name: str = None):
    """data/ 폴더 전체 CSV 벡터화 (하위 폴더 포함)"""
    # ** 로 하위 폴더까지 전부 탐색
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

    if not csv_files:
        logger.warning(f"CSV 파일 없음: {data_dir}")
        return

    logger.info(f"총 {len(csv_files)}개 CSV 파일 발견")
    success, fail = 0, 0

    for filepath in csv_files:
        try:
            ingest_csv_file(filepath, collection_name)
            success += 1
        except Exception as e:
            fail += 1
            logger.error(f"❌ 실패: {filepath} → {e}")

    logger.info(f"=== 완료 | 성공: {success} / 실패: {fail} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV 벡터화 스크립트")
    parser.add_argument("--all", action="store_true", help="data/ 전체 CSV")
    parser.add_argument("--file", type=str, help="특정 CSV 파일 경로")
    parser.add_argument("--collection", type=str, help="컬렉션 이름 (기본값: .env의 COLLECTION_NAME)")
    args = parser.parse_args()

    # pgvector 초기화
    init_pgvector()

    if args.file:
        ingest_csv_file(args.file, args.collection)
    elif args.all:
        ingest_all_csv(collection_name=args.collection)
    else:
        parser.print_help()