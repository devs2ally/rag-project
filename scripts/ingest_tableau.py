"""
Tableau 데이터 수동 벡터화 스크립트
사용법:
  # 전체 워크북 벡터화
  python scripts/ingest_tableau.py --all

  # 특정 프로젝트만
  python scripts/ingest_tableau.py --project 바나프레소

  # 특정 워크북/뷰만
  python scripts/ingest_tableau.py --workbook 대시보드 --view 대시보드 --project 바나프레소
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.services.tableau import TableauService
from app.services.ingest import ingest_tableau_data
from app.services.database import init_pgvector
from app.utils.logger import get_logger

logger = get_logger("ingest_tableau")


def ingest_all(project_filter: str = None):
    """전체 또는 특정 프로젝트 워크북 전부 벡터화"""
    tableau = TableauService()
    metadata = tableau.get_all_metadata()

    # 프로젝트 필터 적용
    if project_filter:
        metadata = [m for m in metadata if m["project"] == project_filter]
        logger.info(f"프로젝트 필터: {project_filter} → {len(metadata)}개 워크북")

    total = sum(len(m["views"]) for m in metadata)
    logger.info(f"총 {len(metadata)}개 워크북 / {total}개 뷰 벡터화 시작")

    success, fail = 0, 0
    for item in metadata:
        project = item["project"]
        workbook = item["workbook"]
        views = item["views"]

        if not views:
            logger.warning(f"뷰 없음 스킵: {project}/{workbook}")
            continue

        for view in views:
            try:
                logger.info(f"처리 중: {project}/{workbook}/{view}")
                ingest_tableau_data(
                    workbook_name=workbook,
                    view_name=view,
                    project_name=project,
                    force_refresh=True  # 캐시 무시하고 항상 새로 받기
                )
                success += 1
                logger.info(f"✅ 완료: {project}/{workbook}/{view}")
            except Exception as e:
                fail += 1
                logger.error(f"❌ 실패: {project}/{workbook}/{view} → {e}")

    logger.info(f"=== 완료 | 성공: {success} / 실패: {fail} ===")


def ingest_single(workbook: str, view: str, project: str = None):
    """특정 워크북/뷰 하나만 벡터화"""
    logger.info(f"단일 벡터화: {project}/{workbook}/{view}")
    try:
        ingest_tableau_data(
            workbook_name=workbook,
            view_name=view,
            project_name=project,
            force_refresh=True
        )
        logger.info(f"완료: {workbook}/{view}")
    except Exception as e:
        logger.error(f"❌ 실패: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tableau 데이터 벡터화 스크립트")
    parser.add_argument("--all", action="store_true", help="전체 벡터화")
    parser.add_argument("--project", type=str, help="특정 프로젝트만")
    parser.add_argument("--workbook", type=str, help="특정 워크북")
    parser.add_argument("--view", type=str, help="특정 뷰")
    args = parser.parse_args()

    # pgvector 초기화
    init_pgvector()

    if args.workbook and args.view:
        # 단일 벡터화
        ingest_single(
            workbook=args.workbook,
            view=args.view,
            project=args.project
        )
    elif args.all or args.project:
        # 전체 또는 프로젝트별
        ingest_all(project_filter=args.project)
    else:
        parser.print_help()