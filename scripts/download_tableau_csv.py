"""
Tableau 뷰 데이터를 CSV로 저장
python scripts/download_tableau_csv.py --project 바나프레소 --workbook 대시보드 --view 대시보드
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.services.tableau import TableauService
from app.utils.logger import get_logger

logger = get_logger("download_csv")


def download_csv(project: str, workbook: str, view: str):
    tableau = TableauService()
    df = tableau.get_view_dataframe(
        workbook_name=workbook,
        view_name=view,
        project_name=project
    )

    os.makedirs("./data", exist_ok=True)
    filename = f"./data/{project}_{workbook}_{view}.csv".replace(" ", "_")
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    logger.info(f"✅ CSV 저장 완료: {filename} ({len(df)}행)")
    print(f"\n저장됨: {filename}")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--workbook", type=str, required=True)
    parser.add_argument("--view", type=str, required=True)
    args = parser.parse_args()

    download_csv(args.project, args.workbook, args.view)