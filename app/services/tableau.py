import os
import io
# from datetime import datetime
import pandas as pd
import tableauserverclient as TSC
from dotenv import load_dotenv
from app.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class TableauService:
    def __init__(self):
        self.server_url = os.getenv("TABLEAU_SERVER_URL")
        self.token_name = os.getenv("TABLEAU_TOKEN_NAME")
        self.token_value = os.getenv("TABLEAU_TOKEN_VALUE")
        self.site_id = os.getenv("TABLEAU_SITE_ID", "")
        self.server = TSC.Server(self.server_url, use_server_version=True)

        if "@" in self.token_name:
            self.auth = TSC.TableauAuth(
                username=self.token_name,
                password=self.token_value,
                site_id=self.site_id
            )
        else:
            self.auth = TSC.PersonalAccessTokenAuth(
                token_name=self.token_name,
                personal_access_token=self.token_value,
                site_id=self.site_id
            )

    def get_all_metadata(self) -> list[dict]:
        metadata = []
        with self.server.auth.sign_in(self.auth):
                
            # 전체 페이지 다 가져오기
            request_opts = TSC.RequestOptions(pagesize=100)
            all_wbs, pagination = self.server.workbooks.get(request_opts)
            
            # 페이지가 더 있으면 전부 가져오기
            total = pagination.total_available
            fetched = len(all_wbs)
            
            while fetched < total:
                request_opts.pagenumber += 1
                next_wbs, _ = self.server.workbooks.get(request_opts)
                all_wbs.extend(next_wbs)
                fetched += len(next_wbs)
            
            logger.info(f"전체 워크북 {len(all_wbs)}개 로드")
            
            for wb in all_wbs:
                try:
                    self.server.workbooks.populate_views(wb)
                    views = [v.name for v in wb.views]
                except Exception:
                    views = []

                metadata.append({
                    "project": wb.project_name,
                    "workbook": wb.name,
                    "workbook_id": wb.id,
                    "views": views
                })

            logger.info(f"메타데이터 로드 완료: {len(metadata)}개 워크북")
            return metadata

    def get_view_dataframe(
        self,
        workbook_name: str,
        view_name: str,
        filters: dict | None = None,
        project_name: str | None = None
    ) -> pd.DataFrame:
        filters = filters or {}
        with self.server.auth.sign_in(self.auth):
            all_wbs, _ = self.server.workbooks.get()

            if project_name:
                wb = next(
                    (w for w in all_wbs
                    if w.name == workbook_name and w.project_name == project_name),
                    None
                )
            else:
                wb = next((w for w in all_wbs if w.name == workbook_name), None)

            if not wb:
                raise ValueError(f"워크북 '{workbook_name}' (프로젝트: {project_name}) 없음")

            self.server.workbooks.populate_views(wb)
            view = next((v for v in wb.views if v.name == view_name), None)
            if not view:
                available = [v.name for v in wb.views]
                raise ValueError(f"뷰 '{view_name}' 없음. 사용 가능: {available}")

            options = TSC.CSVRequestOptions()
            for k, v in filters.items():
                options.vf(k, v)

            self.server.views.populate_csv(view, options)
            csv_bytes = b"".join(view.csv)
            df = pd.read_csv(io.BytesIO(csv_bytes))
            logger.info(f"데이터 로드: {len(df)}행 × {len(df.columns)}열")
            return df