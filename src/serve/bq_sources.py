"""BQ raw 테이블 → 메모리 DataFrame (서빙용). 로컬 parquet 미경유.

팀 수집 파이프라인 산출 테이블에서 직접 읽는다:
  - gdelt_processed_events → base gdelt_ 피처용 (iso3별 events)
  - economic_daily         → econ_ 피처용
gdelt_titles 는 임베딩 추출(01_extract)/Track1(gkg_feature_builder)이 직접 BQ 조회하므로 여기선 미포함.

컬럼은 수집 파이프라인 스키마와 일치(확인됨):
  gdelt_processed_events: GLOBALEVENTID,SQLDATE,ActionGeo_CountryCode,EventCode,EventRootCode,
                          QuadClass,GoldsteinScale,NumMentions,NumArticles,AvgTone,event_date,iso3
  economic_daily:         date,VIX,WTI,Gold,DXY,STLFSI4,*_pct_change,econ_volatility_proxy
"""
from __future__ import annotations
import os

import pandas as pd

GCP_PROJECT = os.getenv("GCP_PROJECT", "conflict-ew-mvp-20260604")
BQ_DATASET = os.getenv("BQ_DATASET", "conflict_ew")


def _client():
    from google.cloud import bigquery
    return bigquery.Client(project=GCP_PROJECT)


def read_gdelt_events(start: str, end: str) -> dict[str, pd.DataFrame]:
    """gdelt_processed_events → {iso3: events df}. feature_builder._build_gdelt_features 입력 형식."""
    sql = f"""
      SELECT GLOBALEVENTID, event_date, iso3, GoldsteinScale, AvgTone, NumMentions, QuadClass
      FROM `{GCP_PROJECT}.{BQ_DATASET}.gdelt_processed_events`
      WHERE DATE(event_date) BETWEEN DATE('{start}') AND DATE('{end}')
    """
    df = _client().query(sql).to_dataframe(create_bqstorage_client=False)
    return {iso3: g.reset_index(drop=True) for iso3, g in df.groupby("iso3")}


def read_economic(start: str, end: str) -> pd.DataFrame:
    """economic_daily → date + VIX/WTI/Gold/DXY/STLFSI4 (feature_builder._build_economic_features 입력)."""
    sql = f"""
      SELECT date, VIX, WTI, Gold, DXY, STLFSI4
      FROM `{GCP_PROJECT}.{BQ_DATASET}.economic_daily`
      WHERE date BETWEEN DATE('{start}') AND DATE('{end}')
      ORDER BY date
    """
    return _client().query(sql).to_dataframe(create_bqstorage_client=False)
