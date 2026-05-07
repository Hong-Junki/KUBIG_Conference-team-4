"""
ACLED 수집기: OAuth 2.0 인증 + 페이지네이션 + 체크포인트
"""

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from .config import (
    ACLED_BASE_URL,
    ACLED_EVENT_TYPES,
    ACLED_PAGE_SIZE,
    ACLED_PASSWORD,
    ACLED_USERNAME,
    COLLECT_END,
    COLLECT_START,
    COUNTRIES,
)
from .utils import Checkpoint, get_logger, retry

logger = get_logger(__name__)

# 저장 경로
RAW_DIR = Path("input/raw/acled")
CKPT_PATH = RAW_DIR / ".ckpt_historical.json"

# OAuth 엔드포인트
OAUTH_URL = "https://acleddata.com/oauth/token"

# 응답에서 수집할 필드 (피처 + 라벨 생성에 필요한 것만)
RESPONSE_FIELDS = "|".join([
    "event_id_cnty",
    "event_date",
    "year",
    "disorder_type",
    "event_type",
    "sub_event_type",
    "actor1",
    "actor2",
    "inter1",
    "inter2",
    "country",
    "iso",
    "admin1",
    "latitude",
    "longitude",
    "fatalities",
    "timestamp",
])


# ──────────────────────────────────────────────
# OAuth 토큰 관리
# ──────────────────────────────────────────────

class TokenManager:
    """access_token 자동 발급 및 만료 전 갱신."""

    def __init__(self):
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._expires_at: datetime | None = None

    def get_token(self) -> str:
        if self._is_expired():
            if self._refresh_token:
                self._refresh()
            else:
                self._fetch_new()
        return self._access_token

    def _is_expired(self) -> bool:
        if not self._access_token or not self._expires_at:
            return True
        # 만료 5분 전에 갱신
        return datetime.utcnow() >= self._expires_at - timedelta(minutes=5)

    @retry(max_attempts=3, base_delay=5.0, exceptions=(requests.RequestException,))
    def _fetch_new(self) -> None:
        logger.info("ACLED OAuth 토큰 신규 발급 중...")
        resp = requests.post(OAUTH_URL, data={
            "username":   ACLED_USERNAME,
            "password":   ACLED_PASSWORD,
            "grant_type": "password",
            "client_id":  "acled",
        }, timeout=30)
        resp.raise_for_status()
        self._store(resp.json())
        logger.info("토큰 발급 완료 (24시간 유효)")

    @retry(max_attempts=3, base_delay=5.0, exceptions=(requests.RequestException,))
    def _refresh(self) -> None:
        logger.info("ACLED OAuth 토큰 갱신 중...")
        resp = requests.post(OAUTH_URL, data={
            "refresh_token": self._refresh_token,
            "grant_type":    "refresh_token",
            "client_id":     "acled",
        }, timeout=30)
        if resp.status_code == 401:
            # refresh_token 만료 → 신규 발급
            logger.warning("refresh_token 만료. 신규 발급으로 전환.")
            self._refresh_token = None
            self._fetch_new()
            return
        resp.raise_for_status()
        self._store(resp.json())
        logger.info("토큰 갱신 완료")

    def _store(self, data: dict) -> None:
        self._access_token  = data["access_token"]
        self._refresh_token = data.get("refresh_token")
        expires_in = data.get("expires_in", 86400)
        self._expires_at = datetime.utcnow() + timedelta(seconds=expires_in)


# 모듈 레벨 싱글톤 (같은 프로세스 내 토큰 재사용)
_token_manager = TokenManager()


# ──────────────────────────────────────────────
# 수집 함수
# ──────────────────────────────────────────────

@retry(max_attempts=5, base_delay=2.0, exceptions=(requests.RequestException,))
def _fetch_page(params: dict) -> dict:
    """단일 페이지 요청."""
    headers = {"Authorization": f"Bearer {_token_manager.get_token()}"}
    resp = requests.get(ACLED_BASE_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise ValueError(f"ACLED API 오류: {data.get('messages')}")
    return data


def fetch_country_period(
    country_name: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    특정 국가 + 기간의 ACLED 이벤트를 전체 페이지네이션하여 반환.

    Returns:
        pd.DataFrame: 수집된 이벤트 (비어있으면 빈 DataFrame)
    """
    event_type_filter = "|".join(ACLED_EVENT_TYPES)
    base_params = {
        "country":          country_name,
        "event_type":       event_type_filter,
        "event_date":       f"{start}|{end}",
        "event_date_where": "BETWEEN",
        "fields":           RESPONSE_FIELDS,
        "limit":            ACLED_PAGE_SIZE,
        "_format":          "json",
    }

    all_records: list[dict] = []
    page = 1

    while True:
        params = {**base_params, "page": page}
        logger.debug(f"  {country_name} 페이지 {page} 요청 중...")
        data = _fetch_page(params)

        records = data.get("data", [])
        if not records:
            break

        all_records.extend(records)
        logger.debug(f"  {country_name} 페이지 {page}: {len(records)}건 수집")

        if len(records) < ACLED_PAGE_SIZE:
            break  # 마지막 페이지
        page += 1
        time.sleep(0.5)  # 레이트 리밋 여유

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["event_date"] = pd.to_datetime(df["event_date"], utc=True)
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).astype(int)
    return df


def collect_historical(
    start: date = COLLECT_START,
    end: date = COLLECT_END,
    countries: list[dict] | None = None,
) -> None:
    """
    과거 데이터 일괄 수집.
    체크포인트 기반 idempotent — 완료된 국가는 재수집하지 않음.

    저장 경로: input/raw/acled/{iso3}.parquet
    체크포인트: input/raw/acled/.ckpt_historical.json
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(CKPT_PATH)

    target = countries or COUNTRIES
    total = len(target)

    for idx, country in enumerate(target, 1):
        name = country["name"]
        iso3 = country["iso3"]
        key  = f"{iso3}_{start}_{end}"

        if ckpt.is_done(key):
            logger.info(f"[{idx}/{total}] {name} — 이미 완료, 건너뜀")
            continue

        logger.info(f"[{idx}/{total}] {name} 수집 시작 ({start} ~ {end})")
        try:
            # acled_names가 있으면 여러 이름으로 쿼리 후 합산 (예: Palestine = West Bank + Gaza Strip)
            acled_names = country.get("acled_names", [name])
            frames = []
            for acled_name in acled_names:
                df_part = fetch_country_period(acled_name, start, end)
                if not df_part.empty:
                    frames.append(df_part)
            df = pd.concat(frames).drop_duplicates("event_id_cnty") if frames else pd.DataFrame()
            out_path = RAW_DIR / f"{iso3}.parquet"

            if df.empty:
                logger.warning(f"  {name}: 수집 데이터 없음")
                ckpt.mark_done(key, {"rows": 0, "path": str(out_path)})
                continue

            # 기존 파일이 있으면 병합 (증분 수집 대비)
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                df = pd.concat([existing, df]).drop_duplicates("event_id_cnty")

            df.to_parquet(out_path, index=False)
            ckpt.mark_done(key, {"rows": len(df), "path": str(out_path)})
            logger.info(f"  {name}: {len(df)}건 저장 → {out_path}")

        except Exception as e:
            logger.error(f"  {name} 수집 실패: {e}")
            # 실패해도 다음 국가 계속 진행


def collect_recent(days: int = 14) -> None:
    """
    최근 N일 증분 수집 (실시간 스케줄러용).
    기존 parquet에 신규 이벤트를 병합하고 중복 제거.
    """
    end   = date.today()
    start = end - timedelta(days=days)
    logger.info(f"증분 수집: {start} ~ {end}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for country in COUNTRIES:
        name = country["name"]
        iso3 = country["iso3"]
        try:
            df_new = fetch_country_period(name, start, end)
            if df_new.empty:
                continue

            out_path = RAW_DIR / f"{iso3}.parquet"
            if out_path.exists():
                df_old = pd.read_parquet(out_path)
                df_new = pd.concat([df_old, df_new]).drop_duplicates("event_id_cnty")

            df_new.to_parquet(out_path, index=False)
            logger.info(f"  {name}: 증분 저장 완료 (총 {len(df_new)}건)")

        except Exception as e:
            logger.error(f"  {name} 증분 수집 실패: {e}")
