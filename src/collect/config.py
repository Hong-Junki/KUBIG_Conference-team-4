"""
수집 설정: 대상 국가 목록, 수집 기간, API 인증 정보

국가 선정 기준:
  1. ACLED 이벤트 수 — 2022년 이후 연간 100건 이상
  2. Fragile States Index — "Alert" 또는 "Warning" 등급
  3. 지역 대표성 — 전 세계 분쟁 패턴 포착을 위한 지역별 최소 커버
"""

import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# 수집 기간
# ──────────────────────────────────────────────
COLLECT_START = date(2014, 1, 1)   # raw_merged 8년+ 합본 시작 (팀원 분담 + 2016-2017 보강 통합)
COLLECT_END   = date(2026, 3, 31)  # GDELT/경제지표 종료 (ACLED는 2025-03까지 — label inner join에서 자동 제외)

# ──────────────────────────────────────────────
# 대상 국가 목록 (57개국)
# iso2: ISO 3166-1 alpha-2 (ACLED/yfinance)
# iso3: ISO 3166-1 alpha-3
# gdelt: GDELT FIPS 10-4 국가 코드
# ──────────────────────────────────────────────
COUNTRIES = [
    # ── 동유럽 (2) ──────────────────────────────
    {"name": "Ukraine",                      "iso2": "UA", "iso3": "UKR", "gdelt": "UP"},
    {"name": "Russia",                       "iso2": "RU", "iso3": "RUS", "gdelt": "RS"},

    # ── 중동 (9) ────────────────────────────────
    {"name": "Palestine (West Bank & Gaza)", "iso2": "PS", "iso3": "PSE", "gdelt": ["WE", "GZ"], "acled_names": ["Palestine"]},
    {"name": "Israel",                       "iso2": "IL", "iso3": "ISR", "gdelt": "IS"},
    {"name": "Syria",                        "iso2": "SY", "iso3": "SYR", "gdelt": "SY"},
    {"name": "Yemen",                        "iso2": "YE", "iso3": "YEM", "gdelt": "YM"},
    {"name": "Iraq",                         "iso2": "IQ", "iso3": "IRQ", "gdelt": "IZ"},
    {"name": "Lebanon",                      "iso2": "LB", "iso3": "LBN", "gdelt": "LE"},
    {"name": "Iran",                         "iso2": "IR", "iso3": "IRN", "gdelt": "IR"},
    {"name": "Turkey",                       "iso2": "TR", "iso3": "TUR", "gdelt": "TU"},
    {"name": "Saudi Arabia",                 "iso2": "SA", "iso3": "SAU", "gdelt": "SA"},

    # ── 북아프리카 (4) ──────────────────────────
    {"name": "Libya",                        "iso2": "LY", "iso3": "LBY", "gdelt": "LY"},
    {"name": "Egypt",                        "iso2": "EG", "iso3": "EGY", "gdelt": "EG"},
    {"name": "Algeria",                      "iso2": "DZ", "iso3": "DZA", "gdelt": "AG"},
    {"name": "Tunisia",                      "iso2": "TN", "iso3": "TUN", "gdelt": "TS"},

    # ── 서아프리카 (12) ─────────────────────────
    {"name": "Mali",                         "iso2": "ML", "iso3": "MLI", "gdelt": "ML"},
    {"name": "Burkina Faso",                 "iso2": "BF", "iso3": "BFA", "gdelt": "UV"},
    {"name": "Niger",                        "iso2": "NE", "iso3": "NER", "gdelt": "NG"},
    {"name": "Nigeria",                      "iso2": "NG", "iso3": "NGA", "gdelt": "NI"},
    {"name": "Cameroon",                     "iso2": "CM", "iso3": "CMR", "gdelt": "CM"},
    {"name": "Chad",                         "iso2": "TD", "iso3": "TCD", "gdelt": "CD"},
    {"name": "Guinea",                       "iso2": "GN", "iso3": "GIN", "gdelt": "GV"},
    {"name": "Guinea-Bissau",                "iso2": "GW", "iso3": "GNB", "gdelt": "PU"},
    {"name": "Senegal",                      "iso2": "SN", "iso3": "SEN", "gdelt": "SG"},
    {"name": "Cote d'Ivoire",                "iso2": "CI", "iso3": "CIV", "gdelt": "IV",  "acled_names": ["Ivory Coast"]},
    {"name": "Sierra Leone",                 "iso2": "SL", "iso3": "SLE", "gdelt": "SL"},
    {"name": "Togo",                         "iso2": "TG", "iso3": "TGO", "gdelt": "TO"},

    # ── 중앙아프리카 (2) ────────────────────────
    {"name": "Democratic Republic of Congo", "iso2": "CD", "iso3": "COD", "gdelt": "CG"},
    {"name": "Central African Republic",     "iso2": "CF", "iso3": "CAF", "gdelt": "CT"},

    # ── 동아프리카 (7) ──────────────────────────
    {"name": "Sudan",                        "iso2": "SD", "iso3": "SDN", "gdelt": "SU"},
    {"name": "South Sudan",                  "iso2": "SS", "iso3": "SSD", "gdelt": "OD"},
    {"name": "Ethiopia",                     "iso2": "ET", "iso3": "ETH", "gdelt": "ET"},
    {"name": "Somalia",                      "iso2": "SO", "iso3": "SOM", "gdelt": "SO"},
    {"name": "Kenya",                        "iso2": "KE", "iso3": "KEN", "gdelt": "KE"},
    {"name": "Uganda",                       "iso2": "UG", "iso3": "UGA", "gdelt": "UG"},
    {"name": "Eritrea",                      "iso2": "ER", "iso3": "ERI", "gdelt": "ER"},

    # ── 남아프리카 (3) ──────────────────────────
    {"name": "Mozambique",                   "iso2": "MZ", "iso3": "MOZ", "gdelt": "MZ"},
    {"name": "Zimbabwe",                     "iso2": "ZW", "iso3": "ZWE", "gdelt": "ZI"},
    {"name": "Madagascar",                   "iso2": "MG", "iso3": "MDG", "gdelt": "MA"},

    # ── 남아시아 (4) ────────────────────────────
    {"name": "Afghanistan",                  "iso2": "AF", "iso3": "AFG", "gdelt": "AF"},
    {"name": "Pakistan",                     "iso2": "PK", "iso3": "PAK", "gdelt": "PK"},
    {"name": "India",                        "iso2": "IN", "iso3": "IND", "gdelt": "IN"},
    {"name": "Bangladesh",                   "iso2": "BD", "iso3": "BGD", "gdelt": "BG"},

    # ── 동남아시아 (4) ──────────────────────────
    {"name": "Myanmar",                      "iso2": "MM", "iso3": "MMR", "gdelt": "BM"},
    {"name": "Philippines",                  "iso2": "PH", "iso3": "PHL", "gdelt": "RP"},
    {"name": "Thailand",                     "iso2": "TH", "iso3": "THA", "gdelt": "TH"},
    {"name": "Indonesia",                    "iso2": "ID", "iso3": "IDN", "gdelt": "ID"},

    # ── 중앙아시아 (2) ──────────────────────────
    {"name": "Tajikistan",                   "iso2": "TJ", "iso3": "TJK", "gdelt": "TI"},
    {"name": "Kyrgyzstan",                   "iso2": "KG", "iso3": "KGZ", "gdelt": "KG"},

    # ── 코카서스 (2) ────────────────────────────
    {"name": "Armenia",                      "iso2": "AM", "iso3": "ARM", "gdelt": "AM"},
    {"name": "Azerbaijan",                   "iso2": "AZ", "iso3": "AZE", "gdelt": "AJ"},

    # ── 중남미 (7) ──────────────────────────────
    {"name": "Haiti",                        "iso2": "HT", "iso3": "HTI", "gdelt": "HA"},
    {"name": "Colombia",                     "iso2": "CO", "iso3": "COL", "gdelt": "CO"},
    {"name": "Mexico",                       "iso2": "MX", "iso3": "MEX", "gdelt": "MX"},
    {"name": "Venezuela",                    "iso2": "VE", "iso3": "VEN", "gdelt": "VE"},
    {"name": "Ecuador",                      "iso2": "EC", "iso3": "ECU", "gdelt": "EC"},
    {"name": "Honduras",                     "iso2": "HN", "iso3": "HND", "gdelt": "HO"},
    {"name": "Guatemala",                    "iso2": "GT", "iso3": "GTM", "gdelt": "GT"},
]

# 빠른 조회용 dict
COUNTRY_BY_ISO2  = {c["iso2"]:  c for c in COUNTRIES}
COUNTRY_BY_ISO3  = {c["iso3"]:  c for c in COUNTRIES}

# GDELT FIPS → country 매핑 (Palestine처럼 다중 FIPS 지원)
COUNTRY_BY_GDELT = {}
for _c in COUNTRIES:
    _codes = _c["gdelt"] if isinstance(_c["gdelt"], list) else [_c["gdelt"]]
    for _code in _codes:
        COUNTRY_BY_GDELT[_code] = _c

# DOC 2.0 검색용 키워드 오버라이드 (특수문자 포함 국가명)
GDELT_DOC_KEYWORDS = {
    "PSE": "Palestine",
    "CIV": "Ivory Coast",
    "COD": "Congo DRC",
}

# ──────────────────────────────────────────────
# ACLED 설정
# ──────────────────────────────────────────────
ACLED_USERNAME = os.getenv("ACLED_USERNAME")
ACLED_PASSWORD = os.getenv("ACLED_PASSWORD")
ACLED_BASE_URL = "https://acleddata.com/api/acled/read"
ACLED_PAGE_SIZE = 5000  # 최대 허용값
ACLED_EVENT_TYPES = [
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
]

# ──────────────────────────────────────────────
# GDELT 설정
# ──────────────────────────────────────────────
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GDELT_BQ_TABLE = "gdelt-bq.gdeltv2.events_partitioned"  # 파티션 필터 실제 적용되는 테이블 (events 대비 스캔량 28배 절감)
GDELT_FIELDS = [
    "SQLDATE",
    "ActionGeo_CountryCode",
    "EventCode",
    "GoldsteinScale",
    "NumMentions",
    "AvgTone",
    "QuadClass",
]

# ──────────────────────────────────────────────
# 경제지표 설정
# ──────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY")

YFINANCE_TICKERS = {
    "VIX":  "^VIX",
    "WTI":  "CL=F",
    "Gold": "GC=F",
    "DXY":  "DX-Y.NYB",
}
FRED_SERIES = {
    "STLFSI4": "STLFSI4",
}

# ──────────────────────────────────────────────
# Reddit 설정
# ──────────────────────────────────────────────
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_SUBREDDITS = [
    "worldnews",
    "geopolitics",
    "CombatFootage",
    "UkraineRussiaReport",
    "syriancivilwar",
]

# ──────────────────────────────────────────────
# Telegram 설정
# ──────────────────────────────────────────────
TELEGRAM_API_ID   = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_PHONE    = os.getenv("TELEGRAM_PHONE")
TELEGRAM_CHANNELS = [
    "inaborov",
    "ryaborofficial",
    "warmonitoreng",
    "militarysummary",
    "osaborman",
    "UkrWeaponsTracker",
    "SudanWarMonitor",
    "GazaNewsNN",
]
# 실시간 수집 간격 (초)
TELEGRAM_REALTIME_INTERVAL_NORMAL = 3600   # 1시간
TELEGRAM_REALTIME_INTERVAL_SURGE   = 600   # 10분 (급증 감지 시)
TELEGRAM_SURGE_THRESHOLD           = 3.0   # 7일 평균 대비 배율
TELEGRAM_SURGE_NORMALIZE_HOURS     = 2     # 정상화 판단 연속 시간
# 과거 수집 딜레이 (초)
TELEGRAM_HIST_DELAY_CHANNEL  = 2
TELEGRAM_HIST_DELAY_BETWEEN  = 5
