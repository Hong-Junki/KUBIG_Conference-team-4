# 데이터 소스 명세

## 핵심 소스 (Phase 1 필수)

---

### 1. ACLED (라벨 + 충돌 피처)

#### 1-1. 개요
- 라이브러리: `acled` (PyPI, Python >=3.10) 또는 REST API 직접 호출
- 수집 범위: 57개국, 2022.01~**2025.03.31** (국가 목록: `src/collect/config.py` → `COUNTRIES`)
- 업데이트 주기: 주간 (목요일 기준)
- 제약: 한 번에 최대 5000건, 페이지네이션 필수
- 대상 이벤트 (라벨/피처): Battles, Explosions/Remote violence, Violence against civilians

> ⚠️ **Research 레벨 데이터 제한 (2026-04-16 확인)**
> ACLED Research 레벨 계정은 개별 이벤트 API를 통해 **최근 12개월 이내** 데이터만 조회 가능.
> 이 프로젝트의 test set 상한을 2025-03-31로 설정한 이유이며, 2025-04-01 이후 데이터는
> API 개별 이벤트 조회 불가 (집계 통계는 가능). 향후 갱신은 온라인 RL 방식으로 대응 예정.

#### 1-2. 인증 방식 (2025년 기준 OAuth 전환 완료)

**변경 이력:**
- 구 방식 (2025.09.15 이전): API key + email을 쿼리 파라미터에 직접 포함
  - `?key=API_KEY&email=USER_EMAIL`
  - 2025.09.15 이후 완전 폐기 (신규 키 발급 중단됨)
- 신 방식 (현재): OAuth 2.0 Password Grant + Bearer Token

**OAuth 토큰 발급:**
```
POST https://acleddata.com/oauth/token
Content-Type: application/x-www-form-urlencoded

username={ACLED_USERNAME}   # ACLED 계정 이메일
password={ACLED_PASSWORD}   # ACLED 계정 비밀번호
grant_type=password          # 고정값
client_id=acled              # 고정값
```

**응답 필드:**
```json
{
  "token_type": "Bearer",
  "expires_in": 86400,
  "access_token": "ACCESS-TOKEN",
  "refresh_token": "REFRESH-TOKEN"
}
```
- `access_token`: 24시간 유효
- `refresh_token`: 14일 유효

**토큰 갱신:**
```
POST https://acleddata.com/oauth/token

refresh_token={REFRESH_TOKEN}
grant_type=refresh_token     # 고정값
client_id=acled              # 고정값
```

**API 요청 시 토큰 사용:**
```
Authorization: Bearer {ACCESS-TOKEN}
```

**환경변수:** `.env` → `ACLED_USERNAME`, `ACLED_PASSWORD`

#### 1-3. 엔드포인트

| 엔드포인트 | URL | 설명 |
|-----------|-----|------|
| ACLED 이벤트 | `https://acleddata.com/api/acled/read` | 핵심 충돌/시위 이벤트 데이터 |
| CAST 예측 | `https://acleddata.com/api/cast/read` | 월별 이벤트 수 예측/실측값 |
| 삭제된 이벤트 | `https://acleddata.com/api/deleted/read` | 삭제된 event_id_cnty 목록 |

응답 포맷: `?_format=json` (기본) 또는 `?_format=csv`

#### 1-4. ACLED 이벤트 엔드포인트 요청 파라미터

| 파라미터 | 기본 연산자 | 타입 | 설명 |
|---------|-----------|------|------|
| `event_id_cnty` | LIKE | string | 국가약자+번호 고유 이벤트 식별자 (예: SYR12345) |
| `event_date` | = | string | 이벤트 날짜 (yyyy-mm-dd) |
| `year` | = | int | 이벤트 발생 연도 |
| `time_precision` | = | int | 날짜 정밀도 (1=정확, 2=대략, 3=추정) |
| `disorder_type` | LIKE | string | 무질서 유형 분류 (Political violence / Demonstrations / Strategic developments) |
| `event_type` | LIKE | string | 주요 이벤트 유형 |
| `sub_event_type` | LIKE | string | 세부 이벤트 유형 |
| `actor1` | LIKE | string | 주요 행위자 이름 |
| `actor2` | LIKE | string | 2차 행위자 이름 |
| `assoc_actor_1` | LIKE | string | 주요 행위자 연관 세력 |
| `assoc_actor_2` | LIKE | string | 2차 행위자 연관 세력 |
| `inter1` | = | string/int | 주요 행위자 유형 코드 (기본: 텍스트, inter_num=1이면 숫자) |
| `inter2` | = | string/int | 2차 행위자 유형 코드 |
| `interaction` | = | string/int | 두 행위자 유형 조합 코드 |
| `inter_num` | = | int | 0(기본, 텍스트) 또는 1(숫자 코드) |
| `civilian_targeting` | LIKE | string | 민간인 표적 여부 |
| `iso` | = | int | ISO 3166 국가 코드 (숫자) |
| `region` | = | int | ACLED 지역 코드 (숫자) |
| `country` | LIKE | string | 국가/영토 이름 |
| `admin1` | LIKE | string | 1단계 행정 구역 (주/도) |
| `admin2` | LIKE | string | 2단계 행정 구역 (시/군) |
| `admin3` | LIKE | string | 3단계 행정 구역 |
| `location` | LIKE | string | 이벤트 발생 지명 |
| `latitude` | = | float | 위도 (EPSG:4326) |
| `longitude` | = | float | 경도 (EPSG:4326) |
| `geo_precision` | = | int | 위치 정밀도 (1=정확, 2=대략, 3=추정) |
| `source` | LIKE | string | 정보 출처 (세미콜론 구분) |
| `source_scale` | LIKE | string | 소스 지리적 근접성 |
| `notes` | LIKE | string | 이벤트 서술 텍스트 |
| `fatalities` | = | int | 사망자 수 (보수적 추정) |
| `tags` | LIKE | string | 구조화된 메타데이터 태그 (세미콜론 구분) |
| `timestamp` | >= | int | Unix timestamp 필터 (최신 데이터 추적용) |
| `export_type` | = | string | `dyadic`(기본, 행위자쌍) 또는 `monadic`(단일 행위자) |
| `population` | = | string | 인구 데이터 포함 여부: `TRUE` 또는 `full` |
| `_format` | = | string | 응답 포맷: `json`, `csv`, `xml`, `txt` |
| `fields` | = | string | 반환할 컬럼 목록 (파이프`\|` 구분, 예: `event_date\|country\|fatalities`) |
| `limit` | = | int | 반환 행 수 (기본: 5000, 최대: 5000) |
| `page` | = | int | 페이지 번호 (1부터 시작, 페이지네이션용) |

**쿼리 연산자 변경:** 파라미터명에 `_where` 접미사를 붙여 연산자 지정
- 예: `year_where=BETWEEN`, `fatalities_where=>`, `event_date_where=BETWEEN`

**다중 값 필터:** 파이프(`|`)로 구분
- 예: `country=Ukraine|Sudan|Gaza`

#### 1-5. ACLED 이벤트 엔드포인트 응답 필드

**JSON 래퍼 필드:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `status` | int | 요청 상태 코드 |
| `success` | boolean | 호출 성공 여부 |
| `last_update` | int | 마지막 데이터 업데이트 이후 경과 시간(시간) |
| `count` | int | 반환된 행 수 |
| `messages` | array | 주의 필요 정보 메시지 |
| `data` | array | 이벤트 레코드 배열 |
| `filename` | string | CSV 다운로드 시 파일명 |
| `data_query_restrictions` | object | 적용된 쿼리 제한 정보 |

**data 배열 개별 레코드 필드:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `event_id_cnty` | string | 국가 약자 + 번호 고유 식별자 (예: `SYR12345`) |
| `event_date` | string (YYYY-MM-DD) | 이벤트 발생 날짜 |
| `year` | int | 이벤트 발생 연도 |
| `time_precision` | int | 날짜 정밀도: `1`=정확한 날짜, `2`=주/월 단위 대략, `3`=분기 이상 추정 |
| `disorder_type` | string | 무질서 유형: `Political violence`, `Demonstrations`, `Strategic developments` |
| `event_type` | string | 주요 이벤트 유형 (6종, 아래 섹션 참조) |
| `sub_event_type` | string | 세부 이벤트 유형 (25종, 아래 섹션 참조) |
| `actor1` | string | 주요 행위자 전체 명칭 |
| `assoc_actor_1` | string | 주요 행위자 연관 세력 (세미콜론 구분) |
| `inter1` | string or int | 주요 행위자 유형 (기본 텍스트, `inter_num=1`이면 숫자) |
| `actor2` | string | 2차 행위자 전체 명칭 |
| `assoc_actor_2` | string | 2차 행위자 연관 세력 (세미콜론 구분) |
| `inter2` | string or int | 2차 행위자 유형 |
| `interaction` | string or int | 두 행위자 유형 조합 (예: `State forces - Rebel groups` 또는 숫자 `12`) |
| `civilian_targeting` | string | 민간인 표적 여부 |
| `iso` | int | ISO 3166-1 숫자 국가 코드 |
| `region` | string | ACLED 지역 명칭 |
| `country` | string | 국가/영토 이름 |
| `admin1` | string | 1단계 행정 구역 |
| `admin2` | string | 2단계 행정 구역 |
| `admin3` | string | 3단계 행정 구역 |
| `location` | string | 이벤트 발생 지명 |
| `latitude` | float | 위도 (EPSG:4326 WGS84) |
| `longitude` | float | 경도 (EPSG:4326 WGS84) |
| `geo_precision` | int | 위치 정밀도: `1`=정확한 지점, `2`=지역 수준, `3`=국가/광역 수준 |
| `source` | string | 정보 출처 목록 (세미콜론 구분) |
| `source_scale` | string | 소스의 지리적 근접성 (예: Local, National, International, New media) |
| `notes` | string | 이벤트 서술 텍스트 |
| `fatalities` | int | 보고된 사망자 수 (복수 출처 시 가장 보수적 수치) |
| `tags` | string | 구조화 메타데이터 (세미콜론 구분, 예: 성별 데이터, 종교 관련 등) |
| `timestamp` | int | 해당 레코드 마지막 업데이트 Unix timestamp |
| `population_1km` | int | 반경 1km 인구 추정치 (`population=TRUE` 시 포함) |
| `population_2km` | int | 반경 2km 인구 추정치 |
| `population_5km` | int | 반경 5km 인구 추정치 |
| `population_best` | int | 최적 반경 인구 추정치 |

#### 1-6. event_type 값 목록 (6종)

| event_type | disorder_type | 설명 |
|-----------|--------------|------|
| `Battles` | Political violence | 두 정치적으로 조직된 무장 집단 간 폭력적 충돌 |
| `Explosions/Remote violence` | Political violence | 원격 무기(포격, 드론, IED 등)를 이용한 일방적 폭력. 대응 능력 비대칭이 특징 |
| `Violence against civilians` | Political violence | 조직 무장 집단이 비무장 민간인을 의도적으로 공격 |
| `Riots` | Demonstrations | 시위대 또는 군중이 재산 파괴 또는 비조직적 폭력 행위 |
| `Protests` | Demonstrations | 참가자가 폭력을 행사하지 않는 공개 시위 (경찰 진압이 있을 수 있음) |
| `Strategic developments` | Strategic developments | 충돌 행위자의 비폭력적이지만 전략적으로 중요한 활동 |

#### 1-7. sub_event_type 값 목록 (25종)

| sub_event_type | event_type | 설명 |
|---------------|-----------|------|
| `Armed clash` | Battles | 두 무장 집단 간 교전 |
| `Government regains territory` | Battles | 정부군이 영토 탈환 |
| `Non-state actor overtakes territory` | Battles | 비국가 행위자가 영토 점령 |
| `Air/drone strike` | Explosions/Remote violence | 항공/드론 공습 (지대지 공격 포함) |
| `Shelling/artillery/missile attack` | Explosions/Remote violence | 포격, 포병, 박격포, 유도 미사일 공격 |
| `Remote explosive/landmine/IED` | Explosions/Remote violence | 지뢰, IED, 차량폭탄(SVBIED) 등 원격/피해자 작동 폭발물 |
| `Suicide bomb` | Explosions/Remote violence | 자살 폭탄 공격 (SVBIED 포함) |
| `Grenade` | Explosions/Remote violence | 수류탄 공격 |
| `Chemical weapon` | Explosions/Remote violence | 화학무기 사용 (화학무기협약 Schedule 1 물질) |
| `Attack` | Violence against civilians | 민간인 대상 공격 |
| `Sexual violence` | Violence against civilians | 성폭력 |
| `Abduction/forced disappearance` | Violence against civilians | 납치/강제 실종 |
| `Violent demonstration` | Riots | 시위대의 폭력적 행위 |
| `Mob violence` | Riots | 군중 폭력 |
| `Peaceful protest` | Protests | 평화 시위 |
| `Protest with intervention` | Protests | 경찰 등 외부 개입이 있는 시위 |
| `Excessive force against protesters` | Protests | 시위대에 대한 과도한 공권력 행사 |
| `Agreement` | Strategic developments | 협정/합의 체결 |
| `Arrests` | Strategic developments | 구금/체포 |
| `Change to group/activity` | Strategic developments | 집단 구조·활동 변화 (해산, 합병, 이름 변경 등) |
| `Disrupted weapons use` | Strategic developments | 무기 사용 차단 (무기 압수, 공격 실패 등) |
| `Headquarters or base established` | Strategic developments | 본부/기지 설치 |
| `Looting/property destruction` | Strategic developments | 약탈/재산 파괴 |
| `Non-violent transfer of territory` | Strategic developments | 비폭력적 영토 이전 |
| `Other` | Strategic developments | 기타 전략적 발전 |

#### 1-8. inter1 / inter2 행위자 유형 코드

2024년 9월 26일부터 기본값이 숫자→텍스트로 변경됨. `inter_num=1` 파라미터로 숫자 코드 요청 가능.

| 숫자 코드 | 텍스트 값 | 설명 |
|---------|---------|------|
| `1` | `State forces` | 국가 군대, 경찰, 정보기관 등 |
| `2` | `Rebel groups` | 반군 집단 |
| `3` | `Political militias` | 정치적 민병대 |
| `4` | `Identity militias` | 민족·종교·공동체 기반 민병대 |
| `5` | `Rioters` | 폭력 시위 참가자 |
| `6` | `Protesters` | 평화 시위 참가자 |
| `7` | `Civilians` | 민간인 |
| `8` | `External/Other forces` | 외국 군대, 민간 군사 기업, 기타 |

`interaction` 필드: inter1+inter2 조합 (예: `12`=State forces vs Rebel groups, `17`=State forces vs Civilians)

#### 1-9. CAST 엔드포인트 응답 필드

월별 국가/admin1 단위 이벤트 수 예측값 및 실측값.

| 필드 | 타입 | 설명 |
|------|------|------|
| `country` | string | 국가명 |
| `admin1` | string | 1단계 행정구역 |
| `month` | string | 월 |
| `year` | int | 연도 |
| `total_forecast` | int | 전체 예측 이벤트 수 |
| `battles_forecast` | int | Battles 예측 수 |
| `erv_forecast` | int | Explosions/Remote violence 예측 수 |
| `vac_forecast` | int | Violence against civilians 예측 수 |
| `total_observed` | int | 전체 실측 이벤트 수 (해당 월 종료 후 집계) |
| `battles_observed` | int | Battles 실측 수 |
| `erv_observed` | int | Explosions/Remote violence 실측 수 |
| `vac_observed` | int | Violence against civilians 실측 수 |
| `timestamp` | int | 마지막 업데이트 Unix timestamp |

#### 1-10. 삭제 엔드포인트 응답 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| `event_id_cnty` | string | 삭제된 이벤트 식별자 |
| `deleted_timestamp` | int | 삭제 시각 Unix timestamp |

#### 1-11. 에러 코드

| 코드 | 메시지 | 원인 |
|------|--------|------|
| 400 | Unrecognized username/password | 로그인 인증 실패 |
| 401 | Resource owner denied request | 잘못된 access_token |
| 403 | Consent acceptance required | 사용자 동의 미완료 |
| 403 | Fill required fields | 프로필 필수 항목 미입력 |
| 403 | Access denied | 미로그인 또는 권한 부족 |
| 403 | Invalid URL | 잘못된 엔드포인트 |

#### 1-12. 페이지네이션 예시

```python
# 전체 데이터 수집 패턴
page = 1
while True:
    resp = requests.get(
        "https://acleddata.com/api/acled/read",
        headers={"Authorization": f"Bearer {access_token}"},
        params={
            "country": "Ukraine",
            "event_date": "2022-01-01|2024-12-31",
            "event_date_where": "BETWEEN",
            "limit": 5000,
            "page": page,
            "_format": "json",
        }
    )
    data = resp.json()["data"]
    if len(data) < 5000:
        break
    page += 1
```

---

### 2. GDELT (뉴스 피처)

#### 2-1. 개요

- 업데이트 주기: 15분 단위 실시간
- 커버리지: 100개 이상 언어, 전 세계 뉴스 소스
- 주의: 정확도 ~55%, 중복률 ~20% → 중복 제거 전처리 필수

---

#### 2-A. 과거 데이터 - BigQuery (`gdelt-bq.gdeltv2.events`)

##### 2-A-1. 테이블 정보

| 항목 | 값 |
|------|-----|
| 테이블 ID | `gdelt-bq.gdeltv2.events` |
| 파티션 테이블 | `gdelt-bq.gdeltv2.events_partitioned` |
| 커버리지 | 2015.02 ~ 현재 |
| 라이브러리 | `google-cloud-bigquery` |
| 인증 | `.env` → `GOOGLE_APPLICATION_CREDENTIALS` (서비스 계정 JSON 경로) |
| 무료 한도 | 1TB/월 |

> **파티션 테이블 사용 권장**: `events_partitioned`에서 `_PARTITIONTIME` 필터를 사용하면 동일 쿼리 대비 스캔량을 최대 28배 절감 (예: 423GB → 15GB).

##### 2-A-2. 전체 컬럼 목록 (61개 필드)

**[이벤트 식별]**

| 컬럼명 | BigQuery 타입 | 설명 |
|--------|-------------|------|
| `GLOBALEVENTID` | INTEGER | 전역 고유 이벤트 식별자 (Primary Key) |
| `SQLDATE` | INTEGER | 이벤트 발생 날짜 (YYYYMMDD 형식, UTC 기준) |
| `MonthYear` | INTEGER | 이벤트 날짜 월·연도 (YYYYMM) |
| `Year` | INTEGER | 이벤트 발생 연도 (YYYY) |
| `FractionDate` | FLOAT | 소수점 날짜 (YYYY.FFFF, FFFF=연도 내 진행율) |

**[Actor1 - 주요 행위자]**

| 컬럼명 | BigQuery 타입 | 설명 |
|--------|-------------|------|
| `Actor1Code` | STRING | Actor1 전체 CAMEO 코드 (국가+그룹+민족+종교+유형 조합) |
| `Actor1Name` | STRING | Actor1 식별 이름 (기사에서 추출) |
| `Actor1CountryCode` | STRING | Actor1 국적 국가 코드 (CAMEO 3자리) |
| `Actor1KnownGroupCode` | STRING | Actor1 알려진 집단 코드 (117종) |
| `Actor1EthnicCode` | STRING | Actor1 민족 코드 (646종) |
| `Actor1Religion1Code` | STRING | Actor1 주요 종교 코드 (31종) |
| `Actor1Religion2Code` | STRING | Actor1 2차 종교 코드 |
| `Actor1Type1Code` | STRING | Actor1 1차 분류 코드 (40종) |
| `Actor1Type2Code` | STRING | Actor1 2차 분류 코드 |
| `Actor1Type3Code` | STRING | Actor1 3차 분류 코드 |

**[Actor2 - 상대 행위자]**

| 컬럼명 | BigQuery 타입 | 설명 |
|--------|-------------|------|
| `Actor2Code` | STRING | Actor2 전체 CAMEO 코드 |
| `Actor2Name` | STRING | Actor2 식별 이름 |
| `Actor2CountryCode` | STRING | Actor2 국적 국가 코드 |
| `Actor2KnownGroupCode` | STRING | Actor2 알려진 집단 코드 |
| `Actor2EthnicCode` | STRING | Actor2 민족 코드 |
| `Actor2Religion1Code` | STRING | Actor2 주요 종교 코드 |
| `Actor2Religion2Code` | STRING | Actor2 2차 종교 코드 |
| `Actor2Type1Code` | STRING | Actor2 1차 분류 코드 |
| `Actor2Type2Code` | STRING | Actor2 2차 분류 코드 |
| `Actor2Type3Code` | STRING | Actor2 3차 분류 코드 |

**[이벤트 분류]**

| 컬럼명 | BigQuery 타입 | 설명 |
|--------|-------------|------|
| `IsRootEvent` | INTEGER | 루트 이벤트 여부 (1=루트, 0=파생) |
| `EventCode` | STRING | CAMEO 이벤트 코드 (4자리, 310종 분류) |
| `EventBaseCode` | STRING | CAMEO 2단계 코드 (EventCode의 상위) |
| `EventRootCode` | STRING | CAMEO 루트 코드 (최상위 20종 분류, 예: "02"=Appeal) |
| `QuadClass` | INTEGER | 4분류 이벤트 유형 (값 목록은 아래 별도 섹션 참조) |
| `GoldsteinScale` | FLOAT | 골드스타인 안정성 점수 (-10.0 ~ +10.0) |
| `NumMentions` | INTEGER | 최초 15분 처리 주기 내 언급 수 |
| `NumSources` | INTEGER | 최초 15분 처리 주기 내 고유 소스 수 |
| `NumArticles` | INTEGER | 이벤트를 언급한 기사 수 |
| `AvgTone` | FLOAT | 전체 언급 기사의 평균 톤 (-100 ~ +100) |

**[Actor1 지리정보]**

| 컬럼명 | BigQuery 타입 | 설명 |
|--------|-------------|------|
| `Actor1Geo_Type` | INTEGER | 지리 정밀도 유형 (1=국가, 2=미국 주, 3=미국 도시, 4=세계 도시, 5=세계 지역) |
| `Actor1Geo_FullName` | STRING | 전체 지명 (예: "Kyiv, Ukraine") |
| `Actor1Geo_CountryCode` | STRING | FIPS 10-4 2자리 국가 코드 |
| `Actor1Geo_ADM1Code` | STRING | 1단계 행정구역 코드 |
| `Actor1Geo_ADM2Code` | STRING | 2단계 행정구역 코드 |
| `Actor1Geo_Lat` | FLOAT | 위도 (WGS84) |
| `Actor1Geo_Long` | FLOAT | 경도 (WGS84) |
| `Actor1Geo_FeatureID` | STRING | GNS/GNIS 지형 ID |

**[Actor2 지리정보]**

| 컬럼명 | BigQuery 타입 | 설명 |
|--------|-------------|------|
| `Actor2Geo_Type` | INTEGER | 지리 정밀도 유형 |
| `Actor2Geo_FullName` | STRING | 전체 지명 |
| `Actor2Geo_CountryCode` | STRING | FIPS 10-4 국가 코드 |
| `Actor2Geo_ADM1Code` | STRING | 1단계 행정구역 코드 |
| `Actor2Geo_ADM2Code` | STRING | 2단계 행정구역 코드 |
| `Actor2Geo_Lat` | FLOAT | 위도 |
| `Actor2Geo_Long` | FLOAT | 경도 |
| `Actor2Geo_FeatureID` | STRING | GNS/GNIS 지형 ID |

**[ActionGeo - 실제 행동 발생 지점 (가장 중요)]**

| 컬럼명 | BigQuery 타입 | 설명 |
|--------|-------------|------|
| `ActionGeo_Type` | INTEGER | 지리 정밀도 유형 |
| `ActionGeo_FullName` | STRING | 전체 지명 |
| `ActionGeo_CountryCode` | STRING | **FIPS 10-4 2자리 국가 코드** ← 국가 필터링에 사용 |
| `ActionGeo_ADM1Code` | STRING | 1단계 행정구역 코드 |
| `ActionGeo_ADM2Code` | STRING | 2단계 행정구역 코드 |
| `ActionGeo_Lat` | FLOAT | 위도 |
| `ActionGeo_Long` | FLOAT | 경도 |
| `ActionGeo_FeatureID` | STRING | GNS/GNIS 지형 ID |

**[메타데이터]**

| 컬럼명 | BigQuery 타입 | 설명 |
|--------|-------------|------|
| `DATEADDED` | INTEGER | 데이터베이스 추가 시각 (YYYYMMDDHHMMSS UTC) |
| `SOURCEURL` | STRING | 최초 보도 기사 URL |

**[파티션 의사 컬럼 - WHERE 절에서만 사용]**

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `_PARTITIONTIME` | TIMESTAMP | 데이터 수집 시각 기준 파티션 경계 (UTC). SELECT에 포함 불가, WHERE 필터 전용 |

> **주의**: `_PARTITIONTIME`은 GDELT가 데이터를 수집한 시각 기준이며, 이벤트 발생일(`SQLDATE`)과 최대 수일 차이가 날 수 있음. 두 조건을 함께 사용할 것.

##### 2-A-3. QuadClass 값 목록

| 값 | 의미 | 설명 |
|----|------|------|
| `1` | Verbal Cooperation | 구두 협력 (외교적 지지, 의도 표명 등) |
| `2` | Material Cooperation | 물질적 협력 (원조, 군사 협력, 협정 이행 등) |
| `3` | Verbal Conflict | 구두 갈등 (비난, 위협, 요구, 거부 등) |
| `4` | Material Conflict | 물질적 갈등 (공격, 폭력, 제재, 군사 행동 등) |

> 충돌 예측에는 `QuadClass IN (3, 4)` 또는 `QuadClass = 4`를 필터로 사용.

##### 2-A-4. GoldsteinScale 범위

- **범위**: -10.0 ~ +10.0
- **의미**: 이벤트 유형이 국가 안정성에 미치는 이론적 영향
  - `-10.0`: 최대 불안정 (전면전, 대량학살 등)
  - `0.0`: 중립
  - `+10.0`: 최대 안정 (평화 협정 체결 등)
- **특성**: 이벤트 유형(`EventCode`)에 고정된 값. 개별 이벤트 규모가 아닌 이벤트 종류에 부여됨
- **NULL 가능**: 일부 이벤트는 GoldsteinScale 없음 (NULL 처리 필요)

##### 2-A-5. EventCode / EventRootCode 구조 (CAMEO 1.1b3)

```
EventRootCode (2자리) → EventBaseCode (3자리) → EventCode (4자리)

예시:
  19 (Use conventional military force)
    └─ 190 (Use conventional military force, NOS)
    └─ 193 (Fight with small arms and light weapons)
    └─ 194 (Fight with artillery and tanks)
    └─ 195 (Employ aerial weapons)
    └─ 196 (Violate ceasefire)
```

**주요 EventRootCode 목록:**

| 코드 | 분류 | QuadClass |
|------|------|-----------|
| `01` | Make public statement | 1~3 혼재 |
| `02` | Appeal | 1~3 혼재 |
| `03` | Express intent to cooperate | 1 |
| `04` | Consult | 1 |
| `05` | Engage in diplomatic cooperation | 2 |
| `06` | Engage in material cooperation | 2 |
| `07` | Provide aid | 2 |
| `08` | Yield | 1~2 |
| `09` | Investigate | 1 |
| `10` | Demand | 3 |
| `11` | Disapprove | 3 |
| `12` | Reject | 3 |
| `13` | Threaten | 3 |
| `14` | Protest | 3 |
| `15` | Exhibit force posture | 3~4 혼재 |
| `16` | Reduce relations | 3~4 혼재 |
| `17` | Coerce | 4 |
| `18` | Assault | 4 |
| `19` | Fight | 4 |
| `20` | Use unconventional mass violence | 4 |

> 참조 파일: `input/reference/` CAMEO 코드북

##### 2-A-6. _PARTITIONTIME 파티션 사용법 및 비용 절약 팁

```python
from google.cloud import bigquery

client = bigquery.Client()

# 1. dry_run으로 스캔량 사전 확인 (쿼리 실행 전 반드시 수행)
query = """
    SELECT
        SQLDATE,
        ActionGeo_CountryCode,
        EventCode,
        EventRootCode,
        QuadClass,
        GoldsteinScale,
        NumMentions,
        NumArticles,
        AvgTone
    FROM `gdelt-bq.gdeltv2.events_partitioned`
    WHERE _PARTITIONTIME BETWEEN TIMESTAMP('2022-01-01') AND TIMESTAMP('2024-12-31')
      AND CAST(SQLDATE AS STRING) BETWEEN '20220101' AND '20241231'
      AND QuadClass IN (3, 4)
      AND ActionGeo_CountryCode IN ('UP', 'SU', 'IZ')  -- FIPS: Ukraine, Sudan, Iraq
"""
job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
dry_job = client.query(query, job_config=job_config)
print(f"예상 스캔량: {dry_job.total_bytes_processed / 1e9:.2f} GB")

# 2. 실제 쿼리 실행
job_config = bigquery.QueryJobConfig(use_query_cache=True)
df = client.query(query, job_config=job_config).to_dataframe()
```

**비용 절약 체크리스트:**
1. `events_partitioned` 테이블 사용 (`events`가 아닌 `events_partitioned`)
2. `_PARTITIONTIME` 필터 필수 → 동일 쿼리 최대 28배 스캔량 절감
3. `_PARTITIONTIME`과 `SQLDATE` 조건 둘 다 지정 (파티션 프루닝 + 데이터 정확도)
4. SELECT에 필요한 컬럼만 명시 (전체 컬럼 조회 금지)
5. `dry_run=True`로 사전 스캔량 확인 후 실행
6. `use_query_cache=True` 설정 (반복 쿼리 캐시 활용)
7. 월 1TB 무료 한도 관리: 각 쿼리 스캔량 로그 기록 권장

**국가 코드 주의**: `ActionGeo_CountryCode`는 FIPS 10-4 코드 사용 (ISO 3166과 다름)
- Ukraine: `UP` (ISO: `UA`)
- Sudan: `SU` (ISO: `SD`)
- Gaza/Palestine: `GZ` (ISO: `PS`)
- Syria: `SY` (동일)
- Iraq: `IZ` (ISO: `IQ`)
- Ethiopia: `ET` (동일)

---

#### 2-B. 실시간 데이터 - DOC 2.0 API (`gdeltdoc` 라이브러리)

##### 2-B-1. 엔드포인트

```
https://api.gdeltproject.org/api/v2/doc/doc
```

- 인증 불필요 (무료 공개 API)
- 설치: `pip install gdeltdoc`

##### 2-B-2. 지원 모드 전체 목록

| 모드 | gdeltdoc 함수 | 설명 |
|------|-------------|------|
| `artlist` | `gd.article_search(f)` | 매칭 기사 목록 (최대 250건) |
| `timelinevol` | `gd.timeline_search("timelinevol", f)` | 전체 모니터링 기사 대비 매칭 비율 (%) |
| `timelinevolraw` | `gd.timeline_search("timelinevolraw", f)` | 실제 기사 수 (절대값) + 전체 기사 수 |
| `timelinelang` | `gd.timeline_search("timelinelang", f)` | 언어별 기사 수 분해 |
| `timelinesourcecountry` | `gd.timeline_search("timelinesourcecountry", f)` | 소스 국가별 기사 수 분해 |
| `timelinetone` | `gd.timeline_search("timelinetone", f)` | 시간대별 평균 톤 점수 |

##### 2-B-3. 요청 파라미터 (Filters 클래스)

```python
from gdeltdoc import GdeltDoc, Filters, near, repeat

f = Filters(
    # 날짜 (둘 중 하나 필수)
    start_date="2025-01-01",      # YYYY-MM-DD (UTC 기준)
    end_date="2025-03-31",        # YYYY-MM-DD
    # timespan="7d",              # 또는 상대 기간: 분(m), 시간(h), 일(d), 주(w), 월(mon)

    # 콘텐츠 필터 (모두 선택)
    keyword="Ukraine war",        # 정확한 구문 매칭
    country="UP",                 # FIPS 2자리 국가 코드 (리스트 가능)
    language="English",           # ISO 639 언어명 (리스트 가능)
    domain="bbc.co.uk",          # 도메인 필터 (부분 매칭, 리스트 가능)
    domain_exact="bbc.co.uk",    # 도메인 정확 매칭
    theme="CONFLICT",             # GDELT GKG 테마 코드

    # 고급 필터
    near=near(10, "ceasefire", "Ukraine"),     # 단어 근접 거리
    repeat=repeat(3, "attack"),               # 단어 반복 임계값
    tone=">5",                                # 톤 필터 (>, <, =)
    # tone_absolute=">10",                   # 감정 강도 (극성 무관)

    num_records=250,              # artlist 모드 최대 250건
)
```

##### 2-B-4. 응답 구조

**article_search() 반환 DataFrame 컬럼:**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `url` | string | 기사 원문 URL |
| `url_mobile` | string | 모바일 버전 URL |
| `title` | string | 기사 제목 |
| `seendate` | string | GDELT 수집 시각 (YYYYMMDDTHHMMSSZ 형식, UTC) |
| `socialimage` | string | 소셜 공유 이미지 URL |
| `domain` | string | 출처 도메인 |
| `language` | string | 기사 언어 |
| `sourcecountry` | string | 소스 국가 |

**timeline_search() 반환 DataFrame 구조:**

| 모드 | datetime 컬럼 | 값 컬럼 | 설명 |
|------|-------------|---------|------|
| `timelinevol` | datetime (UTC) | `Volume Intensity` | 전체 대비 비율 (%) |
| `timelinevolraw` | datetime (UTC) | `Volume Intensity`, `All Articles` | 절대 기사 수, 전체 모니터링 수 |
| `timelinetone` | datetime (UTC) | `Tone` | 평균 톤 점수 (음수=부정, 양수=긍정) |
| `timelinelang` | datetime (UTC) | [언어코드별 컬럼] | 언어별 기사 수 |
| `timelinesourcecountry` | datetime (UTC) | [국가코드별 컬럼] | 국가별 기사 수 |

##### 2-B-5. 날짜 제한 정확한 기준

- **공식 지원**: 최근 3개월 (rolling window)
- **비공식 접근**: 2017년 1월 1일 이후 데이터는 요청 시 반환될 수 있으나 보장되지 않음
- **3개월 이전 데이터**: BigQuery(`events_partitioned`) 사용 필수
- 내부 날짜 형식: `YYYYMMDDHHMMSS` (14자리, UTC). 라이브러리가 자동 변환

##### 2-B-6. 사용 예시

```python
from gdeltdoc import GdeltDoc, Filters

gd = GdeltDoc()

# 우크라이나 관련 최근 7일 기사 수 (절대값)
f = Filters(
    keyword="Ukraine conflict",
    country="UP",
    timespan="7d"
)
vol_df = gd.timeline_search("timelinevolraw", f)
# 반환: datetime 컬럼 + "Volume Intensity" (기사 수) + "All Articles" (전체 수)

# 특정 기간 톤 변화
f2 = Filters(
    keyword="Sudan war",
    country="SU",
    start_date="2025-01-01",
    end_date="2025-03-31"
)
tone_df = gd.timeline_search("timelinetone", f2)
# 반환: datetime 컬럼 + "Tone" (평균 톤)

# 기사 목록 수집 (최대 250건)
articles_df = gd.article_search(f2)
# 반환: url, title, seendate, domain, language, sourcecountry 등
```

---

#### 2-C. BigQuery ↔ DOC API 스키마 매핑

| 파이프라인 목적 | BigQuery 필드 | DOC API 필드 | 변환 |
|--------------|-------------|------------|------|
| 국가 필터 | `ActionGeo_CountryCode` (FIPS) | `country` (FIPS) | 동일 코드 체계 사용 |
| 날짜 | `SQLDATE` (YYYYMMDD INT) | `seendate` (YYYYMMDDTHHMMSSZ) | 날짜 부분만 추출해 통일 |
| 기사 수 | `NumArticles` | `Volume Intensity` (timelinevolraw) | 집계 단위 상이 (이벤트 vs 기사) |
| 톤 | `AvgTone` (-100~+100) | `Tone` (timelinetone) | 동일 스케일 |
| 이벤트 유형 | `QuadClass`, `EventCode` | `theme` (GKG 테마) | 직접 매핑 없음, 별도 처리 |

**통합 시 주의사항:**
- BigQuery는 이벤트 단위 (Actor1 vs Actor2 행위), DOC API는 기사 단위
- 둘의 수치는 직접 비교 불가 → 각각 독립적 피처로 사용
- 타임존: 둘 다 UTC이나 BigQuery `SQLDATE`는 날짜만, DOC API `seendate`는 초 단위까지 포함

---

### 3. 경제 지표

| 지표 | 티커/코드 | 라이브러리 | 업데이트 |
|------|----------|-----------|---------|
| VIX | `^VIX` | `yfinance` | 일간 |
| WTI | `CL=F` | `yfinance` | 일간 |
| Gold | `GC=F` | `yfinance` | 일간 |
| DXY | `DX-Y.NYB` | `yfinance` | 일간 |
| STLFSI4 | `STLFSI4` | `fredapi` | 주간 (매주 금요일 기준) |

#### yfinance 응답 스키마
- 호출: `yf.Ticker(ticker).history(start=..., end=...)`
- 반환 타입: `pd.DataFrame`
- 컬럼: `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends`, `Stock Splits` (모두 float64, Volume은 int64)
- 인덱스: `DatetimeIndex` — **timezone이 티커마다 다름** (UTC 변환 필수)
  - VIX(`^VIX`): `America/Chicago`
  - WTI(`CL=F`), Gold(`GC=F`), DXY(`DX-Y.NYB`): `America/New_York`
- 사용 컬럼: `Close`만 사용 (일간 종가)

#### fredapi 응답 스키마
- 호출: `fred.get_series("STLFSI4", observation_start=..., observation_end=...)`
- 반환 타입: `pd.Series` (dtype: float64)
- 인덱스: `DatetimeIndex` — **timezone 없음 (tz-naive, 미국 동부 시간 기준)**
- 업데이트: 매주 금요일 기준, 약 1주일 후 발표 (지연 있음)
- 단위: Index (양수=긴장, 음수=완화)
- 결측 처리: 주간 데이터이므로 일간 피처 테이블에 `ffill` 적용 필요

#### 인증
- yfinance: 인증 불필요
- fredapi: `.env` → `FRED_API_KEY`
- **macOS Python 3.10 SSL 주의**: fredapi 호출 시 SSL 인증서 오류 발생 가능. `certifi` 패키지로 해결:
  ```python
  import ssl, certifi
  ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
  ```

---

## 선택적 보강 소스 (베이스라인 성능 확인 후 추가 여부 결정)

### 4. Reddit
- 라이브러리: `praw`
- 인증: `.env` → `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`
- 대상: r/worldnews, r/geopolitics, r/CombatFootage, r/UkraineRussiaReport, r/syriancivilwar
- 실시간: API로 일간 수집
- 과거: API는 최근 1000개 제한. Pushshift는 2023년 이후 차단됨. 과거 데이터 의존도를 낮추고, 수집 시작 시점부터 축적하는 전략으로 진행
- 피처: 일간 게시물 수, 댓글 급증, 키워드 빈도

### 5. Telegram OSINT
- 라이브러리: `telethon`
- 인증: `.env` → `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_PHONE`
- 대상 채널: `inaborov`, `ryaborofficial`, `warmonitoreng`, `militarysummary`, `osaborman`, `UkrWeaponsTracker`, `SudanWarMonitor`, `GazaNewsNN`
- 과거 일괄 수집: 대량 요청 발생하므로 채널당 2초 간격, 채널 간 5초 대기 적용
- 실시간 수집 (하이브리드):
  - 기본: 1시간 간격 배치로 신규 메시지 수집
  - 급증 감지: 직전 1시간 메시지 수가 채널별 7일 평균의 3배 초과 시 수집 주기를 10분으로 단축
  - 정상화: 2시간 연속 평균 이하이면 1시간 간격으로 복귀
- 피처: 일간 메시지 수, 급증 탐지 (전일 대비 2배), 키워드 빈도 (airstrike, ceasefire, evacuation 등)
- `.session` 파일 `.gitignore`에 추가 필수
