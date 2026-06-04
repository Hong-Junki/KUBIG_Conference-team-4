# 텔레그램 기반 실시간 무력충돌 모니터링 작업 공유

> 작업 폴더: `New_analyze/telegram_osint`  
> 목적: GDELT의 시차/요약 한계를 보완하기 위해, 공개 텔레그램 채널 기반의 실시간 OSINT 피드를 구축하고 사이트에 표시하는 실무 가능성을 검토

---

## 1. 왜 텔레그램을 보조 데이터로 봤는가

기존 모델링 데이터는 ACLED/GDELT/경제 지표 중심이다. 이 중 GDELT는 전 세계 뉴스 기반이라 범위는 넓지만, 실제 운영 관점에서는 다음 한계가 있다.

- 뉴스 반영까지 시차가 있을 수 있음
- 기사 제목/요약 중심이라 현장 상황 맥락이 부족할 수 있음
- 특정 지역의 속보성 충돌, 공습, 드론 공격, 피난/대피 신호를 빠르게 보기 어려움

그래서 텔레그램은 모델 학습 데이터가 아니라, 사이트의 **실시간 상황판 레이어**로 활용하는 방향이 적절하다고 판단했다.

---

## 2. 현재 구축한 것

현재는 아래 파이프라인이 동작한다.

```text
공개/허가된 Telegram 채널
→ 메시지 수집
→ SQLite DB 저장
→ 충돌 관련 키워드/국가/위치 추출
→ live_events.json 생성
→ Leaflet 지도 + 필터 가능한 HTML 피드 생성
→ 품질진단 audit_report.md 생성
```

생성 결과물:

- `artifacts/live_osint/live_events.db`: 원문 메시지와 추출 이벤트 저장 DB
- `artifacts/live_osint/live_events.json`: 대시보드/프론트엔드에서 읽을 수 있는 JSON
- `site/live_osint.html`: 지도와 필터가 포함된 정적 데모 사이트
- `artifacts/live_osint/audit_report.md`: 자동 품질진단 리포트

---

## 3. 활용 중인 채널

현재 `config/telegram_channels.json` 기준 활성 채널은 다음과 같다.

| 채널 | handle | 역할 | reliability |
|---|---|---|---|
| Liveuamap | `liveuamap` | 글로벌 분쟁 속보/지도형 aggregator | 0.75 |
| The Kyiv Independent | `KyivIndependent_official` | 우크라이나 전쟁 관련 현지 영문 매체 | 0.85 |
| Al Jazeera English | `aljazeeraenglish` | 중동/글로벌 충돌 뉴스 | 0.85 |
| Bellingcat EN | `bellingcat_en` | 속보보다는 검증/OSINT 분석 참고 | 0.90 |

비활성 후보:

| 채널 | handle | 상태 | 이유 |
|---|---|---|---|
| Combat Intel | `combatintel` | disabled | 속보성 aggregator라 노이즈/검증 리스크가 있어 추후 수동 검토 후 활성화 권장 |

주의: 텔레그램은 공개 채널 또는 모니터링 권한이 있는 채널만 대상으로 해야 하며, 개인 대화/비공개방은 수집 대상에서 제외한다.

---

## 4. 이벤트 추출 방식

현재는 투명하고 수정하기 쉬운 rule-based 방식으로 시작했다.

추출 항목:

- 충돌 관련 여부
- 국가 ISO3 코드
- 도시/지역명
- 위도/경도
- 위치 정밀도
- 이벤트 타입
- severity
- confidence
- 매칭 키워드
- 원문 링크

대표 키워드:

```text
airstrike, missile, rocket, drone, shelling, explosion,
clash, battle, attack, killed, wounded, casualties,
troop, evacuation, protest, riot
```

이벤트 타입 예시:

| 타입 | 의미 |
|---|---|
| `strike` | 미사일, 드론, 공습, 로켓 등 |
| `shelling_explosion` | 포격/폭발 |
| `armed_clash` | 교전/전투 |
| `civil_unrest` | 시위/폭동 |
| `military_movement` | 병력 이동/대피 |
| `conflict_signal` | 그 외 충돌 관련 신호 |

---

## 5. 위치 처리 방식

위치는 3단계로 구분한다.

| 값 | 의미 | 지도 표시 |
|---|---|---|
| `city` | 메시지에서 도시/지역명을 잡아 해당 좌표를 사용 | 빨간 마커 |
| `country` | 도시는 못 잡았지만 국가는 잡아서 국가 중심 좌표를 사용 | 파란 마커 |
| `missing` | 국가도 못 잡음 | 지도 표시 어려움 |

예:

```text
"Missile strike in Dnipro" → city 좌표
"Russian attacks across Ukraine" → Ukraine 국가 중심 fallback
```

국가 중심 fallback을 넣은 뒤, 좌표 누락률이 크게 줄었다.

```text
missing_coordinates: 52.3% → 11.4%
```

---

## 6. 현재 결과 요약

최근 실행 기준 자동 audit 결과:

```text
Raw messages: 201
Extracted rows: 201
Conflict events: 44
Fresh conflict events within 14 days: 20
```

채널별 감지율:

| 채널 | 수집 메시지 | 충돌 이벤트 | 감지율 |
|---|---:|---:|---:|
| KyivIndependent_official | 50 | 16 | 32.0% |
| liveuamap | 50 | 16 | 32.0% |
| aljazeeraenglish | 50 | 5 | 10.0% |
| bellingcat_en | 50 | 7 | 14.0% |

품질 지표:

| 항목 | 값 |
|---|---:|
| missing_country | 5건 |
| missing_location | 5건 |
| missing_coordinates | 5건 |
| country_centroid_fallback | 18건 |
| low_confidence_lt_0.75 | 0건 |
| possible_duplicate_groups | 0건 |

최근 14일 사이트 표시 기준:

```text
events: 20
city coordinates: 11
country fallback: 7
missing location: 2
```

---

## 7. 데모 사이트 공유 방법

팀원에게 가장 직관적으로 보여줄 파일은 다음이다.

```text
site/live_osint.html
```

브라우저에서 열면 다음 기능을 확인할 수 있다.

- Leaflet 기반 지도
- city/country fallback 마커 구분
- 국가 필터
- 채널 필터
- 이벤트 타입 필터
- 위치 정밀도 필터
- 최근 24시간/7일 시간 필터
- 지도 클릭 시 오른쪽 이벤트 상세 패널
- GDELT 기사 제목 컨텍스트 섹션
- 키워드/본문 검색
- 원문 텔레그램 링크 이동

로컬에서 열기:

```powershell
start site\live_osint.html
```

주의: Leaflet/OpenStreetMap 타일은 인터넷 연결이 필요하다. 외부 타일 로딩이 실패해도 fallback coordinate view가 나오도록 처리해두었다.

GDELT 기사 제목 컨텍스트는 현재 BigQuery service account 키를 받기 전 단계라 샘플 JSON으로 UI만 먼저 연결했다.

```text
artifacts/live_osint/gdelt_context.sample.json
artifacts/live_osint/gdelt_context.json
```

나중에 팀원이 수집한 `conflict-early-warning.conflict_ew.gdelt_titles`에 접근 가능해지면, BigQuery에서 국가별 최근 24시간/7일 기사 수와 대표 제목을 export해서 `gdelt_context.json`을 교체하면 된다.

---

## 8. 실행 방법

전체 파이프라인은 다음 명령 하나로 실행된다.

```powershell
python scripts\run_live_pipeline.py --limit-per-channel 50 --since-days 14
```

내부 실행 순서:

```text
collect
→ reprocess
→ export
→ build site
→ audit
```

수집 없이 기존 DB 기준으로 사이트/audit만 다시 만들 때:

```powershell
python scripts\run_live_pipeline.py --skip-collect --since-days 14
```

데모 데이터만 사용할 때:

```powershell
python scripts\run_live_pipeline.py --demo
```

---

## 9. 인증/보안 주의

텔레그램 연결에는 아래 값이 필요하다.

```text
TELEGRAM_API_ID
TELEGRAM_API_HASH
TELEGRAM_SESSION
```

이 값은 `.env.local`에 넣어 사용한다.

```text
.env.local
<TELEGRAM_SESSION>.session
```

위 파일들은 민감 파일이다. 절대 GitHub에 올리면 안 된다. 현재 `.gitignore`에 `.env*`, `*.session`, `*.session-journal`을 제외하도록 추가해두었다.

팀원에게 코드를 공유할 때는 `.env.local`과 `.session` 파일을 제외하고 공유해야 한다.

---

## 10. 자동화 방향성

현재는 로컬에서 명령을 실행하면 갱신되는 구조다. 실제 서비스 운영 시에는 아래 방식으로 자동화할 수 있다.

### VPS/서버 + cron 권장

가장 안정적인 방식이다.

```text
서버에 repo 배포
→ 환경변수/세션 등록
→ cron으로 12시간마다 run_live_pipeline_job.sh 실행
→ live_events.json / live_osint.html 갱신
→ Nginx 또는 대시보드 서버가 파일/API 제공
```

배포 상세 문서:

```text
docs/deployment_vps_cron.md
```

cron 예시:

```bash
0 */12 * * * APP_DIR=/opt/kubig-telegram-osint PYTHON_BIN=/opt/kubig-telegram-osint/.venv/bin/python /opt/kubig-telegram-osint/scripts/run_live_pipeline_job.sh >> /opt/kubig-telegram-osint/logs/pipeline.log 2>&1
```

### GitHub Actions도 가능하지만 주의 필요

GitHub Actions도 주기 실행은 가능하지만, Telethon `.session` 파일을 안전하게 Secret으로 관리해야 한다. `.session`은 로그인 토큰에 가까우므로 실운영 메인으로는 VPS 방식이 더 안정적이다.

---

## 11. 현재 한계와 다음 개선 후보

현재 구조는 실무 검토용 MVP에 가깝다. 다음 개선 후보는 아래와 같다.

1. 중복/유사 이벤트 병합
   - 같은 국가, 비슷한 시간대, 비슷한 키워드의 메시지를 하나의 사건 그룹으로 묶기

2. 지도 마커 클러스터링
   - 같은 지역에 마커가 많을 때 겹침 완화

3. 지역 alias 확장
   - 도시/지역명을 더 추가해 `missing`을 줄이기

4. URL slug 기반 국가/지역 추론
   - Bellingcat/Liveuamap 링크에서 국가명이나 지역명을 보조 추출

5. 모델 예측 결과와 결합
   - 국가별 risk score 옆에 최근 Telegram signal 수, 주요 키워드, 대표 이벤트를 표시

6. API화
   - `live_events.json`을 정적 파일로 쓰는 대신 `/api/live-events` 형태로 제공

---

## 12. 결론

이번 작업으로 텔레그램 공개 채널을 활용한 실시간 무력충돌 정보 레이어의 MVP를 구축했다.

핵심은 다음과 같다.

- 텔레그램은 모델 학습용 데이터가 아니라 실시간 운영/상황판 레이어로 활용
- 공개 채널 allowlist 기반 수집
- rule-based 충돌 이벤트 추출
- 국가/도시 좌표 추론 및 국가 fallback
- 지도 기반 HTML 데모 구현
- 자동 품질진단 리포트 생성
- 향후 서버 cron 또는 GitHub Actions 기반 자동화 가능

팀원에게 공유할 때는 `site/live_osint.html`을 먼저 보여주면 가장 직관적이다.
