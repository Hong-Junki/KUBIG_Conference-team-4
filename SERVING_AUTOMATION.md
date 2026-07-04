# 서빙 자동화 배포 지침 (Cloud Run + Scheduler)

> 15분 주기 실시간 스코어링. 컨테이너(`Dockerfile.serve`)를 팀 크레딧 프로젝트
> `conflict-ew-mvp-20260604`에 배포. **배포는 대시보드/DB 담당**, 이 문서로 turnkey.

## 아키텍처 (2개 잡)

```
[일배치 1회]  means-update Job
   scripts/update_means_table.py → gkg_embeddings_means 테이블 갱신 (뉴스 임베딩)
        ↑ 임베딩은 비싸서 하루 1회

[15분]  score Job = Cloud Scheduler(*/15) → Cloud Run Job
   ① features 모드: raw BQ(events·econ·titles) + gkg_embeddings_means → model_input(BQ)
   ② score 모드:    model_input → model_scores(BQ, run_ts=now)
   → 대시보드는 model_scores 의 최신 run_ts 를 읽음
```

- 병현 GDELT 수집(events·titles)은 이미 15분 주기 → raw 는 실시간.
- 15분 잡은 임베딩을 새로 하지 않고 `gkg_embeddings_means`(하루 1회 갱신)를 읽음.
  따라서 스코어는 15분마다 GDELT 이벤트 변화를 반영해 갱신, 뉴스-NLP 신호는 하루 단위.

## 컨테이너 (이미지 빌드 — CI 또는 로컬 docker)

```bash
docker build -f Dockerfile.serve -t <REGION>-docker.pkg.dev/conflict-ew-mvp-20260604/serve/onset-serve:latest .
docker push <REGION>-docker.pkg.dev/conflict-ew-mvp-20260604/serve/onset-serve:latest
```
가중치·아티팩트는 레포에 커밋돼 있어(`output/models/`, `anchor_embeddings.npz`) checkout만으로 빌드됨.

## 필요 secret (Cloud Run 잡 env)
- `GOOGLE_APPLICATION_CREDENTIALS` = 내부 SA 키 마운트 경로 (BQ read/write 권한)
- `OPENAI_API_KEY` = 임베딩용 (means-update 잡만 필요)
- `GCP_PROJECT=conflict-ew-mvp-20260604`, `BQ_DATASET=conflict_ew`

## 잡 1: means-update (일 1회, 예: 매일 KST 11:00)

```bash
# entrypoint 우회하고 직접 실행 (또는 별도 이미지 target)
python scripts/update_means_table.py --start <어제> --end <오늘> --suffix inc<YYYYMMDD>
```
- Cloud Scheduler cron `0 2 * * *` (UTC) → Cloud Run Job.
- 임베딩 비용: 하루치 제목 ≈ 수천 건, ~$0.02/일.

## 잡 2: 15분 스코어 (Cloud Scheduler `*/15 * * * *`)

컨테이너 2번 실행(또는 한 잡에서 순차):
```bash
# ① 피처 (SERVE_MEANS_SOURCE=bq → 임베딩 스킵, means 는 테이블에서)
SERVE_MEANS_SOURCE=bq  onset-serve features --start <오늘-90d> --end <오늘> --run-ts <ISO>
# ② 스코어
onset-serve score --run-ts <ISO>
```
- `--start` 는 LSTM 45일 + rolling 30일 lookback 포함해 넉넉히(≈90일).
- `run-ts` 는 호출측(스케줄러)에서 현재시각 주입.
- 비용: features run 당 BQ 스캔 ~$0.1 → 15분×96회/일 ≈ $10/일(크레딧). 비싸면 주기 완화(스펙 "15분~1시간").

## 대시보드 조회 (최신 run 만)
```sql
SELECT * FROM `conflict-ew-mvp-20260604.conflict_ew.model_scores`
QUALIFY ROW_NUMBER() OVER (PARTITION BY country, date ORDER BY run_ts DESC) = 1
```

## 미완/주의
- 이미지 빌드·Cloud Run/Scheduler 배포는 미실행(배포 담당 몫).
- `update_means_table.py` 는 로컬 검증 전(임베딩 서브프로세스 포함) — 첫 배포 시 dry 확인 권장.
- 완전한 intraday(현재시각 rolling-24h) 피처는 미구현 — 현재는 calendar-day(오늘분 누적).
