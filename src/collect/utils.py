"""
공통 유틸: 재시도, 로깅, 체크포인트
"""

import json
import logging
import time
from datetime import date, datetime
from functools import wraps
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────
# 로깅
# ──────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """수집 스크립트용 로거. 콘솔 + 파일(logs/) 동시 출력."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 중복 핸들러 방지

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # 콘솔
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 파일
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────
# 재시도 데코레이터
# ──────────────────────────────────────────────

def retry(max_attempts: int = 5, base_delay: float = 2.0, exceptions: tuple = (Exception,)):
    """
    지수 백오프 재시도 데코레이터.

    Args:
        max_attempts: 최대 시도 횟수 (초기 시도 포함)
        base_delay:   첫 번째 재시도 대기 시간(초). 이후 2배씩 증가
        exceptions:   재시도할 예외 타입
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} 최종 실패 ({attempt}회): {e}")
                        raise
                    logger.warning(f"{func.__name__} 실패 ({attempt}/{max_attempts}), "
                                   f"{delay:.0f}초 후 재시도: {e}")
                    time.sleep(delay)
                    delay *= 2
        return wrapper
    return decorator


# ──────────────────────────────────────────────
# 체크포인트
# ──────────────────────────────────────────────

class Checkpoint:
    """
    수집 진행 상태를 JSON 파일로 저장/로드.

    사용 예:
        ckpt = Checkpoint("input/raw/acled/.ckpt_historical.json")
        if ckpt.is_done("UKR_2023"):
            continue
        # ... 수집 ...
        ckpt.mark_done("UKR_2023")
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {"done": [], "meta": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2, default=str)

    def is_done(self, key: str) -> bool:
        return key in self._data["done"]

    def mark_done(self, key: str, meta: dict | None = None) -> None:
        if key not in self._data["done"]:
            self._data["done"].append(key)
        if meta:
            self._data["meta"][key] = {**meta, "completed_at": datetime.utcnow().isoformat()}
        self._save()

    def get_meta(self, key: str) -> dict:
        return self._data["meta"].get(key, {})

    def reset(self, key: str) -> None:
        """특정 키 완료 상태 초기화 (재수집 필요 시)."""
        self._data["done"] = [k for k in self._data["done"] if k != key]
        self._data["meta"].pop(key, None)
        self._save()

    def reset_all(self) -> None:
        self._data = {"done": [], "meta": {}}
        self._save()

    @property
    def done_keys(self) -> list[str]:
        return list(self._data["done"])


# ──────────────────────────────────────────────
# 날짜 유틸
# ──────────────────────────────────────────────

def date_range_months(start: date, end: date) -> list[tuple[date, date]]:
    """
    시작~종료 기간을 월 단위 (month_start, month_end) 튜플 리스트로 분할.
    API 페이지네이션 단위 분할에 사용.
    """
    from calendar import monthrange

    ranges = []
    cur = start.replace(day=1)
    while cur <= end:
        last_day = monthrange(cur.year, cur.month)[1]
        month_end = cur.replace(day=last_day)
        ranges.append((cur, min(month_end, end)))
        # 다음 달 1일
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            cur = cur.replace(month=cur.month + 1, day=1)
    return ranges
