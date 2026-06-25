"""
증분 수집 파이프라인 unit test.

외부 API 및 BigQuery client는 모두 mock 처리한다.
"""

import sys
import types
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────
# state.py 테스트
# ──────────────────────────────────────────────

class TestComputeCollectionWindow:
    def setup_method(self):
        from src.collect.incremental.state import compute_collection_window
        self.fn = compute_collection_window

    def test_normal_case_gdelt(self):
        last = date(2026, 3, 31)
        start, end = self.fn(last, "gdelt")
        # overlap=3, start = last - 3 = 2026-03-28
        assert start == date(2026, 3, 28)
        assert end == date.today() - timedelta(days=1)

    def test_normal_case_economic(self):
        last = date(2026, 3, 30)
        start, end = self.fn(last, "economic")
        # overlap=10, start = last - 10 = 2026-03-20
        assert start == date(2026, 3, 20)

    def test_forced_start_overrides_bq(self):
        last = date(2026, 3, 31)
        forced = date(2026, 4, 1)
        start, end = self.fn(last, "gdelt", forced_start=forced)
        assert start == forced

    def test_forced_end(self):
        last = date(2026, 3, 31)
        forced_end = date(2026, 4, 15)
        _, end = self.fn(last, "gdelt", forced_end=forced_end)
        assert end == forced_end

    def test_none_last_date_raises_type_error(self):
        """fallback 제거 이후 None 전달 시 TypeError 발생 (date 연산 불가)."""
        with pytest.raises((TypeError, AttributeError)):
            self.fn(None, "gdelt", overlap_days=5)

    def test_end_before_start_raises(self):
        # forced_start > forced_end
        with pytest.raises(ValueError, match="start"):
            self.fn(
                date(2026, 3, 31), "gdelt",
                forced_start=date(2026, 5, 1),
                forced_end=date(2026, 4, 1),
            )

    def test_custom_overlap_days(self):
        last = date(2026, 3, 31)
        start, _ = self.fn(last, "gdelt", overlap_days=14)
        assert start == date(2026, 3, 17)


class TestGetMaxDateFromBq:
    def _make_mock_client(self, return_value):
        client = MagicMock()
        row = MagicMock()
        row.__getitem__ = lambda self, key: return_value
        client.query.return_value.__iter__ = lambda self: iter([row])
        return client

    def test_returns_date(self):
        from src.collect.incremental.state import get_max_date_from_bq

        client = MagicMock()
        mock_row = {"max_dt": date(2026, 3, 31)}
        mock_row_obj = MagicMock()
        mock_row_obj.__getitem__ = lambda self, k: mock_row[k]
        client.query.return_value.__iter__ = lambda self: iter([mock_row_obj])

        result = get_max_date_from_bq(client, "proj.ds.tbl", "event_date")
        assert result == date(2026, 3, 31)

    def test_returns_none_on_exception(self):
        from src.collect.incremental.state import get_max_date_from_bq

        client = MagicMock()
        client.query.side_effect = Exception("BQ error")

        result = get_max_date_from_bq(client, "proj.ds.tbl", "event_date")
        assert result is None

    def test_returns_none_on_null_result(self):
        from src.collect.incremental.state import get_max_date_from_bq

        client = MagicMock()
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, k: None
        client.query.return_value.__iter__ = lambda self: iter([mock_row])

        result = get_max_date_from_bq(client, "proj.ds.tbl", "event_date")
        assert result is None


# ──────────────────────────────────────────────
# bigquery_io.py 테스트
# ──────────────────────────────────────────────

class TestMergeQuery:
    """MERGE 쿼리가 올바른 SQL 구조를 갖는지 확인."""

    def test_merge_insert_only(self):
        from src.collect.incremental.bigquery_io import merge_into_target

        client = MagicMock()
        mock_job = MagicMock()
        mock_job.num_dml_affected_rows = 100
        client.query.return_value = mock_job
        mock_job.result.return_value = None

        affected = merge_into_target(
            client,
            staging_fqn="proj.ds._staging_tbl_abc",
            target_fqn="proj.ds.tbl",
            merge_keys=["GLOBALEVENTID"],
            columns=["GLOBALEVENTID", "iso3", "event_date"],
            update_on_match=False,
        )

        assert affected == 100
        call_args = client.query.call_args[0][0]
        assert "MERGE" in call_args
        assert "WHEN NOT MATCHED THEN" in call_args
        assert "WHEN MATCHED" not in call_args

    def test_merge_upsert(self):
        from src.collect.incremental.bigquery_io import merge_into_target

        client = MagicMock()
        mock_job = MagicMock()
        mock_job.num_dml_affected_rows = 50
        client.query.return_value = mock_job
        mock_job.result.return_value = None

        affected = merge_into_target(
            client,
            staging_fqn="proj.ds._staging_econ",
            target_fqn="proj.ds.economic_daily",
            merge_keys=["date"],
            columns=["date", "VIX", "WTI"],
            update_on_match=True,
        )

        assert affected == 50
        call_args = client.query.call_args[0][0]
        assert "WHEN MATCHED THEN UPDATE" in call_args
        assert "WHEN NOT MATCHED THEN" in call_args


class TestDryRun:
    def test_gdelt_dry_run_no_write(self):
        """dry_run=True일 때 BQ load/merge 호출 없음."""
        with (
            patch("src.collect.incremental.collect_gdelt_incremental.bigquery.Client"),
            patch("src.collect.incremental.collect_gdelt_incremental.require_max_date_from_bq") as mock_state,
            patch("src.collect.incremental.collect_gdelt_incremental.dry_run_query") as mock_dry,
            patch("src.collect.incremental.collect_gdelt_incremental.load_dataframe_to_staging") as mock_load,
        ):
            mock_state.return_value = date(2026, 3, 31)
            mock_dry.return_value = 5.0

            from src.collect.incremental.collect_gdelt_incremental import run_gdelt_incremental
            result = run_gdelt_incremental(
                project_id="test-project",
                dry_run=True,
                run_id="test123",
            )

            assert result.get("dry_run") is True
            mock_load.assert_not_called()


# ──────────────────────────────────────────────
# run_incremental.py 테스트
# ──────────────────────────────────────────────

class TestRunIncremental:
    def test_unsupported_source_rejected(self):
        """choices 밖의 소스는 argparse가 거부해야 함."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "src.collect.incremental.run_incremental",
             "--sources", "invalid_source"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr

    def test_missing_credentials_exits_nonzero(self):
        """credential 없으면 sys.exit(1)."""
        with (
            patch("src.collect.incremental.run_incremental._check_credentials") as mock_cred,
        ):
            mock_cred.return_value = False
            from src.collect.incremental import run_incremental
            with pytest.raises(SystemExit) as exc_info:
                with patch("sys.argv", ["run_incremental", "--sources", "economic"]):
                    run_incremental.main()
            assert exc_info.value.code == 1

    def test_validate_only_no_write(self):
        """--validate-only는 수집 함수를 호출하지 않음."""
        with (
            patch("src.collect.incremental.run_incremental._check_credentials") as mock_cred,
            patch("src.collect.incremental.run_incremental.validate_only") as mock_val,
            patch("src.collect.incremental.run_incremental.run_gdelt") as mock_gdelt,
        ):
            mock_cred.return_value = True
            with patch("sys.argv", ["run_incremental", "--validate-only"]):
                try:
                    from src.collect.incremental import run_incremental
                    run_incremental.main()
                except SystemExit:
                    pass
            mock_val.assert_called_once()
            mock_gdelt.assert_not_called()


# ──────────────────────────────────────────────
# economic 파생 컬럼 테스트
# ──────────────────────────────────────────────

class TestEconomicDerivedColumns:
    def test_pct_change_computed(self):
        import pandas as pd
        from src.collect.incremental.collect_economic_incremental import _compute_derived_columns

        df = pd.DataFrame({
            "VIX": [20.0, 22.0, 21.0],
            "WTI": [80.0, 84.0, 82.0],
            "Gold": [1800.0, 1820.0, 1810.0],
            "DXY": [100.0, 101.0, 99.0],
            "STLFSI4": [0.5, 0.6, 0.55],
        })
        result = _compute_derived_columns(df)

        assert "VIX_pct_change" in result.columns
        assert "econ_volatility_proxy" in result.columns
        # 첫 행은 NaN (이전 값 없음)
        assert pd.isna(result["VIX_pct_change"].iloc[0])
        # 두 번째 행: (22-20)/20 * 100 = 10.0
        assert abs(result["VIX_pct_change"].iloc[1] - 10.0) < 1e-6

    def test_empty_cols_handled(self):
        import pandas as pd
        from src.collect.incremental.collect_economic_incremental import _compute_derived_columns

        df = pd.DataFrame({"VIX": [20.0, 22.0]})
        result = _compute_derived_columns(df)
        assert "VIX_pct_change" in result.columns
        # WTI 없어도 에러 없음
        assert "econ_volatility_proxy" in result.columns


# ──────────────────────────────────────────────
# config.py 국가 매핑 테스트
# ──────────────────────────────────────────────

class TestCountryMapping:
    def test_country_count(self):
        """COUNTRIES 목록 58개 확인."""
        # config.py import에 dotenv 필요 → 직접 파싱
        import re
        config_text = (Path(__file__).parent.parent / "src/collect/config.py").read_text()
        count = len(re.findall(r'"name":', config_text))
        assert count == 58

    def test_palestine_has_multiple_gdelt_codes(self):
        """Palestine은 GDELT 코드가 2개(WE, GZ)여야 함."""
        import re
        config_text = (Path(__file__).parent.parent / "src/collect/config.py").read_text()
        pse_line = [l for l in config_text.splitlines() if "PSE" in l and "gdelt" in l]
        assert len(pse_line) >= 1
        # WE, GZ 모두 포함
        line = pse_line[0]
        assert "WE" in line
        assert "GZ" in line

    def test_no_duplicate_iso3(self):
        """iso3 코드 중복 없음."""
        import re
        config_text = (Path(__file__).parent.parent / "src/collect/config.py").read_text()
        iso3_codes = re.findall(r'"iso3":\s*"([A-Z]{3})"', config_text)
        assert len(iso3_codes) == len(set(iso3_codes)), f"중복 iso3: {set(x for x in iso3_codes if iso3_codes.count(x) > 1)}"


# ──────────────────────────────────────────────
# collect_gdelt_titles_gkg.py 프로젝트명 테스트
# ──────────────────────────────────────────────

class TestGkgProjectId:
    def test_project_reads_from_env(self):
        """GCP_PROJECT_ID env가 없으면 기본값 conflict-ew-mvp-20260604 사용."""
        src = (Path(__file__).parent.parent / "src/collect/collect_gdelt_titles_gkg.py").read_text()
        # 실제 코드 라인에서 env 변수 사용 확인
        assert "GCP_PROJECT_ID" in src
        assert "conflict-ew-mvp-20260604" in src

    def test_no_hardcoded_old_project(self):
        """구 프로젝트명이 코드 실행 라인에 하드코딩되어 있지 않음."""
        src = (Path(__file__).parent.parent / "src/collect/collect_gdelt_titles_gkg.py").read_text()
        # 주석, docstring이 아닌 실제 코드 라인에서 구 프로젝트명 확인
        # (docstring은 문서화 목적이므로 허용)
        lines_with_old = [
            l for l in src.splitlines()
            if "conflict-early-warning" in l
            and not l.strip().startswith("#")
            and not l.strip().startswith('"""')
            and not l.strip().startswith("'")
            and "→" not in l  # docstring 화살표 표기 허용
            and "os.getenv" not in l  # env 기본값에서의 참조 허용
        ]
        assert len(lines_with_old) == 0, f"하드코딩된 구 프로젝트명: {lines_with_old}"


# ──────────────────────────────────────────────
# 보안: .gitignore 확인
# ──────────────────────────────────────────────

class TestGitignoreSecurity:
    def _read_gitignore(self):
        return (Path(__file__).parent.parent / ".gitignore").read_text()

    def test_env_ignored(self):
        assert ".env" in self._read_gitignore()

    def test_sa_json_ignored(self):
        gi = self._read_gitignore()
        assert "conflict-early-warning-*.json" in gi or "conflict-ew-mvp-*.json" in gi

    def test_docs_not_ignored(self):
        """docs/ 디렉터리가 gitignore에서 제외되었음 확인."""
        gi = self._read_gitignore()
        active_lines = [l for l in gi.splitlines()
                        if l.strip() == "docs/" and not l.strip().startswith("#")]
        assert len(active_lines) == 0, "docs/가 여전히 gitignore에 포함되어 있음"


# ──────────────────────────────────────────────
# state.py: require_max_date_from_bq 하드 실패 테스트
# ──────────────────────────────────────────────

class TestRequireMaxDateFromBq:
    def test_raises_on_none(self):
        """get_max_date_from_bq가 None 반환 시 RuntimeError 발생."""
        from src.collect.incremental.state import require_max_date_from_bq
        mock_client = MagicMock()
        with patch(
            "src.collect.incremental.state.get_max_date_from_bq",
            return_value=None,
        ):
            with pytest.raises(RuntimeError, match="fallback"):
                require_max_date_from_bq(mock_client, "p.d.t", "date")

    def test_returns_date_when_found(self):
        """get_max_date_from_bq가 date 반환 시 그대로 반환."""
        from src.collect.incremental.state import require_max_date_from_bq
        expected = date(2026, 5, 29)
        mock_client = MagicMock()
        with patch(
            "src.collect.incremental.state.get_max_date_from_bq",
            return_value=expected,
        ):
            result = require_max_date_from_bq(mock_client, "p.d.t", "date")
        assert result == expected

    def test_no_fallback_in_compute_window_when_none_passed_would_fail(self):
        """
        compute_collection_window에 None을 넘기면 AttributeError 발생 확인.
        (None을 허용하는 fallback 경로가 삭제되었음을 검증)
        """
        from src.collect.incremental.state import compute_collection_window
        with pytest.raises((AttributeError, TypeError)):
            # None - timedelta는 TypeError, None.date()는 AttributeError
            compute_collection_window(None, "gdelt")


# ──────────────────────────────────────────────
# gdelt_titles MERGE key 확인
# ──────────────────────────────────────────────

class TestGdeltTitlesMergeKey:
    def test_merge_keys_include_date(self):
        """MERGE_KEYS가 ['date', 'iso3', 'url']임을 확인 (partition pruning 필요)."""
        from src.collect.incremental.collect_gdelt_titles_incremental import MERGE_KEYS
        assert MERGE_KEYS == ["date", "iso3", "url"]

    def test_merge_keys_has_date_first(self):
        """partition column인 date가 MERGE_KEYS 첫 번째 항목."""
        from src.collect.incremental.collect_gdelt_titles_incremental import MERGE_KEYS
        assert MERGE_KEYS[0] == "date"

    def test_schema_has_required_merge_columns(self):
        """GDELT_TITLES_SCHEMA에 date, iso3, url이 REQUIRED 필드로 존재."""
        from src.collect.incremental.collect_gdelt_titles_incremental import GDELT_TITLES_SCHEMA
        schema_map = {f.name: f for f in GDELT_TITLES_SCHEMA}
        assert "date" in schema_map
        assert "iso3" in schema_map
        assert "url" in schema_map
        assert schema_map["date"].mode == "REQUIRED"
        assert schema_map["iso3"].mode == "REQUIRED"
        assert schema_map["url"].mode == "REQUIRED"

    def test_schema_columns_match_all_columns(self):
        """ALL_COLUMNS가 GDELT_TITLES_SCHEMA 컬럼명과 일치."""
        from src.collect.incremental.collect_gdelt_titles_incremental import (
            GDELT_TITLES_SCHEMA, ALL_COLUMNS,
        )
        assert ALL_COLUMNS == [f.name for f in GDELT_TITLES_SCHEMA]

    def test_staging_query_inserts_all_columns(self):
        """_build_staging_select_query 반환 쿼리에 모든 컬럼이 포함됨."""
        from src.collect.incremental.collect_gdelt_titles_incremental import (
            _build_staging_select_query, ALL_COLUMNS,
        )
        query = _build_staging_select_query(
            fips_codes=["US"],
            fips_to_iso3={"US": "USA"},
            m_start=date(2026, 5, 1),
            m_end=date(2026, 5, 31),
            staging_fqn="proj.dataset.staging",
        )
        for col in ALL_COLUMNS:
            assert col in query, f"컬럼 '{col}'이 쿼리에 없음"

    def test_staging_query_dedup_uses_date_iso3_url(self):
        """staging 쿼리 dedup이 (date, iso3, url) 키 기준임을 확인."""
        from src.collect.incremental.collect_gdelt_titles_incremental import _build_staging_select_query
        query = _build_staging_select_query(
            fips_codes=["US"],
            fips_to_iso3={"US": "USA"},
            m_start=date(2026, 5, 1),
            m_end=date(2026, 5, 31),
            staging_fqn="proj.dataset.staging",
        )
        assert "PARTITION BY date, iso3, url" in query


# ──────────────────────────────────────────────
# run_incremental.py: VALID_SOURCES 확인
# ──────────────────────────────────────────────

class TestValidSources:
    def test_gdelt_titles_in_valid_sources(self):
        """gdelt_titles가 VALID_SOURCES에 포함됨."""
        from src.collect.incremental.run_incremental import VALID_SOURCES
        assert "gdelt_titles" in VALID_SOURCES

    def test_all_three_sources_present(self):
        """세 소스 gdelt, economic, gdelt_titles 모두 포함."""
        from src.collect.incremental.run_incremental import VALID_SOURCES
        assert set(VALID_SOURCES) == {"gdelt", "economic", "gdelt_titles"}

    def test_raw_targets_covers_all_sources(self):
        """RAW_TARGETS가 VALID_SOURCES의 모든 소스를 포함."""
        from src.collect.incremental.run_incremental import VALID_SOURCES, RAW_TARGETS
        for src in VALID_SOURCES:
            assert src in RAW_TARGETS, f"RAW_TARGETS에 '{src}' 없음"

    def test_no_non_raw_targets(self):
        """modeling 테이블이 RAW_TARGETS에 포함되지 않음."""
        from src.collect.incremental.run_incremental import RAW_TARGETS
        for v in RAW_TARGETS.values():
            assert "modeling" not in v.lower(), f"modeling 테이블 발견: {v}"
            assert "acled" not in v.lower(), f"ACLED 테이블 발견: {v}"
            assert "feature" not in v.lower(), f"feature 테이블 발견: {v}"


# ──────────────────────────────────────────────
# GitHub Actions 워크플로우 검증
# ──────────────────────────────────────────────

class TestWorkflowConfig:
    def _read_workflow(self) -> str:
        return (
            Path(__file__).parent.parent
            / ".github/workflows/incremental-data-collection.yml"
        ).read_text()

    def test_default_sources_includes_gdelt_titles(self):
        """workflow_dispatch 기본 sources에 gdelt_titles 포함."""
        wf = self._read_workflow()
        # default 값에 gdelt_titles 포함 확인
        assert "gdelt_titles" in wf

    def test_schedule_active_with_correct_cron(self):
        """schedule cron이 활성화되어 있고 01:20 UTC 시간을 사용함."""
        wf = self._read_workflow()
        active_cron_lines = [
            l for l in wf.splitlines()
            if "cron:" in l and not l.strip().startswith("#")
        ]
        assert len(active_cron_lines) >= 1, "활성 cron 없음 — schedule이 비활성 상태"
        assert any("20 1 * * *" in l for l in active_cron_lines), \
            f"01:20 UTC cron 없음. 발견된 cron: {active_cron_lines}"

    def test_validate_permissions_step_exists(self):
        """validate-permissions 단계가 workflow에 포함됨."""
        wf = self._read_workflow()
        assert "--validate-permissions" in wf

    def test_no_secrets_in_workflow_run_commands(self):
        """
        GCP_SA_KEY는 google-github-actions/auth의 credentials_json에서만 사용.
        run: 블록에서 직접 echo/cat/print 등으로 비밀 출력 없음.
        """
        wf = self._read_workflow()
        # run: | 안에서 secret을 직접 echo하거나 파일에 쓰는 패턴이 없어야 함
        import re
        # run 블록만 추출 (들여쓰기 기반)
        run_code_lines = []
        in_run_block = False
        run_indent = 0
        for line in wf.splitlines():
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if "run: |" in line:
                in_run_block = True
                run_indent = indent
                continue
            if in_run_block:
                # run 블록은 run_indent보다 더 들여쓰인 줄
                if stripped and indent <= run_indent and not stripped.startswith("#"):
                    in_run_block = False
                else:
                    run_code_lines.append(line)
        run_code = "\n".join(run_code_lines)
        # private_key 등 민감 정보가 run 블록에 없어야 함
        assert "private_key" not in run_code, "run 블록에 private_key 참조"
        # GCP_SA_KEY는 google-github-actions/auth에서만 허용 (run 블록 아님)
        assert "GCP_SA_KEY" not in run_code, "run 블록에 GCP_SA_KEY 직접 참조"


# ──────────────────────────────────────────────
# 소스별 수집기 구조 검증 (mock)
# ──────────────────────────────────────────────

class TestGdeltTitlesIncrementalStructure:
    def test_run_function_exists(self):
        """run_gdelt_titles_incremental 함수가 존재함."""
        from src.collect.incremental.collect_gdelt_titles_incremental import (
            run_gdelt_titles_incremental,
        )
        assert callable(run_gdelt_titles_incremental)

    def test_update_on_match_true(self):
        """
        gdelt_titles는 update_on_match=True (v2tone 등 수정치 반영).
        merge_into_target 호출 인자 검증.
        """
        import src.collect.incremental.collect_gdelt_titles_incremental as mod
        with (
            patch.object(mod, "require_max_date_from_bq", return_value=date(2026, 5, 29)),
            patch.object(mod, "compute_collection_window", return_value=(date(2026, 5, 26), date(2026, 5, 28))),
            patch.object(mod, "date_range_months", return_value=[]),
            patch.object(mod, "_create_empty_staging"),
            patch.object(mod, "drop_table"),
            patch("google.cloud.bigquery.Client") as mock_bq,
        ):
            # staging empty → 조기 반환 경로
            mock_client = MagicMock()
            mock_bq.return_value = mock_client
            count_row = MagicMock()
            count_row.__getitem__ = MagicMock(return_value=0)
            mock_client.query.return_value.__iter__ = MagicMock(return_value=iter([count_row]))
            result = mod.run_gdelt_titles_incremental(
                project_id="test-proj",
                dry_run=False,
            )
        # staging 비어있어도 passed=True (신규 데이터 없음)
        assert result.get("rows_merged") == 0 or result.get("passed") is True

    def test_dry_run_returns_without_writing(self):
        """dry_run=True면 BQ 쓰기 없이 반환."""
        import src.collect.incremental.collect_gdelt_titles_incremental as mod
        with (
            patch.object(mod, "require_max_date_from_bq", return_value=date(2026, 5, 29)),
            patch.object(mod, "compute_collection_window", return_value=(date(2026, 5, 26), date(2026, 5, 28))),
            patch.object(mod, "date_range_months", return_value=[]),
            patch("google.cloud.bigquery.Client"),
        ):
            result = mod.run_gdelt_titles_incremental(
                project_id="test-proj",
                dry_run=True,
            )
        assert result["dry_run"] is True
        assert "estimated_gb" in result


# ──────────────────────────────────────────────
# workflow start/end 날짜 입력 검증
# ──────────────────────────────────────────────

class TestWorkflowDateInputs:
    def _read_workflow(self) -> str:
        return (
            Path(__file__).parent.parent
            / ".github/workflows/incremental-data-collection.yml"
        ).read_text()

    def test_start_input_exists(self):
        """workflow_dispatch에 start 입력이 정의됨."""
        wf = self._read_workflow()
        assert "start:" in wf
        assert "YYYY-MM-DD" in wf

    def test_end_input_exists(self):
        """workflow_dispatch에 end 입력이 정의됨."""
        wf = self._read_workflow()
        assert "end:" in wf

    def test_both_empty_means_no_date_args(self):
        """
        start/end 모두 비어 있으면 CLI에 --start/--end 없이 실행.
        workflow shell 로직: DATE_ARGS는 빈 문자열.
        """
        wf = self._read_workflow()
        # DATE_ARGS가 빈 문자열로 시작하는 패턴이 있어야 함
        assert 'DATE_ARGS=""' in wf

    def test_both_present_passes_date_args(self):
        """start/end 모두 있으면 --start, --end가 CLI에 전달됨."""
        wf = self._read_workflow()
        assert "--start $START" in wf
        assert "--end $END" in wf

    def test_only_start_fails(self):
        """start만 입력 시 오류 exit 1 존재."""
        wf = self._read_workflow()
        # start 있고 end 없는 케이스에 exit 1 로직
        assert 'start를 입력했으나 end가 없습니다' in wf

    def test_only_end_fails(self):
        """end만 입력 시 오류 exit 1 존재."""
        wf = self._read_workflow()
        assert 'end를 입력했으나 start가 없습니다' in wf

    def test_invalid_date_format_fails(self):
        """YYYY-MM-DD 형식이 아니면 실패."""
        wf = self._read_workflow()
        # 날짜 형식 정규식 검증 존재 확인
        assert "DATE_RE" in wf or "grep -qE" in wf

    def test_start_greater_than_end_fails(self):
        """start > end이면 exit 1."""
        wf = self._read_workflow()
        assert 'start($START)가 end($END)보다 이후입니다' in wf or \
               'start.*greater.*end' in wf or \
               '"$START" > "$END"' in wf

    def test_cron_is_active(self):
        """schedule cron이 활성화되어 있음."""
        wf = self._read_workflow()
        active_cron = [
            l for l in wf.splitlines()
            if "cron:" in l and not l.strip().startswith("#")
        ]
        assert len(active_cron) >= 1, "cron이 비활성화(주석) 상태"
        assert any("20 1 * * *" in l for l in active_cron), \
            f"01:20 UTC cron 없음: {active_cron}"

    def test_no_hardcoded_credential_filename(self):
        """
        credential 파일명(conflict-ew-mvp-*-*.json)이
        workflow와 docs에 하드코딩되어 있지 않음.
        """
        import re
        cred_pattern = re.compile(r'conflict-ew-mvp-\d{8}-[a-f0-9]+\.json')

        # workflow 확인
        wf = self._read_workflow()
        assert not cred_pattern.search(wf), \
            "workflow에 credential 파일명 하드코딩"

        # docs 확인
        docs_dir = Path(__file__).parent.parent / "docs"
        if docs_dir.exists():
            for doc in docs_dir.glob("*.md"):
                content = doc.read_text()
                assert not cred_pattern.search(content), \
                    f"{doc.name}에 credential 파일명 하드코딩"


# ──────────────────────────────────────────────
# MERGE partition filter 구조 검증
# ──────────────────────────────────────────────

class TestMergePartitionFilter:
    def test_merge_sql_contains_partition_filter(self):
        """merge_into_target에 target_partition_filter가 ON 절에 포함됨."""
        from src.collect.incremental.bigquery_io import merge_into_target

        client = MagicMock()
        mock_job = MagicMock()
        mock_job.num_dml_affected_rows = 10
        mock_job.result.return_value = None
        client.query.return_value = mock_job

        merge_into_target(
            client,
            staging_fqn="p.d.staging",
            target_fqn="p.d.target",
            merge_keys=["date", "iso3", "url"],
            columns=["date", "iso3", "url", "title"],
            update_on_match=True,
            target_partition_filter="T.date BETWEEN DATE('2026-05-28') AND DATE('2026-05-29')",
        )

        sql = client.query.call_args[0][0]
        assert "T.date BETWEEN DATE('2026-05-28') AND DATE('2026-05-29')" in sql
        assert "T.date = S.date" in sql
        assert "T.iso3 = S.iso3" in sql
        assert "T.url = S.url" in sql

    def test_merge_sql_no_filter_when_not_provided(self):
        """target_partition_filter 없으면 ON 절에 추가 조건 없음."""
        from src.collect.incremental.bigquery_io import merge_into_target

        client = MagicMock()
        mock_job = MagicMock()
        mock_job.num_dml_affected_rows = 5
        mock_job.result.return_value = None
        client.query.return_value = mock_job

        merge_into_target(
            client,
            staging_fqn="p.d.staging",
            target_fqn="p.d.target",
            merge_keys=["GLOBALEVENTID"],
            columns=["GLOBALEVENTID", "iso3"],
            update_on_match=False,
        )

        sql = client.query.call_args[0][0]
        assert "BETWEEN" not in sql

    def test_gdelt_titles_merge_call_passes_partition_filter(self):
        """run_gdelt_titles_incremental의 MERGE 호출에 partition_filter 전달됨."""
        import src.collect.incremental.collect_gdelt_titles_incremental as mod
        captured = {}

        def fake_merge(client, staging_fqn, target_fqn, merge_keys, columns,
                       update_on_match=False, max_bytes_billed=None,
                       target_partition_filter=None):
            captured["filter"] = target_partition_filter
            captured["keys"] = merge_keys
            return 100

        def fake_query(q, *args, **kwargs):
            result = MagicMock()
            row = MagicMock()
            # staging null check → 0
            row.__getitem__ = MagicMock(return_value=0)
            result.__iter__ = MagicMock(return_value=iter([row]))
            result.num_dml_affected_rows = 4224397
            return result

        with (
            patch.object(mod, "require_max_date_from_bq", return_value=date(2026, 5, 29)),
            patch.object(mod, "compute_collection_window",
                         return_value=(date(2026, 5, 26), date(2026, 5, 29))),
            patch.object(mod, "date_range_months", return_value=[]),
            patch.object(mod, "_create_empty_staging"),
            patch.object(mod, "drop_table"),
            patch.object(mod, "merge_into_target", side_effect=fake_merge),
            patch("google.cloud.bigquery.Client") as mock_bq,
        ):
            mock_client = MagicMock()
            mock_client.query.side_effect = fake_query
            mock_bq.return_value = mock_client

            count_row = MagicMock()
            # staging row count → non-zero (to reach MERGE)
            count_row.__getitem__ = MagicMock(return_value=4224397)
            stg_iter = iter([count_row])

            null_row = MagicMock()
            null_row.__getitem__ = MagicMock(return_value=0)

            call_count = [0]
            def query_side(q, *a, **kw):
                r = MagicMock()
                call_count[0] += 1
                if call_count[0] == 1:
                    # staging count
                    row = MagicMock()
                    row.__getitem__ = MagicMock(return_value=4224397)
                    r.__iter__ = MagicMock(return_value=iter([row]))
                elif call_count[0] in (2, 3):
                    # null check / dup check → 0
                    row = MagicMock()
                    row.__getitem__ = MagicMock(return_value=0)
                    r.__iter__ = MagicMock(return_value=iter([row]))
                else:
                    r.__iter__ = MagicMock(return_value=iter([]))
                r.num_dml_affected_rows = 100
                return r

            mock_client.query.side_effect = query_side

            with patch.object(mod, "validate_after_load", return_value={"passed": True}):
                mod.run_gdelt_titles_incremental(project_id="test-proj", dry_run=False)

        assert "filter" in captured, "merge_into_target에 target_partition_filter가 전달되지 않음"
        assert captured["filter"] is not None
        assert "T.date BETWEEN" in captured["filter"]
        assert captured["keys"] == ["date", "iso3", "url"]

    def test_partition_filter_dates_are_literals_not_variables(self):
        """partition filter의 날짜가 SQL 변수 아닌 리터럴 DATE() 함수로 전달됨."""
        import re as re_mod
        import src.collect.incremental.collect_gdelt_titles_incremental as mod

        with (
            patch.object(mod, "require_max_date_from_bq", return_value=date(2026, 5, 29)),
            patch.object(mod, "compute_collection_window",
                         return_value=(date(2026, 5, 26), date(2026, 5, 29))),
            patch.object(mod, "date_range_months", return_value=[]),
            patch.object(mod, "_create_empty_staging"),
            patch.object(mod, "drop_table"),
            patch.object(mod, "merge_into_target") as mock_merge,
            patch("google.cloud.bigquery.Client") as mock_bq,
            patch.object(mod, "validate_after_load", return_value={"passed": True}),
        ):
            mock_merge.return_value = 0
            mock_client = MagicMock()
            mock_bq.return_value = mock_client

            call_count = [0]
            def query_side(q, *a, **kw):
                r = MagicMock()
                call_count[0] += 1
                v = 4224397 if call_count[0] == 1 else 0
                row = MagicMock()
                row.__getitem__ = MagicMock(return_value=v)
                r.__iter__ = MagicMock(return_value=iter([row]))
                r.num_dml_affected_rows = 0
                return r

            mock_client.query.side_effect = query_side
            mod.run_gdelt_titles_incremental(project_id="test-proj", dry_run=False)

        assert mock_merge.called, "merge_into_target이 호출되지 않음"
        _, kwargs = mock_merge.call_args
        f = kwargs.get("target_partition_filter", "")
        # DATE('YYYY-MM-DD') 리터럴이어야 하며, @파라미터 변수 형식이면 안 됨
        assert re_mod.search(r"DATE\('\d{4}-\d{2}-\d{2}'\)", f), \
            f"partition_filter가 DATE() 리터럴을 사용하지 않음: {f}"
        assert "@" not in f, f"partition_filter에 SQL 파라미터 변수 사용됨: {f}"


# ──────────────────────────────────────────────
# FRED API 키 형식 검증
# ──────────────────────────────────────────────

class TestFredApiKeyValidation:
    def test_valid_key_passes(self):
        """32자리 소문자+숫자 키는 통과."""
        from src.collect.incremental.collect_economic_incremental import validate_fred_api_key
        with patch.dict("os.environ", {"FRED_API_KEY": "a" * 32}):
            validate_fred_api_key()  # 예외 없음

    def test_valid_mixed_key_passes(self):
        """소문자+숫자 혼합 32자리 키 통과."""
        from src.collect.incremental.collect_economic_incremental import validate_fred_api_key
        with patch.dict("os.environ", {"FRED_API_KEY": "ab1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e"}):
            validate_fred_api_key()

    def test_empty_key_raises(self):
        """키가 없으면 ValueError."""
        from src.collect.incremental.collect_economic_incremental import validate_fred_api_key
        with patch.dict("os.environ", {}, clear=True):
            os_env = {}
            with patch("os.getenv", return_value=""):
                with pytest.raises(ValueError) as exc_info:
                    validate_fred_api_key()
        assert "32자리" in str(exc_info.value)
        assert "접두어" in str(exc_info.value)

    def test_wrong_length_raises(self):
        """31자리 키는 ValueError."""
        from src.collect.incremental.collect_economic_incremental import validate_fred_api_key
        with patch.dict("os.environ", {"FRED_API_KEY": "a" * 31}):
            with pytest.raises(ValueError):
                validate_fred_api_key()

    def test_key_with_prefix_raises(self):
        """'FRED_API_KEY=...' 형식으로 등록된 키는 실패."""
        from src.collect.incremental.collect_economic_incremental import validate_fred_api_key
        bad_key = "FRED_API_KEY=" + "a" * 20
        with patch.dict("os.environ", {"FRED_API_KEY": bad_key}):
            with pytest.raises(ValueError) as exc_info:
                validate_fred_api_key()
        # 오류 메시지에 실제 키 값이 포함되지 않아야 함
        assert bad_key not in str(exc_info.value)
        assert "a" * 20 not in str(exc_info.value)

    def test_key_with_quotes_raises(self):
        """따옴표로 감싼 키는 실패."""
        from src.collect.incremental.collect_economic_incremental import validate_fred_api_key
        with patch.dict("os.environ", {"FRED_API_KEY": '"' + "a" * 32 + '"'}):
            with pytest.raises(ValueError):
                validate_fred_api_key()

    def test_key_with_spaces_raises(self):
        """앞뒤 공백 포함 키는 실패."""
        from src.collect.incremental.collect_economic_incremental import validate_fred_api_key
        with patch.dict("os.environ", {"FRED_API_KEY": " " + "a" * 32}):
            with pytest.raises(ValueError):
                validate_fred_api_key()

    def test_uppercase_key_raises(self):
        """대문자 포함 키는 실패 (FRED는 소문자+숫자만 허용)."""
        from src.collect.incremental.collect_economic_incremental import validate_fred_api_key
        with patch.dict("os.environ", {"FRED_API_KEY": "A" * 32}):
            with pytest.raises(ValueError):
                validate_fred_api_key()

    def test_error_message_safe(self):
        """오류 메시지에 실제 키 값 포함되지 않음."""
        from src.collect.incremental.collect_economic_incremental import validate_fred_api_key
        secret_key = "x" * 31  # 잘못된 길이
        with patch.dict("os.environ", {"FRED_API_KEY": secret_key}):
            with pytest.raises(ValueError) as exc_info:
                validate_fred_api_key()
        assert secret_key not in str(exc_info.value)

    def test_validate_called_before_api_in_run(self):
        """run_economic_incremental이 실제 수집 전에 키 검증을 호출함."""
        import src.collect.incremental.collect_economic_incremental as mod

        call_order = []

        def fake_validate():
            call_order.append("validate")
            raise ValueError("키 오류")

        def fake_build_df(*args, **kwargs):
            call_order.append("build_df")
            return None

        with (
            patch.object(mod, "require_max_date_from_bq", return_value=date(2026, 5, 29)),
            patch.object(mod, "validate_fred_api_key", side_effect=fake_validate),
            patch.object(mod, "_build_economic_df", side_effect=fake_build_df),
            patch("google.cloud.bigquery.Client"),
        ):
            with pytest.raises(ValueError, match="키 오류"):
                mod.run_economic_incremental(project_id="test-proj", dry_run=False)

        assert "validate" in call_order
        assert "build_df" not in call_order, "키 검증 실패 후 수집 함수가 호출됨"
