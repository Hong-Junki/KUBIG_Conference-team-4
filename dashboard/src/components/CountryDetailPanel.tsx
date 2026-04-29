'use client';

import { Prediction } from '@/lib/types';
import { getRiskColors, getRiskLabel } from '@/lib/riskUtils';
import { getCountryName } from '@/lib/countryNames';
import RiskTimeSeries from './RiskTimeSeries';

interface Props {
  country: string | null;
  history: Prediction[];
  currentData: Prediction | null;
}

export default function CountryDetailPanel({ country, history, currentData }: Props) {
  if (!country || !currentData) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 flex flex-col items-center justify-center min-h-72 text-center px-8">
        <div className="text-3xl mb-3 opacity-20">👆</div>
        <p className="text-gray-400 text-sm font-medium">국가를 선택하면 상세 정보가 표시됩니다</p>
        <p className="text-gray-300 text-xs mt-1">TOP 5 카드 또는 테이블 행을 클릭하세요</p>
      </div>
    );
  }

  const colors = getRiskColors(currentData.risk_level);
  const countryName = getCountryName(country);

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 flex flex-col gap-4">
      {/* Country header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xl font-bold text-gray-900">{country}</div>
          <div className="text-sm text-gray-500">{countryName}</div>
        </div>
        <div className="text-right">
          <div className="text-4xl font-extrabold text-gray-900 tabular-nums leading-none">
            {currentData.risk_score.toFixed(1)}
          </div>
          <div className="flex items-center justify-end gap-1.5 mt-1">
            <span className={`text-xs font-bold px-2.5 py-0.5 rounded-full ${colors.text} ${colors.bg}`}>
              {currentData.risk_level}
            </span>
            <span className="text-xs text-gray-400">{getRiskLabel(currentData.risk_level)}</span>
          </div>
          <div className="text-xs text-gray-400 mt-1">y_prob: {currentData.y_prob.toFixed(4)}</div>
        </div>
      </div>

      {/* Time series */}
      <div className="border-t border-gray-100 pt-4">
        <RiskTimeSeries history={history} countryName={countryName} />
      </div>

      {/* Feature explanation placeholder */}
      <div className="rounded-lg border border-dashed border-gray-200 bg-gray-50 p-3">
        <div className="flex items-center gap-1.5 mb-2">
          <span className="text-xs font-semibold text-gray-500">위험도 상승 요인</span>
          <span className="text-xs bg-gray-200 text-gray-500 rounded px-1.5 py-0.5">준비 중</span>
        </div>
        <ul className="space-y-1 text-xs text-gray-400">
          <li>• SHAP local explanation — Phase 6에서 구현 예정</li>
          <li>• Feature importance TOP 5 표시 예정</li>
          <li>• ACLED / GDELT 신호 요약 예정</li>
        </ul>
      </div>

      {/* News placeholder */}
      <div className="rounded-lg border border-dashed border-gray-200 bg-gray-50 p-3">
        <div className="flex items-center gap-1.5 mb-2">
          <span className="text-xs font-semibold text-gray-500">관련 뉴스</span>
          <span className="text-xs bg-gray-200 text-gray-500 rounded px-1.5 py-0.5">준비 중</span>
        </div>
        <ul className="space-y-1 text-xs text-gray-400">
          <li>• GDELT DOC API 연결 — Phase 6에서 구현 예정</li>
          <li>• 관련 기사 TOP 3 + 한국어 요약 표시 예정</li>
        </ul>
      </div>
    </div>
  );
}
