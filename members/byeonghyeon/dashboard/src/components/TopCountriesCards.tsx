'use client';

import { Prediction } from '@/lib/types';
import { getRiskColors, getRiskLabel } from '@/lib/riskUtils';
import { getCountryName } from '@/lib/countryNames';

interface Props {
  countries: Prediction[];
  selectedCountry: string | null;
  onSelect: (country: string) => void;
}

const RANK_EMOJI = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'];

export default function TopCountriesCards({ countries, selectedCountry, onSelect }: Props) {
  return (
    <div>
      <h2 className="text-sm font-semibold text-gray-700 mb-3">위험도 상위 5개국 (선택 날짜 기준)</h2>
      <div className="grid grid-cols-5 gap-3">
        {countries.map((p, i) => {
          const colors = getRiskColors(p.risk_level);
          const isSelected = selectedCountry === p.country;
          return (
            <button
              key={p.country}
              onClick={() => onSelect(p.country)}
              className={`rounded-xl border-2 p-4 text-left transition-all hover:shadow-md ${colors.border} ${colors.bg} ${
                isSelected ? 'ring-2 ring-blue-500 ring-offset-2 shadow-md' : ''
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-base">{RANK_EMOJI[i]}</span>
                <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${colors.text} bg-white bg-opacity-60`}>
                  {getRiskLabel(p.risk_level)}
                </span>
              </div>
              <div className="font-bold text-gray-900 text-base">{p.country}</div>
              <div className="text-xs text-gray-500 mb-3 truncate">{getCountryName(p.country)}</div>
              <div className="text-3xl font-extrabold text-gray-900 tabular-nums leading-none">
                {p.risk_score.toFixed(1)}
              </div>
              <div className="text-xs text-gray-500 mt-0.5">risk score</div>
            </button>
          );
        })}

        {countries.length === 0 && (
          <div className="col-span-5 text-center text-gray-400 py-10 bg-white rounded-xl border border-gray-200">
            날짜를 선택하면 위험도 상위 국가가 표시됩니다
          </div>
        )}
      </div>
    </div>
  );
}
