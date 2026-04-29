'use client';

import { Prediction } from '@/lib/types';
import { getRiskColors, getRiskLabel } from '@/lib/riskUtils';
import { getCountryName } from '@/lib/countryNames';

interface Props {
  predictions: Prediction[];
  selectedCountry: string | null;
  onSelect: (country: string) => void;
}

export default function CountryTable({ predictions, selectedCountry, onSelect }: Props) {
  return (
    <div className="bg-white rounded-xl border border-gray-200">
      <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-700">전체 국가 위험도</h2>
        <span className="text-xs text-gray-400">{predictions.length}개국 · y_prob 내림차순</span>
      </div>

      <div className="overflow-y-auto max-h-80">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-gray-50 border-b border-gray-100">
            <tr>
              <th className="px-5 py-2.5 text-left text-xs font-medium text-gray-500 w-12">순위</th>
              <th className="px-5 py-2.5 text-left text-xs font-medium text-gray-500">국가</th>
              <th className="px-5 py-2.5 text-right text-xs font-medium text-gray-500">Risk Score</th>
              <th className="px-5 py-2.5 text-right text-xs font-medium text-gray-500">y_prob</th>
              <th className="px-5 py-2.5 text-center text-xs font-medium text-gray-500 w-28">Risk Level</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {predictions.map((p, i) => {
              const colors = getRiskColors(p.risk_level);
              const isSelected = selectedCountry === p.country;
              return (
                <tr
                  key={p.country}
                  onClick={() => onSelect(p.country)}
                  className={`cursor-pointer transition-colors hover:bg-blue-50 ${
                    isSelected ? 'bg-blue-50' : ''
                  }`}
                >
                  <td className="px-5 py-2.5 text-gray-400 text-xs tabular-nums">{i + 1}</td>
                  <td className="px-5 py-2.5">
                    <span className="font-semibold text-gray-800">{p.country}</span>
                    <span className="text-gray-400 text-xs ml-2">{getCountryName(p.country)}</span>
                  </td>
                  <td className="px-5 py-2.5 text-right font-bold text-gray-800 tabular-nums">
                    {p.risk_score.toFixed(1)}
                  </td>
                  <td className="px-5 py-2.5 text-right text-gray-500 text-xs tabular-nums">
                    {p.y_prob.toFixed(4)}
                  </td>
                  <td className="px-5 py-2.5 text-center">
                    <span className={`inline-block text-xs font-medium px-2.5 py-0.5 rounded-full ${colors.text} ${colors.bg}`}>
                      {getRiskLabel(p.risk_level)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
