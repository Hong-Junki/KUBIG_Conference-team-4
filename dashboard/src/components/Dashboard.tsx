'use client';

import { useState, useMemo, useEffect } from 'react';
import { ModelMetrics } from '@/lib/types';
import { usePredictions } from '@/lib/usePredictions';
import Header from './Header';
import ModelPerformanceCard from './ModelPerformanceCard';
import TopCountriesCards from './TopCountriesCards';
import CountryTable from './CountryTable';
import CountryDetailPanel from './CountryDetailPanel';
import MapPlaceholder from './MapPlaceholder';

const MODEL_METRICS: ModelMetrics = {
  model_name: 'Updated(mask=0) + Platt Cal.',
  pr_auc: 0.1741,
  p_at_top5pct: 0.2159,
  recall_at_precision_010: 0.6209,
  ece: 0.0029,
  brier_score: 0.0365,
};

export default function Dashboard() {
  const { predictions, loading, error } = usePredictions();
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);

  // All unique dates sorted descending (latest first)
  const dateOptions = useMemo(() => {
    const dates = [...new Set(predictions.map((p) => p.date))].sort().reverse();
    return dates;
  }, [predictions]);

  // Default to latest date when data loads
  useEffect(() => {
    if (dateOptions.length > 0 && !selectedDate) {
      setSelectedDate(dateOptions[0]);
    }
  }, [dateOptions, selectedDate]);

  const currentDate = selectedDate || dateOptions[0] || '';

  // Predictions for selected date, sorted by risk descending
  const filteredByDate = useMemo(
    () => predictions.filter((p) => p.date === currentDate).sort((a, b) => b.y_prob - a.y_prob),
    [predictions, currentDate]
  );

  // Full history for selected country
  const countryHistory = useMemo(
    () =>
      selectedCountry
        ? predictions
            .filter((p) => p.country === selectedCountry)
            .sort((a, b) => a.date.localeCompare(b.date))
        : [],
    [predictions, selectedCountry]
  );

  const currentCountryData = useMemo(
    () => filteredByDate.find((p) => p.country === selectedCountry) ?? null,
    [filteredByDate, selectedCountry]
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4 animate-pulse">🌍</div>
          <p className="text-gray-500 font-medium">데이터 로딩 중...</p>
          <p className="text-gray-400 text-sm mt-1">15,718개 예측값을 불러오고 있습니다</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center max-w-md">
          <div className="text-3xl mb-3">⚠️</div>
          <p className="text-red-700 font-medium">데이터 로드 실패</p>
          <p className="text-red-500 text-sm mt-1">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header
        dateOptions={dateOptions}
        selectedDate={currentDate}
        onDateChange={setSelectedDate}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">
        <ModelPerformanceCard metrics={MODEL_METRICS} />

        <TopCountriesCards
          countries={filteredByDate.slice(0, 5)}
          selectedCountry={selectedCountry}
          onSelect={setSelectedCountry}
        />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <MapPlaceholder />
          <CountryDetailPanel
            country={selectedCountry}
            history={countryHistory}
            currentData={currentCountryData}
          />
        </div>

        <CountryTable
          predictions={filteredByDate}
          selectedCountry={selectedCountry}
          onSelect={setSelectedCountry}
        />
      </main>

      <footer className="max-w-7xl mx-auto px-6 py-4 text-center text-xs text-gray-400">
        Conflict Temperature Map · KUBIG 26-1 Vibe Coding ·{' '}
        데이터: ACLED + GDELT + Economic Indicators (2024-07 ~ 2025-03)
      </footer>
    </div>
  );
}
