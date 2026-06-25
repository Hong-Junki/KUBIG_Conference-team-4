'use client';

import { useState, useEffect } from 'react';
import Papa from 'papaparse';
import { Prediction } from './types';
import { getRiskLevel } from './riskUtils';

const CSV_PATH = '/data/predictions__lightgbm_se_updated_mask0_only_platt__byeonghyeon.csv';

interface RawRow {
  date: string;
  country: string;
  y_prob: string | number;
}

export function usePredictions() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Papa.parse<RawRow>(CSV_PATH, {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (result) => {
        const data: Prediction[] = result.data
          .filter((row) => row.date && row.country && row.y_prob != null)
          .map((row) => {
            const y_prob = typeof row.y_prob === 'string' ? parseFloat(row.y_prob) : row.y_prob;
            return {
              date: String(row.date),
              country: String(row.country),
              y_prob,
              risk_score: Math.round(y_prob * 1000) / 10, // 1 decimal place
              risk_level: getRiskLevel(y_prob),
            };
          });
        setPredictions(data);
        setLoading(false);
      },
      error: (err: Error) => {
        setError(err.message);
        setLoading(false);
      },
    });
  }, []);

  return { predictions, loading, error };
}
