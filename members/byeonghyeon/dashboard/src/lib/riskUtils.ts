import { RiskLevel } from './types';

export function getRiskLevel(y_prob: number): RiskLevel {
  if (y_prob < 0.30) return 'Low';
  if (y_prob < 0.60) return 'Moderate';
  if (y_prob < 0.80) return 'High';
  return 'Critical';
}

export interface RiskColors {
  bg: string;
  text: string;
  border: string;
  hex: string;
}

export function getRiskColors(level: RiskLevel): RiskColors {
  switch (level) {
    case 'Low':      return { bg: 'bg-blue-50',   text: 'text-blue-700',   border: 'border-blue-200',   hex: '#3b82f6' };
    case 'Moderate': return { bg: 'bg-amber-50',  text: 'text-amber-700',  border: 'border-amber-200',  hex: '#f59e0b' };
    case 'High':     return { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200', hex: '#f97316' };
    case 'Critical': return { bg: 'bg-red-50',    text: 'text-red-700',    border: 'border-red-200',    hex: '#ef4444' };
  }
}

export function getRiskLabel(level: RiskLevel): string {
  switch (level) {
    case 'Low':      return '낮음';
    case 'Moderate': return '주의';
    case 'High':     return '높음';
    case 'Critical': return '매우 높음';
  }
}
