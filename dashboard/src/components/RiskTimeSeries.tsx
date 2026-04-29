'use client';

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, TooltipProps,
} from 'recharts';
import { Prediction } from '@/lib/types';

interface Props {
  history: Prediction[];
  countryName: string;
}

interface ChartPoint {
  date: string;
  risk_score: number;
}

function CustomTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null;
  const value = payload[0].value as number;
  return (
    <div className="bg-white border border-gray-200 rounded-lg px-3 py-2 shadow-md text-xs">
      <div className="text-gray-500 mb-1">{label}</div>
      <div className="font-bold text-gray-800">Risk Score: {value.toFixed(1)}</div>
    </div>
  );
}

export default function RiskTimeSeries({ history, countryName }: Props) {
  const data: ChartPoint[] = history.map((p) => ({
    date: p.date,
    risk_score: Math.round(p.risk_score * 10) / 10,
  }));

  const interval = Math.max(1, Math.ceil(data.length / 8) - 1);

  return (
    <div>
      <p className="text-xs text-gray-500 mb-2">
        {countryName} — risk score 시계열 ({data.length}일)
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
          <XAxis
            dataKey="date"
            interval={interval}
            tickFormatter={(v: string) => v.slice(5)}
            tick={{ fontSize: 10, fill: '#9ca3af' }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 10, fill: '#9ca3af' }}
            tickLine={false}
            axisLine={false}
            width={28}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Risk level thresholds */}
          <ReferenceLine y={30} stroke="#93c5fd" strokeDasharray="4 2" strokeWidth={1} />
          <ReferenceLine y={60} stroke="#fcd34d" strokeDasharray="4 2" strokeWidth={1} />
          <ReferenceLine y={80} stroke="#fb923c" strokeDasharray="4 2" strokeWidth={1} />

          <Line
            type="monotone"
            dataKey="risk_score"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3, fill: '#3b82f6', strokeWidth: 0 }}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Threshold legend */}
      <div className="flex gap-4 mt-1 justify-end">
        {[
          { color: '#93c5fd', label: 'Low (30)' },
          { color: '#fcd34d', label: 'Moderate (60)' },
          { color: '#fb923c', label: 'High (80)' },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1">
            <div className="w-4 border-t-2 border-dashed" style={{ borderColor: color }} />
            <span className="text-xs text-gray-400">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
