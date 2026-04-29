'use client';

interface HeaderProps {
  dateOptions: string[];
  selectedDate: string;
  onDateChange: (date: string) => void;
}

export default function Header({ dateOptions, selectedDate, onDateChange }: HeaderProps) {
  return (
    <header className="bg-gray-900 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Conflict Temperature Map</h1>
          <p className="text-gray-400 text-xs mt-0.5">
            ACLED + GDELT + Economic Indicators 기반 무력충돌 조기경보 대시보드
          </p>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400">기준 날짜</span>
            <select
              value={selectedDate}
              onChange={(e) => onDateChange(e.target.value)}
              className="bg-gray-800 text-white border border-gray-600 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              {dateOptions.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2 bg-gray-800 rounded-md px-3 py-1.5">
            <span className="text-xs text-gray-400">Model</span>
            <span className="text-sm font-semibold text-blue-400">LightGBM + SE</span>
          </div>
        </div>
      </div>
    </header>
  );
}
