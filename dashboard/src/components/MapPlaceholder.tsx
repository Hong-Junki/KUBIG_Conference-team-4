export default function MapPlaceholder() {
  return (
    <div className="bg-white rounded-xl border-2 border-dashed border-gray-200 flex flex-col items-center justify-center p-10 min-h-72">
      <div className="text-5xl mb-4 opacity-30">🗺️</div>
      <div className="text-gray-400 font-semibold text-sm">세계 지도 (Phase 6 예정)</div>
      <div className="text-gray-300 text-xs mt-2 text-center leading-relaxed max-w-48">
        국가별 위험도 choropleth 지도를
        <br />
        이 영역에 연결할 수 있습니다
      </div>
      <div className="mt-5 px-3 py-1.5 bg-gray-50 rounded-lg text-xs text-gray-400 font-mono">
        {'<WorldMap predictions={data} />'}
      </div>
    </div>
  );
}
