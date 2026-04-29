import { ModelMetrics } from '@/lib/types';

interface Props {
  metrics: ModelMetrics;
}

interface MetricItemProps {
  label: string;
  value: string;
  direction: 'up' | 'down';
  description: string;
}

function MetricItem({ label, value, direction, description }: MetricItemProps) {
  return (
    <div className="text-center px-4 py-2">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className="text-2xl font-bold text-gray-800 tabular-nums">{value}</div>
      <div className="text-xs text-gray-400 mt-0.5">
        {direction === 'up' ? '▲' : '▼'} {description}
      </div>
    </div>
  );
}

export default function ModelPerformanceCard({ metrics }: Props) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-700">모델 성능 요약</h2>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">val set 기준</span>
          <span className="text-xs font-medium bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">
            {metrics.model_name}
          </span>
        </div>
      </div>

      <div className="flex divide-x divide-gray-100">
        <MetricItem label="PR-AUC"     value={metrics.pr_auc.toFixed(4)}                    direction="up"   description="높을수록 good" />
        <MetricItem label="P@top5%"    value={metrics.p_at_top5pct.toFixed(4)}              direction="up"   description="높을수록 good" />
        <MetricItem label="R@P≥0.10"   value={metrics.recall_at_precision_010.toFixed(4)}   direction="up"   description="높을수록 good" />
        <MetricItem label="ECE"        value={metrics.ece.toFixed(4)}                        direction="down" description="낮을수록 good" />
      </div>

      <p className="text-xs text-gray-400 mt-3 pt-3 border-t border-gray-100">
        ※ y_prob는 calibration이 미적용된 모델 신호입니다. 실제 발생 확률이 아닌 escalation 위험 신호(risk score)로 해석하세요.
      </p>
    </div>
  );
}
