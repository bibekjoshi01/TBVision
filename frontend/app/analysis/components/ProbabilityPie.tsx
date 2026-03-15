"use client";

import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";

const COLOR_MAP: Record<string, string> = {
  TB: "#c9122aff",
  Normal: "#17c964",
};
const getColor = (label: string) => COLOR_MAP[label] || "#f5a524";

interface ProbabilityPieProps {
  probabilities: Record<string, number>;
  prediction: string;
}

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const { name, value } = payload[0];
    return (
      <div className="bg-white border border-default-200 rounded-lg px-3 py-2 shadow-lg">
        <p className="text-xs font-black uppercase tracking-wide" style={{ color: getColor(name) }}>{name}</p>
        <p className="text-sm font-bold text-default-800">{(value * 100).toFixed(1)}%</p>
      </div>
    );
  }
  return null;
};

export function ProbabilityPie({ probabilities, prediction }: ProbabilityPieProps) {
  const data = Object.entries(probabilities).map(([name, value]) => ({ name, value }));
  const confidence = ((probabilities[prediction] ?? 0) * 100).toFixed(1);
  const textColor = getColor(prediction);

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative w-[140px] h-[140px]">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={38}
              outerRadius={60}
              paddingAngle={3}
              dataKey="value"
              strokeWidth={0}
            >
              {data.map((entry) => (
                <Cell key={entry.name} fill={getColor(entry.name)} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>
        {/* Center label */}
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <span className="text-lg font-black leading-none" style={{ color: textColor }}>{confidence}%</span>
          <span className="text-[8px] font-black uppercase tracking-wide text-default-400 mt-0.5">{prediction}</span>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-x-3 gap-y-1 justify-center">
        {data.map(({ name, value }) => (
          <div key={name} className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full shrink-0" style={{ background: getColor(name) }} />
            <span className="text-[9px] font-bold uppercase text-default-600 tracking-wide">{name}</span>
            <span className="text-[9px] font-medium text-default-400">{(value * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
