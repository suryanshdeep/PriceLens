import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatPercent } from "../utils/formatters";

const metricCards = [
  { label: "SMAPE", value: 18.42, suffix: "%" },
  { label: "MAE", value: 5.73, suffix: "" },
  { label: "RMSE", value: 8.91, suffix: "" },
  { label: "R2", value: 0.71, suffix: "" },
];

const actualVsPredicted = [
  { index: 1, actual: 9.99, predicted: 10.8 },
  { index: 2, actual: 15.49, predicted: 14.1 },
  { index: 3, actual: 22.0, predicted: 24.2 },
  { index: 4, actual: 41.5, predicted: 38.9 },
  { index: 5, actual: 64.0, predicted: 68.4 },
];

const bucketErrors = [
  { bucket: "$0-10", smape: 22.1 },
  { bucket: "$10-25", smape: 16.4 },
  { bucket: "$25-50", smape: 14.8 },
  { bucket: "$50+", smape: 19.7 },
];

export default function Analysis() {
  return (
    <div className="page-shell">
      <div className="mb-6">
        <h1 className="text-3xl font-semibold tracking-normal text-ink">Model Analysis</h1>
        <p className="mt-2 text-sm text-slate-600">Mock metrics until the first baseline notebook run is promoted.</p>
      </div>
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {metricCards.map((metric) => (
          <article key={metric.label} className="panel p-5">
            <p className="text-sm font-medium text-slate-500">{metric.label}</p>
            <p className="mt-2 text-3xl font-semibold tracking-normal text-ink">
              {metric.suffix === "%" ? formatPercent(metric.value) : metric.value.toFixed(2)}
            </p>
          </article>
        ))}
      </section>
      <section className="mt-6 grid gap-6 lg:grid-cols-2">
        <div className="panel p-5">
          <h2 className="text-base font-semibold text-ink">Actual vs predicted</h2>
          <div className="mt-4 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={actualVsPredicted}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="index" />
                <YAxis />
                <Tooltip />
                <Line dataKey="actual" stroke="#0f9f7a" strokeWidth={2} />
                <Line dataKey="predicted" stroke="#d95f59" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="panel p-5">
          <h2 className="text-base font-semibold text-ink">Error by price bucket</h2>
          <div className="mt-4 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bucketErrors}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="smape" fill="#3f5f73" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>
    </div>
  );
}
