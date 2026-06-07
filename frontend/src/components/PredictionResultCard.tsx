import { CheckCircle2, Layers } from "lucide-react";

import type { PredictionResponse } from "../types";
import { formatCurrency } from "../utils/formatters";

type PredictionResultCardProps = {
  result: PredictionResponse | null;
};

export default function PredictionResultCard({ result }: PredictionResultCardProps) {
  if (!result) {
    return (
      <section className="panel flex min-h-72 items-center justify-center p-6 text-center">
        <div>
          <Layers className="mx-auto h-9 w-9 text-steel" aria-hidden="true" />
          <p className="mt-3 text-sm font-medium text-slate-800">Prediction output</p>
          <p className="mt-1 text-sm text-slate-500">Submit product details to generate a mock v1 estimate.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="panel p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-slate-500">Predicted price</p>
          <p className="mt-2 text-4xl font-semibold tracking-normal text-ink">
            {formatCurrency(result.predicted_price)}
          </p>
        </div>
        <CheckCircle2 className="h-6 w-6 text-mint" aria-hidden="true" />
      </div>
      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <div className="rounded-md bg-slate-50 p-3">
          <p className="text-xs font-medium uppercase text-slate-500">Confidence band</p>
          <p className="mt-1 text-sm text-slate-800">
            {formatCurrency(result.confidence_band.low)} to {formatCurrency(result.confidence_band.high)}
          </p>
        </div>
        <div className="rounded-md bg-slate-50 p-3">
          <p className="text-xs font-medium uppercase text-slate-500">Model version</p>
          <p className="mt-1 text-sm text-slate-800">{result.model_version}</p>
        </div>
      </div>
      <div className="mt-4 rounded-md bg-slate-50 p-3">
        <p className="text-xs font-medium uppercase text-slate-500">Features used</p>
        <div className="mt-2 flex flex-wrap gap-2">
          {result.features_used.map((feature) => (
            <span key={feature} className="rounded-md bg-white px-2 py-1 text-xs text-slate-700">
              {feature}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
