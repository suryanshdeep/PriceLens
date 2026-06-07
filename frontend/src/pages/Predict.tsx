import { useEffect, useState } from "react";

import { getModelInfo, predictPrice } from "../api/priceLensApi";
import ProductInputForm from "../components/ProductInputForm";
import PredictionResultCard from "../components/PredictionResultCard";
import type { ModelInfo, PredictionResponse, PredictPayload } from "../types";

export default function Predict() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    getModelInfo()
      .then(setModelInfo)
      .catch(() => setModelInfo(null));
  }, []);

  async function handleSubmit(payload: PredictPayload) {
    setIsSubmitting(true);
    setError(null);
    try {
      setResult(await predictPrice(payload));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Prediction request failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="page-shell">
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-normal text-ink">Price Prediction</h1>
          <p className="mt-2 max-w-2xl text-sm text-slate-600">
            Text-first baseline with optional image upload support in the API contract.
          </p>
        </div>
        <div className="rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600">
          {modelInfo ? modelInfo.model_version : "Backend offline"}
        </div>
      </div>
      {error ? (
        <div className="mb-4 rounded-md border border-coral/30 bg-coral/10 px-4 py-3 text-sm text-coral">
          {error}
        </div>
      ) : null}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(340px,0.9fr)]">
        <ProductInputForm isSubmitting={isSubmitting} onSubmit={handleSubmit} />
        <PredictionResultCard result={result} />
      </div>
    </div>
  );
}
