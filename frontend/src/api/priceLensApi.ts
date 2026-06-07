import type { HealthResponse, ModelInfo, PredictionResponse, PredictPayload } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function healthCheck(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function getModelInfo(): Promise<ModelInfo> {
  return request<ModelInfo>("/model-info");
}

export function predictPrice(payload: PredictPayload): Promise<PredictionResponse> {
  if (payload.image) {
    const formData = new FormData();
    formData.append("catalog_content", payload.catalog_content);
    if (payload.brand) formData.append("brand", payload.brand);
    if (payload.category) formData.append("category", payload.category);
    formData.append("image", payload.image);

    return request<PredictionResponse>("/predict-with-image", {
      method: "POST",
      body: formData,
    });
  }

  return request<PredictionResponse>("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      catalog_content: payload.catalog_content,
      brand: payload.brand || undefined,
      category: payload.category || undefined,
    }),
  });
}
