export type PredictPayload = {
  catalog_content: string;
  brand?: string;
  category?: string;
  image?: File | null;
};

export type PredictionResponse = {
  predicted_price: number;
  confidence_band: {
    low: number;
    high: number;
  };
  model_version: string;
  features_used: string[];
  explanation_tokens: string[];
  image_received: boolean;
};

export type ModelInfo = {
  model_version: string;
  model_type: string;
  status: string;
  supports_images: boolean;
  features: string[];
};

export type HealthResponse = {
  status: string;
  service: string;
  version: string;
};
