import { API_BASE } from "@/lib/constansts";
import type { PredictionResponse } from "@/types";

export async function runPrediction(
  file: File,
  metadata: Record<string, unknown>
): Promise<PredictionResponse> {
  const body = new FormData();
  body.append("image", file);
  body.append("metadata", JSON.stringify(metadata));

  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    body,
  });

  if (!response.ok) {
    throw new Error(`Analysis failed: ${response.statusText}`);
  }

  return response.json() as Promise<PredictionResponse>;
}
