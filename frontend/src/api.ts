import type { PredictResponse } from "./types/api";

export async function predict(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/predict", { method: "POST", body: formData });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed: ${res.status}`);
  }
  return res.json();
}
