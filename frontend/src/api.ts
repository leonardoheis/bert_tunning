import type { PredictJob, PredictJobCreated, PredictResponse, PredictStage } from "./types/api";

const POLL_INTERVAL_MS = 500;

export async function predict(
  file: File,
  onStage?: (stage: PredictStage) => void,
): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/predict", { method: "POST", body: formData });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed: ${res.status}`);
  }
  const created: PredictJobCreated = await res.json();

  while (true) {
    const statusRes = await fetch(`/predict/status/${created.jobId}`);
    if (!statusRes.ok) {
      throw new Error(`Status check failed: ${statusRes.status}`);
    }
    const job: PredictJob = await statusRes.json();
    onStage?.(job.stage);
    if (job.stage === "done") {
      if (!job.result) throw new Error("Job marked done with no result");
      return job.result;
    }
    if (job.stage === "error") {
      throw new Error(job.error ?? "Prediction failed");
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
}
