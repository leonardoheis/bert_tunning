export interface OodMetrics {
  mahalanobisPValue: number;
  mahalanobisPValueTheoretical: number;
  cosineZ: number;
  knnDistance: number;
  tfidfCosineZ: number | null;
  inDistribution: boolean;
  mahalanobisCalibrationStatus: "calibrated" | "not_calibrated" | "refused_degenerate";
  cosineCalibrationStatus: "calibrated" | "not_calibrated";
  knnDistanceCalibrationStatus: "calibrated" | "not_calibrated";
  tfidfCalibrationStatus: "calibrated" | "not_calibrated" | null;
}

export type ReviewRoute = "" | "accept" | "llm_judge" | "human_review";

export interface PredictResponse {
  filename: string;
  label: string | null;
  confidence: number;
  certain: boolean;
  allScores: Record<string, number>;
  error: string | null;
  oodMetrics: OodMetrics | null;
  extractedText: string;
  extractorUsed: string;
  reviewRoute: ReviewRoute;
  foreignMunicipality: string | null;
  foreignMunicipalityContext: string | null;
  svmScores: Record<string, number>;
  svmPredictedLabel: string;
  svmAgreesWithPrediction: boolean;
}

/** A file that failed before/during the request -- never reached a real PredictResponse. */
export interface PredictFailure {
  filename: string;
  error: string;
}

export type PredictOutcome = PredictResponse | PredictFailure;

export type PredictStage = "queued" | "extracting" | "classifying" | "done" | "error";

export interface PredictJob {
  stage: PredictStage;
  result: PredictResponse | null;
  error: string | null;
}

export interface PredictJobCreated {
  jobId: string;
}

export function isPredictFailure(outcome: PredictOutcome): outcome is PredictFailure {
  return !("label" in outcome);
}
