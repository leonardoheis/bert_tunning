import type { PredictOutcome } from "../types/api";
import { isPredictFailure } from "../types/api";

export interface FlatResultRow {
  filename: string;
  label: string | null;
  confidence: number | null;
  certain: boolean | null;
  mahalanobisPValue: number | null;
  mahalanobisPValueTheoretical: number | null;
  cosineZ: number | null;
  knnDistance: number | null;
  tfidfCosineZ: number | null;
  inDistribution: boolean | null;
  mahalanobisCalibrationStatus: string | null;
  cosineCalibrationStatus: string | null;
  knnDistanceCalibrationStatus: string | null;
  tfidfCalibrationStatus: string | null;
  foreignMunicipality: string | null;
  foreignMunicipalityContext: string | null;
  reviewRoute: string | null;
  extractorUsed: string | null;
  error: string | null;
  svmScores: Record<string, number>;
  svmPredictedLabel: string | null;
  svmAgreesWithPrediction: boolean | null;
}

const NULL_ROW: Omit<FlatResultRow, "filename" | "error"> = {
  label: null,
  confidence: null,
  certain: null,
  mahalanobisPValue: null,
  mahalanobisPValueTheoretical: null,
  cosineZ: null,
  knnDistance: null,
  tfidfCosineZ: null,
  inDistribution: null,
  mahalanobisCalibrationStatus: null,
  cosineCalibrationStatus: null,
  knnDistanceCalibrationStatus: null,
  tfidfCalibrationStatus: null,
  foreignMunicipality: null,
  foreignMunicipalityContext: null,
  reviewRoute: null,
  extractorUsed: null,
  svmScores: {},
  svmPredictedLabel: null,
  svmAgreesWithPrediction: null,
};

/** Mirrors flatten_predict_result() (src/schema.py) -- turns one PredictOutcome into a
 * flat row keyed by the same columns _PREDICTION_COLUMNS (src/wandb.py) already picked. */
export function flattenResult(outcome: PredictOutcome): FlatResultRow {
  if (isPredictFailure(outcome)) {
    return {
      filename: outcome.filename,
      error: outcome.error,
      ...NULL_ROW,
    };
  }
  const ood = outcome.oodMetrics;
  return {
    filename: outcome.filename,
    label: outcome.label,
    confidence: outcome.confidence,
    certain: outcome.certain,
    mahalanobisPValue: ood?.mahalanobisPValue ?? null,
    mahalanobisPValueTheoretical: ood?.mahalanobisPValueTheoretical ?? null,
    cosineZ: ood?.cosineZ ?? null,
    knnDistance: ood?.knnDistance ?? null,
    tfidfCosineZ: ood?.tfidfCosineZ ?? null,
    inDistribution: ood?.inDistribution ?? null,
    mahalanobisCalibrationStatus: ood?.mahalanobisCalibrationStatus ?? null,
    cosineCalibrationStatus: ood?.cosineCalibrationStatus ?? null,
    knnDistanceCalibrationStatus: ood?.knnDistanceCalibrationStatus ?? null,
    tfidfCalibrationStatus: ood?.tfidfCalibrationStatus ?? null,
    foreignMunicipality: outcome.foreignMunicipality,
    foreignMunicipalityContext: outcome.foreignMunicipalityContext,
    reviewRoute: outcome.reviewRoute,
    extractorUsed: outcome.extractorUsed,
    error: outcome.error,
    svmScores: outcome.svmScores,
    svmPredictedLabel: outcome.svmPredictedLabel,
    svmAgreesWithPrediction: outcome.svmAgreesWithPrediction,
  };
}
