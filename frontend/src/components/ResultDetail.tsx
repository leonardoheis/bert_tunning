import type { PredictResponse } from "../types/api";

function ScoreList({ scores }: { scores: Record<string, number> }) {
  const entries = Object.entries(scores);
  if (entries.length === 0) {
    return <span className="text-gray-400">none</span>;
  }
  return (
    <ul className="space-y-0.5">
      {entries.map(([name, value]) => (
        <li key={name}>
          {name}: {value.toFixed(4)}
        </li>
      ))}
    </ul>
  );
}

export function ResultDetail({ result }: { result: PredictResponse }) {
  const { oodMetrics } = result;

  return (
    <div className="grid grid-cols-1 gap-4 border-t border-gray-200 bg-gray-50 p-4 text-sm sm:grid-cols-3">
      <div>
        <h4 className="mb-1 font-semibold text-gray-700">All class scores</h4>
        <ScoreList scores={result.allScores} />
      </div>

      <div>
        <h4 className="mb-1 font-semibold text-gray-700">SVM reviewer</h4>
        {Object.keys(result.svmScores).length === 0 ? (
          <span className="text-gray-400">no SVM classifiers loaded</span>
        ) : (
          <>
            <p>Predicted: {result.svmPredictedLabel}</p>
            <p>Agrees with softmax: {result.svmAgreesWithPrediction ? "yes" : "no"}</p>
            <ScoreList scores={result.svmScores} />
          </>
        )}
      </div>

      <div>
        <h4 className="mb-1 font-semibold text-gray-700">OOD metrics</h4>
        {oodMetrics === null ? (
          <span className="text-gray-400">no ood_stats.npz loaded</span>
        ) : (
          <ul className="space-y-0.5">
            <li>In distribution: {oodMetrics.inDistribution ? "yes" : "no"}</li>
            <li>
              Mahalanobis p (empirical): {oodMetrics.mahalanobisPValue.toFixed(6)} (
              {oodMetrics.mahalanobisCalibrationStatus})
            </li>
            <li>
              Mahalanobis p (theoretical): {oodMetrics.mahalanobisPValueTheoretical.toFixed(6)}
            </li>
            <li>
              Cosine z: {oodMetrics.cosineZ.toFixed(4)} ({oodMetrics.cosineCalibrationStatus})
            </li>
            <li>
              k-NN distance: {oodMetrics.knnDistance.toFixed(4)} (
              {oodMetrics.knnDistanceCalibrationStatus})
            </li>
            <li>
              TF-IDF cosine z:{" "}
              {oodMetrics.tfidfCosineZ === null ? "n/a" : oodMetrics.tfidfCosineZ.toFixed(4)} (
              {oodMetrics.tfidfCalibrationStatus ?? "n/a"})
            </li>
          </ul>
        )}
        {result.foreignMunicipality !== null && (
          <p className="mt-2 text-amber-700">
            Foreign municipality detected: {result.foreignMunicipality}
          </p>
        )}
      </div>

      {result.extractedText !== "" && (
        <div className="sm:col-span-3">
          <h4 className="mb-1 font-semibold text-gray-700">
            Extracted text ({result.extractorUsed})
          </h4>
          <p className="max-h-40 overflow-y-auto whitespace-pre-wrap rounded border border-gray-200 bg-white p-2 text-xs text-gray-600">
            {result.extractedText}
          </p>
        </div>
      )}
    </div>
  );
}
