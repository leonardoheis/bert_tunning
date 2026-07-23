import type { PredictStage } from "../types/api";

const STEPS: PredictStage[] = ["queued", "extracting", "classifying", "done"];

/** App.tsx removes a file's entry as soon as it settles (success or error), so only
 * "queued"/"extracting"/"classifying" are ever actually rendered here -- a settled file's
 * outcome shows in the results table instead, there's no need to duplicate it here too. */
export function PredictionProgress({ stages }: { stages: Record<string, PredictStage> }) {
  const entries = Object.entries(stages);
  if (entries.length === 0) return null;

  return (
    <ul className="mb-4 space-y-1 text-sm">
      {entries.map(([filename, stage]) => (
        <li key={filename} className="flex items-center gap-2">
          <span className="w-48 truncate text-gray-100">{filename}</span>
          <span className="flex gap-1">
            {STEPS.map((step) => (
              <span
                key={step}
                className={`h-2 w-2 rounded-full ${
                  STEPS.indexOf(step) <= STEPS.indexOf(stage) ? "bg-blue-600" : "bg-gray-700"
                }`}
              />
            ))}
          </span>
          <span className="text-gray-400">{stage}</span>
        </li>
      ))}
    </ul>
  );
}
