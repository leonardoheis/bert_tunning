import { Fragment, useState } from "react";
import type { PredictOutcome } from "../types/api";
import { isPredictFailure } from "../types/api";
import { formatConfidence, reviewRouteLabel } from "../utils/format";
import { ResultDetail } from "./ResultDetail";

export function ResultsTable({ results }: { results: PredictOutcome[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (results.length === 0) {
    return null;
  }

  return (
    <table className="w-full border-collapse text-left text-sm">
      <thead>
        <tr className="border-b border-gray-300 text-gray-500">
          <th className="py-2 pr-4 font-medium">File</th>
          <th className="py-2 pr-4 font-medium">Label</th>
          <th className="py-2 pr-4 font-medium">Confidence</th>
          <th className="py-2 pr-4 font-medium">Certain</th>
          <th className="py-2 pr-4 font-medium">Review route</th>
        </tr>
      </thead>
      <tbody>
        {results.map((outcome, index) => {
          const key = `${outcome.filename}-${index}`;

          if (isPredictFailure(outcome)) {
            return (
              <tr key={key} className="border-b border-gray-100">
                <td className="py-2 pr-4">{outcome.filename}</td>
                <td colSpan={4} className="py-2 pr-4 text-red-600">
                  {outcome.error}
                </td>
              </tr>
            );
          }

          const isExpanded = expanded === key;
          return (
            <Fragment key={key}>
              <tr
                onClick={() => setExpanded(isExpanded ? null : key)}
                className="cursor-pointer border-b border-gray-100 hover:bg-gray-50"
              >
                <td className="py-2 pr-4">{outcome.filename}</td>
                <td className="py-2 pr-4">{outcome.label ?? "—"}</td>
                <td className="py-2 pr-4">{formatConfidence(outcome.confidence)}</td>
                <td className="py-2 pr-4">{outcome.certain ? "yes" : "no"}</td>
                <td className="py-2 pr-4">{reviewRouteLabel(outcome.reviewRoute)}</td>
              </tr>
              {isExpanded && (
                <tr>
                  <td colSpan={5} className="p-0">
                    <ResultDetail result={outcome} />
                  </td>
                </tr>
              )}
            </Fragment>
          );
        })}
      </tbody>
    </table>
  );
}
