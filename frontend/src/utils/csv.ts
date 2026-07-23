import type { FlatResultRow } from "./flatten";

/** Shared with PredictionsTable.tsx's FIXED_COLUMNS -- a single source of truth for the
 * on-screen table and the CSV export, so a new field can't silently show up in one but
 * not the other (this repo already hit that exact drift once, on the Python side --
 * see _PREDICTION_COLUMNS in src/wandb.py's history). */
export const RESULT_COLUMNS: { key: keyof FlatResultRow; header: string }[] = [
  { key: "filename", header: "File" },
  { key: "label", header: "Label" },
  { key: "confidence", header: "Confidence" },
  { key: "certain", header: "Certain" },
  { key: "mahalanobisPValue", header: "Mahalanobis p" },
  { key: "mahalanobisPValueTheoretical", header: "Mahalanobis p (theoretical)" },
  { key: "cosineZ", header: "Cosine z" },
  { key: "knnDistance", header: "k-NN distance" },
  { key: "tfidfCosineZ", header: "TF-IDF cosine z" },
  { key: "inDistribution", header: "In distribution" },
  { key: "mahalanobisCalibrationStatus", header: "Mahalanobis calibration" },
  { key: "cosineCalibrationStatus", header: "Cosine calibration" },
  { key: "knnDistanceCalibrationStatus", header: "k-NN calibration" },
  { key: "tfidfCalibrationStatus", header: "TF-IDF calibration" },
  { key: "foreignMunicipality", header: "Foreign municipality" },
  { key: "foreignMunicipalityContext", header: "Foreign municipality context" },
  { key: "reviewRoute", header: "Review route" },
  { key: "extractorUsed", header: "Extractor used" },
  { key: "error", header: "Error" },
  { key: "svmPredictedLabel", header: "SVM predicted label" },
  { key: "svmAgreesWithPrediction", header: "SVM agrees" },
];

function escapeCsvValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  let str = typeof value === "boolean" ? (value ? "yes" : "no") : String(value);
  // Prevent CSV/formula injection when the export is opened in Excel/Sheets -- several
  // exported fields (filename, error, foreignMunicipalityContext) are derived from
  // untrusted PDF content, and a value starting with =, +, -, or @ would run as a live
  // formula on open otherwise. Prefixing with a quote neutralizes it without changing
  // the visible text.
  if (/^[=+\-@]/.test(str)) {
    str = `'${str}`;
  }
  return /[",\r\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
}

/** Always includes every column, independent of the table's current column-visibility
 * state -- an export is a record to keep, not a view of what's currently on screen. */
export function resultsToCsv(rows: FlatResultRow[]): string {
  const svmClassNames = Array.from(
    new Set(rows.flatMap((row) => Object.keys(row.svmScores))),
  ).sort();

  const headers = [
    ...RESULT_COLUMNS.map((c) => c.header),
    ...svmClassNames.map((name) => `svm_scores.${name}`),
  ];
  const lines = rows.map((row) => {
    const fixed = RESULT_COLUMNS.map((c) => escapeCsvValue(row[c.key]));
    const svm = svmClassNames.map((name) => escapeCsvValue(row.svmScores[name]));
    return [...fixed, ...svm].join(",");
  });

  return [headers.join(","), ...lines].join("\n");
}

export function downloadCsv(csv: string, filename: string): void {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
