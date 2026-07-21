import { useState } from "react";
import { predict } from "./api";
import type { PredictOutcome, PredictStage } from "./types/api";
import { FileUploadForm } from "./components/FileUploadForm";
import { PredictionProgress } from "./components/PredictionProgress";
import { PredictionsTable } from "./components/PredictionsTable";

function App() {
  const [results, setResults] = useState<PredictOutcome[]>([]);
  const [stages, setStages] = useState<Record<string, PredictStage>>({});
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(files: File[]) {
    setSubmitting(true);
    setResults([]);
    setStages({});

    await Promise.allSettled(
      files.map(async (file) => {
        try {
          const result = await predict(file, (stage) =>
            setStages((prev) => ({ ...prev, [file.name]: stage })),
          );
          setResults((prev) => [...prev, result]);
        } catch (err) {
          setResults((prev) => [
            ...prev,
            { filename: file.name, error: err instanceof Error ? err.message : String(err) },
          ]);
        } finally {
          setStages((prev) => {
            const next = { ...prev };
            delete next[file.name];
            return next;
          });
        }
      }),
    );

    setSubmitting(false);
  }

  return (
    <div className="mx-auto max-w-7xl p-6">
      <h1 className="mb-1 text-2xl font-semibold text-gray-900">Bert Tunning</h1>
      <p className="mb-6 text-sm text-gray-500">
        Classify one or more Spanish municipal PDF documents.
      </p>

      <FileUploadForm onSubmit={handleSubmit} submitting={submitting} />

      <div className="mt-6">
        <PredictionProgress stages={stages} />
        <PredictionsTable results={results} />
      </div>
    </div>
  );
}

export default App;
