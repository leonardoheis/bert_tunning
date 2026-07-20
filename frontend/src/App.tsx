import { useState } from "react";
import { predict } from "./api";
import type { PredictOutcome } from "./types/api";
import { FileUploadForm } from "./components/FileUploadForm";
import { ResultsTable } from "./components/ResultsTable";

function App() {
  const [results, setResults] = useState<PredictOutcome[]>([]);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(files: File[]) {
    setSubmitting(true);
    setResults([]);

    await Promise.allSettled(
      files.map(async (file) => {
        try {
          const result = await predict(file);
          setResults((prev) => [...prev, result]);
        } catch (err) {
          setResults((prev) => [
            ...prev,
            { filename: file.name, error: err instanceof Error ? err.message : String(err) },
          ]);
        }
      }),
    );

    setSubmitting(false);
  }

  return (
    <div className="mx-auto max-w-4xl p-6">
      <h1 className="mb-1 text-2xl font-semibold text-gray-900">Bert Tunning</h1>
      <p className="mb-6 text-sm text-gray-500">
        Classify one or more Spanish municipal PDF documents.
      </p>

      <FileUploadForm onSubmit={handleSubmit} submitting={submitting} />

      <div className="mt-6">
        <ResultsTable results={results} />
      </div>
    </div>
  );
}

export default App;
