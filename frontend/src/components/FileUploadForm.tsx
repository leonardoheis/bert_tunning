import { useState } from "react";
import type { FormEvent } from "react";

interface FileUploadFormProps {
  onSubmit: (files: File[]) => void;
  submitting: boolean;
}

export function FileUploadForm({ onSubmit, submitting }: FileUploadFormProps) {
  const [files, setFiles] = useState<File[]>([]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (files.length === 0) {
      return;
    }
    onSubmit(files);
  }

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-3">
      <input
        type="file"
        multiple
        accept="application/pdf"
        onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
        className="block flex-1 rounded border border-gray-700 bg-gray-900 p-2 text-sm text-gray-100 file:mr-3 file:rounded file:border-0 file:bg-gray-800 file:px-3 file:py-1.5 file:text-sm file:text-gray-100"
      />
      <button
        type="submit"
        disabled={files.length === 0 || submitting}
        className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
      >
        {submitting ? "Classifying…" : "Classify"}
      </button>
    </form>
  );
}
