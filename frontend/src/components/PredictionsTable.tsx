import { useMemo, useState } from "react";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import type { ColumnDef, VisibilityState } from "@tanstack/react-table";
import type { PredictOutcome } from "../types/api";
import type { FlatResultRow } from "../utils/flatten";
import { flattenResult } from "../utils/flatten";

function formatCell(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  return String(value);
}

function formatConfidence(value: unknown): string {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : formatCell(value);
}

const columnHelper = createColumnHelper<FlatResultRow>();

const FIXED_COLUMNS: ColumnDef<FlatResultRow, string>[] = [
  columnHelper.accessor("filename", { header: "File" }),
  columnHelper.accessor("label", { header: "Label", cell: (c) => formatCell(c.getValue()) }),
  columnHelper.accessor("confidence", {
    header: "Confidence",
    cell: (c) => formatConfidence(c.getValue()),
  }),
  columnHelper.accessor("certain", { header: "Certain", cell: (c) => formatCell(c.getValue()) }),
  columnHelper.accessor("mahalanobisPValue", {
    header: "Mahalanobis p",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("mahalanobisPValueTheoretical", {
    header: "Mahalanobis p (theoretical)",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("cosineZ", { header: "Cosine z", cell: (c) => formatCell(c.getValue()) }),
  columnHelper.accessor("knnDistance", {
    header: "k-NN distance",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("tfidfCosineZ", {
    header: "TF-IDF cosine z",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("inDistribution", {
    header: "In distribution",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("mahalanobisCalibrationStatus", {
    header: "Mahalanobis calibration",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("cosineCalibrationStatus", {
    header: "Cosine calibration",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("knnDistanceCalibrationStatus", {
    header: "k-NN calibration",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("tfidfCalibrationStatus", {
    header: "TF-IDF calibration",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("foreignMunicipality", {
    header: "Foreign municipality",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("foreignMunicipalityContext", {
    header: "Foreign municipality context",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("reviewRoute", {
    header: "Review route",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("extractorUsed", {
    header: "Extractor used",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("error", { header: "Error", cell: (c) => formatCell(c.getValue()) }),
  columnHelper.accessor("svmPredictedLabel", {
    header: "SVM predicted label",
    cell: (c) => formatCell(c.getValue()),
  }),
  columnHelper.accessor("svmAgreesWithPrediction", {
    header: "SVM agrees",
    cell: (c) => formatCell(c.getValue()),
  }),
] as unknown as ColumnDef<FlatResultRow, string>[];

export function PredictionsTable({ results }: { results: PredictOutcome[] }) {
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});
  const [showColumnPicker, setShowColumnPicker] = useState(false);

  const rows = useMemo(() => results.map(flattenResult), [results]);

  const svmClassNames = useMemo(() => {
    const names = new Set<string>();
    for (const row of rows) {
      for (const name of Object.keys(row.svmScores)) {
        names.add(name);
      }
    }
    return Array.from(names).sort();
  }, [rows]);

  const columns = useMemo(() => {
    const svmColumns = svmClassNames.map((name) =>
      columnHelper.accessor((row) => row.svmScores[name], {
        id: `svmScores.${name}`,
        header: `svm_scores.${name}`,
        cell: (c) => formatCell(c.getValue()),
      }),
    );
    return [...FIXED_COLUMNS, ...svmColumns] as ColumnDef<FlatResultRow, unknown>[];
  }, [svmClassNames]);

  const table = useReactTable({
    data: rows,
    columns,
    state: { columnVisibility },
    onColumnVisibilityChange: setColumnVisibility,
    getCoreRowModel: getCoreRowModel(),
  });

  if (rows.length === 0) {
    return null;
  }

  return (
    <div className="relative">
      <div className="mb-2 flex justify-end">
        <button
          type="button"
          onClick={() => setShowColumnPicker((prev) => !prev)}
          className="rounded border border-gray-300 px-3 py-1 text-sm hover:bg-gray-50"
        >
          Columns
        </button>
        {showColumnPicker && (
          <div className="absolute top-8 right-0 z-10 max-h-80 w-64 overflow-y-auto rounded border border-gray-200 bg-white p-2 shadow-lg">
            {table.getAllLeafColumns().map((column) => (
              <label key={column.id} className="flex items-center gap-2 px-2 py-1 text-sm">
                <input
                  type="checkbox"
                  checked={column.getIsVisible()}
                  onChange={column.getToggleVisibilityHandler()}
                />
                {typeof column.columnDef.header === "string" ? column.columnDef.header : column.id}
              </label>
            ))}
          </div>
        )}
      </div>

      <div className="overflow-x-auto rounded border border-gray-200">
        <table className="w-full border-collapse text-left text-sm">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="border-b border-gray-300 bg-gray-50">
                {headerGroup.headers.map((header) => (
                  <th key={header.id} className="px-3 py-2 font-medium whitespace-nowrap">
                    {flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="border-b border-gray-100 hover:bg-gray-50">
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-3 py-2 whitespace-nowrap">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
