import type { ReactNode } from "react";

const NAV_ITEMS = [{ key: "table" as const, label: "Predictions" }];

export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen bg-gray-950 text-gray-100">
      <aside className="w-56 shrink-0 border-r border-gray-800 bg-gray-900 p-4">
        <h1 className="mb-6 text-lg font-semibold">Bert Tunning</h1>
        <nav className="space-y-1">
          {NAV_ITEMS.map((item) => (
            <span
              key={item.key}
              className="block w-full rounded bg-gray-800 px-3 py-2 text-left text-sm text-white"
            >
              {item.label}
            </span>
          ))}
        </nav>
      </aside>
      <main className="flex-1 overflow-x-auto p-6">{children}</main>
    </div>
  );
}
