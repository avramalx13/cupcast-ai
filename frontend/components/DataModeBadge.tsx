"use client";

import { Database } from "lucide-react";
import { useEffect, useState } from "react";
import { getJson } from "@/lib/api";

type DataStatus = {
  mode: string;
  matches_loaded: number;
  last_validation_valid: boolean;
};

export function DataModeBadge() {
  const [status, setStatus] = useState<DataStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    getJson<DataStatus>("/data/status")
      .then((value) => {
        if (!cancelled) setStatus(value);
      })
      .catch(() => {
        if (!cancelled) setStatus(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const label = status
    ? status.mode === "real"
      ? "API Mode: Real Historical Dataset"
      : "API Mode: Synthetic Demo"
    : "Static Portfolio Demo";
  const detail = status ? `${status.matches_loaded.toLocaleString()} matches` : "API offline";

  return (
    <div className="fixed right-4 top-4 z-30 flex items-center gap-2 rounded-md border border-white/20 bg-[#07120d]/90 px-3 py-2 text-xs font-semibold text-white shadow-lg shadow-black/20 backdrop-blur">
      <Database aria-hidden className="h-4 w-4 text-emerald-300" />
      <span>{label}</span>
      <span className="hidden text-white/50 sm:inline">/ {detail}</span>
    </div>
  );
}
