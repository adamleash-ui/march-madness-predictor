import type { BracketData } from "./types";

let _cache: BracketData | null = null;

export async function getBracketData(): Promise<BracketData> {
  if (_cache) return _cache;
  // Works both server-side (fs) and client-side (fetch)
  if (typeof window === "undefined") {
    const { readFileSync } = await import("fs");
    const { join } = await import("path");
    const file = join(process.cwd(), "public", "data", "bracket.json");
    _cache = JSON.parse(readFileSync(file, "utf-8")) as BracketData;
  } else {
    const res = await fetch("/data/bracket.json");
    _cache = (await res.json()) as BracketData;
  }
  return _cache!;
}
