import { NextRequest, NextResponse } from "next/server";
import { getBracketData } from "@/lib/bracket";
import type { PredictResult } from "@/lib/types";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const t1id = parseInt(searchParams.get("t1") ?? "0");
  const t2id = parseInt(searchParams.get("t2") ?? "0");

  if (!t1id || !t2id || t1id === t2id) {
    return NextResponse.json({ error: "Invalid team IDs" }, { status: 400 });
  }

  const data = await getBracketData();
  const lo = Math.min(t1id, t2id);
  const hi = Math.max(t1id, t2id);
  const rawProb = data.pairProbs[`${lo}_${hi}`] ?? 0.5;

  // rawProb = P(lo team wins)
  const probT1 = t1id === lo ? rawProb : 1 - rawProb;
  const probT2 = 1 - probT1;

  const t1 = data.tourneyTeams.find((t) => t.id === t1id);
  const t2 = data.tourneyTeams.find((t) => t.id === t2id);

  const result: PredictResult = {
    probT1: Math.round(probT1 * 1000) / 1000,
    probT2: Math.round(probT2 * 1000) / 1000,
    favored: probT1 >= probT2 ? "t1" : "t2",
    confidence: Math.round(Math.max(probT1, probT2) * 1000) / 1000,
    t1: t1 ?? { id: t1id, name: "Unknown", seed: 0, region: "" },
    t2: t2 ?? { id: t2id, name: "Unknown", seed: 0, region: "" },
  };

  return NextResponse.json(result);
}
