"use client";

import GameSlot from "./GameSlot";
import type { Region } from "@/lib/types";

const ROUND_LABELS = ["Round of 64", "Round of 32", "Sweet 16", "Elite Eight"];

interface Props {
  regionCode: string;
  region: Region;
}

export default function BracketRegion({ regionCode, region }: Props) {
  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <div className="w-1 h-6 bg-orange rounded-full" />
        <h2 className="text-lg font-bold text-white">{region.name} Region</h2>
        <span className="text-xs text-slate-500 font-mono">{regionCode}</span>
      </div>

      {/* Rounds scrollable horizontally */}
      <div className="overflow-x-auto pb-2">
        <div className="flex gap-3 min-w-max">
          {region.rounds.map((games, roundIdx) => (
            <div key={roundIdx} className="flex flex-col">
              <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2 text-center">
                {ROUND_LABELS[roundIdx] ?? `Round ${roundIdx + 1}`}
              </p>

              {/* Games spaced to align with bracket structure */}
              <div
                className="flex flex-col"
                style={{
                  // Each game occupies a proportional vertical slot
                  // R1: 8 games → base. R2: 4 → 2x slot. etc.
                  gap: `${(Math.pow(2, roundIdx) - 1) * 8 + (roundIdx === 0 ? 6 : 8)}px`,
                  paddingTop: `${(Math.pow(2, roundIdx) - 1) * 4}px`,
                }}
              >
                {games.map((game, gi) => (
                  <GameSlot key={gi} game={game} compact={roundIdx >= 2} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
