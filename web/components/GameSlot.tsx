"use client";

import { CheckCircle, XCircle } from "lucide-react";
import type { Game } from "@/lib/types";
import clsx from "clsx";

interface Props {
  game: Game;
  compact?: boolean;
}

function TeamRow({
  name,
  seed,
  probPct,
  isWinner,
  isCorrect,
  showResult,
  compact,
}: {
  name: string;
  seed: number;
  probPct: number;
  isWinner: boolean;
  isCorrect: boolean | null;
  showResult: boolean;
  compact?: boolean;
}) {
  return (
    <div
      className={clsx(
        "flex items-center gap-1.5 px-2 transition-colors",
        compact ? "py-1" : "py-1.5",
        isWinner
          ? "bg-orange/15 border-l-2 border-orange"
          : "border-l-2 border-transparent"
      )}
    >
      {/* Seed badge */}
      <span
        className={clsx(
          "text-xs font-bold w-5 h-5 flex items-center justify-center rounded",
          compact ? "text-[10px] w-4 h-4" : "",
          isWinner ? "bg-orange text-white" : "bg-slate-700 text-slate-400"
        )}
      >
        {seed}
      </span>

      {/* Name */}
      <span
        className={clsx(
          "flex-1 font-medium truncate",
          compact ? "text-xs" : "text-sm",
          isWinner ? "text-white" : "text-slate-300"
        )}
      >
        {name}
      </span>

      {/* Probability */}
      <span
        className={clsx(
          "font-mono tabular-nums shrink-0",
          compact ? "text-[10px]" : "text-xs",
          isWinner ? "text-orange font-bold" : "text-slate-500"
        )}
      >
        {probPct}%
      </span>

      {/* Result icon */}
      {showResult && isWinner && (
        isCorrect ? (
          <CheckCircle className="w-3.5 h-3.5 text-green shrink-0" />
        ) : (
          <XCircle className="w-3.5 h-3.5 text-red shrink-0" />
        )
      )}
    </div>
  );
}

export default function GameSlot({ game, compact = false }: Props) {
  const { t1, t2, probT1, predictedWinner, actualWinner } = game;
  const probT1Pct = Math.round(probT1 * 100);
  const probT2Pct = 100 - probT1Pct;

  const t1IsWinner = predictedWinner === 1;
  const t2IsWinner = predictedWinner === 2;

  const hasResult = actualWinner !== null;
  const t1Correct = hasResult ? actualWinner === 1 : null;
  const t2Correct = hasResult ? actualWinner === 2 : null;

  // Determine if prediction was right at all
  const predictedCorrectly = hasResult ? actualWinner === predictedWinner : null;

  return (
    <div
      className={clsx(
        "bg-card rounded border border-slate-700/50 overflow-hidden",
        "hover:border-slate-600 transition-all group",
        compact ? "min-w-[140px]" : "min-w-[180px]"
      )}
    >
      <TeamRow
        name={t1.name}
        seed={t1.seed}
        probPct={probT1Pct}
        isWinner={t1IsWinner}
        isCorrect={t1Correct}
        showResult={hasResult}
        compact={compact}
      />
      <div className="h-px bg-slate-700/50 mx-2" />
      <TeamRow
        name={t2.name}
        seed={t2.seed}
        probPct={probT2Pct}
        isWinner={t2IsWinner}
        isCorrect={t2Correct}
        showResult={hasResult}
        compact={compact}
      />

      {/* Probability bar at bottom */}
      <div className="h-0.5 w-full bg-slate-700">
        <div
          className="h-full bg-orange transition-all"
          style={{ width: `${probT1Pct}%` }}
        />
      </div>

      {/* Model accuracy indicator on hover */}
      {hasResult && (
        <div
          className={clsx(
            "text-center py-0.5 text-[10px] font-semibold",
            predictedCorrectly
              ? "bg-green/10 text-green"
              : "bg-red/10 text-red"
          )}
        >
          {predictedCorrectly ? "✓ Correct" : "✗ Upset"}
        </div>
      )}
    </div>
  );
}
