import { getBracketData } from "@/lib/bracket";
import BracketRegion from "@/components/BracketRegion";
import GameSlot from "@/components/GameSlot";
import { Trophy, CheckCircle, XCircle, HelpCircle } from "lucide-react";

export default async function BracketPage() {
  const data = await getBracketData();

  // Calculate accuracy stats
  let correct = 0;
  let total = 0;
  const allGames = [
    ...Object.values(data.regions).flatMap((r) => r.rounds.flat()),
    ...data.finalFour,
    ...(data.championship ? [data.championship] : []),
  ];
  for (const g of allGames) {
    if (g.actualWinner !== null) {
      total++;
      if (g.actualWinner === g.predictedWinner) correct++;
    }
  }
  const accuracy = total > 0 ? Math.round((correct / total) * 100) : null;

  const predictedChamp =
    data.championship
      ? data.championship.predictedWinner === 1
        ? data.championship.t1
        : data.championship.t2
      : null;

  const actualChamp =
    data.championship?.actualWinner != null
      ? data.championship.actualWinner === 1
        ? data.championship.t1
        : data.championship.t2
      : null;

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Trophy className="w-5 h-5 text-orange" />
            <h1 className="text-2xl font-black text-white">
              {data.season} Tournament Predictions
            </h1>
          </div>
          <p className="text-slate-400 text-sm">
            Model picks shown · Orange = predicted winner · ✓/✗ = actual result
          </p>
        </div>

        {/* Accuracy badge */}
        {accuracy !== null && (
          <div className="bg-card border border-slate-700 rounded-xl px-5 py-3 text-center">
            <p className="text-3xl font-black text-orange">{accuracy}%</p>
            <p className="text-xs text-slate-400">
              Model accuracy ({correct}/{total} games)
            </p>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 mb-8 text-xs text-slate-400">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 bg-orange/30 border-l-2 border-orange rounded-sm" />
          Predicted winner
        </div>
        <div className="flex items-center gap-1.5">
          <CheckCircle className="w-3.5 h-3.5 text-green" />
          Correct prediction
        </div>
        <div className="flex items-center gap-1.5">
          <XCircle className="w-3.5 h-3.5 text-red" />
          Upset (model was wrong)
        </div>
        <div className="flex items-center gap-1.5">
          <HelpCircle className="w-3.5 h-3.5 text-slate-500" />
          Result pending
        </div>
      </div>

      {/* Predicted champion callout */}
      {predictedChamp && (
        <div className="mb-8 bg-orange/10 border border-orange/30 rounded-xl p-5 flex items-center gap-4">
          <Trophy className="w-8 h-8 text-orange shrink-0" />
          <div>
            <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider">
              Predicted Champion
            </p>
            <p className="text-xl font-black text-white">
              [{predictedChamp.seed}] {predictedChamp.name}
            </p>
            {actualChamp && (
              <p className="text-sm mt-0.5">
                {actualChamp.id === predictedChamp.id ? (
                  <span className="text-green font-semibold">✓ Model called it!</span>
                ) : (
                  <span className="text-red">
                    ✗ Actual champion: [{actualChamp.seed}] {actualChamp.name}
                  </span>
                )}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Regional brackets */}
      <div className="space-y-10">
        {Object.entries(data.regions).map(([code, region]) => (
          <BracketRegion key={code} regionCode={code} region={region} />
        ))}
      </div>

      {/* Final Four */}
      {data.finalFour.length > 0 && (
        <div className="mt-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-1 h-6 bg-orange rounded-full" />
            <h2 className="text-lg font-bold text-white">Final Four</h2>
          </div>
          <div className="flex flex-wrap gap-4">
            {data.finalFour.map((game, i) => (
              <GameSlot key={i} game={game} />
            ))}
          </div>
        </div>
      )}

      {/* Championship */}
      {data.championship && (
        <div className="mt-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-1 h-6 bg-orange rounded-full" />
            <h2 className="text-lg font-bold text-white">Championship</h2>
          </div>
          <div className="max-w-xs">
            <GameSlot game={data.championship} />
          </div>
        </div>
      )}
    </div>
  );
}
