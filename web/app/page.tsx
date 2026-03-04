import { getBracketData } from "@/lib/bracket";
import MatchupPredictor from "@/components/MatchupPredictor";
import Link from "next/link";
import { Trophy, TrendingUp, Database, Target } from "lucide-react";

const STATS = [
  { icon: Target, label: "Accuracy", value: "69.8%", sub: "season-aware CV" },
  { icon: Database, label: "Seasons", value: "21", sub: "years of data" },
  { icon: TrendingUp, label: "Features", value: "22", sub: "matchup metrics" },
  { icon: Trophy, label: "Games", value: "1,382", sub: "in training set" },
];

export default async function HomePage() {
  const data = await getBracketData();

  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden">
        {/* Background glow */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-orange/5 rounded-full blur-3xl" />
        </div>

        <div className="relative max-w-5xl mx-auto px-4 pt-16 pb-10 text-center">
          <div className="inline-flex items-center gap-2 bg-orange/10 border border-orange/20 rounded-full px-4 py-1.5 mb-6">
            <span className="w-2 h-2 bg-orange rounded-full animate-pulse" />
            <span className="text-orange text-sm font-semibold">
              {data.season} Tournament · ML Predictions
            </span>
          </div>

          <h1 className="text-4xl sm:text-6xl font-black text-white leading-tight mb-4">
            Who&apos;s Cutting Down{" "}
            <span className="text-orange">the Nets?</span>
          </h1>
          <p className="text-slate-400 text-lg max-w-xl mx-auto mb-10">
            A stacked ensemble model trained on 21 seasons of NCAA data predicts
            every bracket matchup — down to the final buzzer.
          </p>

          {/* Matchup predictor */}
          <MatchupPredictor data={data} />

          {/* CTA */}
          <div className="mt-8">
            <Link
              href="/bracket"
              className="inline-flex items-center gap-2 bg-orange hover:bg-orange-dark text-white font-bold px-6 py-3 rounded-xl transition-colors"
            >
              <Trophy className="w-5 h-5" />
              See the Full Bracket
            </Link>
          </div>
        </div>
      </section>

      {/* Stats strip */}
      <section className="border-y border-slate-800 bg-card/30">
        <div className="max-w-5xl mx-auto px-4 py-6 grid grid-cols-2 sm:grid-cols-4 gap-4">
          {STATS.map(({ icon: Icon, label, value, sub }) => (
            <div key={label} className="text-center">
              <Icon className="w-5 h-5 text-orange mx-auto mb-1" />
              <p className="text-2xl font-black text-white">{value}</p>
              <p className="text-xs text-slate-400 font-medium">{label}</p>
              <p className="text-xs text-slate-600">{sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="max-w-5xl mx-auto px-4 py-14">
        <h2 className="text-2xl font-bold text-white mb-2">How it works</h2>
        <p className="text-slate-400 mb-8">
          Three models combined into one ensemble — each brings something different.
        </p>
        <div className="grid sm:grid-cols-3 gap-4">
          {[
            {
              title: "Logistic Regression",
              weight: "+2.43",
              desc: "Clean, interpretable. Learns that seed differential and net efficiency are the biggest predictors.",
            },
            {
              title: "Random Forest",
              weight: "+1.09",
              desc: "Captures non-linear interactions. Strong on detecting when hot teams over-perform their seeds.",
            },
            {
              title: "Gradient Boosting",
              weight: "+0.59",
              desc: "Powerful but overfit in season-aware CV — the meta-learner correctly down-weights it.",
            },
          ].map((m) => (
            <div
              key={m.title}
              className="bg-card border border-slate-700 rounded-xl p-5 hover:border-slate-600 transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                <h3 className="font-bold text-white text-sm">{m.title}</h3>
                <span className="text-orange font-mono text-xs bg-orange/10 px-2 py-0.5 rounded">
                  {m.weight}
                </span>
              </div>
              <p className="text-slate-400 text-sm leading-relaxed">{m.desc}</p>
            </div>
          ))}
        </div>

        <div className="mt-6 bg-card border border-slate-700 rounded-xl p-5">
          <h3 className="font-bold text-white text-sm mb-3">
            Top features by importance
          </h3>
          <div className="space-y-2">
            {[
              { name: "Seed differential", pct: 100 },
              { name: "Massey consensus ranking", pct: 78 },
              { name: "End-of-season trend", pct: 40 },
              { name: "Offensive efficiency", pct: 38 },
              { name: "Win rate", pct: 29 },
            ].map(({ name, pct }) => (
              <div key={name} className="flex items-center gap-3">
                <span className="text-xs text-slate-400 w-44 shrink-0">{name}</span>
                <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-orange rounded-full"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="text-xs text-slate-500 font-mono w-8 text-right">{pct}%</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
