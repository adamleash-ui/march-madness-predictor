"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, ChevronDown, Zap } from "lucide-react";
import type { BracketData, PredictResult, TeamInfo } from "@/lib/types";
import clsx from "clsx";

function TeamSelect({
  label,
  teams,
  selected,
  onSelect,
  exclude,
}: {
  label: string;
  teams: TeamInfo[];
  selected: TeamInfo | null;
  onSelect: (t: TeamInfo) => void;
  exclude: number | null;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  const filtered = teams
    .filter((t) => t.id !== exclude)
    .filter((t) => t.name.toLowerCase().includes(query.toLowerCase()));

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setQuery("");
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  return (
    <div ref={ref} className="relative flex-1 min-w-0">
      <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-1.5">
        {label}
      </p>
      <button
        onClick={() => setOpen(!open)}
        className={clsx(
          "w-full flex items-center gap-2 px-3 py-2.5 rounded-lg border text-left transition-all",
          "bg-card hover:bg-card-hover",
          open
            ? "border-orange shadow-[0_0_0_2px_#F9731630]"
            : "border-slate-600 hover:border-slate-500"
        )}
      >
        {selected ? (
          <>
            <span className="w-6 h-6 bg-orange text-white text-xs font-bold rounded flex items-center justify-center shrink-0">
              {selected.seed}
            </span>
            <span className="font-semibold text-white truncate">{selected.name}</span>
            <span className="text-slate-500 text-xs ml-auto shrink-0">
              {selected.region} Region
            </span>
          </>
        ) : (
          <span className="text-slate-400">Select a team…</span>
        )}
        <ChevronDown
          className={clsx(
            "w-4 h-4 text-slate-400 shrink-0 transition-transform ml-auto",
            open && "rotate-180"
          )}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 w-full mt-1 bg-card border border-slate-600 rounded-lg shadow-xl overflow-hidden"
          >
            {/* Search input */}
            <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-700">
              <Search className="w-4 h-4 text-slate-400 shrink-0" />
              <input
                autoFocus
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search teams…"
                className="flex-1 bg-transparent text-sm text-white placeholder-slate-500 outline-none"
              />
            </div>

            <div className="max-h-60 overflow-y-auto">
              {filtered.length === 0 ? (
                <p className="px-3 py-3 text-sm text-slate-500 text-center">No teams found</p>
              ) : (
                filtered.map((team) => (
                  <button
                    key={team.id}
                    onClick={() => {
                      onSelect(team);
                      setOpen(false);
                      setQuery("");
                    }}
                    className="w-full flex items-center gap-2.5 px-3 py-2 hover:bg-slate-700 transition-colors text-left"
                  >
                    <span className="w-6 h-6 bg-slate-600 text-slate-300 text-xs font-bold rounded flex items-center justify-center shrink-0">
                      {team.seed}
                    </span>
                    <span className="text-sm text-white font-medium">{team.name}</span>
                    <span className="text-xs text-slate-500 ml-auto">
                      {team.region}
                    </span>
                  </button>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function CountUp({ target, duration = 800 }: { target: number; duration?: number }) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    setValue(0);
    const steps = 40;
    const inc = target / steps;
    let current = 0;
    const id = setInterval(() => {
      current = Math.min(current + inc, target);
      setValue(Math.round(current));
      if (current >= target) clearInterval(id);
    }, duration / steps);
    return () => clearInterval(id);
  }, [target, duration]);

  return <>{value}</>;
}

export default function MatchupPredictor({ data }: { data: BracketData }) {
  const [t1, setT1] = useState<TeamInfo | null>(null);
  const [t2, setT2] = useState<TeamInfo | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);

  const teams = [...data.tourneyTeams].sort((a, b) => {
    if (a.region !== b.region) return a.region.localeCompare(b.region);
    return a.seed - b.seed;
  });

  useEffect(() => {
    if (!t1 || !t2) {
      setResult(null);
      return;
    }
    setLoading(true);
    fetch(`/api/predict?t1=${t1.id}&t2=${t2.id}`)
      .then((r) => r.json())
      .then((d) => {
        setResult(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [t1, t2]);

  const favored = result
    ? result.favored === "t1"
      ? t1
      : t2
    : null;
  const favoredProb = result
    ? Math.round(result.confidence * 100)
    : null;

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Team selectors */}
      <div className="flex items-end gap-3">
        <TeamSelect
          label="Team 1"
          teams={teams}
          selected={t1}
          onSelect={setT1}
          exclude={t2?.id ?? null}
        />
        <div className="pb-2.5 shrink-0 text-slate-500 font-bold text-sm">vs</div>
        <TeamSelect
          label="Team 2"
          teams={teams}
          selected={t2}
          onSelect={setT2}
          exclude={t1?.id ?? null}
        />
      </div>

      {/* Result */}
      <AnimatePresence mode="wait">
        {t1 && t2 && (
          <motion.div
            key={`${t1.id}-${t2.id}`}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3 }}
            className="mt-5 bg-card border border-slate-700 rounded-xl p-5 overflow-hidden"
          >
            {loading ? (
              <div className="flex items-center justify-center h-24 text-slate-400">
                <div className="w-6 h-6 border-2 border-orange border-t-transparent rounded-full animate-spin" />
              </div>
            ) : result ? (
              <>
                {/* Probability bar */}
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-sm font-semibold text-white truncate min-w-0 max-w-[35%]">
                    {t1.name}
                  </span>
                  <div className="flex-1 h-3 bg-slate-700 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: "50%" }}
                      animate={{ width: `${Math.round(result.probT1 * 100)}%` }}
                      transition={{ duration: 0.6, ease: "easeOut" }}
                      className="h-full bg-orange rounded-full"
                    />
                  </div>
                  <span className="text-sm font-semibold text-white truncate min-w-0 max-w-[35%] text-right">
                    {t2.name}
                  </span>
                </div>

                <div className="flex justify-between text-xs text-slate-400 -mt-2 mb-4 px-0">
                  <span className="font-mono">{Math.round(result.probT1 * 100)}%</span>
                  <span className="font-mono">{Math.round(result.probT2 * 100)}%</span>
                </div>

                {/* Result callout */}
                <div className="flex items-center gap-3 bg-navy rounded-lg px-4 py-3">
                  <Zap className="w-5 h-5 text-orange shrink-0" />
                  <div>
                    <p className="text-slate-400 text-xs">Model favors</p>
                    <p className="text-white font-bold">
                      <span className="text-orange">
                        <CountUp target={favoredProb ?? 50} />%
                      </span>{" "}
                      {favored?.name}
                    </p>
                  </div>
                  <div className="ml-auto text-right">
                    <p className="text-slate-400 text-xs">Seeds</p>
                    <p className="text-white font-mono text-sm">
                      #{t1.seed} vs #{t2.seed}
                    </p>
                  </div>
                </div>
              </>
            ) : null}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
