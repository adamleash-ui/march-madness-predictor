export interface TeamInfo {
  id: number;
  name: string;
  seed: number;
  region: string;
}

export interface Game {
  t1: TeamInfo;
  t2: TeamInfo;
  prob: number;       // probability of the favored team
  probT1: number;     // probability team 1 wins specifically
  predictedWinner: 1 | 2;
  actualWinner: 1 | 2 | null;
}

export interface Region {
  name: string;
  rounds: Game[][];   // rounds[0] = round of 32, rounds[3] = elite eight
}

export interface BracketData {
  season: number;
  teams: Record<string, string>;
  tourneyTeams: TeamInfo[];
  pairProbs: Record<string, number>;
  regions: Record<string, Region>;
  finalFour: Game[];
  championship: Game | null;
}

export interface PredictResult {
  probT1: number;
  probT2: number;
  favored: "t1" | "t2";
  confidence: number;
  t1: TeamInfo;
  t2: TeamInfo;
}
