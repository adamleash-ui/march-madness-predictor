import { NextResponse } from "next/server";
import { getBracketData } from "@/lib/bracket";

export async function GET() {
  const data = await getBracketData();
  return NextResponse.json(data);
}
